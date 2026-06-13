# ResponseMetric Reward RLOO 研究計画 v2

作成日: 2026-06-09（v1 = `rloo_response_metric_reward_research_plan.md` の要点を再構成）

## 0. この文書の目的

2D CAD sketch の primitives が与えられたとき、その間に入る **design-intent-aware な
constraints** を推論するタスクを RLOO で強化学習する。design intent の入れ方は多義的
（同じ意図でも annotation が違う）なので、その差分を吸収しつつ「人間 response との整合」を
測れる指標 **motion / leak / response angle** を設計し、これを報酬に使う。

本文書の主張は2層:

1. **評価器として**: `response_full_angle`（canonical 指標）は ablation で想定通りに振る舞う
   （§2）。これは評価器単独の主張であり、RL の成否に依存しない。
2. **報酬として**: 同一足場上の ablation で、`motion_relational_angle` 報酬が
   design-intent 側へ方策を動かす（§3）。small scale で「強い成功」を確定（§3.3）、
   現在は全データ scale での挙動を確認中（§4）。

> v1 にあった `d_feas` の連続性問題、`d_H` / `d_H_angle` の命名探索、案 A–O の比較などの
> 過去の試行錯誤は本 v2 では省く。現行の指標名は下記 §1 の two-axis 命名に統一する。
> （`d_sub_angle`→`motion_full_angle`、`leak_angle`→`leak_relational_angle`、
> `d_H_angle`→`response_full_angle` に対応。過去ログは v1 / `summary_*_v4*.md` 参照。）

実装: [src/models/rewards/response_metric_reward.py](../src/models/rewards/response_metric_reward.py)

---

## 1. 指標の定義: motion / leak / response angle

すべて Pred sketch と GT sketch の **1次応答（first-order response）** から作る部分空間量で、
すべて `[0, 1]` スケール（低い = 人間 response に整合）。命名規約は `<component>_<scope>_angle`。

### 1.1 構成要素

各 sketch の constraint 集合は、native dimensional parameter の変化を primitive の運動へ
写す応答写像 `R_C = -pinv(J_x) J_p` を定める。比較は primitive 座標そのものではなく、
`J_phi`（pairwise-distance observable。Circle/Arc は radius scalar を追加）で射影した
**共通観測空間**で行う（Pred と GT で primitive 数が違っても比較可能にするため）。

- **`motion_full_angle`** — 全運動部分空間 `W_C = range(J_phi R_C)` の食い違い。
  dimensional driver と relational null-space の両方を含む。padded principal angle:

  ```
  motion_full_angle² = 1 − Σ_i cos²θ_i / max(dim W_P, dim W_G)
  ```

  次元差は 90°（直交）として `max(dim_P, dim_G)` 正規化で罰す
  → **under/over-dimensioning に感応**。

- **`motion_relational_angle`** — 報酬用の sibling。dimensional driver を意図的に無視し、
  relational constraints が許す運動 `null(J_R)` のみを同一 state 上で比較。
  raw 角度を mode 固有の最大値で正規化して `[0, 1]` に乗せる。
  既定 mode は `symmetric`:

  ```
  raw² = (dim_P + dim_G − 2 Σ cos²θ_i) / max(dim_P, dim_G)
  ```

  他 mode: `pred_subset_gt`（Pred⊄GT 方向のみ＝Pred の関係不足を罰す）、
  `mixed_subset`（GT⊄Pred を `reverse_weight` で加重）。
  → **dimensioning placement には構造的に盲目**（意図的。多義的な placement 差を吸収）。

- **`leak_relational_angle`** — Pred が許す運動が GT relation の禁止方向へ漏れる量。
  scale-free な RMS cosine:

  ```
  leak² = ‖orth(row(J_R_G))ᵀ U_P‖_F² / dim(U_P)
  ```

  → **relational mismatch に感応**（部分空間の傾き差を拾う）。

### 1.2 合成スカラ

2成分（motion + leak）は同じ angular 単位なので RMS で合成し、`/√2` で `[0, 1]` に正規化:

- **`response_full_angle` = RMS(`motion_full_angle`, `leak_relational_angle`)**
  — canonical 指標（**評価器としての本指標**）。coverage（under/over-dim）も relational
  correctness も両方拾う。
- **`response_relational_angle` = RMS(`motion_relational_angle`, `leak_relational_angle`)**
  — 報酬の既定値。dimensional placement への一致を直接 reward しないことで、ある特定の
  annotator の寸法の置き方に過適合するのを避ける。dimensional parsimony / FC は composer の
  sibling reward（SolverReward, DimParsimony）が担当する。
- **`response_relational_weighted_angle` = (1−w)·`motion_relational_angle` + w·`leak_relational_angle`**
  — 上の重み付き変種（2026-06-12 追加）。RMS の 50:50 固定だと、leak channel を「寸法ゼロ」で
  自明に 0 にする under-game の利得が大きすぎる（§4.4 scaffold10 collapse）ため、混合比 `leak_weight`
  を config knob 化した線形版。線形なのは説明性のため（no-dims の利得が正確に `w×leak` に比例）。

### 1.3 報酬化

`ResponseIntentReward` は選択した1つの angle に clip 済み線形 bonus を掛け、生成系列の最初の
STOP token に置く。`gate`（`fc` / `converged` / `always`）で solver status により採点を絞る。
既定 reward metric は `response_relational_angle`、実験では `motion_relational_angle` を多用。

---

## 2. ablation で想定通りに振る舞うこと（評価器の検証）

検証データ: 6 batches × 5 humans、有効 rotation 396（heldout）、Vitruvion model paired
base=307 / solver_rl=366。詳細・全候補比較は
[summary_histcad_response_metric_v4_results.md](summary_histcad_response_metric_v4_results.md)。
以下は現行命名に読み替えた要点。

### 2.1 劣化に対する単調性（dim/rel dropout sweep）

GT から dimensional / relational constraint を段階的に落とすと、指標が単調に悪化する
（median, n=396）:

| variant | motion_full_angle | leak_relational_angle | response_full_angle |
|---|---:|---:|---:|
| source_self / heldout | 0.000 | 0.000 | 0.000 |
| dim_drop25 | 0.500 | 0.000 | 0.561 |
| dim_drop50 | 0.707 | 0.000 | 0.707 |
| dim_drop100 | 1.000 | 0.000 | 1.000 |
| rel_drop50 | 0.662 | 0.658 | 0.939 |
| rel_drop100 | 0.786 | 0.792 | 1.117 |

**成分の役割分担が想定通り**: dimension を落とすと `motion_full_angle` だけが立ち
`leak` は 0、relation を落とすと `leak` が立つ。coverage と relational correctness が
別チャネルに分離している。

### 2.2 人間 vs model の方向と強度（paired Wilcoxon）

`z > 0` = 人間優位（lower-is-better）。

| model | metric | z | p |
|---|---|---:|---:|
| base (under-constrained) | motion_full_angle | +13.34 | 1.5e-40 |
| base | response_full_angle | +13.49 | 1.7e-41 |
| solver_rl (fully-constrained, RL) | motion_full_angle | +11.96 | 6.1e-33 |
| solver_rl | response_full_angle | +12.59 | 2.4e-36 |
| solver_rl | leak_relational_angle | +13.25 | 4.6e-40 |

base / solver_rl の両方で人間優位を強く検出。特に solver_rl（fully-constrained で
token metric では人間に近い）でも、relational mismatch を `leak_relational_angle` が
最強に拾う。HistCAD OES のような sign inversion は起きない。

### 2.3 annotator variance の吸収

heldout（人間どうし）の median は全成分で 0、`response_full_angle` の near-zero 率
（frac<1e-4）は約 0.53。残る tail は「一方が冗長な dim を付けた」等の**本物の構造差**で、
指標の失敗ではなく真の annotation 差分の捕捉と読める。

### 2.4 per-rotation 連続性

dim_drop を 10% 刻みで振った per-rotation 軌跡で、旧 `d_feas`（gain が binary）より
滑らか（distinct values/rotation 4→6、monotonic transition 34%→49%）。

### 2.5 設計上の役割の裏取り（§19.3 の RL probe）

「`motion_full_angle`（=full）は under/over に感応、`motion_relational_angle`（=rel）は
寸法盲目」という設計を、学習側でも確認した。FC-bonus を 0 にして関係偏重にした run
(M4) は distFam~0.02 / keyrel 最高（関係は完璧）なのに **`response_full_angle` 最悪 0.660**、
一方 `motion_relational_angle` は 0.417 で良いまま。**寸法盲目な指標を最適化すると喜んで
under-constraint に落ちる**ことの実データ証拠で、`response_full` がそれを正しく罰すことを示す。

> **重要な但し書き（§4.2 で詳述）**: `response_full` のこの coverage 感応は、val GT が人間
> （= 寸法フル拘束）であることに由来する。同じ性質ゆえ、**`response_full` は over-dim 側にも
> 反応せず、むしろ「人間の寸法付けの再現」= over-dimensioning を *報酬* する**（`motion_full`
> は `dimP≈dimG` で最小）。したがって `response_full` は評価器としては coverage を正しく測るが、
> **over-dim を抑える学習実験の成否判定・checkpoint 選択には使えない**。そこでは
> `distFam` / `motion_relational` / `leak`（leak は寸法増で悪化＝over-dim を正しく罰す）で判定する。

---

## 3. small scale で報酬が効くこと

### 3.1 実験構造

同一「足場（scaffold）」の上で intent reward の有無だけを ablation する。足場 =
SolverReward + dimensional hygiene penalty（極小 Distance / axis-Angle を罰す）+
DimParsimony + RelationalClassPrior（rare relational token を探索空間に出す prior）。
足場は「Equal / Tangent / Parallel 等の候補を方策分布に乗せる」ためのもので、それ自体は
design intent を証明しない。比較は必ず同一 step で行う。

報酬 GT は **rule-augmenter で生成した pseudo-GT**: 人間 annotation は疎なので、primitives の
幾何から成り立つ関係を述語で決定的に検出し closure を取って relational constraints を充足する。
fiverr 人間データはベンチマーク（評価専用）で学習・報酬 GT には使わない。

**relation type の取捨**: false positive（幾何的にはほぼ成立するが意図でない検出）を懸念して
**Symmetric と Equal を pseudo-GT から抜くことを検討した**が、**現在は全 type を入れている**。
そのほうが生成が relational 多めへうまく誘導されたため（多少の false positive より、rare
relational を方策分布に乗せる効果が勝った）。

### 3.2 第一段階: 「弱い成功」（v1 GT, scaffold）

`motion_relational_angle` を報酬化した S+R は、no-intent の S に対し:

- canonical `response_full_angle` ↓（0.495 vs 0.526）、`leak_relational_angle` ↓、
  `motion_relational_angle` ↓ をすべて一貫して改善。
- 生成側で dimensional（特に Radius）を抑制、Equal を復活、rare rel（Eq+Tan+Par+Perp）を
  ~2.2×。
- ただし FC が ~5pt 低い（no-intent は Radius 等で FC を稼ぐ。intent はそれを抑えるので
  関係で賄いきれない分だけ FC が下がる）= 典型的な trade-off。

報酬は意図通りに効くが、絶対品質は不十分で単独主張としては弱い。律速を
「探索オフ（mode-collapse）」「hygiene 弱」「報酬 GT の不完全さ（per-sketch 推論漏れ）」
と特定し、探索ノブ ON（entropy_bonus, temperature↑）、hygiene 強化、pseudo-GT v2（closure で
per-sketch 完全化）、flip による counterfactual 生成、reward rebalance を導入した。

### 3.3 第二段階: 「強い成功」を shuffle 負例対照で確定

足場を v2 GT × flip × rebalance × `symmetric` に固定し、intent を {ON / OFF / shuffled} で
ablation（shuffled = `shuffle_within_group=True`。per-sample 報酬を RLOO group 内で
permutation し、報酬 scale と group baseline は保ったまま **sample↔GT 対応だけを破壊**）。

同一 step (~15.6k) 比較（低い方が良い: resp/mrel/leak/distFam、高い方が良い: keyrel/FC）:

| run | response_full | motion_rel | leak | FC | distFam | keyrel |
|---|---:|---:|---:|---:|---:|---:|
| **S+R** intent | **0.428** | **0.405** | **0.391** | 0.503 | **0.217** | **0.428** |
| S no-intent | 0.475 | 0.442 | 0.447 | 0.570 | 0.246 | 0.413 |
| S+R̃ shuffled | 0.477 | 0.439 | 0.434 | 0.597 | 0.241 | 0.417 |

**決定的な読み**: S+R̃（対応を壊した版）は全 design-intent 軸で no-intent S に潰れる。
intent の改善は **scale を揃えて対応だけ壊すと消える** → 改善は「各サンプルを自分自身の
motion-subspace 標的に向ける GT 対応信号」に由来し、報酬の大きさ/分散ではない。これは
「強い成功」の署名。FC は intent ありで下がる（trade-off、想定済み・guardrail 扱い）。

補足: headline の `response_full_angle` は訓練報酬（`motion_relational_angle`）とは別 metric の
sibling であり、報酬が直接最適化しない非報酬指標（distFam/keyrel/FC）も同方向に動くので、
自己評価のトートロジーではない。

**この時点で補助 RL 実験としての主張（report 用）は確定**。

---

## 4. 現状: 全データ scale での挙動確認（2026-06-09）

「強い成功」を確定後、production として全データ scale（v2 closure 1.67M, `scaffold7`）へ。
そこで **full data だと distFam の intent 効果が希釈し、定性的にも dimensioning 過多**が出た。

切り分け（matched-step gap, intent − no-intent）:

| | mrel Δ | response_full Δ | distFam Δ | keyrel Δ |
|---|---:|---:|---:|---:|
| small 終盤 | −0.041 | −0.041 | −0.026 | +0.017 |
| full 終盤 | −0.039 | −0.046 | −0.006 | +0.005 |

- **intent が直接最適化する角度軸（mrel / response_full）の gap は full でも減衰しない（−0.04）**。
  希釈したのは構成比軸（distFam / keyrel）だけ。
- distFam 希釈は **データ量でなく FC 収束の関数**（full は長く回り FC 0.66–0.75 と高く、
  両アームが寸法を積む）。「GT が 0 dim で閉じる所をモデルが関係で取り切らず寸法で代用」する
  substitution headroom が実在し、回収可能。
- ngrp check: 全 substitution relation で counterfactual が揃い（PASS）、Parallel/Perp/Equal の
  within-group advΔ は既に正 = 報酬は正しく関係を選ぶ。律速はミスランクでなく
  **サンプリング頻度**（Distance 71% vs Equal 25% / Tangent 12%）。Tangent は solver-risk で
  死亡（別問題）。

### 4.1 次手: FC-bonus を主レバーとした operating point 探索

- **over↔under は FC-bonus 一軸の trade-off**（2.0 = over-dim / 0 = under-constraint 崩落）。
  intent も dof_reduction も「閉じ手段」に無関心なのでこの軸を差別的に動かせない →
  両極とも不可、**中間点を探す**。
- 固定 FC で distFam を下げるには「閉じ手段を関係へ寄せる関係優先レバー」（intent 増幅 or
  DimParsimony）が要る。FC-bonus を戻すと「閉じよ＋でも関係で」が拮抗する領域を狙う。

投入中の 5 config（`scaffold8`, 全 intent-on）:

| # | config | FC | DimParsimony | coeff_h | 役割 |
|---|---|---:|---:|---:|---|
| 1 | `rloo_response_scaffold8_fc05_intent` | 0.5 | 0.30 | 3 | FC sweep 下 |
| 2 | `rloo_response_scaffold8_fc10_intent` | 1.0 | 0.30 | 3 | FC sweep 中（hinge） |
| 3 | `rloo_response_scaffold8_fc15_intent` | 1.5 | 0.30 | 3 | FC sweep 上 |
| 4 | `rloo_response_scaffold8_fc10_hicoeff_intent` | 1.0 | 0.30 | **6** | FC=1.0 で intent-bend |
| 5 | `rloo_response_scaffold8_fc10_dimp09_intent` | 1.0 | **0.90** | 3 | FC=1.0 で parsimony-bend |

判定: (1,2,3) で `solver_fc_rate` の operating curve を描き健全 FC（>~0.4–0.5）に戻る点を
特定。(2 vs 4 vs 5) で同一 FC=1.0 のもと `eval_distance_family_ratio_mean` を最も下げる
レバーを分離。**FC を保ったまま distFam↓** なら成功。判定軸は **distFam / motion_relational /
leak / FC**（**`response_full` は使わない** — §4.2 の通り pro-dim バイアスで over-dim ほど良く
見えるため）。under へ落ちていないか（`solver_under_rate` / `solver_valid_dof_mean`）も併読。
当たりの動作点で改めて intent on/off（+shuffled）を同一足場上で回し claim。

> claim の headline は relational 軸（motion_relational / leak / keyrel）に置き、distFam は
> 二次軸とする（§3.3 の整理を維持）。DimParsimony を両アームで強めると distFam の intent gap を
> 食う恐れがあるため。

### 4.2 結果と発見: FC が over↔under の主軸 / `response_full` は over-dim を報酬する

scaffold8 sweep（早期 step、W&B id は v1 §20.1）の operating curve:

| run | FC | DoF | distFam | leak | mfull | **dimP** |
|---|---:|---:|---:|---:|---:|---:|
| sc7 base (FC2.0) | 0.67 | 3.92 | 0.248 | 0.415 | 0.461 | **7.21** |
| fc15 | 0.44 | 5.57 | 0.207 | 0.368 | 0.465 | 5.83 |
| fc10 | 0.31 | 6.70 | 0.177 | 0.332 | 0.491 | 4.87 |
| fc05 | 0.25 | 6.53 | 0.161 | 0.331 | 0.500 | 4.64 |
| hicoeff(c6) | 0.17 | 6.92 | 0.144 | 0.295 | 0.527 | 4.16 |
| (GT dimG) | | | | | | **5.72** |

- **FC↑ で distFam↑ が単調**（fc05 0.161 → sc7 0.248）= over↔under は FC-bonus 一軸という §4.1 を実測確認。
  FC=0（§v1 19.3）は全 run が FC~0.05 / under~0.9 へ崩落、parsimony 単独強化（dimp09）も under へ落ちる。
- **【発見】`motion_full` / `response_full` は over-dimensioning を *報酬* する**: val GT = 人間 fiverr（寸法フル拘束、
  `dimG=5.72`）なので、`mfull` は **`dimP≈dimG` で最小**（dimP 7.21→4.16 に対し mfull 0.461→0.527 と単調悪化、sc7 が最小寄り）。
  ＝関係的（寸法少）な Pred ほど人間の駆動部分空間を張れず `mfull` が悪化。これは paper §6 が `motion_full` を学習報酬に
  *使わなかった* 理由そのもの。**逆に `leak` は寸法増で悪化（sc7 0.415 → hicoeff 0.295）= over-dim を正しく罰す唯一の angle 軸**。
  `respF = RMS(mfull[pro-dim], leak[anti-dim])` は打ち消し合い平坦（収束 run 0.43–0.45）→ 分解して `leak` を使う。
- 学習報酬 `motion_relational_angle` は寸法盲目なので無傷。問題は eval 指標の選択であって報酬設計ではない。

### 4.3 scaffold9: `leak` を substitution-direction 報酬に（DimParsimony は落とす）

`leak² = ‖range(R_pred) を GT 関係 row 空間へ射影‖²_F / dim(range(R_pred))` より、**leak は Pred の寸法ドライバのうち
GT が関係で閉じる方向を駆動している割合＝substitution-Distance を特異的に罰す**（正当寸法は GT 関係空間と直交し leak≈0）。
これを報酬化（`motion_relational_angle` → `response_relational_angle` = RMS(motion_relational, leak)）すれば over-dim
（substitution）に効き、かつ FC-bonus を原状(2.0)へ戻して収束を速められる。

**DimParsimony は落とす**（leak と相補でなく、もはや不要〜微有害）:
- leak を希釈する経路（正当寸法で分母増）は、冗長寸法では rank が増えず希釈不能＋SolverReward redundant penalty(−1.0) が直撃。
  rank を増やすのは正当寸法だけ＝望ましい closure。→ **有害な希釈経路は無い**＝DimParsimony を希釈の番人とする理由が無い。
- 役割: substitution=leak / 冗長=SolverReward redundant / no-dims への under-game=**FC-bonus**。DimParsimony は寸法数を GT≈0 へ
  寄せる under-pull で微有害。**leak+FC が揃えば DimParsimony は冗長**。

scaffold9 3 config（anchor=sc7 `itboolr0`、各段1変更で attribution）:

| config | FC | metric | DimParsimony |
|---|---:|---|---|
| `rloo_response_scaffold9_leak_fc20_dimP_intent` | 2.0 | response_relational | あり0.30（=sc7+leak のみ、leak 純効果） |
| `rloo_response_scaffold9_leak_fc20_intent` | 2.0 | response_relational | **なし**（主筋・最速） |
| `rloo_response_scaffold9_leak_fc15_intent` | 1.5 | response_relational | なし |

canary: `leak`↓ かつ `distFam` < sc7 0.248（FC 維持）／under 暴騰なし（FC が番人）／drop 側で `solver_redundant_rate` が
上がらない（redundant penalty が番人の実証）。判定は distFam/mrel/leak/FC。caveat: response_relational は mrel:leak=50:50 固定、
over-rotate なら leak 小係数化のコード改修へ。
→ 結果は §4.4–4.5。

### 4.4 結果: leak は最強レバーだが「増幅＝崩壊」、s9 自体も軽症の同病理 (2026-06-12)

**判定枠組み = matched-FC operating curve**（閉じ途中なので生 distFam でなく同 FC 率での distFam で比較）:

| レバー | 同FC点の distFam（vs sc7 曲線） | 判定 |
|---|---|---|
| FC bonus sweep（s8_fc05 @FC0.40: 0.199） | シフトゼロ | **曲線上の移動のみ。レバーでない** |
| intent 増幅（s8_hicoeff @FC0.31: 0.155） | −0.03 | 曲線を下げる（中） |
| **leak**（s9_fc20 @FC0.29: **0.104**） | **−0.06〜0.08** | **最大のレバー** |

**leak 報酬（scaffold9, coeff3, DimParsimony drop）は曲線を最も下げる**。DimParsimony drop も裏取り済み
（redundant 率 0.03–0.05 で暴れず＝SolverReward redundant penalty が冗長寸法の番人）。

**ただし s9 は本命ではなく軽症の崩壊**（§4.5 で確定）。s9_fc15 @2493: FC 0.52 / **DoF mean 7.5**。FC 半分が DoF≤2 なので
平均 7.5 は残り ~48% が DoF ~14＝**深く未拘束**。distFam/leak の低下は「関係で置換」でなく「閉じるのをやめた」分が混入。

**増幅の試み（scaffold10/11）は全て no-dims collapse**（s10_c6 / s11_c6: FC ~0.02 / dimP ~0.1 / distF ~0.005＝寸法を全放棄）。
機序: 増幅した intent の大域最大は **「寸法ゼロの under-constrained 隅」**（leak=0 が自明、motion_relational も relations だけで良好）で、
`gate=converged` がその隅を報酬する。**leak_weight を 0.25 へ下げても効かず**（原因は leak の重みでなく coeff×gate）。
※この過程で **`response_relational_weighted_angle` = (1−w)·motion_relational + w·leak を実装**（`leak_weight` knob、線形なのは
説明性＝no-dims 利得が `w×leak` に正確に比例。[response_metric_reward.py](../src/models/rewards/response_metric_reward.py)）。増幅路線自体が
筋悪と判明したので、leak_weight は今後の微調整用に保持。

**補助発見**:
- `train/reward_contrib_*_seq_mean` は gate 発火/飽和/スケール監視に有用（sc7 solver:intent 3:1、hicoeff 1:1）。勾配シェアの
  精密測定は `diag_sample_vs_select.py` の per-reward contrast が担当。
- 人間 dim 比率 24.8%（paper §3）と比較すると、モデルは sc7 0.315 と**高い上に閉じ効率が悪い**（人間 FC100%@24.8% vs
  sc7 FC67%@31.5%）。＝正方向は「寸法を増やす」でなく**「寸法を減らして閉じを上げる」**。over-dim の定性印象は count でなく
  substitution/placement 由来。

### 4.5 統一診断: close-or-abandon 二峰 と leak_gate=dof 治療（scaffold12）(2026-06-13)

s9→s10→s11 は **同一病理の連続体**: 学習報酬の標的（v2 closure）が relational-only＝under-constrained なので、relational 報酬は
**under-constrained 出力に寄る**。coeff がそれを増幅（s11 で完全崩壊）し、coeff3 でも s9 の DoF 7.5 として既に出ている。

**二峰を DoF 分布で裏取り**（[scripts/diag_dof_distribution.py](../scripts/diag_dof_distribution.py)、fiverr val 全件、val は dof Counter を
落とすので ckpt から再生成。[artifacts/dof_distribution/dof_distribution.png](../artifacts/dof_distribution/dof_distribution.png)）:

| ckpt | FC峰(≤2) | 中間(3-6) | 放棄(≥7) | mean |
|---|---:|---:|---:|---:|
| sc7（過剰寸法） | **0.64** | 0.20 | 0.15 | 4.36 |
| s8_fc05（中間） | 0.36 | 0.42 | 0.21 | 5.06 |
| s11_c6（崩壊） | 0.02 | 0.16 | **0.82** | 11.27 |

**2 つの mode が明確**: FC mode（DoF≤2）と**放棄 mode（DoF ~8–12 の山）**。closure 圧を弱めるほど質量が前者→後者へ移る。
放棄 mode は DoF 8–12＝「設計寸法を数本開ける」(DoF 3–5) でなく**深い未拘束**＝明確な放棄（s9 はこの中間で最も二峰的）。
機序: `gate=converged` 下で、**閉じるのに leaky な substitution-Distance が要る「難しい」スケッチを方策が放置する**（開けた関係
スケッチでも leak=0・mrel 良好で報酬が取れるため）。`solver_valid_dof_mean` が真の健全指標（fc_rate は二峰を隠す）。

**治療 = `leak_gate=dof`（成分別・連続 gating。当初案の「報酬全体を gate=fc」は誤り）**。崩壊の縮退最小点を持つのは **leak だけ**
（`leak=0` は寸法ゼロで自明達成＝under 隅）。**motion_relational は縮退最小点を持たない**（GT に合わせるには正しい関係を実際に
出す必要があり、空では null space が全空間＝悪化。実証: s8_hicoeff は motion_relational 単体・coeff6 で崩れない）。

初期案の **`leak_gate=fc`** は no-dims Under 隅への leak=0 ボーナスを消す点では正しいが、学習初期に FC サンプルがほぼ無いため
leak が効かない。また hard FC cliff は DoF≈3 から一気に FC 境界を越える over-dim 圧を作り得る。そこで
**motion_relational は常時 `gate=converged`（Under も採点）、leak だけ `gate=dof`**にする:

```text
w_leak(sample) = clip((orig_dof - DoF) / (orig_dof - fully_constrained_dof), 0, 1)
leak_bonus = w_leak * coeff_leak * (1 - clip(leak)/cap)
```

これで no-dims Under 隅は leak 報酬をほぼ得られず、閉じに近づくほど substitution 抑制が徐々に効く。「常に欲しい関係信号は
dense に保ち、leak は closure progress に応じて ramp」することで over↔under を分離する。

**実装初版**（[response_metric_reward.py](../src/models/rewards/response_metric_reward.py)）: 線形
`response_relational_weighted_angle` に **`leak_gate` パラメータ**を追加（default=`gate`＝後方互換）。この後、§4.6 の
producer/consumer refactor で motion/leak を別報酬へ分離し、`LeakAngleReward(gate=dof)` が `SolverReward` 由来の
`per_sample_dof_reduction_rate` を読む形に更新した。

**scaffold12 = `leak_gate=dof` の dose-response**（motion gate=converged / leak gate=dof / w=0.25 / FC1.5 / DimParsimony なし。
判定主指標 = `solver_valid_dof_mean`↓ ＋ FC↑＝放棄峰の解消、distFam/leak < sc7 維持、dimP≥~2 が collapse 警報）:

| config | motion coeff | leak coeff | 役割 |
|---|---:|---:|---|
| `rloo_response_scaffold12_m2.25_l0.75_leakdof_intent` | 2.25 | 0.75 | **最優先**: s9 と同 total coeff=3 で放棄峰(DoF7.5)が lift するか（増幅と分離） |
| `rloo_response_scaffold12_m4.5_l1.5_leakdof_intent` | 4.5 | 1.5 | leak_gate=dof 下の安全な増幅（total coeff=6、s11_c6 は崩壊した） |
| `rloo_response_scaffold12_m4.5_l1.5_leakdof_dimprior_intent` | 4.5 | 1.5 | 上記 + DimPrior（type-mix polish） |

**no-leak 対照は既存 s8_hicoeff**（`kmcne55j`、motion_relational+DimParsimony, gate=converged）をそのまま使う＝gating 変更不要
（leak が無いので縮退最小点も無い）。s8_hicoeff の DoF6 二峰は leak 由来でなく FC-bonus 圧不足由来。scaffold12 vs s8_hicoeff で
「leak（leak_gate=dof）が substitution 抑制を上乗せするか」を読む。完走後 `diag_dof_distribution.py` に ckpt を足して
**「gate=converged 二峰 → leak_gate=dof 単峰化」figure**（paper 用）を作る。

**もし leak_gate=dof も二峰を治せない / distFam が戻るなら**: 増幅・leak 路線は打ち止め、**s9 系を本命に固定し claim ablation
（intent on/off/shuffled を s9 足場で）へ**移る（s9 は不完全でも sc7 を全関係軸で上回る）。

### 4.6 実装済み refactor: `ResponseIntentReward` を motion / leak に分離 (2026-06-13)

§4.5 の `leak_gate` 初版は `ResponseIntentReward` 内で motion と leak を再重み付けしていたが、実験ノブが
`coeff_h_angle × leak_weight × gate/leak_gate × BaseReward.weight` に分散し、どの成分の寄与か読みにくくなった。そこで
`ResponseIntentReward` は deprecated とし、以下の producer/consumer 構成へ移行した。

- `ResponseMetricEvaluator(name="response_eval")`: 重い Hθ 計算を一度だけ実行し、`ctx.h_theta_metrics[i]` に
  `HThetaMetrics` を cache する。訓練報酬への寄与は 0。既存の `intent_*` eval/log metric はここが維持する。
- `MotionAngleReward(name="motion_intent")`: cache から `motion_relational_angle` を読み、STOP 位置に
  `coeff * (1 - clip(angle)/cap)` を置く。既定 gate は `converged`。
- `LeakAngleReward(name="leak_intent")`: cache から `leak_relational_angle` を読み、同じ線形 bonus を置く。
  `gate=dof` では `ctx.per_sample_dof_reduction_rate` をサンプルごとの連続重みにする。

線形 weighted 版は、通常域（各 angle ≤ cap）では
`MotionAngleReward(coeff=(1-w)C) + LeakAngleReward(coeff=wC)` として厳密に再パラメータ化できる。Under サンプルで
旧 `include_leak=False` が motion に full coeff を与えていた過大評価は、分離後は意図的に除去される（Under は motion coeff 分だけ）。

移行済み config 名は component coeff を明示する形へ変更した。

| old | new |
|---|---|
| `rloo_response_scaffold12_leakgatefc_c3_intent` | `rloo_response_scaffold12_m2.25_l0.75_leakdof_intent` |
| `rloo_response_scaffold12_leakgatefc_c6_intent` | `rloo_response_scaffold12_m4.5_l1.5_leakdof_intent` |
| `rloo_response_scaffold12_leakgatefc_c6_dimprior_intent` | `rloo_response_scaffold12_m4.5_l1.5_leakdof_dimprior_intent` |
| `rloo_response_scaffold12_hicoeff_dimprior_intent` | `rloo_response_scaffold12_m6.0_dimprior_intent` |

検証: `tests/test_response_metric_reward_split.py` で gate / STOP placement / FC 通常域の weighted identity /
Under の期待差分 / composer cache propagation を固定。`tests/test_pairwise_distance_observable.py` と合わせて通過。
4 つの scaffold12 config は Hydra compose + reward/val composer instantiate 済み。

---

## 5. 補足: motion_angle が触れない軸（参考）

`motion_relational_angle` は dimensioning **placement**（どの primitive 対に寸法を置くか）に
構造的に盲目（§1.1、意図的）。この軸を測るため別指標 **participation**（dim_uglify 合成統制で
p=3.8e-59）と **set_sim**（motion-basis 一致）を検討したが、いずれも GT 慣習依存・count
混同で実 model 評価には使えず、diversity descriptor 止まり。**ランキングの主指標は
`motion_full_angle` + `leak_relational_angle` のまま**。詳細は v1 §17 / §18。

> 更新 (2026-06-10, v1 §18.8): set_sim の max-matching を Hungarian 1:1 割当に替えた
> **`asgn_sim`**（max-count 正規化）が dim/rel drop 単調・model 分離 z≈+12 で、しかも
> `motion_full` と逆に **over-dimensioning を正しく罰す**（余剰列が割当から溢れる）。§4 系の
> over-dim 抑制実験の eval / checkpoint 選択の併置候補。placement 純粋軸は `asgn_min`
> （hh-vs-hu 0.666 / multi-ref 0.703、combo 0.736）。新 intrinsic 軸 amp（レバー増幅、統制域
> AUROC 0.71 で歴代最強）は coverage 交絡で model-eval 不可＝thesis instrument 限定。
