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

### 4.7 結果: motion+leak+dimprior が暫定ゴール、各成分の役割が分離できた (2026-06-15)

**暫定の最終 config = `rloo_response_scaffold12_m4.5_l1.5_leakdof_dimprior_intent`（`28xstb2f`）**
＝ scaffold（v2GT × flip × FC1.0 × DimParsimony0.30 × RelClassPrior）の上に
**MotionAngle(coeff4.5, gate=converged) + LeakAngle(coeff1.5, gate=dof) + DimPrior(Distance−/DistanceX,Y+/Angle−)**。

**(1) refactor は挙動保存（§4.6 の健全性確認）**。`m6.0_dimprior` から DimPrior だけ抜いた対照
`rloo_response_scaffold12_m6.0_nodimprior_intent`（`srips9h5`）が、旧 `ResponseIntentReward` 版の
`scaffold8_fc10_hicoeff`（`kmcne55j`）を matched-step で再現（DoF/FC/distFam/Dist が seed 差内で一致）。
→ `ResponseIntentReward → ResponseMetricEvaluator+MotionAngleReward` の分解は等価。以降の差は報酬成分に帰属できる。

**(2) DimPrior の効果＝Distance→軸寸法の型切替（同 coeff で DimPrior だけ違う 2 本の matched 比較）**。
`28xstb2f`(DimPrior有) vs `oyvonqrm`(=`m4.5_l1.5_leakdof`, DimPrior無)、step≈25k:

| run | DoF | FC | distFam | leak | **Dist** | DistX | DistY |
|---|---:|---:|---:|---:|---:|---:|---:|
| `28xstb2f` (M+L+D) | 7.25 | 0.43 | 0.131 | 0.167 | **0.000** | 0.095 | 0.081 |
| `oyvonqrm` (M+L) | 7.92 | 0.39 | 0.108 | 0.119 | **0.098** | 0.021 | 0.025 |

同じ FC/DoF 帯で、DimPrior の有無だけが raw Distance を 0.098→**0.000**、軸寸法を 0.046→0.176 に置換。

**(3) 型切替は「高 coeff × ~15k step 遅延」の現象**。`m6.0_dimprior`(`a1e6s9en`) の cmix 推移:
Distance は step3k で即 0.003 に圧殺されるが、置換先（DistanceX/Y）を獲得するのは step15k 以降で、それと同期して
DoF が再降下（10→6.9）。**DimPrior が収束を遅くする正体＝「Distance を捨てるのは一瞬・軸寸法を再学習するのに ~15k step」**。
低 coeff（`s9rxgu3v`=`m2.25_l0.75_leakdof_dimprior`）は step4k で Distance がまだ 0.13＝**低 coeff では切替が鈍い可能性**（要観察）。

**(4) leak は FC 補償に必須（観察④）**。DimPrior 単独＝leak 無し（`m6.0_dimprior`）は最終 DoF6.9 だが **FC0.12**（同 DoF の
s8 は 0.38）＝「途中まで寸法を入れて閉じ切らない」。leak（gate=dof, closure に応じ ramp）を足すと FC0.43 に回復。
→ **leak と DimPrior は分業**: leak=raw/斜め Distance 抑制＋closure 補償、DimPrior=型の付け替え。両方要る。

**(5) 残る根本トレードオフ**: FC↑ につれ distFam↑ / relRat↓ は全 run 共通で未解決（`s9_leak_fc15`=`iw2i8e0d` は
FC0.63 だが distFam0.198・Dist0.108・leak0.280＝over-dim 端）。`28xstb2f` はこの緊張は破れていないが、**matched-DoF で
raw Distance≈0・leak・distFam が baseline を Pareto 優位**＝intent 品質では現状最良。

**留保**: `Distance=0` は極端で、DimPrior coeff のハイパラ探索余地あり（優先度は下げる）。over-dim 監視 KPI は distFam 単体でなく
**Distance vs DistanceX/Y 比 ＋ leak** に置く（distFam は FC を押せば必ず上がる）。eval は収束完了を待たず、現 ckpt の
`28xstb2f` vs `oyvonqrm` を held-out(fiverr) で確定させれば因果主張は取れる段階。

### 4.8 paper 用 ablation 設計（表A）と評価方針 (2026-06-15)

**最終 config = `rloo_response_scaffold12_m4.5_l1.5_leakdof_dimprior_intent`（`28xstb2f`）**。これを成分分解した build-up
＋ DimParsimony 置換を **表A（main）** とする。共通 scaffold S = Solver(FC1.5) + RelPrior + DimPrior + flip + v2GT
（distortion `null`→1/64 で全 run 統一、pseudoGT-v2）。**M/L/DimParsimony だけ**を動かす:

| arm | M | L | DimParsim | config / run | 役割 |
|---|:-:|:-:|:-:|---|---|
| no-intent | ✗ | ✗ | ✗ | `scaffold12_ablA_noML_dimprior`（新規） | over-dim floor |
| +M | 4.5 | ✗ | ✗ | `scaffold12_ablA_m4.5_dimprior_intent`（新規） | motion の寄与（兼 L-off 腕） |
| +M+DimParsim | 4.5 | ✗ | ✓ | `scaffold12_ablA_m4.5_dimprior_dimpars_intent`（新規） | **Leak↔DimParsimony 置換** |
| +M+L（最終） | 4.5 | 1.5 | ✗ | `..._m4.5_l1.5_leakdof_dimprior_intent`（`28xstb2f`） | full |

読み: no-intent→+M→+M+L で M・L 各寄与。**+M+DimParsim vs +M+L** で「Leak(方向) が count-parsimony を性能で包含」
→ DimParsimony 不採用を**内部で**正当化。3 新規 config は Hydra compose + 2-step CPU smoke 通過済み（成分一致を確認）。
表の列 = resp↓ / θ_motion^rel↓ / dim-ratio＋(Distance vs DistanceX/Y/Angle 内訳) / key-rel↑ / RelDimAlign↓ / FC。

**設計判断（確定）**:
- **DimPrior は ablation しない**（RelPrior と同系の固定 prior、全 main arm に常時 ON）。main arm の
  DimPrior 係数は `Distance=-0.03, DistanceX=DistanceY=0.02, Angle=-0.05` で統一する。
  例外: 既存の低係数 run `s9rxgu3v` (`m2.25_l0.75_leakdof_dimprior`) は
  `Distance=-0.02, DistanceX=DistanceY=0, Angle=-0.03` なので、M/L 低係数だけでなく DimPrior も
  弱い confounded reference として appendix 扱いにする。
- **faithful-Casey は DROP**（実装非公開・Fusion360・ハイパラ未公開＝忠実再現不可）。その役割は上記 DimParsimony↔Leak の
  内部置換 ablation が代替。
- **distortion_threshold は全 run 1/64 で統一済み**（原典 Casey ≈ 0.25 より厳格）＝内部整合 OK、再走不要。論文に一文明記。
- **pseudoGT の RL ablation（旧 表B）は廃止**（素データ `sg_filtered_unique_valid` は under-constrained で、symmetric reward だと
  「Pred も under」になるのが自明＝canonicalization の効果でない。subset mode 説明も重い）。pseudoGT の価値は Sec.5 既出の
  command-F1 の不備で代替（指標レベルで充足）。
- **shuffle 負例・FlipAug on/off は APPENDIX**（shuffle は旧 `buxh9c4w` で既出、flip は自明）。

**APPENDIX**: DimParsimony 冗長性（+M+L+DimParsim 任意1本）、dose-response（`tjxf91fo` 等）、leak_gate＋dof分布 二峰→単峰
figure、低coeff切替（`s9rxgu3v`）、FC崩壊（`a1e6s9en`）、flip on/off、shuffle。

**評価 checkpoint 方針**:
- preliminary / sanity check は matched step で見てよい。ただし最終比較は §4.10 P2 の通り
  **FC-rate / solver status を揃えた operating point** と dense Pareto frontier で行う。
- **`solver_valid_dof_mean` 最小での ckpt 選択は禁止**（DoF 最小＝over-constrained 偏り＝over-dim 主張と逆）。
  代表点は train-val 上で事前に決めた FC 帯に最も近い ckpt を選び、held-out Fiverr を見て選ばない。
- eval は **held-out fiverr**、ckpt 選択基準は train-val のみ（test 選択回避）。

---

### 4.9 Fiverr 全体評価: 「差分が出ない」問題の解消候補 (2026-06-18)

> ⚠️ **更新 (2026-06-19): 本節は評価 iter5（solver-in-the-loop）の結果。§4.10「評価プロトコル変更」で
> iter1 単発を既定にしたため、(a) 本節の **絶対 FC / 各指標の数値は superseded**（dim-ratio で Ours が
> DP frontier の下、という方向の結論のみ不変・むしろ強化）、(b) **`S_place`/`asgn_sim` を headline の「勝ち軸」に置く方針は撤回**
> （iter5 でも非有意、iter1 でも operating point ではタイ＝負けではないが勝ちでもない。S_place は未解決軸として正直に提示）。
> 以下、矛盾箇所は取り消し線。→ §4.10 P4 直前の「評価プロトコル変更」と P4 を参照。**

**Metric update**: paper 用の over-dim 軸は以後 `dimensional_ratio =
(all dimensional constraints) / n_constraints` とする。旧 `distance_family_ratio`
（Distance/DistanceX/DistanceY のみ）は後方互換・診断用に残すが、headline plot / table では使わない。

検証を進める中で「SolverReward + DimParsimony のベースラインが over-dimensional になるのは、単に
DimParsimony の weight が低いからではないか」という疑問が出た。そこで DimParsimony weight=1.0 の
強い baseline (`DP_c1`, W&B `2ij8cdav`) を確認したところ、over-dim はかなり抑制され、学習曲線も
Ours とかなり重なることが分かった。

保存済み run / local directory:

- Main: `seq_M4.5_L1.5` / `28xstb2f` ==> `2026-06-13_19-03-56`
- Baseline: `DP_c1` / `2ij8cdav` ==> `2026-06-16_22-29-57`
- 補助: `DP_c0.3` ==> `2026-06-16_22-40-00`
- 補助: `DP_c3` ==> `2026-06-16_22-30-26`
- 補助: `rloo_response_scaffold12_m2.25_l0.75_leakdof_dimprior_intent` ==> `2026-06-14_16-41-23`
  - 注意: この run は DimPrior 係数も弱い（`Distance=-0.02, DistanceX/Y=0, Angle=-0.03`）ため、
    低 M/L 係数だけの ablation ではない。main claim ではなく参考点に落とす。
- 補助: `rloo_response_scaffold12_ablA_m4.5_dimprior_dimpars_intent` ==> `2026-06-15_14-39-13`

当初の懸念は、`MotionReward(+LeakReward)` と `SolverReward+DimParsimony` が学習曲線上ではほぼ同じ
終点へ落ち、**定性的にも定量的にも差分が見えない**ことだった。特に `motion_relational_angle` は
「GT relational が張る motion subspace を真似する」だけなので、SolverReward が DoF を下げるために
relational constraints を張る圧と区別しにくい可能性があった。一方 `LeakAngleReward` は
「GT relational を violate する dimensional driver だけを選択的に罰する」ので、DimParsimony
（寸法数を一律に罰する）とは質的に違う効果が期待された。

この懸念に対し、Fiverr 全体（12 annotator split, n=292）で既存 ckpt と追加生成を評価した。
集計は [artifacts/fiverr_eval/summary.md](../artifacts/fiverr_eval/summary.md)。図は2種類に分ける:

- **headline / non-circular**:
  [artifacts/fiverr_eval/fiverr_pareto_fc_vs_independent.png](../artifacts/fiverr_eval/fiverr_pareto_fc_vs_independent.png)
  (`dimensional_ratio`, `theta_motion_full`, `asgn_sim` → §4.10: `asgn_sim`/S_place は **headline の「勝ち軸」からは外す**が図・表には残し、「どの手法も改善しない未解決軸＝今後の design-intent 研究の良い target 指標」として正直に提示)
- **mechanism check**:
  [artifacts/fiverr_eval/fiverr_pareto_fc_vs_angles.png](../artifacts/fiverr_eval/fiverr_pareto_fc_vs_angles.png)
  (`motion_relational_angle`, `leak_relational_angle` = RelDimAlign angle, `response_relational_angle`)

Fiverr 評価 JSON から表と図を再生成するスクリプトは
[scripts/plot_fiverr_eval_pareto.py](../scripts/plot_fiverr_eval_pareto.py):

```bash
conda run -n freecad python scripts/plot_fiverr_eval_pareto.py
```

W&B の validation history から学習ログ上の Pareto/frontier 図を作るスクリプトは
[scripts/diag_reward_pareto_frontier.py](../scripts/diag_reward_pareto_frontier.py)。

| model | FC | valid DoF | dim-ratio↓ | θ_full↓ | θ_rel↓ | θ_align↓ | S_place↑ | rawDist↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S step12700 | 0.702 | 3.484 | 0.343 | 0.413 | 0.365 | 0.397 | 0.590 | 0.007 |
| S+M step12200 | 0.651 | 3.380 | 0.292 | 0.350 | 0.295 | 0.301 | **0.665** | 0.119 |
| Ours M4.5/L1.5 step52200 | 0.647 | 4.327 | **0.244** | 0.394 | **0.281** | **0.241** | 0.598 | 0.014 |
| Ours M4.5/L1.5 step59800 | 0.712 | 3.918 | 0.252 | 0.386 | **0.272** | **0.230** | 0.608 | 0.005 |
| DP c1 step21200 | 0.640 | 4.965 | 0.278 | 0.439 | 0.345 | 0.353 | 0.573 | 0.014 |
| DP c0.3 step22500 | **0.774** | 3.682 | 0.328 | 0.402 | 0.353 | 0.380 | 0.603 | 0.006 |
| DP c3 step15400 | 0.158 | 7.400 | 0.191 | 0.579 | 0.344 | 0.300 | 0.499 | 0.036 |
| M+DimParsim step13300 | 0.596 | 3.927 | 0.273 | 0.371 | 0.296 | 0.285 | 0.652 | 0.128 |
| M2.25/L0.75 step36000 | 0.736 | 4.093 | 0.281 | 0.417 | 0.310 | 0.310 | 0.593 | 0.068 |

**critical review の採否**:

- **採用**: headline を `leak/response/motion_relational_angle` に置くのは循環論法に見える。
  これらは Motion/Leak reward と同一ファミリなので、main claim の縦軸は報酬非依存の
  `dimensional_ratio` と ~~`S_place/asgn_sim`~~ に置き、必要なら `theta_motion_full` を隣接して示す。
  報酬ファミリに近い angle 系は「報酬が設計通り効いた」機構確認として解釈する。
  （→ **更新 §4.10**: 報酬非依存 headline は `dimensional_ratio`（＋隣接 `theta_motion_full`）に一本化。
  ~~`S_place/asgn_sim`~~ は iter5 でも非有意・iter1 でも operating point ではタイのため headline の「勝ち軸」から外す（負けではない／未解決軸として提示）。）
- **採用**: raw Distance は Ours vs DP c1 ではほぼ同じ（0.0136 vs 0.0140）なので、Ours-vs-DP の
  headline 証拠にはならない。raw Distance が効くのは Ours vs M+DimParsim / M2.25 などの比較限定。
- **採用**: `S+M` は floor を全面改善するが、独立指標 `dim-ratio` では `S+DP c1` とほぼ同じ frontier に
  乗る。これは「Motion 単体は Solver/DP に subsume されやすい」という当初の疑問の実証になる。
- **採用**: DP frontier を越える候補は Leak を足した `S+M+L`。ただし現状では `S+M step12200` と
  `Ours step52200` の step が大きく違うため、Leak の寄与を確定するには dense trajectory / 再走が必要。

**現時点の読み**:

1. `S -> S+M` はほぼ同 step (12.7k vs 12.2k) で比較でき、Motion は floor を改善する
   (~~`S_place` 0.590->0.665~~ ← iter5 限定・S_place 降格, `dim-ratio` 0.343->0.292, RelDimAlign angle 0.397->0.301)。MotionReward は無効ではない。
2. ただし `S+M` は `S+DP c1` より dim-ratio が高い（0.292 vs 0.278）。**Motion 単体では DP frontier を
   越えない**可能性が高い。
3. `Ours step52200` と `DP c1 step21200` は FC がほぼ一致（0.647 vs 0.640）し、Ours は
   independent headline の `dim-ratio` で 0.278->0.244 改善する。~~`S_place/asgn_sim` で 0.573->0.598 と改善する~~
   （→ **更新 §4.10**: S_place の差は iter5 でも paired 非有意 (p=0.16)。iter1 単発 operating point でも有意差なし
   （高FC終端 p=0.19）＝ headline 勝ち軸からは外すが負けではない）。
   angle 系でも Ours が良いが、これは機構確認として扱う。
4. `DP c0.3` は FC だけ高いが dim-ratio / angle は悪い。単純な FC 最大化や count-parsimony 強化では
   design intent は保てない。
5. したがって「差分が出ない」問題は、**step ではなく FC operating point / Pareto frontier で比較する**と
   解消できる可能性が高い。ただし最終 claim は ~~`dim-ratio/S_place`~~ `dim-ratio`（→ §4.10: S_place 降格）の
   held-out Fiverr dense Pareto と paired 統計で支える。

### 4.10 今後の実験計画と TODO (2026-06-18)

#### P0. 既存 ckpt の dense Pareto 化（学習なし・最優先）

目的: 現在の Fiverr 図はまだスパースで、代表点 cherry-pick に見える。まず再学習せず、保存済み
`stepsave_{5000*n}.ckpt` と既存 ckpt を Fiverr 全体で評価し、FC-rate vs `dim-ratio` の trajectory / frontier を描く（→ §4.10: 評価 iter1 単発。第3パネルは `S_place` を正直に併置＝勝ち軸ではなく未解決軸）。
angle 図は mechanism check として併置する。

推論:

```python
import subprocess
for i in range(12):
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES=1 python src/test.py "
        f"--ckpt logs/train/runs/2026-06-dd_hh-mm-ss/checkpoints/stepsave_xxxxx.ckpt "
        f"--data data_eval/{i}/ "
        f"--outdir model_outputs/Vitruvion_scaffold_v12_<config名>_step_xxxxx/{i}",
        shell=True,
        check=True,
    )
```

評価:

```bash
conda run -n freecad env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  python fusion360_annotation/evaluate.py model_outputs/<pred_dir> data_eval \
  --json artifacts/fiverr_eval/<label>.json
```

作図:

```bash
conda run -n freecad python scripts/plot_fiverr_eval_pareto.py \
  --input "LABEL=artifacts/fiverr_eval/<label>.json" ...
```

P0 対象:

- `rloo_response_scaffold12_ablA_noML_dimprior_dimpars_c1`（DP c1 baseline）
- `rloo_response_scaffold12_ablA_noML_dimprior_dimpars_c0.3` / `c3`（DimParsimony dose）
- `rloo_response_scaffold12_ablA_m4.5_dimprior_dimpars_intent`（S+M+DP）
- `rloo_response_scaffold12_m2.25_l0.75_leakdof_dimprior_intent`（低係数 Ours; DimPrior 係数も違うので appendix/reference）
- 既存 Ours `28xstb2f` は保存 ckpt が少ないので、`52200/59800/60700` など使える点だけ載せる。
- 既存 S / S+M (`s_noml_step12700`, `sm_m4.5_step12200`) は preliminary ablation として既に入れる。

検証仮説:

- 学習ログ Pareto で見えた seq_M4.5_L1.5 vs DP_c1 の FC-matched frontier 差を、held-out Fiverr の
  independent metric (`dim-ratio` ~~`/S_place`~~) で再現できる（→ §4.10: S_place 降格）。
- DP 系は FC を上げるほど dim-ratio が悪化する trade-off 上に乗る。
- Ours は同 FC で DP frontier より低い dim-ratio を作る ~~/ 高い `S_place`~~（→ §4.10: S_place 降格。iter1 では非優位）。

#### P1. 学習が必要な run 一覧

**現在の W&B running 状態（2026-06-18, Singapore）**

W&B project: `kakuda40611-nanyang-technological-university-singapore/vitruvion`

| priority | run id | config / name | 位置づけ |
|---|---|---|---|
| A1 | `ad7gt99w` | `rloo_response_scaffold12_m4.5_l1.5_leakdof_dimprior_intent_retry` | Ours full stack retry。最優先で dense checkpoint を確保。 |
| A2 | `7y0bg0oa` | `rloo_response_scaffold12_ablA_m4.5_dimprior_intent` | S+M。Motion 単体 frontier の確定。 |
| A3 | `2ij8cdav` | `rloo_response_scaffold12_ablA_noML_dimprior_dimpars_c1` | DP c1 strongest baseline。既存 run で FC 帯が足りれば再走不要。 |
| B4 | `cgraga0i` | `rloo_response_scaffold12_ablA_m4.5_dimprior_dimpars_intent` | S+M+DP。Leak と DimParsimony の置換比較。 |
| B5 | `0ku9his9` | `rloo_response_scaffold12_ablA_noML_dimprior` | S floor。S→S+M 改善量の基準。 |
| B6 | `ybk2qgdz` | `rloo_response_scaffold12_ablA_l1.5_leakdof_dimprior_intent` | **Leak-only。格上げ(2026-06-19): appendix → main-table control。** 「L 単体で足りるか／M は冗長か」を切る factorial の欠けセル（~~M=S_place, L=dim-ratio の軸別相補~~ → §4.10: S_place 降格で改訂）。学習中 (`2026-06-18_20-36-15`)。 |
| C8 | `gef5u539`, `y2yu0h8b` | scaffold13 PrefixMotion / PrefixMotion+Leak | 勝った場合だけ appendix/future から昇格。 |
| C9 | `d4hgnece`, `5103sq9n` | DP c0.3 / c3 | DimParsimony dose。既存 ckpt で足りるなら追加学習不要。 |

**A. 必須（fast GPU 優先）**

1. `rloo_response_scaffold12_m4.5_l1.5_leakdof_dimprior_intent`
   - 目的: full stack Ours の dense checkpoint を確保する。`28xstb2f` は本命だが初期から stepsave が揃っていない。
   - 保存: `stepsave_{5000*n}`、少なくとも 60k steps。
   - 検証: Ours frontier が DP frontier より lower dim-ratio になるか ~~/ higher `S_place`~~（→ §4.10: S_place 降格）。

2. `rloo_response_scaffold12_ablA_m4.5_dimprior_intent`
   - 目的: S+M の dense trajectory。Motion 単体が DP frontier に collapse するか、独自 frontier を持つかを確定する。
   - 保存: `stepsave_{5000*n}`、Ours と同じ上限。
   - 検証: Motion は floor を改善するが、DP を越えないという thesis を確定する。

3. `rloo_response_scaffold12_ablA_noML_dimprior_dimpars_c1`
   - 目的: strongest DP baseline。論文上の必須評価対象だが、既存 run (`2ij8cdav`) の dense checkpoint / FC 帯が足りるなら高速 GPU 枠で再走しない。
   - 保存: `stepsave_{5000*n}`、少なくとも Ours と同じ FC 帯を覆うまで。
   - 検証: DP frontier の本当の上限を与える。

**B. 強く推奨（slow GPU でも可）**

4. `rloo_response_scaffold12_ablA_m4.5_dimprior_dimpars_intent`
   - 目的: S+M+DP。Leak と DimParsimony の置換比較。
   - 検証: 同じ Motion 下で、DimParsimony が dim-ratio / `S_place` / rawD をどこまで改善するか。

5. `rloo_response_scaffold12_ablA_noML_dimprior`
   - 目的: S floor。Motion / DP / Leak の改善量の基準。既存 run (`0ku9his9`) を継続して dense trajectory を補えばよい。
   - 検証: S->S+M の効果を dense trajectory で確認。

6. `SolverReward+LeakAngleReward`（Leak 単体, Motion なし。既存 config: `rloo_response_scaffold12_ablA_l1.5_leakdof_dimprior_intent`、run `2026-06-18_20-36-15`）
   - **格上げ (2026-06-19): appendix → main-table control。** ~~理由: §4.9 のデータで M は S_place を 0.590→0.665 と強く改善する一方、dim-ratio headline では DP frontier を越えない＝「M は placement 軸で効く / L は over-dim 軸で効く」という**軸ごとの相補**ストーリー~~（→ **更新 §4.10**: S_place は降格したので「M=placement軸」基盤は撤回）。**改訂理由**: M と L はどちらも over-dim を抑え得るが経路が違う（M=relational subspace alignment / L=dimensional driver の選択的抑制）。「L 単体で足りるのか／M は冗長か」を切り分けるには S+L が必要、という factorial 上の必須セルである点は不変。これが無いと「Leak だけで十分」という contribution を弱める読みを排除できない。
   - 検証（改訂）: S+L は dim-ratio を下げる（L の active ingredient 性）一方、~~S_place は DP 並みに留まる（M なしでは placement が出ない）か~~ relational alignment（motion_rel）は S+M+L より弱い、を確認。これが言えれば Ours（M+L 両方）が正当化される。逆に S+L ≈ Ours なら M 冗長＝再フレーミングが必要。
   - 注意: relational target を増やす信号が弱いので FC は弱めに出る想定。
   - 既に学習中（5k/10k と若い）。Ours retry と同様、stepsave が増えるたび dense 評価する。

**C. 余裕があれば（appendix / reviewer 対応）**

7. `S+M+L+DimParsimony`（既存 `rloo_response_scaffold12_ablA_m4.5_l1.5_leakdof_dimprior_dimpars`）
   - 目的: Leak と DimParsimony が相補か冗長か。
   - 判断: 主張には必須でない。DimParsimony が Leak 上に乗っても frontier を改善しない、という冗長性確認が必要になった場合だけ回す。

8. PrefixMotion / PrefixMotion+Leak (`scaffold13` 系)
   - 目的: STOP 版と同 frontier でも学習効率が良いか、または lower frontier を作るか。
   - 判断: main claim には使わず、勝った場合のみ appendix / future から昇格。

9. DP dose 追加 (`c0.3`, `c3`, `c9`)
   - 目的: DimParsimony 強度 sweep の operating curve。
   - 既存 ckpt で十分なら再走しない。足りない FC 帯だけ補う。

**推奨マシン割当**:

- 高速GPU 1: full stack Ours (`m4.5_l1.5_leakdof_dimprior_intent`)
- 高速GPU 2: S+M (`ablA_m4.5_dimprior_intent`)
- 低速GPU: DP c1 / S+M+DP / S floor を順に流す。既存 ckpt 評価で十分なら訓練を省く。

#### P2. 代表点選択・定性評価・User Study

step 数を揃えるより、**FC-rate または solver status を揃える**ほうが公平。学習の進み具合が違う run を
同 step で比較すると、未収束 run を不当に弱く見せる危険がある。

最終手順は paper で事前宣言する（定性/US だけでなく quantitative の代表点比較にも適用）:

1. train-val 上で目標 FC 帯（例: 0.63--0.67）に最も近い ckpt を選ぶ。
2. 選ばれた ckpt を full Fiverr で一度だけ評価する。
3. dense Pareto 図は全 stepsave を Fiverr 評価して trajectory として描いてよい。ただし本文で報告する
   単一代表点は train-val 選択で決め、Fiverr を見て選ばない。
4. User Study は両者が valid/FC で比較可能な同一 sketch subset から、dim-ratio / `S_place` / motion 指標の差が大きい例と
   ランダム例を混ぜる。出力順は blind/randomize。

現時点の provisional FC-matched 候補:

- Ours `step52200`（FC 0.647） vs DP c1 `step21200`（FC 0.640）
- ただし最終採用は dense trajectory 完成後、train-val rule で決める。

#### P3. 図と統計

main paper の representative table:

- FC rate / valid DoF
- dimensional ratio（全 dimensional constraints の比率; lower is better）
- `theta_motion_full`（lower is better）
- `theta_motion_relational`（lower is better; MotionReward 系の mechanism check でもある）
- `theta_align_relational` / RelDimAlign angle（lower is better; RelDimAlignReward の mechanism check）
- ~~`S_place` / `asgn_sim`（higher is better）~~ → §4.10: 降格。表には完全性のため残すが headline ではない。

main paper の Pareto 図（→ **更新 §4.10**: 評価 iter1 単発・[scripts/run_iter1_pareto.py](../scripts/run_iter1_pareto.py)）:

- FC-rate vs dimensional ratio（headline）
- FC-rate vs `S_place` / `asgn_sim` → 第3パネルとして残す（**勝ち軸ではなく**、どの手法も改善しない未解決軸として正直に提示）。機構角 `leak_relational_angle` は表＋appendix へ。
- FC-rate vs `theta_motion_full` を併置する。

mechanism / appendix 図:

- FC-rate vs `motion_relational_angle`
- FC-rate vs RelDimAlign angle (`leak_relational_angle` in code)
- FC-rate vs `response_relational_angle`
- FC-rate vs raw/axis Distance 内訳
- FC-rate vs `valid_dof_mean`

統計:

- Ours vs DP c1 at FC-matched point の paired bootstrap / Wilcoxon。
- headline は FC-matched Pareto 上の dimensional ratio / ~~`S_place`~~ / `theta_motion_full`（→ §4.10: S_place 降格）。
  `theta_motion_relational` と RelDimAlign angle は提案指標として表に併記するが、報酬同一ファミリなので mechanism confirmation として解釈する。

**予備 paired 統計 (2026-06-19, FC-matched preliminary)**: `evaluate.py` に per-sketch dump を追加
（`composition.per_sketch` / `reference.per_sketch`、batch-qualified id）。
[scripts/paired_stats_fc_matched.py](../scripts/paired_stats_fc_matched.py) で Ours `step52200`(FC 0.647)
vs DP c1 `step20000`(FC 0.658) を paired 比較（Wilcoxon 符号順位 two-sided + bootstrap mean-Δ 95%CI、
delta=ours−base）:

| metric | meanΔ | 95%CI | Wilcoxon p | 判定 |
|---|---:|---|---:|---|
| dimensional_ratio↓ | **−0.051** | [−0.062,−0.040] | **2e-19** | Ours 有意に低い（headline） |
| raw_distance_ratio↓ | −0.0045 | [−0.0087,−0.0003] | 3e-2 | Ours 微減 |
| asgn_sim (S_place)↑ | +0.013 | [−0.011,+0.037] | 0.16 | 方向は Ours だが**この1点では非有意** |
| motion_full_angle↓ | −0.032 | [−0.060,−0.004] | 0.31 | bootstrap-sig / rank-test 非有意（境界） |
| motion_relational_angle↓ | −0.062 | [−0.081,−0.045] | 3e-13 | Ours（報酬同族 = 機構確認） |
| leak_relational_angle↓ | −0.120 | [−0.149,−0.091] | 4e-11 | Ours（報酬同族 = 機構確認） |

→ （上表は iter5 = solver-in-the-loop の旧結果。下記 2026-06-19 の評価プロトコル変更で **iter1 単発を既定**としたため絶対値は superseded。dim-ratio の結論は不変・むしろ強化。S_place は headline の「勝ち軸」からは外すが operating point ではタイ＝負けではない。）

**評価プロトコル変更: iter5 → iter1 単発に統一 (2026-06-19, 重要)**

W&B 学習ログ（`validation_gen/solver_fc_rate` ≈ 0.30 @ step5k）と Pareto 評価（FC 0.473）の乖離を追跡した結果、
[test.py](../src/test.py) の **`n_iter=5` solver-in-the-loop 自己修正**（生成→ソルバ→壊れた制約を再生成、最大5回）が
原因と判明（学習時 val は[単発1回](../src/models/constraint_ar_module.py#L195)）。同 step5000・全Fiverr で
`--n_iter 1` にすると **FC 0.281**（W&B の ~0.30 と一致 ＝ データ分割の問題ではない）。

solver 介入は **FC だけでなく出力スケッチ自体を書き換える**ため、dim-ratio / S_place / motion 角度まで「policy
ではなく solver 後処理の産物」になる（compound effect）。本論文の主張は「報酬が policy に何を教えたか」なので、
**全評価を単発 (`n_iter=1`) に統一**。`src/test.py` の既定を `n_iter=1` に変更。Sec.5 は Base/Solver-RL が iter5
だったが、Solver-RL は今回 c1 に合わせ再走するので追加コストは **Base を iter1 で再評価するだけ**。

**iter1 検証ゲート結果（FC-matched paired, single-shot）**:
- 低FC寄り (Ours 52200 FC0.551 vs DP c1 20k FC0.579): dim-ratio meanΔ −0.058, CI[−.070,−.047], **p=6e-22**（Ours 決定的）。
  S_place meanΔ −0.034, **p=0.058**（DP 寄りだが境界）。θ_full ほぼタイ。
- **高FC終端 (Ours 59800 FC0.640 vs DP c1 35k FC0.644): dim-ratio p=2e-16（Ours 勝ち）、θ_full p=0.07 (n.s.)、
  S_place p=0.19 (n.s.) ＝ operating point ではクリーンなタイ。** motion_rel/leak_rel は両点とも強く Ours
  （p≈1e-10〜1e-19, leak 中央値 0.000）＝機構確認。

**ストーリーライン（確定 2026-06-19, ユーザ合意）**:
> **序盤（低FC・under-constrained）では θ_full / S_place で DP に負ける。しかし operating point（高FC終端）では
> θ_full・S_place ともに有意差なし（タイ）になり、それでいて dim-ratio では勝つ。よって「over-dimensioning を
> 有意に減らしつつ、redesign 挙動 (θ_full) と placement (S_place) は犠牲にしない」で OK。**
> 補足: **dim-ratio は低FC・高FC両方で Ours 勝ち**（序盤負けは θ_full/S_place のみ）。「タイ」は図の見た目だけでなく
> **paired Wilcoxon（θ_full p=0.07, S_place p=0.19）を本文・キャプションに併記**して担保する
> （集計平均の Pareto 図だけだと手前で小差に見えるため）。

**iter1 / iter5 の採否（決定 2026-06-19）**:
> 両プロトコルを出し、**効果がより明確かつ正直に示せる方を採用**する。
> [run_iter1_pareto.py](../scripts/run_iter1_pareto.py) に `--iter N` を追加し出力を完全名前空間分離：
> iter1 → `model_outputs/` + `iter1_*.json` + `paper/figs/rl_reward_pareto_iter1.pdf`、
> iter5 → `model_outputs_iter5/` + `iter5_*.json` + `..._iter5.pdf`（上書きなし）。論文は採用側を
> `\includegraphics` で指定（現状 `_iter1`、切替は1行）。**integrity 条件**: どちらでも (a) 全 variant に同条件適用、
> (b) プロトコル明記、(c) iter5 を採るなら「なぜ solver 介入が Ours を助けるか」（仮説: Ours は GT に自然完備する
> design-intent スキャフォールドを予測 → 開けた DoF が GT 一致で埋まる）の説明/検証を付す。後付けで都合の良い数字だけ拾わない。
> iter5 は 2026-06-19 にバックグラウンド実行開始。

**図方針**: main 図は **FC vs {dim-ratio, θ_motion_full, S_place} の Pareto 軌跡**（モデル成長を見せる）。
forest plot（1点2モデル比較で成長を見せられない）は不採用。operating point のタイは上記 paired 統計で支える。

**一括評価スクリプト**: [scripts/run_iter1_pareto.py](../scripts/run_iter1_pareto.py)（推論 `test.py --n_iter {1|5}` →
`evaluate.py` → Pareto 図 → paired統計を1ファイル・冪等・stepsave 増加に追従）。出力規約
`model_outputs[_iter5]/Vitruvion_scaffold_v12_<config>_step_<step>`（base→`model_outputs/base`）。

#### P4. 論文 thesis の整理

本文 thesis は **design intent != DoF 削減** に寄せる。**評価は iter1（単発, 既定）／ iter5（deployed）両方を出し、
効果が明確かつ honest な方を採用**（§4.10 の採否条件）。ストーリーは「序盤負け→ operating point で θ_full/S_place タイ
→ dim-ratio で勝ち」。

- Motion: relational target alignment を改善（motion_rel）。MotionReward は placement に構造的に盲目。
- Leak: 「どの方向を自由に残すか」の選択性を担い、dimensional substitution を抑える active ingredient
  （leak_rel を ~0 にする）。LeakReward は placement を間接的に動かしうるが、**実測では S_place を有意に改善しない**。
- **S_place の扱い（2026-06-19 決定, ユーザ方針）**: 「効果なしだから除く」ではなく**正直に提示**。どの手法
  （Solver/DimParsimony/Motion/Leak）も S_place を改善しない＝ placement intent は現行 reward 群が触れない**未解決軸**。
  これは「S_place は今後の design-intent-aware constraint prediction の良い discriminative target 指標」という
  **締めくくり（future-work hook）**になる。よって**表・図ともに S_place を残し**、headline の「勝ち軸」からのみ外す。
  図 第3パネル＝S_place（全手法横並び）、機構角 leak_rel/motion_rel は表＋appendix。
- DimPrior: raw Distance -> axis Distance への型切替を助ける固定 prior。
- したがって main claim は「Motion+Leak(+DimPrior) が **independent な dim-ratio** 上で Solver+DimParsimony frontier
  の下に出る（単発で p<1e-20）」。θ_motion_full は隣接 redesign 軸、motion_rel/leak_rel は機構確認。

#### TODO 優先順位

> **更新 (2026-06-19)**: 評価は iter1 単発に統一・一括スクリプト [scripts/run_iter1_pareto.py](../scripts/run_iter1_pareto.py)。
> headline は `dim-ratio`（＋隣接 `theta_motion_full`）、S_place は降格。下記の `S_place` は取り消し線。→ §4.10。

1. **P0評価**: 既存 stepsave/ckpt を full Fiverr（iter1）で評価し、`dim-ratio` ~~`/S_place`~~ Pareto を dense 化する。
2. **必須学習A**: Ours full stack / S+M / DP c1 を stepsave 付きで揃える。
3. **推奨学習B**: S+M+DP / S floor / **S+L（Leak-only, 格上げ control）** を揃え、Motion・Leak・DimParsimony の成分分解を完成させる。S+L は「L 単体で足りるか／M 冗長か」を切る factorial の欠けセル（~~M=S_place / L=dim-ratio の軸別相補~~ → §4.10 で改訂）。
4. **統計**: FC-matched Ours vs DP c1 の paired bootstrap / Wilcoxon（evaluate.py に per-sketch dump 追加済み 2026-06-19。iter1 で dim-ratio p=6e-22）。
5. **定性/US準備**: train-val rule で代表点を選び、同一 sketch subset を blind 化する。
6. **appendix学習C**: S+M+L+DimParsimony / Prefix / DP dose を必要に応じて追加。
7. **論文反映**: Sec.6 reward と results を `dim-ratio` ~~`/S_place`~~ headline + angle mechanism の構成へ修正（→ 反映済み 2026-06-19）。

#### Handoff Checklist

この節だけで別エージェントへ引き継ぐ場合、次の順に進める:

1. W&B で上表の running run が継続中か確認し、各 local run dir に `stepsave_{5000*n}.ckpt` が保存されているか見る。
2. A1/A2/A3 を優先し、少なくとも Ours と DP c1 が同じ FC 帯（例: 0.63--0.72）を覆うまで待つ。
3. ~~各 checkpoint について `src/test.py` で `data_eval/{0..11}` へ推論を出す。~~
4. ~~`fusion360_annotation/evaluate.py` で full Fiverr JSON を作る。~~
5. ~~`scripts/plot_fiverr_eval_pareto.py` で `FC-rate vs dim-ratio/S_place/theta_motion_full` を更新する。~~ angle 図は mechanism check として見る。
   → **更新 (2026-06-19)**: 手順 3–5 は **一括 [scripts/run_iter1_pareto.py](../scripts/run_iter1_pareto.py)** に統合（推論 `test.py --n_iter 1`→evaluate→図→paired統計、冪等）。`src/test.py` 既定は `n_iter=1`。図は `FC-rate vs dim-ratio / theta_motion_full / leak_relational_angle`（S_place 降格）。
6. 本文代表点は Fiverr 結果ではなく train-val 上の FC 帯で選ぶ。Fiverr は最終評価として一度だけ使う。
7. Ours vs DP c1 の FC-matched paired bootstrap / Wilcoxon と、同一 sketch subset の定性/User Study 候補を作る。

### 4.11 確定: Ch.5 再走（c1-70k/iter1）と Ch.6 Table1 充填 (2026-06-22)

**Ch.5（[5_experiments.tex](../paper/sec/5_experiments.tex)）= Solver-RL を DimParsimony c1 の最大実力点で再評価。**
- 代表点: DP c1 は `2ij8cdav` が **70k で停止＝収束**、step70000 が FC0.750/DoF4.64 で最大。Ours の 60k に matched せず Solver-RL 本来の最大で見る方針（ユーザ決定）。
- 全評価を **iter1 単発**に統一（Base も iter1 で再評価）。eda_code は `config.py` の `MODEL_OUTPUT_ROOTS["solver_rl"]` を c1-70k に向けて 04/06/07/08/11 を再走。
- **新数値（multi-ref median）**: Base 0.839/0.707/0.283/OES0.792/F1 0.486（n552）、Solver-RL 0.488/0.471/0.628/OES0.200/F1 0.479（n810）、Human 0/0/0.831/0.343/0.700。
- **大きく動いたのは Base（iter5→iter1 が主因、c1 でなく）**: θ_full 0.667→0.839 等。Solver-RL(c1) 自体は旧c0とほぼ同等。**論旨は保持・強化**（Base が iter1 で明確に under-constrained＝OES バイアス露呈が鮮明）。tex の Table・inが既に反映済み（ユーザ承認 A）。

**Ch.6 Table1 + inline（[6_rl_reward.tex](../paper/sec/6_rl_reward.tex)）充填完了。**
- **代表点 = 提案モデル最終 ckpt（Ours `ad7gt99w` step60000）の train-val FC 0.591 に最近接**（train-val FC は W&B `validation_gen/solver_fc_rate` を `trainer/global_step` で stepsave に対応づけ取得）。held-out では選ばない＝6.3＋appendix A.4.4 の宣言通り。
- 選ばれた点（step / held-out FC / dimR / θfull / θrel / leak / Splace）:
  - Solver-RL(DP c1) 25k / 0.610 / 0.262 / 0.477 / 0.357 / 0.328 / 0.533
  - +M(no DP) `7y0bg0oa` 35k / 0.630 / 0.299 / 0.382 / 0.322 / 0.328 / 0.627
  - +L(no DP) `ybk2qgdz` 30k / 0.613 / 0.238 / 0.515 / 0.353 / 0.190 / 0.470
  - Solver-RL+M(=S+M+DP) `cgraga0i` 35k / 0.565 / 0.289 / 0.373 / 0.310 / 0.297 / 0.637
  - **Ours** `ad7gt99w` 60k / 0.620 / **0.205** / 0.485 / **0.280** / **0.133** / 0.489
- **headline paired（Ours 60k vs DP c1 25k, FC 0.620/0.610）**: dim-ratio meanΔ **−0.063**, CI[−0.051,−0.075], **p=1.3e-24**。θ_full タイ（meanΔ+0.002, CI 0 跨ぎ）。mrel p=6e-13 / leak p=3e-23（機構）。
- **factorial 確定（表で読める）**: +M 単体は dim-ratio で DP frontier を越えない（0.299>0.262）／+L 単体は dim-ratio↓（0.238）だが θ_full 回帰（0.515）＋Splace 低下（0.470）／**M+L だけ** dim-ratio 最小かつ θ_full 非回帰＝両方必要。
- **⚠️ S_place の storyline 変化（§4.10 から要更新）**: 旧 §4.10 は operating point で「S_place タイ」としたが、train-val 選定の新点では **S_place は paired で p=0.018 で僅かに DP 有利**（Ours 0.489 < DP 0.533）。さらに **+M / +M+DP は S_place を baseline 超えに上げる**（0.627 / 0.637）＝「どの手法も S_place を改善しない」は誤り。正しい整理: **M は S_place を上げるが L を足すと baseline 付近へ戻り、Ours は S_place を勝たない／人間(0.831)には全手法届かない**。tex（Table キャプション・S_place 段落）はこの正しい表現に修正済み。**§4.10 の「S_place タイ」記述はこの 4.11 で上書き。**
- **matched-strength control（M+DP c1）**: smdp は DimParsimony 0.30、baseline c1 は 1.0 で強度不一致だったため、**config [rloo_response_scaffold12_ablA_m4.5_dimprior_dimpars_c1_intent.yaml](../configs/experiment/rloo_response_scaffold12_ablA_m4.5_dimprior_dimpars_c1_intent.yaml)** を新規作成（coeff 0.30→1.0 のみ差分）。**別マシンで学習予定**（このリポジトリでは投入しない）。leak の weight-robust 機構（DimParsimony 0.3/1/3 全域で leak 高止まり・Leak のみ低下）で公平性は既に担保されるため load-bearing ではない。
- **図**: [run_iter1_pareto.py](../scripts/run_iter1_pareto.py) の `AXES` に θ_motion^rel / θ_align^rel(leak) を追加し **5パネル**化（mechanism として明示）。smdp は `plot=False` で評価のみ（表用、メイン図には出さない）。
- **更新 (2026-06-22, Ours を 70k へ延長)**: oursretry が stepsave_70000 まで完了。**新ターゲット = Ours@70k train-val FC 0.624**。再選定 → Table1 を 70k operating point に更新（tex は 60k 行を `%` コメント保存・70k 行を下に記載）。新点（step / hoFC / dimR / θfull / θrel / leak / Splace）: DP c1 30k/0.630/0.278/0.449/0.349/0.336/0.563、+M 40k/0.654/0.305/0.367/0.309/0.314/0.633、+L 40k/0.644/0.239/0.513/0.344/0.186/0.476、S-RL+M 40k/0.596/0.302/0.379/0.318/0.328/0.633、**Ours 70k/0.627/0.221/0.448/0.279/0.159/0.530**。
  - **headline paired（Ours 70k vs DP c1 30k, FC 0.627/0.630）**: dim-ratio meanΔ **−0.061**, CI[−0.049,−0.073], **p=9.9e-23**（gap は 60k と同じ Δ≈0.057・不変）。**θ_full clean tie（p=0.72）、S_place clean tie（p=0.17）** ＝ 60k で出た S_place 弱有意（p=0.018）は解消し、**§4.10 の「operating point で θ_full・S_place ともにタイ」storyline に綺麗に一致**。S_place の tex 表現（M が上げる/Ours は勝たない/人間未達）はそのまま有効。
  - factorial も不変（+M dim-ratio 越えず 0.305、+L θ_full 回帰 0.513、M+L のみ両立）。

- **残**: User Study（ユーザが 300 件目視 → Table2）、LaTeX 実ビルド確認、必要なら M+DP c1 完了後に smdp 行を差し替え。

---

## 5. 補足: motion_angle が触れない軸（参考）

`motion_relational_angle` は dimensioning **placement**（どの primitive 対に寸法を置くか）に
構造的に盲目（§1.1、意図的）。この軸を測るため別指標 **participation**（dim_uglify 合成統制で
p=3.8e-59）と **set_sim**（motion-basis 一致）を検討したが、いずれも GT 慣習依存・count
混同で実 model 評価には使えず、diversity descriptor 止まり。**RL 報酬の機構確認には
`motion_relational_angle` / `leak_relational_angle` を使うが、paper headline の model 比較は
~~§4.9 の通り `dimensional_ratio` / `S_place(asgn_sim)` / `theta_motion_full` を主軸にする~~**。
（→ **更新 §4.10**: `S_place(asgn_sim)` は headline から降格。主軸は `dimensional_ratio`＋隣接 `theta_motion_full`。
そもそも本段落が言う通り motion は placement に盲目なので、S_place を M の勝ち筋にするのは機構的に無理筋だった
＝ iter1 の非有意・逆転結果と整合。）詳細は v1 §17 / §18。

> 更新 (2026-06-10, v1 §18.8): set_sim の max-matching を Hungarian 1:1 割当に替えた
> **`asgn_sim`**（max-count 正規化）が dim/rel drop 単調・model 分離 z≈+12 で、しかも
> `motion_full` と逆に **over-dimensioning を正しく罰す**（余剰列が割当から溢れる）。§4 系の
> over-dim 抑制実験の eval / checkpoint 選択の併置候補。placement 純粋軸は `asgn_min`
> （hh-vs-hu 0.666 / multi-ref 0.703、combo 0.736）。新 intrinsic 軸 amp（レバー増幅、統制域
> AUROC 0.71 で歴代最強）は coverage 交絡で model-eval 不可＝thesis instrument 限定。
