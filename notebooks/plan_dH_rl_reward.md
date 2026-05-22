# d_H を強化学習報酬として使うための設計検討

作成日: 2026-05-22

## この文書の位置付け

本プロジェクトの evaluation metric である `d_H`
([v4](summary_histcad_response_metric_v4.md),
[v4_results](summary_histcad_response_metric_v4_results.md))
を、constraint inference model の RL fine-tuning 報酬として使えるかを検討する
設計メモ。

前提として:

- `d_H`, `d_sub`, `leak_prime`, `R_C`, `V_C` などの記号は v4 で導入したものを
  そのまま使う。これらの数学的背景は v3 / v4 を参照
- d_feas を報酬化する議論 (intrinsic probe, pseudo-relation pipeline,
  `bonus + fc` reward mode 等) は
  [d_feas reward roadmap](summary_dfeas_reward_rloo_roadmap.md) を前提とする

**主結論**:

> `d_H` をそのまま報酬関数に投入するのは安全ではない。`d_H` の成分を
> 報酬向けに分解 / 置換 / 補強し、**`leak_prime` + `d_relstruct` +
> `gain_intrinsic`** の 3 軸 + non-redundant FC-gate + parsimony penalty
> という合成構成 (本文書では **案 A″** と呼ぶ) を提案する。
> このうち `d_relstruct` は本検討で新規提案する補完項で、`d_H` が捕捉
> できない over-relation-constraining を捕えつつ、RL の文脈で問題になる
> dim count matching pressure を回避する。

---

## 1. 背景と動機

### 1.1 タスク

2D CAD sketch (primitives 集合) を入力として、その primitive 間にかかる
constraints (dimensional + relational) を推論する。出力 constraint set は
solver で FC (fully constrained, DoF=0) になることが望ましいが、それだけでは
不十分で **design intent aware** であってほしい — つまり「人間設計者が意図
した parametric 関係」を反映した constraint 構成であってほしい。

### 1.2 評価指標の歴史

- **token-level exact match**: 同じ design intent でも constraint の書き方は
  複数あるため過度に厳しく、不適切
- **HistCAD OES**: edit success + preservation の binary 集計。base benchmark
  だが、dimension を落とすほど OES が改善する failure mode が判明 (v3.md §6.1)
- **d_feas** (v3.md): edit response `S_C^(q)` の first-order residual。
  HistCAD OES の sign inversion を回避できる連続値指標
- **d_H** (v4.md): probe-free な observable subspace 距離 + operator leak。
  d_feas の gain term が binary aggregate な問題を解消し、paper main metric
  として確定

### 1.3 RL への動機

base model (token-level supervised) は under-constrained / inconsistent な
constraint set を出しがち。CAD solver feedback を使った RLOO で FC rate を
押し上げられるが、SolverReward 単独だと dimensional constraints に寄りすぎ、
relational constraint が薄い "trivial FC" に collapse する failure mode が
ある (roadmap.md §背景)。

評価指標 `d_H` が「dim と rel が両方とも design intent と整合する」状態を
連続値で measure できるなら、それを RL 報酬に入れて学習を design-intent-
aware の方向に押せるはずである。本文書はその実現可能性を検討する。

---

## 2. 評価指標と RL 報酬は同じではない

両者は似た役割を持つが、要求される性質は異なる。

| 要件 | 評価指標 | RL 報酬 |
|---|---|---|
| 表現非依存 (intent-equiv な書き換えに不変) | ✓ 必須 | ✓ 必須 |
| 連続値 (paired test の検出力に効く) | ✓ 必須 | △ 望ましいが gradient signal の方が重要 |
| annotator variance 耐性 | ✓ 必須 (heldout median ≈ 0) | ✓✓ より重要 (40-50% noise だと学習が破綻) |
| 単一スコアで Goal を網羅 | ✓ 必須 (paper の主張がまとまる) | △ 信号源は合成可能 |
| 失敗形 (gaming) 耐性 | △ paper では事後検証で済む | ✓✓ 必須 (model がそこに lock-in する) |
| GT 依存度の低さ | △ heldout で代用可 | ✓✓ より重要 (training GT 品質に直結) |
| 計算コスト | △ batch evaluation | ✓ rollout per step に乗る |

要点:

- **評価指標**は Goal を直接 measure する **一発の数値**が望ましく、annotator
  variance はある程度の paired test 検出力を保てれば許容される
- **RL 報酬**は複数信号の合成で良いが、各成分の **annotator variance 耐性と
  gaming 耐性** が evaluation より強く求められる

→ 「評価として best な指標 = 報酬として best」とは限らない。`d_H` の各成分が
報酬要件を満たすかを個別に評価する必要がある。

---

## 3. RL の Goal の再確認

`d_H` paper の文脈では「人間と model の差を識別する」のが目的だった。
RL 文脈で同じ指標を使うと、その差を **減らす方向に model を動かす** ことに
なる。具体的に何を望むかというと:

- **Goal 1**: 十分な数の relational constraints が推論される
  (現状の base model は relation が薄く、SolverReward 単独で fine-tune すると
  dim 寄りに偏る)
- **Goal 2**: dimensional constraints が **自然な位置** に入る
  (line ごとに Length を貼り散らすのではなく、human designer が選ぶような
  parsimonious な driving dimension の集合)

副次的に:

- **Goal 3 (parsimony)**: 全体の constraint 数が GT と同程度 (膨張しない)
- **Goal 4 (FC)**: solver 的に FC を維持

Goal 4 は既存 SolverReward が担う。本文書は Goal 1-3 を新報酬成分で扱う。

---

## 4. d_H の各成分が報酬として持つ問題

`d_H = sqrt(d_sub² + leak_prime²)` の各成分を報酬要件で評価する。

### 4.1 d_sub

```text
d_sub(P, G) = ||P_{V_P} - P_{V_G}||_F^2 / max(dim V_P, dim V_G)
              V_C = range(J_phi R_C)
```

**懸念**:

- **annotator variance**: v4_results.md §7 で heldout `frac<1e-4 = 0.583`
  (d_feas の 0.730 より明確に悪い)。RL 報酬として 40% の人間-人間ペアが
  非ゼロ値を持つのは、学習で「正解」を罰する直接ノイズ
- **dim count matching pressure**: `dim V_P` と `dim V_G` の差が d_sub に
  直接効くため、「特定 annotator の dim 数構成」に Pred を引き寄せる
  incentive がある。これが reward だと model は intent-equivalent な
  range of valid dim 構成のうち一つに mode collapse する
- **bootstrap 困難**: pretrain 段階で `d_sub` が 0.5-1.0 範囲に頭打ち
  (v4_results.md §3 base=0.77、dim_drop100=1.00)。学習初期の gradient
  signal が薄い

### 4.2 leak_prime

```text
leak_prime(P; R_G) = ||J_{R_G} U_P||_F / sqrt(n_R · dim V_P)
                     U_P = orthonormal basis of V_P
```

**良い性質**:

- 連続値、annotator variance への耐性は d_sub より良い (heldout median ≈ 0)
- v4_results.md §4.2 で solver_rl 検出力が paired z=+12.68 と全候補で最強
- d_feas leak の operator 化版として、roadmap.md の pseudo-relation
  pipeline をそのまま流用できる

**懸念**:

- **GT 表現依存**: `J_{R_G}` が pseudo-relation の品質に依存
  ([§7](#7-pseudo-relation-との相互作用) で扱う)
- **over-constraining への盲点**: Pred が R_G より多い relations を入れた
  場合、`U_P` は更に小さい部分空間に入るだけで `J_{R_G} U_P` は依然 0。
  Pred が GT より制約過多になっても leak_prime は罰しない
- **Goal 2 への寄与なし**: dim 配置の自然さを直接見ない

### 4.3 d_H 全体

成分ごとの懸念を継承し、特に `d_sub` 由来の annotator variance ノイズと
dim matching pressure が両方乗る。**そのまま投入は危険**。

---

## 5. 報酬成分の段階的検討

### 5.1 案 1: d_H をそのまま投入

`r = r_solver + α · max(0, 1 − d_H / cap_dH) · 1[FC]`

→ **NO**. §4.3 で議論した通り、d_sub の懸念がそのまま顕在化。論文評価指標と
完全一致する魅力はあるが、学習が annotator variance ノイズと dim count
matching pressure で歪む。

### 5.2 案 2: leak_prime のみ投入

`r = r_solver + α · max(0, 1 − leak_prime / cap_leak) · 1[FC]`

これは安全側で、d_feas reward の `bonus + fc` mode (roadmap.md L295-303) と
同型。

**Goal 1 への寄与**: R_G にある関係を Pred に追加することが leak_prime を
0 に近づける最効率な方法 → 部分的に正しい方向の圧

**Goal 2 への寄与**: ほぼ無い。leak_prime は dim 配置を直接見ない

**欠陥**: §4.2 の over-relation-constraining 盲点が残る → model が
"safe over-constraining" 戦略に逃げうる

### 5.3 案 3: leak_prime + gain_intrinsic

dim placement の信号を別途追加する。

```text
Q_intr := { Length(ent) | ent is Line,       ¬construction } ∪
          { Radius(ent) | ent is Arc/Circle, ¬construction }

gain_intrinsic(P) = (#unattainable intrinsic edits in Q_intr) / |Q_intr|
```

**問題点**: gain_intrinsic を 0 にする最安直な方法は「全 entity に直接
Length/Radius を貼る」 → Pred が **dim 数 inflation pathology** に落ちる
可能性。

**緩和**: leak_prime + non-redundant FC-gate との組み合わせで、関係経由の
dim 削減 incentive が働く (relations を入れた方が gain も leak も同時に
下がる)。ただし pseudo-R_G の品質に強く依存。

**dim_coverage (gain_intrinsic の binary 版) は更に悪化**: 階段状 gradient
で dim inflation lock-in が強くなる → 採用すべきでない。

### 5.4 案 4: ユーザー直観由来の新提案 `d_relstruct`

「GT が relation のみで FC になった世界を仮想して、Pred の dim 出力を
抑制できないか」という直観 (DoF=2 までしか relations では削れないので
原理的にあり得ないが、思考実験として) から出発する。

字面通りの定式化は degenerate:
- `p_G` が空 → `R_G` は空行列 → `V_G = {0}` → `d_sub` は定数 1 で
  gradient 消失

しかしこれを「`J_{R_G}` の null space で比較する」と読み替えると、意味の
ある新指標になる:

```text
N_C := null(J_{R_C}) ⊂ R^{n_x}              # C の relations だけで許される
                                              # primitive motion
                                              # dim = n_x - rank(J_{R_C})
N_C_obs := range(J_phi · basis(N_C))         # 観測空間に投影

d_relstruct(P, G) := ||P_{N_P_obs} - P_{N_G_obs}||_F^2
                     / max(dim N_P_obs, dim N_G_obs)
```

#### 性質

**(a) dim 構成に依存しない**: `N_C` は relation のみから決まる。annotator が
dim 数を変えても `N_C` は不変 → annotator variance 耐性が `d_sub` より良い
**見込み** (要 smoke test 検証)。

**(b) 両側対称**: subspace 距離は対称なので、Pred が relations を「足りない」
(`N_P ⊋ N_G`) でも「過剰」(`N_P ⊊ N_G`) でも罰する → **leak_prime の
over-constraining 盲点を補完**。

| シナリオ | leak_prime | d_relstruct |
|---|---|---|
| relations 一致 | 0 | 0 |
| relations 不足 (`R_P ⊊ R_G` 等価) | > 0 | > 0 |
| **relations 過剰** (`R_P ⊋ R_G` 等価) | **0 (見逃す)** | **> 0 (捕捉)** |
| relations 異種 (部分一致) | > 0 | > 0 |

**(c) soft mimicry**: 字面通り `R_G` の行を copy する必要はなく、「null
space が等価」な representation で 0 が出る (例: Mirror ↔ Equal+Coincident+
Midpoint のような intent-equivalent な書き換えが許容される) → pseudo の
完全模倣を回避する soft regularizer として機能。

**(d) dim 抑制への間接効果**: relations が match すれば `dim(N_P) ≈ dim(N_G)`
になる。non-redundant FC-gate のもとで Pred が FC を達成する最小 dim 数は
`dim(N_P)` 個なので、Pred の dim 数は GT の dim 数に自然に収束する。

> つまり「GT を no-dim と錯覚させて dim を抑制」というユーザー直観は、
> 「relations を match させれば必然的に最小 dim 数になる」という形で
> 機能的に実現される。

#### 評価指標としての d_relstruct vs d_H

| 観点 | d_H (v4 paper main) | d_relstruct |
|---|---|---|
| 測定対象 | dim direction + relation 構造 | relation 構造のみ |
| dim placement の自然さ | △ 弱く measure (`d_sub` 経由) | ✗ measure しない |
| relation 等価性 | △ leak_prime で部分的 (一方向) | ✓✓ 直接 (両方向) |
| over-relation-constraining 検出 | ✗ | ✓ |
| annotator variance | △ heldout 0.583 | 良い見込み (要検証) |
| paper narrative の richness | ✓✓ Goal 1+2 を一発 | △ Goal 1 のみ |

→ **評価指標としては d_H が優位**。d_relstruct は Goal 2 を測れないため
paper main metric としては情報量が落ちる。

#### 報酬としての d_relstruct vs d_sub

| 観点 | d_sub (d_H の dim 軸) | d_relstruct |
|---|---|---|
| annotator variance | △ dim 数差で明確に劣化 | ✓ dim 非依存 |
| pseudo-R_G への耐性 | △ V_G 全体が依存 | △ N_G が依存 (同等) |
| mimicry の soft 度 | ✗ 字面 + dim match まで強要 | ✓ null space 等価で OK |
| over-constraining 検出 | ✓ | ✓ |
| dim 数 implicit 制御 | ✗ 直接の matching pressure | ✓ FC-gate と組んで自然に |

→ **報酬としては d_relstruct の方が安全**。dim 配置の信号は別途
`gain_intrinsic` などで補強する分業が可能。

> 要約: **d_relstruct は paper main metric としては d_H に劣るが、
> RL 報酬としては d_H 成分 (特に d_sub) より優れている見込み**。

---

## 6. 統合案 A″

これまでの分析を統合した報酬構成。各成分が複数 Goal に対し直交的に
寄与する設計を狙う。

```text
r_total = r_solver_total
        + α  · max(0, 1 − leak_prime    / cap_leak)   · 1[FC ∧ ¬Redundant]
        + α' · max(0, 1 − d_relstruct   / cap_rs)     · 1[Converged]
        + β  · max(0, 1 − gain_intrinsic / cap_gain)  · 1[Converged ∧ ¬Redundant]
        − η  · #dims_in_Pred
```

### 6.1 推奨初期値 (offline rescore で sweep して固定する)

| パラメータ | 値 | 出どころ |
|---|---|---|
| `α` (leak_prime weight) | 5.0 | roadmap.md L295-303 を踏襲 |
| `α'` (d_relstruct weight) | 3.0 | novel signal なので保守的 |
| `β` (gain_intrinsic weight) | 3.0 | |
| `η` (parsimony) | 0.05 | dim あたり -0.05 |
| `cap_leak` | 0.3 | v4_results.md §3 solver_rl=0.073, rel_drop50=0.195 |
| `cap_rs` | 0.6 | d_sub と同 scale 仮置き、要 smoke test |
| `cap_gain` | 0.5 | |
| FC-gate | DoF==0 ∧ ¬Redundant | CAD solver Redundant status を弾く |
| Convergence gate | status ∈ {FC, Under} (Distorted/SolverError は除外) | v4_results gate と同型 |

### 6.2 成分と Goal の対応

| 成分 | Goal 1 (rel sufficient) | Goal 2 (dim natural) | Goal 3 (parsimony) | Goal 4 (FC) |
|---|---|---|---|---|
| `r_solver` | — | — | partial (Over 罰) | ✓✓ |
| `leak_prime` | ✓ (R_G respect, one-sided) | — | — | — |
| **`d_relstruct`** | **✓✓ (R_G equivalence, two-sided)** | — | ✓ (FC-gate 経由 implicit) | — |
| `gain_intrinsic` | — | ✓ (drive coverage) | — | — |
| non-redundant FC-gate | — | — | ✓ (max dim count cap) | — |
| `η · #dims` | — | — | ✓✓ (明示) | — |

### 6.3 Reward mode と gate

全 bonus 項について `bonus + fc` または `bonus + converged` mode を採用し、
penalty mode は使わない。理由は roadmap.md L188-220 で d_feas について
議論された通り、penalty mode は FC 化途中の DoF improvement と競合する
リスクがあるため。

`d_relstruct` は Converged gate (FC でなくとも relations 構造が比較可能)。
これは d_relstruct が dim を見ないため、Pred が under-constrained でも
意味のある signal を出せるという特性を活かす。一方 `leak_prime` は
`V_P = range(R_P)` を要するので Pred dim ≠ 0 が必要 → FC gate で安全側。

---

## 7. Pseudo-relation との相互作用

SketchGraphs GT relations は sparse なので、roadmap.md §A の pipeline で
rel_only RLOO 出力を pseudo-R_G として使うことが前提。

### 7.1 各成分の pseudo 依存

| 成分 | pseudo 依存 | mimicry リスク |
|---|---|---|
| `leak_prime` | ✓ (`J_{R_G}` に pseudo) | 字面 copy で 0 達成 |
| `d_relstruct` | ✓ (`N_G` に pseudo) | **null space 等価 OK → 字面 copy 不要** |
| `gain_intrinsic` | ✗ | n/a |
| parsimony / FC-gate | ✗ | n/a |

`d_relstruct` の soft mimicry 性質は、pseudo に false-positive relation が
混ざった場合の影響を一部緩和する: 字面 mimicry より自由度があるので、Pred は
「pseudo の意図しない false positive」を避けて「pseudo の null space を
表現する別の relation set」を選ぶことができる (理論上)。

### 7.2 必須の安全策

roadmap.md §A の要件 (CAD solver-valid, consensus filter, GT precision/
coverage) に加えて以下を要求する:

- **M1. Pseudo weight 分け**: `R_G = R_G_gt ∪ R_G_pseudo` で、leak_prime と
  d_relstruct の両方で pseudo に低 weight (ρ ≈ 0.5)。
  `J_{R_G}` の row scaling を `1.0 if GT else ρ` にする
- **M2. Stochastic R_G subsampling**: 各 RL step で `R_G` の subset (例 80%)
  を random sample。pseudo の特定 false positive への over-fit を防ぐ
- **M3. KL regularization to pretrain**: 標準的な RLOO KL ペナルティ。
  完全模倣に明示的なコストが乗る
- **M4. Pseudo precision ≥ 0.9 floor**: rel_only 出力に consensus filter
  (例 K=8 中 ≥ 7 で同意) を強くかけ、false positive 率を下げる

### 7.3 leak_prime と d_relstruct の役割分担

両者が pseudo-R_G を共有するので冗長に見えるが、実際は補完関係:

- `leak_prime`: pseudo に含まれる **個別の関係** が破られないようにする
  (under-respect 検出)
- `d_relstruct`: pseudo 全体の **null space 構造** に Pred の構造を合わせる
  (over/under 両方向で構造一致を強要)

これは「pseudo に書かれた個別関係を respect する」(局所) と「pseudo の
全体的 motion 自由度に合わせる」(大域) の二段構成。pseudo に部分的な
false positive があっても、d_relstruct は「representation を変える」自由度を
残すので、Pred は局所的には false positive を避けつつ大域 null space に
合わせる経路を持つ。

---

## 8. 検証手順

### 8.1 Smoke test (rel_only RLOO 完了前にできる)

1. **d_relstruct の annotator variance 計測**
   (`notebooks/test_d_relstruct_cohort.py` を新規作成):
   - v4 と同じ heldout cohort で d_relstruct を計算
   - 期待: heldout `frac<1e-4 ≥ 0.7` (`d_sub` の 0.583 を超える)
   - 失敗時: d_relstruct は d_sub と同程度の annotator variance を持つことに
     なり、報酬としての優位性が落ちる → 別の補完項を再検討

2. **d_relstruct と既存指標の correlation**:
   - 同 cohort で d_H, d_sub, leak_prime, d_relstruct を計算、Spearman 相関
   - 期待:
     - heldout cohort で `d_relstruct ↔ leak_prime` 相関 ≤ 0.5 (直交情報)
     - base / solver_rl cohort で `d_relstruct ↔ leak_prime` 相関 ≥ 0.7
       (failure 軸では一致)
   - 直交性が薄いと d_relstruct を追加する価値が薄い

3. **Over-constraining 検出力**:
   - 人為的に R を増やした synthetic Pred (新規 `rel_add 50 / 100` という
     劣化版を作る) で d_relstruct が monotonic に上がるか確認
   - 同 sample で leak_prime は 0 のままのはず → これが d_relstruct の
     存在価値の crux 証拠

### 8.2 Offline rescore (rel_only RLOO 完了後)

pretrain checkpoint の K=32 sample に対し A″ reward で rescore
(roadmap.md L137-156 と同型):

1. **Winner flip 分析**: 各 component の coeff を 0 にした版との winner 比較
   - leak_prime のみ vs leak_prime + d_relstruct: relation 過剰候補の rank が
     下がるか確認
   - + gain_intrinsic: dim 配置が natural な候補が rank up するか
   - 全成分: 自然な解への concentration が改善するか

2. **Mimicry check**: `leak_prime = 0` を達成した candidate が pseudo-R_G を
   字面 copy しているか測る (pseudo に対する token-level recall)。
   d_relstruct の soft mimicry が機能していれば recall は 100% 未満で
   d_relstruct も 0

3. **Dim count distribution**: A″ reward 上位候補の dim count 分布が
   GT dim count 分布に近いか (Goal 3)

4. **Risky flip 評価**: roadmap.md L194-211 と同様、reward 上位候補が DoF
   悪化や rel count 減少を伴っていないか

### 8.3 Canary (offline rescore で問題なければ)

500-1000 step の短期 RL を A″ で実走 (roadmap.md L437-447 と同型)。
追加で監視すべき W&B metric:

- `d_relstruct_median`, `d_relstruct_p90`
- `leak_prime_median`
- `gain_intrinsic_median`
- `pred_dim_count_mean`, `pred_dim_count_vs_gt_ratio`
- `pred_rel_count_mean`, `pred_rel_count_vs_gt_ratio`
- `pseudo_token_recall` (mimicry check の連続値版)

監視シグナル:

- `pred_dim_count` が GT の 0.8-1.2× に収まるか
- `pred_rel_count` が膨張しないか (over-constraint cascade のサイン)
- `pseudo_token_recall` が 1.0 に張り付かないか (mimicry サイン)

### 8.4 対照実験 (roadmap.md §D 拡張版)

最終的に以下の 4 arm を比較:

| Arm | 報酬 | 期待される結果 |
|---|---|---|
| A0 | r_solver のみ | baseline (solver-only RLOO) |
| A1 | r_solver + leak_prime | d_feas leak 単独効果と等価 |
| A2 | r_solver + leak_prime + d_relstruct | over-constraining 抑制効果 |
| **A″** | A2 + gain_intrinsic + η penalty | フル合成 |

各 arm の評価指標: d_H, d_feas, token-level F1, HistCAD OES, predicted
constraint type distribution, dim/rel count distribution。
d_relstruct と gain_intrinsic の限界寄与が arm 間 diff で測れる。

---

## 9. 開いた論点

### 9.1 d_relstruct の R3 (fully-constrained cohort) 性能

v4_results.md §13.3 では fully-constrained cohort (`n_dim_G ≤ 3`) で
d_feas_gain が縮退する一方、leak_prime は z=+8 を維持した。d_relstruct が
ここでどう動くかは未検証。relation 数は dim 数とは別軸なので、dim 少なくても
relation 多い sketch では d_relstruct が機能するはずだが、empirical な
確認が必要。

### 9.2 N_C の rank deficient 対策

`J_{R_C}` が rank deficient (redundant relations を含む annotation) なら
null space dim が増えるが意味は same。SVD で安全に扱えるが、threshold
(SVD tau) を v4 case D と統一すべきか別管理かは要決定。
[v4.md §7](summary_histcad_response_metric_v4.md) の rank jump 不連続性
の議論を参照。

### 9.3 Multi-reference aggregation

`min_i d_relstruct(P, G_i)` で集約するか。直感的には d_H と同様 `min_i` で
良いが、relation null space は annotator 間で共通な部分が大きいはずなので
`mean_i` でも機能する可能性。smoke test §8.1 で multi-ref distribution を
見て決める。

### 9.4 論文上の位置付け

d_relstruct を paper に書くか reward 専用に留めるか。書く場合は v4 case
D / H と並列の理論導出として appendix で扱えるが、paper main metric は
d_H のまま。**reward の節を v4 paper に追加する場合、d_relstruct を
そこで初出させる構成が clean**。

### 9.5 pseudo-relation pipeline との timing

A″ canary を走らせるには rel_only RLOO が完了していて pseudo-R_G が
用意されている必要がある。一方 §8.1 の smoke test は GT relations だけで
できる。**smoke test → rel_only RLOO 完了 → offline rescore → canary**
の順で進めるのが現実的。

---

## 10. 実装確認後の結論と実行方針

2026-05-22 に既存 RLOO 実装を確認した結果、結論を次のように明確化する。

> **A″ は「ダメ」ではないが、いきなり本命 reward として full RLOO に投入する
> 段階ではない。まずは既存 `gt_reward_coeff` の効果を切り分け、その後
> `leak_prime` → `d_relstruct` → `gain_intrinsic / dim penalty` の順に
> 段階導入する。**

### 10.1 既存 `experiment=rloo` の注意点

`conda run -n freecad python src/train.py experiment=rloo` は、名前上は
SolverReward RLOO だが、現行 config では `gt_reward_coeff: 1.0` が有効である。
つまりこれは厳密には

```text
SolverReward + exact GT constraint-block matching reward
```

であり、pure SolverReward baseline ではない。

GT-match reward は、生成された各 constraint block が SketchGraphs GT 内の
どれかと完全一致した場合、その block 末尾に `gt_reward_coeff` を足す。
これは SG GT に含まれる確実な constraints への anchor として有用だが、
design-intent-equivalent な別表現は報酬しない。また SG GT が sparse なら、
GT に存在しない relational intent を増やす力はない。

したがって、最初に比較すべき baseline は次の 2 本である。

```bash
# pure SolverReward に近い baseline
conda run -n freecad python src/train.py experiment=rloo model.gt_reward_coeff=0.0

# 現行 rloo 設定
conda run -n freecad python src/train.py experiment=rloo model.gt_reward_coeff=1.0
```

ここで `gt_reward_coeff=1.0` が relation count / dim count / d_H / leak_prime を
どれだけ改善するかを確認する。もしこの差が小さいなら、GT-match は補助的
anchor に留め、A″ 系の追加 reward が必要になる。もし差が大きいなら、
A″ は GT-match と併用する形で、過剰な exact mimicry を避ける regularizer として
設計する。

### 10.2 A″ に対する最終判断

A″ の役割は、`d_H` そのものを reward にすることではなく、`d_H` の失敗分解から
得た複数の補助信号を **FC 候補間の intent-aware reranker / tie-breaker** として
使うことである。

採用可能性はあるが、次の理由で staged rollout が必須:

- `d_relstruct` は relation 構造の両側一致を見られるが、pseudo relation の
  semantic false positive には弱い。表現違いには強いが、間違った独立 relation
  が入ると目標 null space 自体が歪む。
- `gain_intrinsic` は Line Length / Arc Radius の drive 可能性を見る proxy であり、
  human-like な offset / angle / inter-entity distance placement を直接は見ない。
- dim penalty は既存 SolverReward の `penalty_num_total_constraints` /
  `penalty_num_dimensional_constraints` と重なる。全候補に強く掛けると、FC 化の途中で
  dimensional constraints を足す必要のある候補まで不利にする。
- 既存実装の FC は `DoF <= fully_constrained_dof` であり、現在は
  `fully_constrained_dof: 2`。A″ の gate も実装上は `per_sample_status == "FC"` に
  合わせるべきで、数式上の `DoF==0` をそのまま使わない。

したがって、A″ は「一発で使う reward」ではなく、以下の段階を通った場合にのみ
full reward として走らせる。

### 10.3 推奨する段階的ロードマップ

#### Step 0: baseline の切り分け

上記の `gt_reward_coeff=0.0 / 1.0` の 2 run を比較する。

見るべき指標:

- `validation_gen/solver_fc_rate`
- `pred_rel_count_mean`, `pred_dim_count_mean`
- token-level relation / dimension F1
- `d_H`, `d_sub`, `leak_prime`
- qualitative examples

この step の目的は、「SketchGraphs GT exact matching だけで十分に relation-rich に
なるのか」を確認すること。十分なら A″ は小さく入れる。足りなければ A″ の必要性が
強まる。

#### Step 1: `d_relstruct` smoke test

RL に入れる前に、GT relations だけで `d_relstruct` を実装・評価する。

必須条件:

- heldout の median が near-zero
- `rel_drop` で上がる
- synthetic `rel_add` で上がる
- `leak_prime` と完全には冗長でない

ここで失敗した場合、A″ の full 構成には進まない。代わりに
`leak_prime` + existing SolverReward + GT-match を優先する。

#### Step 2: offline rescore

pretrain checkpoint から K=32 などで候補を出し、RL ではなく reranking として
次を比較する。

```text
R0 = solver
R1 = solver + GT-match
R2 = solver + GT-match + leak_prime bonus
R3 = R2 + d_relstruct bonus
R4 = R3 + gain_intrinsic bonus + weak dim penalty
```

期待する winner flip:

- FC rate / DoF が悪化しない
- relation count が増える
- dim count が膨張しない
- duplicate GT-match や pseudo-token copy に寄らない
- qualitative に relation-rich scaffold + 少数の自然な driving dimensions へ寄る

この offline rescore で risky flip が出る構成は、そのまま RL に入れない。

#### Step 3: canary RL

offline rescore が通った最小構成から 500-1000 step の canary を行う。
いきなり A″ full ではなく、次の順で足す。

1. `solver + GT-match + leak_prime`
2. `+ d_relstruct`
3. `+ gain_intrinsic`
4. `+ weak dim penalty`

途中で dim count 膨張、rel count 膨張、pseudo token recall の張り付き、
FC rate 低下が出たら、その成分は外す。

#### Step 4: 最終対照実験

最終的な論文用比較は、少なくとも以下を含める。

| Arm | 報酬 | 目的 |
|---|---|---|
| B0 | SolverReward (`gt_reward_coeff=0`) | pure solver baseline |
| B1 | SolverReward + GT-match (`gt_reward_coeff=1`) | SG GT anchor の効果 |
| B2 | B1 + `leak_prime` | relation preservation bonus |
| B3 | B2 + `d_relstruct` | relation 構造一致 / over-relation 抑制 |
| B4 | B3 + `gain_intrinsic` + weak dim penalty | A″ full candidate |

結論は B4 が必ず勝つ必要はない。もし B2 や B3 が最も安定なら、それを採用する。
A″ は「全成分を入れること」が目的ではなく、SolverReward が作る dimension-heavy な
FC 解を、どの最小追加信号で relation-rich / design-intent-aware に寄せられるかを
調べるための設計空間である。

### 10.4 現時点の実務上の結論

- すぐ full A″ RLOO を走らせない。
- まず `experiment=rloo model.gt_reward_coeff=0.0` と現行
  `model.gt_reward_coeff=1.0` を比較し、GT-match の限界を測る。
- `d_relstruct` は有望だが、reward 前に smoke test 必須。
- `leak_prime` は最も安全な最初の追加成分。FC candidate 間の nonnegative bonus として
  入れるのがよい。
- `gain_intrinsic` と dim penalty は最後に足す。dimensional naturalness の proxy であり、
  ここが最も gaming されやすい。
- 最終報酬は A″ full とは限らない。実験で B2/B3 が勝つなら、その方が論文上も防御しやすい。

---

## 関連文書

- 評価指標 d_feas: [v3](summary_histcad_response_metric_v3.md)
- 評価指標 d_H: [v4 設計](summary_histcad_response_metric_v4.md),
  [v4 実験結果](summary_histcad_response_metric_v4_results.md)
- d_feas 報酬化先行検討:
  [d_feas reward roadmap](summary_dfeas_reward_rloo_roadmap.md)
