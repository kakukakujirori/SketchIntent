# `d_feas` の連続性問題と response 指標再設計

作成日: 2026-05-21

> **この文書の位置付け**
> v3.md (= `summary_histcad_response_metric_v3.md`) の内容を仮定する。すなわち
> response-map `S_C^(q) = R_C pinv(J_q R_C)` に基づく現行 `d_feas`、その HistCAD OES
> に対する優位性 (§6.1, §6.2, §7)、および d_resp が棄却された経緯を既知とする。
>
> 本文書は、現行 `d_feas` の **gain term が per-probe では本質的に discrete である**
> という構造的問題を出発点に、まず probe-free / subspace 系の fix 案 (A–H) を比較検討する。
> その後、`q(x)` が Pred と GT を比較するための共通 observable / edit axis として不可欠である
> 可能性を踏まえ、`q` を残したまま gain を連続化する追加案 (I–O) を追記する。
> 実装と数値検証はこの後に続く。

## 1. 問題提起: 現行 `d_feas` の gain term は連続値ではない

v3.md line 186–188 で `d_feas` の利点として 2 つが挙げられている。

> 1. `q(x)` が target sketch に含まれているかどうかに関わらず計算できる
> 2. **上式の差分は連続値のエラーとして観測可能である**

主張 2 が、gain term `T_P^(q) = J_q S_P^(q) - I_q` については **per-probe で偽** である
ことが本検討の出発点。これは数学的に厳密に示せる。

### 1.1 数学的構造

`S_C^(q) = R_C pinv(J_q R_C)` を代入すると、

```text
M := J_q R_C                            (k_q × p 行列、p = R_C の dimensional column 数)
J_q S_C^(q) = M M^+
```

ここで `M M^+` は **range(M) への直交射影**。固有値は 0 または 1 しか取らない。したがって

```text
||T||_F^2 = ||M M^+ - I_{k_q}||_F^2
         = trace(I - M M^+)             (I - M M^+ も idempotent な射影)
         = k_q - rank(M)
```

すなわち per-probe gain は `(k_q - rank(M)) / k_q ∈ {0, 1/k_q, 2/k_q, …, 1}` という
**有限離散値**しか取らない。

### 1.2 単一 dimension probe (k_q=1) では完全に二値

実験で使う probe `q(x) = Distance(...)` や `Radius(...)` は scalar 出力なので
`k_q = 1`。このとき `rank(M) ∈ {0, 1}` となり、**gain は厳密に {0, 1} の binary**。

これは HistCAD の `T` (edit が finite solve で通るか) の線形化版にすぎず、`d_feas`
独自の付加情報を何も持っていない。

### 1.3 連続性は probe aggregation からのみ生まれている

§6.2 dropout sweep の median gain `0.000 / 0.000 / 0.167 / 0.412 / 1.000` は連続的に
見えるが、これは

- per-probe で {0, 1}
- `Q_G` 内 (典型的に数〜十数 probe) で平均 → 0 と 1 の Bernoulli rate
- rotation 集合で median

という二重集約の結果にすぎず、**指標自体に連続な勾配があるわけではない**。

50% dropout の median 0.167 ≈ 1/6 という値は「中央 rotation で 6 probe のうち 1 個が
rank-deficient」というカウントを反映しているだけで、design intent の連続的な乖離度を
測っているわけではない。

### 1.4 一方 leak term は per-probe でも真に連続

```text
L = J_{R_G} S_P^(q) = J_{R_G} R_P (J_q R_P)^+
```

これは射影ではなく `R_P` の滑らかな関数。`R_P` が GT の relation Jacobian の
null space に乗っていればちょうど 0、外れれば滑らかに増える。§6.1 で Vitruvion leak が
0.0558 / 2.552 など実数値で出ているのはこの理由。

### 1.5 帰結

- v3.md line 188 ②「上式の差分は連続値のエラーとして観測可能」は、**leak について
  は正しく、gain については aggregation 後にのみ正しい** と書き直すべき。
- gain term は実質「`q` 方向が `R_C` の dimensional column span に乗っているか」の
  binary test で、HistCAD `T` の linearization 版にすぎない。`d_feas` が HistCAD に
  対して持つ本質的優位は **leak の連続性** に集約される。
- gain term は指標として informative でないので、再設計すべき。

## 2. 修正案の探索

7 つの方向性を順に検討する。

### 2.1 案 A/B: Tikhonov 正則化された pseudo-inverse

`pinv(M)` を damped least-squares で置き換える:

```text
S_C^(q, λ) := R_C M^T (M M^T + λ I_{k_q})^{-1}
J_q S_C^(q, λ) = M M^T (M M^T + λI)^{-1}
T_λ = J_q S_C^(q, λ) - I = -λ (M M^T + λI)^{-1}
```

`M` の特異値を `σ_i` とすると `T_λ` の固有値は `-λ / (σ_i^2 + λ) ∈ (-1, 0)` で
σ_i に対して連続。`||T_λ||_F^2 = Σ_i λ^2 / (σ_i^2 + λ)^2`、`k_q=1` でも連続。

#### 解釈

`λ` は「`dp_C` を動かすコスト」。linear program では

```text
min_{dp} ||M dp - dq||^2 + λ ||dp||^2
```

を解いており、`||M||` が小さい (probe `q` が dimensional driver から弱くしか駆動
されない) ほど gain residual が滑らかに 1 に近づく。

#### 評価

- **長所**: 数学的に clean、`λ → 0` で現指標に reduce、per-probe で真に連続。
- **短所**: `λ` の選び方が hyperparameter として残り、calibration が必要。なにより
  ユーザ指摘の通り「`q` を probe として固定する」発想自体は維持されたままで、
  本質的な指標観の変更を伴わない **patchwork** の印象が強い。

### 2.2 案 C: probe を多次元化

`q(x)` を scalar から「dimension family の同時編集」のような vector probe にし、
`k_q > 1` にする。すると `(k_q - rank)/k_q ∈ {0, 1/k_q, ..., 1}` で granularity が
細かくなる。しかし依然 discrete で、根本解決ではない。**補助のみ採用可。**

### 2.3 案 D: response 部分空間そのものを比較する (probe-free)

`q` を介する発想は HistCAD の「edit が通るか」バイナリ判定を引きずったもの。
本来の目的は「Pred と GT が同じ可動多様体 (= 同じ design intent) を表すか」を測ること
なので、`q` を捨てて **`R_C` の像 (range) 自体** を比較すれば足りる。

```text
V_C := range(R_C) ⊆ T_{x_0} R^{n_x}    (sketch C が許す primitive motion 全体)
P_C := V_C への直交射影
d_sub(P, G) := ||P_P - P_G||_F^2 / d_norm
            = (dim V_P + dim V_G - 2 Σ_i cos^2 θ_i) / d_norm
```

ここで θ_i は `V_P` と `V_G` の principal angles、`d_norm = max(dim V_P, dim V_G)`。

#### 性質

- **probe を選ぶ必要が一切ない**。`Q_G` 設計、`source_validated` フィルタ等が消える。
- **per-rotation で連続値**。`R_C` の SVD は smooth (rank jump 以外で)、principal
  angles も smooth。
- **両方向の failure を一発で捕捉**:
  - Pred が relation を壊す → `V_P ⊄ V_G`、principal angle が立つ
  - Pred が under-constrained → `dim V_P` が `dim V_G` を超過、または向きが異なる
  - Pred が over-constrained → `dim V_P < dim V_G`、対称的に捕捉
- **diagnostic decomposition**:

  ```text
  align_loss = Σ sin^2 θ_i / min(dim_P, dim_G)     # 部分空間の傾き (旧 leak 対応)
  rank_loss  = |dim V_P - dim V_G| / d_norm         # 可動次元の不足/過剰 (旧 gain 対応)
  ```

  `rank_loss` は整数差なので依然 discrete だが、これは sketch のトポロジ的な構造差を
  反映しているので「悪い discrete」ではない。

#### dropout sweep / Vitruvion leak が正しく動くか

- 50% dropout: dimension column が半分減るので `dim V_P` が縮む → `rank_loss > 0`、
  `align_loss ≈ 0` (残った方向は GT に乗っている)。§6.2 と同方向。
- 100% dropout: `V_P = {0}`、`d_sub = dim V_G / d_norm = 1`。
- Vitruvion relation mismatch: `dim V_P ≈ dim V_G` だが向きが違う → `align_loss > 0`。
  §6.1 と同方向。

### 2.4 案 E: 観測空間での subspace 距離 (d_resp のリハビリ版)

primitive 空間 `R^{n_x}` は位置・半径・角度が混じり scale が不均一。`J_phi` (vertex
観測 Jacobian) で観測空間に projection してから比較する:

```text
W_C := range(J_phi R_C) ⊆ R^{2 · N_vertices}
d_obs(P, G) := ||P_{W_P} - P_{W_G}||_F^2 / d_norm
```

#### v3.md の d_resp 却下理由の再検討

v3.md は d_resp を「Vitruvion vs heldout で p=0.495、区別不能」として棄却したが、
この却下理由には 2 つの混在がある。

1. **annotator variance**: heldout 同士で `J_phi S_C^(q)` がそもそも揺れる
2. **指標構造の問題**: `J_phi S_C^(q)` を **行列要素として** 比較したため、`pinv` が
   選ぶ min-norm 方向が `R_C` のパラメトリゼーションに依存し、design-intent-
   equivalent な sketch でも数値が違ってしまう

(1) は却下理由として弱い (annotator 間のブレは前提として正しい挙動)。問題は (2) で、
これは **entry-wise comparison** ゆえの artefact。**subspace 化すれば (2) が消える**:

- `pinv` の min-norm tiebreak は range には影響しない。`range(J_phi R_C)` は `R_C` の
  min-norm 選択と独立。
- annotator が dimension の表現を変えても、同じ motion を許すなら同じ subspace。

つまり **「d_resp は失敗だが、d_resp の subspace 化 (案 E) は別物」**。v3.md の却下は
entry-wise 比較固有の失敗であって、subspace 比較の失敗ではない。再検討の価値が十分にある。

### 2.5 案 F: kernel (凍結方向) 表現

`R_C` の像でなく orthogonal complement (= 凍結方向) を見る:

```text
N_C := V_C^⊥
d_kerel(P, G) = ||P_{N_P} - P_{N_G}||_F^2
```

subspace 距離は補空間でも値が変わらないので、案 D と数学的に等価:

```text
||P_V - P_W||_F = ||P_{V^⊥} - P_{W^⊥}||_F
```

ただし「どの方向の動きが禁じられているか」という説明の方が relational design intent
として自然な場合があり、論文上の表現として案 D の代替として候補。

### 2.6 案 G: q を bank 化して expectation を取る

`q` を捨てたくない場合のみ。`q` を多次元 random probe にして `k_q` を大きくとり、
さらに probe を多数 random sample して expectation:

```text
d_feas_smooth = E_q [ ||T||_F^2 / k_q + ||L||_F^2 / n_R ]^{1/2}
```

`q` が Haar 分布 (orthogonal group) からくる random subspace なら、closed form で

```text
E_q[T^T T] ∝ (I - P_{V_C})
```

となり、結局 **案 D に漸近**する。したがって案 G を真面目にやると有限 sample の noisy
estimator にしかならず、**案 D の解析的版が strictly 優越**する。

### 2.7 案 D 単体の弱点と、それを補う案 H

user の指摘: 「subspace 比較だと fully constrained に近い場合に比較できる情報が限定的に
ならないか」。これは構造的に正しい懸念で、定量化すると:

| 量 | 個数 (n_dim = k) |
|---|---|
| principal angles `θ_i` | `min(dim V_P, dim V_G)` |
| dimension mismatch | 1 (整数) |
| 計 | **`~ k + 1` 個** |

これに対して現状の `d_feas` は per rotation で:

| 量 | 個数 (Q_G = k 個の probe) |
|---|---|
| gain (binary) | k |
| leak vector `J_{R_G} S_P^(q_i)` | **`k × n_R` 要素** |

raw 情報密度は d_feas が大きい。しかし以下の補題がここで本質的:

#### 補題: d_feas の leak 情報は subspace + operator 1 個から再構成可能

`S_P^(q) = R_P pinv(J_q R_P)` の range は `range(R_P) = V_P` (`J_q R_P` が full row rank
なら)。したがって `J_{R_G} S_P^(q)` は **常に `J_{R_G} R_P` の column space に乗る**。
probe を振っても、出てくる leak vector は

```text
J_{R_G} R_P : R^p → R^{n_R}
```

という (n_R × dim V_P) operator の (probe 方向で重み付けされた) 部分像にすぎない。
d_feas の「k 個の probe で得た k × n_R 個の leak entry」は、実は

```text
||J_{R_G} R_P||_F^2     ─ G の relation を P の motion が破る総量
```

の (n_R × dim V_P) 個の entry から完全に再構成可能。**d_feas の probe-by-probe leak は
operator の sampling、operator を直接見る方が ceiling に届く。**

#### 案 H: subspace + operator leak の hybrid (probe-free)

`q` を完全に捨てた上で、 fully constrained 時の薄さ問題を operator-based leak で
補う 2 成分指標:

```text
gain'(P, G) = ||P_{V_P} - P_{V_G}||_F^2 / d_norm        # subspace 構造差
leak'(P, G) = ||J_{R_G} R_P^orth||_F^2 / n_R             # P motion が G relation を破る operator norm
                                                          # R_P^orth = R_P を column 正規化したもの
d_H(P, G)   = sqrt(gain' + leak')
```

`R_P^orth` は `R_P` の SVD `R_P = U Σ V^T` で `U` (column orthonormal) を取ったもの。
これで `dp_C` の単位任意性が消える。

##### 性質

1. **probe 不要** — `q` の方向選択依存性が完全に消える
2. **per-rotation で連続** — `R_P` の SVD は smooth、`J_{R_G} U_P` も smooth
3. **gain' / leak' の分解の意味が保たれる**:
   - `gain' = 0 ∧ leak' = 0` ⇔ `V_P = V_G` (perfect intent equiv)
   - `gain' > 0 ∧ leak' = 0`: `V_P ⊂ kernel(J_{R_G})` だが `V_G` と次元 or 向きが違う
     (over-constrained Pred、dimension 構造差)
   - `gain' ≈ 0 ∧ leak' > 0`: 次元数は揃うが Pred motion が relation を破る (現 leak)
4. **fully constrained でも情報が痩せない** ─ `leak'` は `(n_R × dim V_P)` operator で、
   `n_R` (= 関係制約の本数) は fully constrained でも大きい (むしろ最大)。**user 指摘の
   regime では `leak'` が主要 channel になり、continuous discrimination が確保される。**
5. **d_feas の上限に届く** ─ §7 の「solver-RL は leak を閉じない」観察は
   `||J_{R_G} U_P||_F / sqrt(n_R · dim V_P)` で同じ向きの結論が出る (scale 不変)

##### fully constrained 時の役割分担

| regime | gain' (subspace) | leak' (operator) |
|---|---|---|
| dim V_G 大 (under-constrained 領域) | 主役 (次元差で discrimination) | 補助 |
| dim V_G 小 (fully constrained) | 痩せる (次元差が少ない) | **主役 (n_R で discrimination)** |

両者は補完的で、`V_G` の dimension 数に応じて自動的に役割分担される。

#### 案 Hθ: subspace + angular leak の hybrid (probe-free, 2026-05-24 追記)

実験後の再検討で、案 H の `leak_prime` は paired test では強い signal を持つ一方、
絶対値スケールでは `d_sub` よりかなり小さく、`d_H = sqrt(d_sub^2 + leak_prime^2)`
の total score では実質的に `d_sub` が支配的になることが分かった。これは
`d_sub` が principal-angle / projection distance であるのに対し、`leak_prime` が
`J_R` の residual operator norm であり、両者の単位が揃っていないためである。

そこで GT relation の forbidden normal subspace を

```text
N_G := orth(row(J_{R_G}))
```

Pred の許す primitive motion subspace を

```text
U_P := orth(range(R_P))
```

として、scale-free な angular leak を定義する:

```text
leak_angle(P, G)^2 = ||N_G^T U_P||_F^2 / dim(V_P)
```

これは「Pred が許す motion が、GT relation が禁じる normal direction にどれだけ
乗るか」を平均 squared cosine として測る。`J_R` の絶対 scale を捨て、
`d_sub` と同じ subspace angle 系の量に揃える。

`d_sub` 側も同じ angular scale に合わせるため、padded principal-angle form を併記する:

```text
d_sub_angle(P, G)^2 = 1 - Σ_i cos^2 θ_i / max(dim V_P, dim V_G)
```

足りない次元は 90 度の angle とみなす。したがって nested dropout では現行 `d_sub`
と同じ値を保ち、等次元の傾き差では `d_sub_angle = d_sub / sqrt(2)` になる。

単一 scalar が必要な場合は

```text
d_H_angle(P, G) = sqrt(d_sub_angle^2 + leak_angle^2)
```

を使う。これは heuristic な重み付け和ではなく、両成分を「正規化された angular error」
という同じ単位に変換してから合成する設計である。

実験結果は
[`summary_histcad_response_metric_v4_results.md`](summary_histcad_response_metric_v4_results.md)
§14 に追記した。結論として、paper では `d_sub_angle` / `leak_angle` の 2 軸 report を
主診断とし、単一 scalar が必要な表では `d_H_angle` を使うのが最も defensible。
最大限 parsimonious な主指標が必要な場合のみ、`d_sub` または `d_sub_angle` を main、
`leak_angle` を mandatory diagnostic とする。

## 3. 比較表

| 案 | per-rotation 連続性 | `q` 設計依存 | gain/leak 分離 | annotator var. 耐性 | 実装コスト |
|---|---|---|---|---|---|
| 現状 d_feas | **弱** (gain が binary aggregation) | 強 | 明示 | 中 | (実装済) |
| A/B: Tikhonov | 強 | 強 | 明示 | 中 | 低 (パッチ) |
| C: probe 多次元化 | 中 (依然 discrete) | 強 | 明示 | 中 | 中 |
| D: subspace (primitive) | **強** (rank jump 除く) | **無** | 再現 (align/rank) | **強** | 中 |
| E: subspace (観測, d_resp 改) | 強 | 無 | 再現 | 強 | 中 |
| F: kernel 表現 (D の双対) | 強 | 無 | 再現 | 強 | 中 |
| G: q bank | 中 (MC noise) | 弱 | 明示 | 中 | 高 |
| **H: subspace + operator leak** | **強** | **無** | **明示 (gain'/leak')** | **強** | 中 |
| **Hθ: subspace + angular leak** | **強** | **無** | **明示 (subspace/leak angle)** | 中〜強 | 中 |

## 4. 指標選定のための meta-evaluation criteria

候補を絞り込むには「良い指標とは何か」を先に定義する必要がある。本節はそれを
**v3.md の ablation 集合から逆算**して整理する。提案手法側が拡張されたため、必要条件
/ 順位付け基準は提案された全候補 (現 d_feas / Tikhonov / D / E / F / H / I–O) を同じ
ものさしで測れる形にしてある。特に I–O の追加により、`q` を残す候補に固有の
「共通土俵としての probe 妥当性」「正則化 hyperparameter による退化」「relation leak を
小さく見せる trivial response」を明示的に評価する必要がある。

### 4.1 v3.md の ablation から抽出できる criteria

v3.md の各実験は、指標が満たすべきプロパティを陰に検証している:

| v3.md の項目 | 暗に testing しているプロパティ |
|---|---|
| §6.3 `source_self` ≈ 0 (frac<1e-4 = 1.00) | **同一性** ─ Pred = GT で指標 0 |
| §6.3 `heldout` の median ~0, frac<1e-4 = 0.73, Q90 = 0.13 | **annotator variance 吸収** ─ 別 human を near-zero に収める |
| §6.1 paired test (Vit vs heldout) p = 1.5e-19 / 1.2e-31 | **model vs human の方向正しさ + 強度** |
| §6.1 base の OES sign inversion (p = 2.2e-19, 反対方向) | **under-constrained model に対する sign correctness** ─ HistCAD はここで失格 |
| §6.2 dropout sweep `0 → 0.167 → 0.444 → 1` の monotonic | **degradation monotonicity** (集約 median) |
| §6.2 within-rotation up-rate (35.6 → 83.0%) | **per-rotation monotonicity** (集約後だけでない) |
| §6.3 `dim_drop100 = 1`, `rel_drop50/100` で tail | **definitional saturation + relation 劣化検出** |
| §6.3 eps ∈ {1e-11, 1e-9, 1e-7} で完全一致 | **eps 数値安定性** |
| §6 model-output gate sensitivity (14% vs 77% admit) | **gate 選択への robustness** |
| §6 bbox normalization (commit 8e6945d) | **coordinate scale 不変性** |
| §7 RL の gain (0.121 → 0.042) vs leak (0.059 → 0.068) | **failure mode 分解の診断性** (gain と leak で動きが違う) |
| §7 fc-rate / eval-rate / OES / F1 の整合 | **token-level metric と方向一致** |
| §8 CDF / paired Wilcoxon | **分布形状の正しさ** (model 分布が右シフト) |
| §9 divergent cases | **qualitative interpretability** (OES が高いのに指標が立つ case を列挙可能) |
| source-conditioned setup (`Q_i`, `R_i` を各 source GT から作る) | **common-ground validity** ─ Pred と GT を同じ edit / observable axis で比較しているか |
| dim/drop と rel/drop を独立に作る control | **component selectivity** ─ dimension loss と relation loss を別成分で説明できるか |
| finite edit local check | **first-order / finite consistency** ─ 線形 metric が小さい finite edit と局所的に矛盾しないか |
| suppl PP-Coincident ablation (p = 6.3e-17 維持) | **単一 relation 種への過依存性が無い** |
| suppl smatch ablation | **token-level baseline からの独立性** |
| §3 外れ値除外 (median / quantile を使う) | **heavy-tail への robustness** |

### 4.2 必要条件 N1–N13 (pass/fail) ─ 落ちたら候補から外す

これらは「指標 X が論文の主指標に成り得るか」の足切り。1 つでも落ちたら却下する。
閾値は v3.md の現 `d_feas` の数値を起点に設定している。

- **N1. 同一性**: `source_self` の `frac < 1e-4` ≥ 0.99
- **N2. annotator variance 吸収**: `heldout` の median が `dim_drop50` の median より明確に
  小さい (現 d_feas で `~0 vs 0.151`)
- **N3. 方向正しさ**: paired test (model vs heldout) で **human-better > model-better**、
  `p < 1e-5`。HistCAD OES の sign inversion に陥らない
- **N4. dropout 単調性**: drop 0 ≤ 25 ≤ 50 ≤ 75 ≤ 100 が median で成立、かつ within-
  rotation up-rate が 75 → 100 transition で `> 50%`
- **N5. relation 劣化検出**: `rel_drop50/100` の median が `heldout` より明確に大きい
- **N6. eps 安定性**: eps を 2 桁動かして median / paired stat が一致
- **N7. gate 不変性**: strict / relaxed gate 両方で結論 (sign + p の order of magnitude) 一致
- **N8. coordinate scale 不変性**: bbox 正規化の dst_length を 2 倍にしても結論一致
- **N9. admit rate**: 評価可能 rotation 割合 `≥ 70%` (現 base 77%, solver_rl 94%)
- **N10. common-ground validity**: Pred と GT を native parameter 空間で直接比較しない。
  `q`-retaining 候補では `q` が source GT 由来の同一 semantic edit axis であること、
  probe-free 候補では primitive / observable subspace が `q` の代替として比較可能な共通空間に
  なっていることを明示する。
- **N11. component selectivity**: `dim_drop` は主に driver/gain/effort 系成分で、
  `rel_drop` は主に relation/leak 系成分で悪化する。total score だけで両者を区別できない
  候補は、主指標ではなく補助指標扱い。
- **N12. hyperparameter non-degeneracy**: `λ`, `μ`, `τ`, normalization などを極端に振っても、
  全サンプルが near-zero / ceiling に collapse しない。main setting は test result を見てから
  選ばず、事前 grid / validation split / scale heuristic で固定する。
- **N13. no failure hiding**: 最適化型候補 (K/M/L/N) は、response を小さくして leak だけを
  下げる、または relation を無視して gain だけを下げる trivial solution を total score が必ず
  罰する。gain / leak / effort は必ず別カラムで報告する。

### 4.3 順位付け基準 R1–R12 (degree) ─ N を全部通った候補の中で差を付ける

ここからは pass/fail でなく degree。指標選定の主戦場。

#### R1. discrimination 強度

Vit vs heldout の paired Wilcoxon の `|z|` (大きいほど良い)。base / solver_rl 別々で
評価する。現 d_feas: base `p = 1.2e-31`、solver_rl `p = 1.5e-19` が baseline。

#### R2. **per-rotation での連続性** ─ 本作業の主目的

- **テスト**: drop rate を {0, 10, 20, ..., 90, 100}% で細かく振り、per-rotation で指標値が
  monotonic に滑らかに変化するかを確認
- 現 d_feas の gain は per-probe binary aggregation なので、refinement で sample 数を
  増やすと階段状のヒストグラムが出るはず。**真に連続な候補ではヒストグラムが滑らか**
- **定量化**: 各 rotation について隣接 drop rate での値差分の分布、その IQR が drop rate
  に対して滑らかに変動するか。あるいは「per-rotation values が `< 1/Q_G` 単位の有限
  集合に集中する」かどうかを KS-test で確認

#### R3. **subset cohort discrimination** ─ fully constrained 時の薄さ問題

- heldout cohort を `n_dim_G ∈ [1, 3]` に絞って §6.1 paired test を再計算
- 現 d_feas / 案 D / 案 H で `p` が落ちないかを比較
- **落ちる候補は user 懸念の通り情報量不足**で減点

#### R4. 診断的分解の richness

- 分解 (gain/leak、align/rank、gain'/leak' 等) が:
  - 互いに直交的な情報を運ぶか (Pearson `|ρ|` が小さい方が良い、目安 `< 0.5`)
  - §7 の model 改良軸 (RL は gain' を直すが leak' を直さない) を再現できるか
- **RL improvement decomposition が再現できないと §7 paper claim が崩れる**

#### R5. ablation robustness

- PP-Coincident を抜く / Equal を抜く / 距離系のみ等の relation subset で結論が安定か
- smatch ablation (相補的 metric との独立性) で結論が変わらないか
- **量**: 各 ablation 下での paired `p` の log-scale 変動幅。1 桁以内が望ましい

#### R6. design parsimony

- hyperparameter 数 (Tikhonov の λ、δ scale、ε、normalization choice 等)
- probe 設計の必要性 (案 D/E/F/H は probe 不要、現状 / Tikhonov / 案 G は probe 設計依存)
- 論文への書きやすさ (導出が paper 本文 1 ページに収まるか)

#### R7. heavy-tail への robustness

- median と Q75, Q90 の order が一致 (mean ベースで主張を組まないこと)
- §3 の外れ値 (`9/55749...` 等) を除外しても結論不変

#### R8. qualitative interpretability

- §9 divergent case を生成できるか (OES と乖離する rotation を上位/下位で列挙)
- principal angles / leak 方向の visualization が可能か

#### R9. edit-axis coverage / task relevance

- `q`-retaining 候補では、`Q_G` が source GT の edit axes をどれだけ覆っているかを見る。
  例えば probe 数、dimension type coverage、source validation pass rate、`J_q R_G` の rank /
  singular spectrum を report する。
- probe-free 候補では、`q` を捨てても edit-response task と関係が保たれているかを見る。
  具体的には `Q_G` に対する downstream gain/leak や finite soft-OES と相関するかを確認する。

#### R10. first-order / finite consistency

- linear metric が低い case で finite edit residual も低いか、高い case で finite residual が
  高いかを見る。完全一致は不要だが、rank correlation が正であることは欲しい。
- FreeCAD branch change の影響があるため、これは pass/fail ではなく diagnostic。論文では
  appendix evidence として扱う。

#### R11. human-distribution calibration

- `source_self`, `heldout`, `model`, `dropout controls` を単に median で比べるだけでなく、
  human-human 分布を「許容帯」として使う。
- 例: `P(score(model) > Q90(heldout))`, AUROC(heldout vs model), AUROC(heldout vs dropout),
  nearest-human / multi-reference likelihood。annotator 間差が大きい sketch は「指標失敗」
  ではなく「human intent が多峰的」と扱えるかを見る。

#### R12. computational and reward usability

- full evaluation だけでなく、RLOO reward / candidate reranking に使える計算量かを見る。
  K/M/N は solve or linear system を probe 数だけ解くため、H/D より重い可能性がある。
- 1 sample あたりの runtime、failure rate、GPU/CPU 依存、batch 化可能性を report する。
  主論文 metric と training reward は同一でなくてもよいが、reward 化できる候補は加点。

### 4.4 候補絞り込みへの適用 (scorecard)

現時点で v3.md / 本文書から推定できる範囲で各候補を埋めると:

| 候補 | N1–13 | R1 | R2 (連続) | R3 (薄さ) | R4 (分解) | R5 (ablation) | R6 (parsimony) |
|---|---|---|---|---|---|---|---|
| 現 d_feas | ✓ | 強 (実証) | **✗ (gain binary)** | 未検証 | 強 (実証) | ✓ (実証) | 中 (probe 必要) |
| A/B Tikhonov | ✓ 予想 | 同等? | ✓ (理論) | 未検証 | 強 (継承) | ✓ 予想 | 低 (λ 追加) |
| D subspace | ✓ 予想 | 未検証 | ✓ (rank jump 除く) | **△ (薄くなる)** | 中 (align/rank) | 未検証 | **高 (probe 不要)** |
| E observable subspace | ✓ 予想 | 未検証 | ✓ | △ | 中 | 未検証 | 高 |
| H subspace + leak op | ✓ 予想 | 未検証 | ✓ | **✓ (leak で補う)** | **強 (gain'/leak')** | 未検証 | 高 |
| I/J effort/budget | ✓ 予想 | 同等? | ✓ (理論) | 未検証 | 中 (gain 修正中心) | 未検証 | 中〜低 (scale/τ) |
| **K relation-preserving controllability** | ✓ 予想 | 未検証 | **✓** | ✓ 予想 | **強 (gain/leak/effort)** | 未検証 | 中 (λ/μ) |
| **M stacked K / spectrum** | ✓ 予想 | 未検証 | **✓** | ✓ 予想 | **強** | 未検証 | 中 (λ/μ + Q stack) |
| L canonical response match | ? (N2/N13 要注意) | 未検証 | ✓ | 未検証 | 中 | 未検証 | 低 (定式化複雑) |
| N soft-OES | ? (solver failure 依存) | 未検証 | ✓ (finite residual) | 未検証 | 中 | 未検証 | 低 (重い) |
| O human distribution | protocol 改善 | AUROC で評価 | n/a | n/a | 強 (分布化) | 未検証 | 低 (refs 必要) |

優先度の高い実験は scorecard の空欄を埋める順に決まる:

1. **R2 (連続性 test) を全候補で**: drop rate を 10% 刻みで振って per-rotation 軌跡を見る。
   「現 d_feas より良い」と言える最も直接的な証拠。**まずこれを現 d_feas に対して
   行って "本当に階段状か" を実データで確認**するのが、本検討の出発点 (= "gain は
   離散" claim) を経験的に裏付けるためにも有意義
2. **N10 / R9 (common-ground validity) を D/H と K/M で比較**: `q` を捨てることで
   edit-response task から外れていないか、逆に `q` を残すことで source edit axes を
   十分覆えているかを確認する
3. **N12 / N13 を K/M で**: `λ, μ` grid によって trivial solution や collapse が起きないか、
   gain / leak / effort の個別成分で確認する
4. **R3 (subset cohort) を D/H/K/M で**: user 懸念が実データで具現化するか、案 H/K/M の
   operator leak で実際に補えるかの crux 検証
5. **R4 (分解の richness) を H/K/M で**: §7 の RL diagnostic が `gain'` / `leak'`
   または `gain_K` / `leak_K` / `effort_K` の動きで再現するかを
   `model_ranking_table.csv` と同形式で出す
6. **N9 (admit rate)** を案 D/E/F/H/K/M で確認: probe validation を捨てる/変える代わりに `R_C` の
   rank deficiency が極端な rotation を除外する必要が出るかの sanity check

### 4.5 注意点

- **R2 の per-rotation 連続性**は「集約後の median が滑らか」だけでなく「個々の rotation
  で値が滑らかに動く」ことが本質。v3.md §6.2 sweep は集約値の monotonicity しか
  見ていないので、**per-rotation trajectory を新規に出す実験が必要**
- **R5 ablation** は paper supplement の主軸で reviewer 効果が大きい。新指標でも同じ
  ablation を回す必要がある (PP-Coincident、smatch、gate sensitivity の 3 つは特に)
- **N3 (sign correctness)** は HistCAD OES が落とした最重要 criterion。新指標でも
  compliance を必ず数値で示す必要がある
- **N10 (common-ground validity)** は `q` を捨てる案と残す案の本質的な分岐。D/H が
  ここで弱ければ main metric ではなく subspace diagnostic、K/M がここを満たして N12/N13
  も通れば main metric 候補として最優先になる
- **N12/N13** は K/M/L のような最適化型 metric で特に重要。total score が良く見えても、
  成分を見ると `u≈0` で leak を隠しているだけ、または `μ` が小さすぎて relation を無視している
  だけなら却下する
- scorecard は **検証実験の優先度を決める** ためのもの。N にひっかかれば即却下、R は
  全候補で埋めてから総合判定する。一発の R-score で決めない

## 5. 検証すべき問い

実装に先立ち、以下を実データで確認する。各項目は §4 の criteria のどれを埋めるかを
明示する。

### 5.1 案 D の妥当性検証 (N1–5, R1, R7 を埋める)

- §6.1 paired test を `d_sub` で再計算: Vitruvion vs heldout-drop0 の discrimination は
  現 `d_feas` (p = 1.5e-19 / 1.2e-31) と同等以上か?
- §6.2 dropout sweep が単調に劣化するか? (現 `d_feas`: median 0/0/0.167/0.444/1.000)
- §7 model ranking で solver_rl の改善 (`d_feas` 0.121 → 0.042) が `d_sub` でも検出できるか?

### 5.2 案 D の弱点 (fully constrained 時の薄さ) の定量化 (R3 を埋める)

- heldout cohort を `n_dim_G ≤ 3` の subset に絞り、`d_sub` の discrimination power が
  落ちるかを確認
- 落ちる場合、案 H の `leak'` がそれを補えるかを同じ subset で確認

### 5.3 案 H の全面検証 (N1–13, R1, R2, R4, R9 を埋める)

- §6.1, §6.2, §7 すべてを `gain'` / `leak'` / `d_H` で再計算
- 現 `d_feas` との paired comparison: tie/win/loss を rotation 単位で見る
- §7 の「RL は leak を閉じない」観察が `leak'` でも再現するか
- §6.2 dropout sweep で `gain'` が真に連続な単調増加を見せるか (R2 の核心テスト)
- `q` を使わない H が edit-response task から外れていないか、`Q_G` に対する downstream
  gain/leak または finite soft-OES との相関で確認する (N10/R9)

### 5.4 案 E vs D の比較 (R6 と R8 を埋める)

- primitive 空間 vs 観測空間 (J_phi 投影) の選択は別レイヤの decision
- 両方で同じ実験を回し、観測空間版が discrimination で勝るか確認
- 論文上の物語性 (geometry に対する design intent) では案 E の方が書きやすい

### 5.5 ablation robustness の継承確認 (R5 を埋める)

- PP-Coincident / smatch / gate sensitivity の 3 ablation を新指標で再 run
- 各 ablation 下で paired `p` の order of magnitude が変わらないことを確認

### 5.6 q-retaining family K/M の検証 (N10–13, R2, R4, R9, R12 を埋める)

- 案 K (`relation-preserving controllability`) と案 M (`Q_G` stack 版) を最優先 q-retaining
  候補として実装し、`gain_K`, `leak_K`, `effort_K`, `d_K` を CSV に出す
- `λ, μ` grid を固定して sweep し、total score が near-zero / ceiling に collapse しないか、
  `u≈0` で leak を隠すだけになっていないかを成分別に確認する (N12/N13)
- `Q_G` coverage を report する: source ごとの probe 数、dimension type coverage、
  `J_Q R_G` の rank / singular spectrum、source validation pass rate (N10/R9)
- dim/drop と rel/drop の独立 control で、dim は `gain_K` / `effort_K`、rel は `leak_K`
  を主に動かすかを確認する (N11/R4)
- H/D と同じ paired test, dropout sweep, model ranking を回し、`q` を残す利点が
  discrimination / interpretability / reward usability に出るか比較する

### 5.7 human-distribution calibration と finite consistency (R10, R11 を埋める)

- human heldout 分布を許容帯として、各候補の `P(model > Q90(heldout))`,
  AUROC(heldout vs model), AUROC(heldout vs dropout) を出す
- raw median だけでなく、multi-reference nearest-human distance / Mahalanobis distance の
  形で annotator variance を吸収できるか確認する
- linear candidate (H/K/M など) と finite soft-OES / native finite feasibility の rank
  correlation を計算し、first-order metric が finite edit diagnostic と大きく矛盾しないかを見る

## 6. 推奨と実装方針

### probe-free 側の暫定推奨

この時点では、probe-free family の中では **案 H (subspace gain' + operator leak') を
最有力候補**、観測空間投影 (案 E 統合) も同時適用とする。**案 D は H の特殊ケース
(leak' を捨てる) として論文の理論導出に残す**。

> 2026-05-24 追記: 実験後の scale 再検討により、`leak_prime` は単独では強いが
> `d_H` total score への寄与が小さすぎることが確認された。現在の probe-free 側の
> 推奨は **案 Hθ (`d_sub_angle`, `leak_angle`, `d_H_angle`)**。特に paper では
> `d_sub_angle` / `leak_angle` の 2 軸を主診断として report し、単一 scalar が必要な
> 場面で `d_H_angle` を使う。

ただし §8–§9 の q-retaining family 追加後は、主指標候補は H に固定せず、
§4 の scorecard に従って **H/D と K/M を並列比較**する。`q` が共通土俵として不可欠
という立場を採る場合、K/M が N10–N13 を通れば main metric 候補として H より優先される。

### 実装段階

1. **R_C 抽出 layer**: 既存の `notebooks/exp_constrained_response_metric.py` から
   `R_C` を取り出す (現状は `S_C^(q)` の中で internal に計算しているはず)。SVD で
   `U_C, Σ_C, V_C^T` を出し、`V_C = range(U_C)` を保持。
2. **subspace 距離**: `d_norm = max(dim V_P, dim V_G)`、`||P_P - P_G||_F^2` を SVD 経由で。
3. **operator leak / angular leak**: `J_{R_G}` を v3.md と同じ relation Jacobian から取る。
   旧 H では `J_{R_G} U_P` の Frobenius norm を `sqrt(n_R · dim V_P)` で正規化する。
   Hθ では `orth(row(J_{R_G}))` と `U_P` の subspace overlap を `leak_angle` として測る。
4. **観測空間版**: `J_phi` を bbox 正規化された vertex Jacobian として、`J_phi U_C` の
   range で同じことを繰り返す。
   > **2026-05-28 追記 (X6)**: `J_phi` の subnode 構成では Circle が `{eid}.center`
   > のみだったため、radius DOF が `J_phi` の kernel に落ち `Equal(Circle, Circle)`
   > のような radius-only relation が観測空間で見えない構造的盲点があった。
   > Circle / Arc radius は rigid-motion-invariant な scalar 量なので
   > `state.get_radius(eid)` を scalar observable として append すれば不変性は
   > 保たれ盲点が解消される ([src/metrics/edit_response_distance.py:599-678](../src/metrics/edit_response_distance.py#L599-L678))。
   > `center + (r, 0)` のような virtual perimeter point を ref に加える方式は
   > offset 方向が sketch frame に固定されているため pairwise distance の
   > rigid-motion invariance を破る — scalar radius を append するのが clean な
   > 唯一の道。実験結果は
   > [`summary_histcad_response_metric_v4_results.md`](summary_histcad_response_metric_v4_results.md)
   > 冒頭の "2026-05-28 更新" を参照。
5. **paired 統計**: 現 `d_feas` の dataset (`v3_controls_scores_full.csv` の cohort) を
   そのまま再利用。新指標を同 CSV にカラム追加 (`d_sub`, `gain_prime`, `leak_prime`,
   `d_H`, `d_sub_angle`, `leak_angle`, `d_H_angle`, `d_obs`)。
6. **可視化**: §8 と同形式の CDF と paired Wilcoxon を新指標で再生成。

### 想定工数

- 1–3 (案 D, H の primitive 空間版): 半日
- 4 (観測空間版): 半日
- 5–6 (paired stat と plot): 半日
- 4.2 (fully constrained subset 検証): 半日

合計 2 日程度を見込む。

### 既存 artifact との互換性

- v3 の CSV / details は **削除しない**。`*.v3.csv` として保存し、新指標は
  `*_v4.csv` で並列出力する。
- v3 の paper section の対応する数値は、v4 検証完了後に置き換える前提で、まず supplement
  table として両者並記する形が無難。

## 7. オープン問題

検証中に明らかになり得る論点を列挙する。

- **R_C の scale invariance**: column normalize で本当に十分か? `dp_C` の物理単位
  (mm vs deg) の影響をどう中和するかは、bbox 正規化と SVD 正規化の二重で吸収できる
  と期待しているが、要 sanity check。
- **`d_norm` の選択**: `max(dim V_P, dim V_G)` か `(dim V_P + dim V_G)/2` か。前者は
  jump を抑える、後者は連続性 (片方が縮むときの挙動) で勝る。実験で決める。
- **leak' の正規化**: `n_R · dim V_P` で割る operator norm 版 (`leak_prime`) は実験では
  単独 signal が強い一方、`d_sub` と合成すると scale が小さすぎた。Hθ ではこの問題を
  `orth(row(J_R))` による `leak_angle` で解消する。旧 `leak_prime` は residual-scale
  diagnostic として残す。
- **rank jump の不連続性**: 実 sketch でこれが頻繁に起きるか? 起きる場合の影響は
  `gain'` の jump 幅でどれくらいか。soft rank approximation (Tikhonov 風だが subspace
  に適用) を検討すべきか。
- **多人数 reference の aggregation**: 現 `d_feas` は `min_i d_feas` を使う。`d_H` でも
  同じ `min_i` で良いか、それとも `gain'` と `leak'` で別々に集約すべきか。

## 8. 次のアクション

1. 案 D (subspace 距離) と案 H (subspace + operator leak) を実装
2. §6.1 / §6.2 / §7 / §8 を新指標で再現
3. fully constrained subset で案 D が痩せるか、案 H で補えるかを確認
4. 観測空間版 (案 E) も並列で出して比較
5. paired Wilcoxon、CDF、quantile table、divergent case list を生成
6. 結果に基づき、v4.md にこのセクションの後ろから「実験結果」「結論」を追記

## 9. 追加候補: `q` を共通土俵として残す再設計

上の案 D/H は probe-free 化を強く押しているが、ここには重要な反論がある。
**Pred と GT を共通の土俵で比較するには、何らかの共通 observable / edit axis が必要**であり、
その役割を担うのが `q(x)` である。`R_P` と `R_G` はそれぞれ異なる native dimensional
parameter 空間 `p_P`, `p_G` から primitive motion への写像なので、native parameter を
直接比較しても意味がない。`q` はこの任意性を消し、「同じ編集要求に対してどう反応するか」
という比較可能な座標を与える。

したがって、`q` を完全に捨てる案 D/H は、もし採用するにしても
「edit request response metric」ではなく「motion subspace metric」として位置付けるべき。
一方、評価目的が HistCAD-style editability / source-conditioned response にあるなら、
`q` を残したまま gain の binary 問題を直す方向が自然である。

以下では、`q` を維持する追加案を列挙する。

### 8.1 案 I: 最小 knob effort 指標

現行 gain は `J_q R_C dp = 1` が解けるかどうかだけを見るため、scalar probe では binary になる。
代わりに、**その edit を達成するためにどれだけ native dimensions を動かす必要があるか**
を測る。

```text
A_C := D_q^{-1} J_q R_C D_p

e_q(C) = min_u ||u||_2
         s.t. A_C u = 1
```

ここで `D_p` は native dimensional parameters の単位正規化、`D_q` は probe value の
単位正規化である。`u = D_p^{-1} dp` と考えれば、`e_q(C)` は normalized knob effort。
`A_C` が rank deficient なら `e_q = ∞` とするか、後述の soft residual 付き問題に落とす。

連続値として使うには、例えば

```text
gain_effort = e_q(C)^2 / (1 + e_q(C)^2)
```

または

```text
gain_effort = log(1 + e_q(C))
```

を使う。これは「edit が数学的に可能か」ではなく、「自然な大きさの dimension knob で
edit できるか」を測る。`J_q R_C` が非ゼロなら常に gain=0 になる現行版より、conditioning
と controllability を反映できる。

#### 期待される挙動

- dimension dropout: 残った dimensions だけで `q` を動かす必要があり、effort が増える。
- ill-conditioned response: rank は足りていても effort が大きくなり、連続的に悪化する。
- 100% dropout: `A_C=0` なので最大ペナルティ。

### 8.2 案 J: bounded-budget reachability

案 I の双対として、一定の knob budget 内で `q` をどれだけ動かせるかを見る。

```text
reach_q(C) = max_{||u||_2 <= 1} ||A_C u||_2
           = ||A_C||_2
```

scalar `q` なら `reach_q(C) = ||A_C||_2`。これを

```text
gain_budget = 1 / (1 + reach_q(C)^2 / τ^2)
```

のように変換すれば、0 から 1 の連続ペナルティになる。案 I が「達成に必要な effort」
を見るのに対し、案 J は「同じ effort でどれだけ edit が効くか」を見る。

この指標は特に、dimension が存在するが `q` にほとんど効かないケースを検出しやすい。
現行 `rank(J_q R_C)` では非ゼロなら成功扱いになるが、案 J では小さい singular value が
そのまま悪化として出る。

### 8.3 案 K: relation-preserving controllability

最も有望な `q` 維持案。gain と leak を後から足すのではなく、最初から
**source relations を保存しながら `q` を動かせる native dimension update が存在するか**
を解く。

```text
A_q := D_q^{-1} J_q R_P D_p
A_R := D_R^{-1} J_{R_G} R_P D_p

u* = argmin_u
       ||A_q u - I||_F^2
     + μ ||A_R u||_F^2
     + λ ||u||_F^2
```

ここで `u` は normalized native dimension update、`D_R` は relation residual の
row normalization。得られた `u*` から

```text
gain_K = ||A_q u* - I||_F / sqrt(k_q)
leak_K = ||A_R u*||_F / sqrt(n_R)
effort_K = ||u*||_F / sqrt(k_q)
d_K = sqrt(gain_K^2 + μ leak_K^2 + λ effort_K^2)
```

を定義する。

これは現行 `d_feas` の構造をかなり保っている:

- `q` は source edit axis として残る
- `J_{R_G}` による source relation preservation も残る
- gain / leak decomposition も残る
- ただし `pinv(J_q R_P)` による hard attainability をやめ、soft constrained optimization
  にするため、scalar probe でも per-probe で連続値になる

#### 現行 `d_feas` との関係

`μ=0`, `λ→0` として、かつ `A_q` が full row rank なら、解は現行の
`u = A_q^+ I` に近づく。つまり案 K は現行 `d_feas` の連続化かつ relation-aware 化。

現行版では

```text
S_P^(q) = R_P (J_q R_P)^+
```

を先に作ってから `J_{R_G} S_P` を測る。一方案 K では

```text
J_q R_P dp ≈ I
J_{R_G} R_P dp ≈ 0
```

を同時に満たす `dp` を探す。したがって「edit は達成できるが relation を壊す最小ノルム解」
を自動的に避けられる。これは CAD 的にはより自然で、「relation-preserving edit knob」
そのものを測っている。

#### 注意点

`μ` と `λ` は hyperparameter だが、解釈は明確:

- `μ`: relation preservation を edit attainability に対してどれだけ重く見るか
- `λ`: native dimension knob effort の budget

論文では sweep して robustness を示し、main table では固定値を使うのがよい。

### 8.4 案 L: canonical response match

`S_P^(q)` と `S_G^(q)` を直接比較する案は v3.md で失敗した。理由は、
`S_C^(q)=R_C pinv(J_qR_C)` の min-norm が **各 sketch 固有の native dimension 空間**
で定義されるため、chain vs fan のような intent-equivalent な annotation でも `S` が変わるから。

しかし、native parameter 空間で min-norm を取るのではなく、共通 primitive / observable 空間で
canonical response を選べば、この問題を弱められる。

```text
S_can(C, q)
  = argmin_S ||J_phi S||_F^2
    s.t. S ∈ range(R_C)
         J_q S = I
```

relation preservation も入れるなら

```text
S_can(C, q; R_G)
  = argmin_S ||J_phi S||_F^2
    s.t. S ∈ range(R_C)
         J_q S = I
         J_{R_G} S = 0
```

hard constraint が厳しすぎる場合は案 K と同じ soft formulation にする。

この指標では

```text
d_can(P, G)
  = ||J_phi S_can(P, q; R_G) - J_phi S_can(G, q; R_G)||_F
```

のように、共通 observable 空間で canonical response を比較できる。raw `S_C^(q)` match
とは異なり、native dimension basis の違いを直接比較しない。

#### 位置付け

案 L は「Pred と GT の応答そのものを比較したい」という直感を救う再検討案。
ただし、constraint set が同じ feasible condition を満たすなら response が一意であるべきか、
それとも複数の human-like response を許すべきかは未解決。主指標よりは diagnostic / appendix
候補。

### 8.5 案 M: `Q_G` 全体の controllability spectrum

個別 probe の binary 問題を避ける別方向として、source GT の dimensional probes をまとめて
1 つの multi-output edit map として扱う。

```text
J_Q := stack_q J_q
A_Q := D_Q^{-1} J_Q R_P D_p
```

そして `A_Q` の singular values を評価する。

```text
d_spec(P, G)
  = mean_i ρ(σ_i(A_Q))
```

例えば

```text
ρ(σ) = λ / (σ^2 + λ)
```

または

```text
ρ(σ) = -log(σ^2 + ε)
```

を使う。個別 `q` ではなく `Q_G` 全体として source edit axes をどれだけ覆うかを見るため、
aggregation 後の Bernoulli rate より滑らかになる。

source relation preservation も入れるなら、案 K の stack 版として

```text
min_U
  ||A_Q U - I||_F^2
  + μ ||A_R U||_F^2
  + λ ||U||_F^2
```

を解く。これは案 K を per-probe で平均するより安定で、probe 間の相関も扱える。

### 8.6 案 N: finite residual を連続化した soft-OES

HistCAD OES の binary 判定を捨て、finite edit 後の residual magnitude を直接使う。

```text
soft_gain = |q(x_solved) - q_target| / |delta|
soft_leak = RMS( r_G(x_solved) - r_G(x_0) ) / |delta|
d_softOES = sqrt(soft_gain^2 + soft_leak^2)
```

これは `q` を保持し、finite nonlinear solve / branch change / FreeCAD roundtrip の影響も含めて
評価できる。現行 `d_feas` の first-order diagnostic とは別物だが、HistCAD への対案として
は説明しやすい。

欠点は計算コストと solver 依存性。main metric にするには重いが、linear metric の妥当性を
確認する appendix diagnostic として価値が高い。

### 8.7 案 O: human annotator distribution を明示的に扱う

複数 human annotation の差を「失敗」ではなく「許容分布」として扱う。
ただし raw `S_C^(q)` は native parameterization artifact を含むので、分布化する特徴量は
invariant 寄りに選ぶ。

候補:

```text
z_C(q) = [
  gain_effort(q; C),
  relation_preserving_effort(q; C, G_i),
  ||J_{R_G} S_C^(q)||,
  singular values of D_q^{-1}J_q R_C D_p,
  J_phi S_can(C, q)
]
```

human refs `{H_i}` から file/probe ごとの分布を作り、

```text
score(P) = dist( z_P, distribution{z_{H_i}} )
```

を測る。annotator 間の差分が大きい場合、それは指標の失敗ではなく
「この sketch/probe では human design intent が複数ある」ことを意味する。従って、
単一 GT に対する距離より、multi-reference likelihood / nearest-human distance /
Mahalanobis distance の方が自然。

#### 位置付け

これは主指標というより evaluation protocol の改善案。`min_i` aggregation の原理的な
拡張として扱える。

### 8.8 案 P: `T_C` の SVD spectrum と truncated subspace match

> 追記日 2026-05-21。本案は §9 priority 整理および
> [`summary_histcad_response_metric_v4_results.md`](summary_histcad_response_metric_v4_results.md)
> の主結果 (d_H 推奨) 確定後に検討された後発候補である。smoke test は
> v4_results.md §9 末尾の追記 (「後追い: d_SVD smoke test」) を参照。

#### 動機

v3 で棄却された案 (= d_resp) は、`S_C^(q) = R_C pinv(J_q R_C)` を **行列要素として
そのまま** Pred / GT で比較していた。棄却理由は、`pinv` の min-norm が native
dimensional parametrization に依存し、annotator が chain vs fan のような異なる dim
構成を選ぶと entry が変わってしまうため、heldout (intent-equivalent な人間アノテータ
ペア) でも `d_resp` が 0 にならず paired test で discrimination 不能 (`p = 0.495`)
となった点だった (v3.md §6.1)。

しかし v3 の棄却は「`S_C^(q)` という構造に意味がない」ことを示したのではなく、
**「`S_C^(q)` の生 entry には parametrization 起源のノイズが大きすぎる」** ことを示した
だけだとも読める。本案は、その noise を **SVD truncation** で自動的に取り除いた残り
(=「主要な edit response の構造」) を比較する。具体的には、

1. Pred と GT に対して共通の probe stack `Q_G` (= source GT 由来の dimensional
   constraints) を当て、それぞれの観測空間応答 `T_C` を構成する。
2. `T_C` を SVD で分解し、stable な主成分 (大きい singular value とそれに対応する
   左特異ベクトル) のみを残す。
3. 残った成分どうしを Pred / GT で比較する。

「重要な挙動を機械的に抽出する」発想は、`q(x)` を共通土俵として残しつつ、case D / H
の subspace 抽出 (= `range(R_C)` の SVD で得られる直交基底) を **probe-conditioned** に
行うものと位置付けられる (case D / H は probe を使わないので、Q_G が表す edit task に
対する応答の「強さ」を測れない)。

#### 定式化

GT 由来の probe stack を `Q_G = stack_q J_q` とし、それぞれの sketch `C` で

```text
M_C := J_Q R_C                         (|Q_G| × n_p)
S_C := R_C pinv(M_C)                   (n_x × |Q_G|)     # 各列が "edit q_i 達成の min-norm 応答"
T_C := J_phi S_C                       (m_phi × |Q_G|)   # 観測空間 (pairwise distance) に投影
T_C = U_C diag(sigma_C) V_C^T          # thin SVD
```

を計算する。`T_P` と `T_G` の列数は同じ |Q_G| なので、列ごとの対応をとって比較できる
(=「同じ edit task」に対する応答の比較になる)。共通の probe stack が必要なので、Pred と
GT の両方で `J_q` が計算可能な probe のみを採用する。

具体的に以下のサブ指標を定義する:

```text
d_SVD_resp     := ||T_P - T_G||_F / sqrt(m_phi · |Q_G|)        # 旧 d_resp の素朴 baseline
d_SVD_sigma    := ||sigma_P - sigma_G||_2                        # 特異値スペクトルの絶対差
d_SVD_sigma_rel := d_SVD_sigma / (||sigma_G||_2 + eps)           # 同 GT スケール正規化版
d_SVD_top1     := sqrt(2 − 2|<u_1^P, u_1^G>|)                    # 主成分方向のみの距離
d_SVD_sub_trunc :=
    truncated principal-angle distance of (U_P, U_G)
    keeping only columns i with sigma[i] >= tau · max(sigma_P[0], sigma_G[0])
    (rank-mismatch handled via the same (k_P + k_G - 2 Σ cos²θ) / max(k_P, k_G)
     formula as case D)
```

`tau` (推奨初期値 `0.01`) は「主成分」の閾値で、これより小さい特異値に対応する
左特異ベクトルは数値的に不安定 (実質 null-space の任意方向) なので、annotator 間で
偶然方向がずれた成分が原因で principal angle に大きな値が出るのを防ぐ。

#### 案 D / H との関係

- **共通点**: `T_C = J_phi @ R_C @ pinv(J_Q R_C)` の像 = `range(J_phi R_C)` (Q_G が
  `range(R_C)` を spanning するなら) なので、`d_SVD_sub_trunc` の極限 (tau → 0) は
  case D の `d_sub` に漸近する。
- **新規性**: `d_SVD_sigma` / `d_SVD_sigma_rel` は **特異値の magnitude そのもの** を見る
  ため、case D / H では捕捉できない「Pred の edit response が縮退する failure mode」
  (= 制約過多で「Q_G の edit を flow できない sketch」) を直接検出できる可能性がある。
  例: 21803 sketch 上で dfeas_rl の `T_P` は `sigma_P = [0.000, 0.000]` まで縮退し、
  `d_SVD_sigma_rel = 1.0` (= GT スケール 100% の差) になる。同じケースで案 H の
  `leak_prime` は 0.39 と中程度の値しか付かない (v4_results.md §9 後追い参照)。

#### 期待される弱点

1. **annotator variance 吸収力は case H と同程度** — smoke test (v4_results.md §9 後追い)
   で `d_SVD_sub_trunc` の heldout `frac<1e-3` が 0.417 と、`d_feas` (0.792) より明確に
   悪く、`d_H` / `d_sub` (0.375) と同水準に留まった。`d_SVD_sigma_rel` (0.583) は
   `d_SVD` variants の中では最も heldout 吸収が良いが、それでも `d_feas` には及ばない。
2. **サブ指標が多く main metric 化が難しい** — `d_SVD_resp / sigma / sigma_rel /
   top1 / sub_trunc` の中でどれを main にするか、どう合成するかが open。
3. **`tau` 依存** — truncation threshold の妥当値が経験則。`tau` を変えると per-sketch
   挙動が大きく変わる可能性があり、§4.2 N12 (hyperparameter non-degeneracy) の検証が
   必要。

#### 位置付け

主指標 case H を置き換える候補ではなく、**`d_H` の failure-mode-detection を補強する
diagnostic** として位置付けるのが妥当 (smoke test の段階)。具体的には、

- `d_H` が low なのに `d_SVD_sigma_rel` が high → Pred の subspace は GT と似て見えるが
  edit task に対する応答が縮退している (= overconstrained predicted sketch)
- `d_H` が high なのに `d_SVD_sigma_rel` が low → subspace は違うが Q_G 方向には応答する
  (= under-constrained predicted sketch でも probe task には対応できる)

という対角的 diagnostic を提供しうる。フル評価は本案 H の v4 実験と並列の
ablation column として実施する想定。

## 9. 追加候補を踏まえた暫定優先順位

現時点での優先順位は以下。

1. **案 K: relation-preserving controllability**
   - `q` を共通土俵として残す
   - gain binary 問題を解く
   - leak / effort / gain の分解ができる
   - 現行 `d_feas` からの連続的な拡張として説明しやすい

2. **案 M: `Q_G` 全体の controllability spectrum / stack 版 K**
   - per-probe の不安定さを減らす
   - source edit axes 全体の coverage を測れる

3. **案 I/J: knob effort / budget reachability**
   - gain term だけを最小変更で修正する候補
   - 実装と ablation が簡単

4. **案 L: canonical response match**
   - v3 で棄却した raw `S` match の再検討版
   - diagnostic として有用だが、主指標には慎重

5. **案 N/O: soft-OES / human distribution**
   - main metric より validation protocol / appendix 向け

6. **案 P: `T_C` SVD spectrum + truncated subspace match** (2026-05-21 追記)
   - 主指標候補ではなく **`d_H` の補助 diagnostic** として位置付ける
   - `d_SVD_sigma_rel` は overconstrained pred の応答縮退を磁気的に検出する独自シグナル
     (smoke test で確認、v4_results.md §9 後追い参照)
   - heldout 吸収では `d_H` と同水準。`d_feas` には及ばない

この整理に従うなら、v4 の実装フェーズは当初の案 H だけでなく、
**q-retaining family (I/J/K/M) と probe-free family (D/H) を並列比較**するのが安全。
もし `q` が共通土俵として不可欠という立場を採るなら、主指標候補は案 H ではなく
案 K または案 M に寄せるべきである。

実装フェーズに進む。

> 2026-05-21 追記: 上記実装フェーズの結果 (主指標は `d_H` で固定) を反映した実験は
> [`summary_histcad_response_metric_v4_results.md`](summary_histcad_response_metric_v4_results.md)
> 参照。その後発候補 (案 P) も同 doc の §9 後追い節で smoke test 結果を整理してある。
>
> 2026-05-24 追記: `d_H` の scale issue を受けて Hθ (`d_sub_angle` + `leak_angle`) を
> 追加実装・フル評価した。結果は同 doc §14。結論は「`d_sub` だけでは solver_rl の
> relational mismatch を十分に表現しきれないが、旧 `leak_prime` をそのまま足すより
> `leak_angle` として angle scale に揃える方がよい」。
