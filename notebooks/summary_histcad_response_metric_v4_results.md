# V4 候補メトリクス実験結果

作成日: 2026-05-21

> **この文書の位置付け**
> [v4.md](summary_histcad_response_metric_v4.md) で提案された複数候補
> (Case D / H / Hθ / I / J / K / M) を実装し、v3 と同じ source-conditioned
> multi-reference setting で評価した結果をまとめる。実装は
> [notebooks/v4_exp/](v4_exp/) 下、artifact は
> [artifacts/v4_exp/](../artifacts/v4_exp/) および
> [artifacts/v4_exp_angle/](../artifacts/v4_exp_angle/) 下。

> **2026-05-28 更新 (X6: phi に Circle/Arc radius を追加)**
> [src/metrics/edit_response_distance.py](../src/metrics/edit_response_distance.py)
> の `pairwise_distance_observable` が `Circle` の `center` のみを subnode と
> していたため、`Equal(Circle, Circle)` のような radius のみを束縛する
> relational constraint が J_phi の kernel に落ち、`d_sub` / `d_H` / `d_H_angle`
> がそうした design intent を検知できない構造的盲点があった。修正として、
> radius は rigid-motion-invariant な scalar 量なので
> `Circle.radius` / `Arc.radius` を phi に scalar observable として append。
>
> 修正後の v4 controls フル再走 (n=3846, 6 batches, dim_drop {25,50,75,100} ×
> rel_drop {50,100}) の artifact は
> [artifacts/v4_exp_x6/](../artifacts/v4_exp_x6/)。本文書の §3–§5, §5.1, §7,
> §13.1, §15 の表は再走の数値に上書き済み。
> §6 (R2 連続性 sweep), §13.2 (sensitivity sweep),
> §13.3 (R3 subset cohort), §13.4 (PP-Coincident), §13.5 (smatch),
> §13.6 (strict-gate), §14 (case P d_SVD) は別データセットに依存する
> ablation のため pre-X6 数値のまま。再走の優先度は本 doc §16 で議論する。

## TL;DR

**2026-05-24 更新: 最終推奨は Case Hθ (`d_sub_angle` + `leak_angle`)。単一 scalar が必要な
表では `d_H_angle = sqrt(d_sub_angle^2 + leak_angle^2)` を使う。**

**2026-05-28 更新 (X6): `pairwise_distance_observable` に Circle/Arc radius を
scalar として append。Equal(Circle, Circle) のような radius-only relation が
phi の kernel に落ちる構造的盲点を解消。下記の z 値は X6 後の re-run の数値。**

- v4.md §9 で暫定 1 位だった案 K (relation-preserving controllability) と案 M
  (stack 版) は、本実験で **solver_rl に対し sign inversion** を起こすことが判明し
  失格。effort 項 `λ ||u||^2` が "controllable な model 出力" を不当に reward し、
  HistCAD OES が base を不当に reward した failure mode と同型の問題を起こす。
- 案 H は v3 `d_feas` より paired discrimination が同等以上 (base z: 11.59 →
  13.64, solver_rl z: 9.06 → 11.95)、かつ per-rotation の連続性 (R2) も改善
  (distinct values per rotation: 4 → 6, monotonic transitions: 34% → 49%)。
- ただし旧 `d_H = sqrt(d_sub^2 + leak_prime^2)` は total score 上ほぼ `d_sub` に支配される。
  `leak_prime` は単独 signal は強いが、scale が小さく scalar 合成に効きにくい。
- `leak_angle = ||orth(row(J_R))^T U_P||_F / sqrt(dim V_P)` に変更すると、`d_sub_angle`
  と同じ angular scale に乗る。solver_rl では `d_H_angle` が依然 `d_H` を上回り
  (`z=+12.59` vs `+11.95`)、`leak_angle` 単独は最強 (`z=+13.25`)。
- X6 で `d_sub` 単体の solver_rl 識別力が +10.70 → +11.91 と大幅強化されたため、
  `d_H_angle` との差は従来の z=1.66 → 0.68 まで縮まった (`d_sub` が radius 系
  relational mismatch を自前で拾えるようになったため)。
- したがって paper では **`d_sub_angle` / `leak_angle` の 2 軸 report を主診断** とし、
  scalar が必要な場合に `d_H_angle` を使う。最大限 parsimonious な main metric が
  必要なら `d_sub` または `d_sub_angle` main + `leak_angle` mandatory diagnostic。

## 1. 実装した候補

実装は [notebooks/v4_exp/metrics_v4.py](v4_exp/metrics_v4.py).

| Case | 説明 | probe 依存 |
|---|---|---|
| **D** `d_sub` | 観測空間 `range(J_phi R_C)` の subspace 距離 | 不要 |
| **H** `d_H` | D + operator leak `||J_{R_G} U_P||_F` | 不要 |
| **Hθ** `d_H_angle` | angular subspace distance + angular leak `||N_G^T U_P||_F` | 不要 |
| **K** `d_K` | per-probe LS: `min ||A_q u - I||^2 + μ||A_R u||^2 + λ||u||^2` | 必要 |
| **M** `d_M` | Case K の `Q_G` stack 版 | 必要 |
| **I** `gain_I` | `e_q² / (1+e_q²)` (effort transform) | 必要 |
| **J** `gain_J` | `1 / (1 + reach² / τ²)` (budget transform) | 必要 |

実装上の規約:

- `D_p` = `R_P` の column norm (eps floor) → `R_tilde` は unit-column-norm
- `D_R` = `J_R R_tilde` の row norm
- `D_q` = 1 (sketch は bbox 正規化済みのため省略)
- subspace 比較の共通空間は `J_phi C R_C` (pairwise distance observable);
  Pred と GT の primitive count が異なる場合 (model がエンティティを欠落させる時)
  でも比較できるようにするため必須
- 既定 hyperparameter: `μ=1.0`, `λ=1e-3`, `τ=1.0`, `eps=1e-9`
- `d_K_pure` / `d_M_pure` は `sqrt(gain² + μ leak²)` (effort 項なし) として
  analyze 側で派生
- Hθ 追加カラム:
  - `d_sub_angle² = 1 - Σ cos²θ_i / max(dim V_P, dim V_G)`
  - `leak_angle² = ||orth(row(J_R))^T U_P||_F² / dim(V_P)`
  - `d_H_angle = sqrt(d_sub_angle² + leak_angle²)`
  旧 `leak_prime` は residual-scale operator diagnostic として残す

## 2. データセット

v3 と同設定: 6 batches `{0, 4, 6, 8, 9, 11}` × 5 humans
`{Eko, Johnny, Ryotaro, Sagar, Tanaka}`. 共通 files: 159, 有効
rotations: 396 (heldout), Vitruvion paired: base=307, solver_rl=366.

## 3. Variant 別 median (full 6 batches, n=396) — X6 後

| variant | n | d_feas | d_sub | d_H | leak_prime | d_K | d_M | gain_K | leak_K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source_self | 396 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0202 | 0.0129 | 0.0022 | 0.0000 |
| heldout | 396 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0224 | 0.0183 | 0.0027 | 0.0000 |
| dim_drop25 | 396 | 0.0002 | 0.5348 | 0.5424 | 0.0000 | 0.0665 | 0.4578 | 0.0252 | 0.0000 |
| dim_drop50 | 396 | 0.1547 | 0.7071 | 0.7071 | 0.0000 | 0.4093 | 0.7072 | 0.4083 | 0.0000 |
| dim_drop75 | 396 | 0.4430 | 0.8660 | 0.8660 | 0.0000 | 0.7087 | 0.8452 | 0.7071 | 0.0000 |
| dim_drop100 | 396 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| rel_drop50 | 396 | 0.2975 | 0.8783 | 0.9031 | 0.1935 | 0.6895 | 0.6720 | 0.6872 | 0.0482 |
| rel_drop100 | 396 | 0.4284 | 1.0409 | 1.0834 | 0.2530 | 0.7184 | 0.7843 | 0.7145 | 0.0468 |
| **base** | 309 | 0.1213 | 0.7188 | 0.7354 | 0.0212 | 0.3786 | 0.5290 | 0.3780 | 0.0002 |
| **solver_rl** | 369 | 0.0421 | 0.5195 | 0.5321 | 0.0732 | **0.0201** | **0.0151** | 0.0026 | 0.0002 |

`d_K` / `d_M` で **solver_rl が heldout (0.022/0.018) より小さい** ことに注意。
これが下記の sign inversion の原因。

X6 で `d_sub` / `d_H` の絶対値は dim_drop sweep の ceiling 形状 (0.7071, 0.8660,
1.0000) を完全保存。rel_drop / base は `max(dim_P, dim_G)` 正規化の分母が
radius 列を含むぶん増えるため median 値が数 % 下がる。dim_drop25 だけは
0.5000 → 0.5348 と上昇するが、これは radius observable が dim drop の被害を
受けやすい (Radius constraint が drop されると radius scalar の rank が直接
減る) ためで、monotonicity / ceiling 性質には影響しない。

## 4. Paired Wilcoxon (model vs heldout)

`h_better` は人間優位 rotation 数、`m_better` は model 優位 rotation 数、
`z>0` は人間優位 (lower-is-better).

### 4.1 base モデル (under-constrained gate=Converged を通る) — X6 後

| metric | n | h | m | t | z | p |
|---|---:|---:|---:|---:|---:|---:|
| d_feas | 306 | 216 | 27 | 63 | +11.59 | 4.5e-31 |
| d_sub | 306 | 254 | 31 | 21 | +13.67 | 1.6e-42 |
| **d_H** | 306 | 253 | 35 | 18 | **+13.64** | **2.3e-42** |
| leak_prime | 306 | 166 | 44 | 96 | +9.33 | 1.1e-20 |
| d_K | 306 | 224 | 69 | 13 | +11.26 | 2.1e-29 |
| d_M | 306 | 236 | 53 | 17 | +11.94 | 7.1e-33 |
| gain_I | 306 | 183 | 105 | 18 | +6.86 | 7.1e-12 |
| gain_J | 306 | 171 | 113 | 22 | +5.45 | 5.1e-08 |

base に対しては全候補が同方向 (人間優位) で検出。X6 で `d_sub` の z は
+13.63 → +13.67、`d_H` は +13.62 → +13.64 と僅かに強化。`leak_prime` は
ほぼ不変 (+9.39 → +9.33、X6 が primitive 空間 leak を変えないため)。

### 4.2 solver_rl モデル (fully-constrained, RL fine-tuned) — X6 後

| metric | n | h | m | t | z | p |
|---|---:|---:|---:|---:|---:|---:|
| d_feas | 365 | 195 | 38 | 132 | +9.06 | 1.3e-19 |
| d_sub | 365 | 259 | 66 | 40 | **+11.91** | **1.0e-32** |
| **d_H** | 365 | 262 | 68 | 35 | **+11.95** | **6.8e-33** |
| leak_prime | 365 | 229 | 26 | 110 | +12.65 | 1.2e-36 |
| **d_K** | 365 | **108** | **233** | 24 | **−4.55** | **5.5e-6 (逆方向)** |
| d_K_pure | 365 | 151 | 182 | 32 | +0.15 | 0.88 (n.s.) |
| **d_M** | 365 | **104** | **230** | 31 | **−8.62** | **6.6e-18 (逆方向)** |
| d_M_pure | 365 | 168 | 151 | 46 | −2.82 | 0.005 (弱逆方向) |
| gain_I | 365 | 140 | 196 | 29 | −1.73 | 0.083 (n.s.) |
| gain_J | 365 | 110 | 219 | 36 | −6.65 | 3.0e-11 (逆方向) |

**Sign inversion 構造は維持** (X6 で probe-free / probe-retaining の本質的差異は
変わらない)。`d_K` / `d_M` / `gain_J` の逆方向は v3.md §6.1 で HistCAD OES が
base に対し起こしていた sign inversion (h=75 vs m=204, p=2.2e-19) と同型の
failure mode で、effort 項 / probe optimization 由来。

**X6 で `d_sub` の判別力が大幅強化** (z: +10.69 → +11.91, p: 10^−26 →
10^−33)。これは radius 系 relational mismatch が phi で見えるようになった
直接の効果で、solver_rl が落とす radius driver / Equal(circle) を `d_sub`
side で初めて検出できるようになったため。`d_H` も同様 (z: +11.23 → +11.95)。
`leak_prime` は X6 と独立 (primitive 空間) なのでほぼ不変 (+12.68 → +12.65)。

## 5. d_H decomposition (full 6 batches) — X6 後

| variant | d_sub (subspace) | leak_prime (operator) | d_H |
|---|---:|---:|---:|
| source_self | 0.000 | 0.000 | 0.000 |
| heldout | 0.000 | 0.000 | 0.000 |
| dim_drop50 | 0.707 | 0.000 | 0.707 |
| rel_drop50 | 0.878 | 0.194 | 0.903 |
| **base** | 0.719 | 0.021 | 0.735 |
| **solver_rl** | 0.520 | **0.073** | 0.532 |

base と solver_rl で `leak_prime` を比べると **solver_rl のほうが大きい (0.073 vs 0.021)**。
これは:

- base は under-constrained で subspace 構造自体が大きく異なる (d_sub=0.72)。
  relational 情報を持つ次元自体が少ないため leak_prime が小さくなる
- solver_rl は dim 構造を人間に近付けたが (d_sub=0.52)、追加された relations が
  source GT の relations とは異なるため leak_prime が増えた

これは v3.md §7 の「solver_rl は gain は閉じるが leak は閉じない、むしろ若干悪化
(0.059 → 0.068)」観察と整合する。

**leak_prime 単独でも solver_rl 検出力が最強 (z=+12.65, p=1.2e-36)**。
X6 で `d_sub` の z は base で +13.63 → +13.67、solver_rl で +10.69 → +11.91 に
強化されたため、`d_sub` 単独で見ても solver_rl の relational mismatch を旧よりも
強く拾うようになっている。それでも solver_rl では `leak_prime` 単独が依然
最強で、`d_H` ≈ `d_sub` < `leak_prime` という順序は維持。

### 5.1 2026-05-24 追試: angular leak 版 (2026-05-28 X6 反映)

上記の解釈は component diagnostic としては正しいが、scalar `d_H` では
`leak_prime` の絶対値が小さく、total score はほぼ `d_sub` になる。そこで
`leak_prime` を `leak_angle` に置き換え、`d_sub` 側も `d_sub_angle` として
angular scale に揃えた。

| variant | d_sub | d_H | leak_prime | d_sub_angle | leak_angle | d_H_angle |
|---|---:|---:|---:|---:|---:|---:|
| source_self | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heldout | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dim_drop25 | 0.5348 | 0.5424 | 0.0000 | 0.5000 | 0.0000 | 0.5613 |
| dim_drop50 | 0.7071 | 0.7071 | 0.0000 | 0.7071 | 0.0000 | 0.7071 |
| rel_drop50 | 0.8783 | 0.9031 | 0.1935 | 0.6616 | 0.6584 | 0.9387 |
| rel_drop100 | 1.0409 | 1.0834 | 0.2530 | 0.7863 | 0.7921 | 1.1167 |
| **base** | 0.7188 | 0.7354 | 0.0212 | 0.6602 | 0.2634 | 0.8202 |
| **solver_rl** | 0.5195 | 0.5321 | 0.0732 | 0.5000 | 0.4472 | 0.6831 |

`leak_angle` は `leak_prime` と違い、solver_rl / rel_drop で `d_sub_angle` と同程度の
スケールに乗る。したがって `d_H_angle` では relation mismatch が total score に
実際に効く。

Paired Wilcoxon の主要比較 (X6 後):

| model | metric | n | h | m | t | z | p |
|---|---|---:|---:|---:|---:|---:|---:|
| base | d_sub | 306 | 254 | 31 | 21 | +13.67 | 1.6e-42 |
| base | d_sub_angle | 306 | 248 | 37 | 21 | +13.34 | 1.5e-40 |
| base | d_H | 306 | 253 | 35 | 18 | +13.64 | 2.3e-42 |
| base | d_H_angle | 306 | 256 | 32 | 18 | +13.49 | 1.7e-41 |
| base | leak_angle | 306 | 167 | 45 | 94 | +9.31 | 1.3e-20 |
| solver_rl | d_sub | 365 | 259 | 66 | 40 | +11.91 | 1.0e-32 |
| solver_rl | d_sub_angle | 365 | 260 | 66 | 39 | +11.96 | 6.1e-33 |
| solver_rl | d_H | 365 | 262 | 68 | 35 | +11.95 | 6.8e-33 |
| solver_rl | d_H_angle | 365 | 267 | 63 | 35 | +12.59 | 2.4e-36 |
| solver_rl | leak_angle | 365 | 239 | 15 | 111 | +13.25 | 4.6e-40 |

結論 (X6 後の更新):

- base では `d_H_angle` は `d_sub` / 旧 `d_H` とほぼ同水準で、僅かに弱い (X6 前と
  同じ性質)。
- solver_rl では `d_H_angle` が依然 `d_sub` を上回る (+12.59 vs +11.91) が、
  X6 で `d_sub` 自身が +10.70 → +11.91 と大幅強化されたため、両者の差は
  従来の 1.66 → 0.68 まで縮まった。これは「X6 後の `d_sub` が radius 系
  relational mismatch をだいぶ自前で拾えるようになった」ことの数値的裏付け。
- `leak_angle` 単独は solver_rl の relational mismatch 検出で依然最強 (z=+13.25)。
  X6 と独立 (primitive 空間) なため数値も完全保持。
- 推奨は維持: 主診断は `d_sub_angle` / `leak_angle` の 2 軸、scalar が必要な
  場合に `d_H_angle`。X6 後は `d_sub` (angular でない素朴版) も solver_rl
  identification に対しほぼ同等の強さを持つ。

## 6. R2 連続性 (dim_drop sweep, 10% 刻み, batch 0+9)

| metric | n_tr | frac↑ | frac↓ | frac= | med\|Δ\| | distinct/rotation |
|---|---:|---:|---:|---:|---:|---:|
| d_feas | 1350 | 34% | 11% | 54% | 0.000 | 4 |
| **d_H** | 1350 | **49%** | 11% | 41% | 0.048 | **6** |
| d_sub | 1350 | 47% | 10% | 43% | 0.046 | 5 |
| d_K | 1350 | 58% | 20% | 22% | 0.031 | 9 |
| d_M | 1350 | 59% | 11% | 30% | 0.027 | 8 |

`d_K` / `d_M` は per-rotation の distinct values で最も滑らかだが、frac↓ (rate と
逆方向に動く割合) が大きく "signal" に占める noise も多い。`d_H` は v3 `d_feas`
(distinct=4, monotonic 34%) より明確に改善 (distinct=6, monotonic 49%) し、かつ
frac↓ は v3 と同じ 11% を維持。

## 7. Annotator variance absorption (heldout 分布) — X6 後

| metric | median | Q75 | Q90 | frac<1e-4 |
|---|---:|---:|---:|---:|
| d_feas | 0.0000 | 0.0009 | 0.1226 | 0.735 |
| d_sub | 0.0000 | 0.4082 | 0.5774 | 0.577 |
| **d_H** | 0.0000 | 0.4123 | 0.5774 | **0.526** |
| d_K | 0.0224 | 0.0460 | 0.3445 | 0.000 |
| d_M | 0.0183 | 0.2099 | 0.4573 | 0.000 |

Hθ 追試での angular 版 (X6 後):

| metric | median | Q75 | Q90 | frac<1e-4 |
|---|---:|---:|---:|---:|
| d_sub | 0.0000 | 0.4082 | 0.5774 | 0.577 |
| d_H | 0.0000 | 0.4123 | 0.5774 | 0.526 |
| leak_prime | 0.0000 | 0.0015 | 0.0872 | 0.727 |
| d_sub_angle | 0.0000 | 0.4082 | 0.5345 | 0.577 |
| leak_angle | 0.0000 | 0.1313 | 0.4472 | 0.709 |
| d_H_angle | 0.0000 | 0.4971 | 0.6364 | 0.526 |

**Trade-off** (X6 後でも構造は維持):
- `d_feas` は heldout の frac<1e-4=0.735 (annotator variance を強く吸収)
- `d_H` は frac<1e-4=0.53 (約 1/2 の human-human ペアに near-zero、残りに非ゼロ
  tail)。X6 前 (0.524) とほぼ同じ
- `d_H_angle` の Q90 は X6 で 0.671 → 0.636 と僅かに縮小 (radius observable で
  共通土俵が広がるため human-human の subspace mismatch が相対的に小さくなる)。
  near-zero 率は X6 と無関係に旧 `d_H` と同水準
- `leak_angle` 系は X6 と独立 (primitive 空間)、絶対値も完全同一

`d_H` の heldout 上位 tail は dim count mismatch (一方の human が 4 obs dims、
他方が 16 obs dims) の場合に発生する。これは **intent-equivalent でない真の
annotation 差分** (一方が冗長な dim を付けた) と解釈できるため、d_H が罰するのは
"指標の失敗" でなく "本物の構造差を捕捉" と読める。とはいえ、d_feas より絶対的な
human-human distance 分布が広がるのは事実。

## 8. v4.md §4 criteria に対する評価

### N (足切り) 評価

| criterion | d_feas | d_sub | **d_H** | **d_H_angle** | d_K | d_M |
|---|---|---|---|---|---|---|
| N1 source_self frac<1e-4≥0.99 | ✓ (1.00) | ✓ (1.00) | ✓ (1.00) | ✓ (1.00) | ✗ (0.00) | ✗ (0.00) |
| N2 heldout med << dim50 med | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| N3a sign correct (base, p<1e-5) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **N3b sign correct (solver_rl)** | ✓ | ✓ | ✓ | **✓** | **✗ p=5.8e-6** | **✗ p=4e-17** |
| N4 dropout monotonic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| N5 rel_drop > heldout | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| N11 component selectivity | ✓ | partial | ✓ (gain'/leak') | **✓ (subspace/leak angle)** | partial | partial |
| **N13 no failure hiding** | ✓ | ✓ | ✓ | ✓ | **✗ effort game** | **✗ effort game** |

`d_K` / `d_M` は **N3b** および **N13** で失格。

### R (degree) 評価 — X6 後の R1 / R11 更新

| criterion | d_feas | d_sub | **d_H** | **d_H_angle** |
|---|---|---|---|---|
| R1 paired \|z\| (base / solver_rl) | 11.6 / 9.1 | **13.7 / 11.9** | **13.6 / 11.9** | **13.5 / 12.6** |
| R2 連続性 (distinct/rotation, monotonic↑) | 4 / 34% | 5 / 47% | **6 / 49%** | (要 R2 sweep) |
| R3 subset cohort discrimination | (pre-X6) | (pre-X6) | (pre-X6) | (pre-X6) |
| R4 分解 richness | gain/leak (gain binary) | rank/align (rank integer) | gain' / leak' | **subspace angle / leak angle** |
| R5 ablation robustness | ✓ (v3 で確認) | (要検証) | (要検証) | (要検証) |
| R6 design parsimony | 中 (probe必要) | **高** | **高** | **高** |
| R10 finite consistency | (v3 で部分検証) | (要検証) | (要検証) | (要検証) |
| R11 human-distribution calibration | heldout frac<1e-4=0.735 | 0.577 | 0.526 | 0.526 |
| R12 computational usability | (probe LS) | SVD のみ | **SVD のみ** | **SVD のみ** |

X6 で R1 が顕著に動いた: solver_rl `d_sub` z が 10.7 → 11.9、`d_H` z が
11.2 → 11.9。`d_H_angle` の優位はやや薄まる (12.4 → 12.6、`d_sub` との差が
1.7 → 0.7 に)。base 側はほぼ同水準。

## 9. 結論と推奨

### 推奨

**主診断**: `d_sub_angle` + `leak_angle` の 2 軸 report (Case Hθ)

**単一 scalar が必要な場合**: `d_H_angle = sqrt(d_sub_angle² + leak_angle²)`

**旧 Case H**: `d_H = sqrt(d_sub² + leak_prime²)` は residual-scale operator diagnostic
として残すが、main scalar としては Hθ に置き換える。

**理由**:

1. v4.md の N1–N13 必要条件をすべて満たす唯一の "新しい" 候補
   (v3 `d_feas` も満たすが、R1/R4 で `d_H_angle` が改善)
2. **base / solver_rl 両方の paired z が `d_feas` 比で強化**
   (base 11.7→13.5, solver_rl 9.1→12.4)
3. **per-rotation の R2 連続性が改善** — distinct values per rotation が
   旧 `d_H` で 4 → 6 に、monotonic 率が 34% → 49% に向上。Hθ でも同じ
   probe-free SVD family なので、追加 R2 sweep は残るが構造上は同等
4. **probe 設計に依存しない** — `Q_G` の選び方や source validation を省略可能
5. `d_sub_angle` と `leak_angle` の **同一 scale の連続値分解** で v3.md §7 の
   RL diagnostic ("RL は gain を改善するが leak は閉じない") を再現し、solver_rl では
   旧 `d_H` より強い識別力を示す

**Diagnostic 補助**: `d_sub` / `d_sub_angle` (subspace only)、`leak_prime` (residual-scale
operator)、`leak_angle` (scale-free angular leak) を独立成分として report し、§7 タイプの
model improvement axis 分解を提供。

### v4.md §9 暫定優先順位の修正

v4.md §9 では:

> 1. 案 K: relation-preserving controllability
> 2. 案 M
> 3. 案 I/J
> 4. 案 L
> 5. probe-free family (D, H)

を提案していたが、本実験の結果以下に修正:

1. **案 Hθ (probe-free subspace angle + angular leak)** — 最有力
2. 案 H (probe-free subspace + residual-scale operator leak) — diagnostic / historical baseline
3. 案 D (subspace) — parsimonious main candidate or diagnostic
4. v3 `d_feas` — baseline として残す
5. 案 K / 案 M — **N3 (sign correctness) 不合格**、controllability bias で却下
6. 案 I / J — 弱識別力 (および sign inversion)

**§9 の "q が共通土俵として不可欠" という直感は本実験では裏付けられなかった**。
観測空間 `J_phi R_C` 自体が十分な共通土俵になっており、probe を介する必要はない。
むしろ probe を介する K / M は LS の `u*` が relation を壊さない方向を選ぶため
"best-case" leak しか測れず、solver_rl の relational mismatch を隠してしまう。

### 失敗の構造的理解 — なぜ probe-free が勝つのか

- **probe-free metric (`d_H`) は subspace の "intrinsic" な不一致を測る**
  : `range(R_P)` 上の任意の orthonormal direction が GT relation を破れば
  `leak_prime` に必ず表れる
- **q-retaining metric (`d_K`, `d_M`) は最適化された "best-case" を測る**
  : LS の `u*` は relation を破らない方向を選ぶため、subspace に bad direction が
  あっても total residual に出ない
- **`d_K` / `d_M` の sign inversion の root cause**:
  - solver_rl は多数の dim constraints を持つ → controllable (effort_K 小)
  - `d_K = sqrt(gain_K² + μ leak_K² + λ effort_K²)` で effort_K の小ささが
    total score を heldout 以下に押し下げる
  - leak_K は LS が自動的に下げるため、relational mismatch を捕捉できない
- v4.md §4.2 N13 ("no failure hiding") の懸念がこの形で発現した

この対比は CAD 評価指標としての **"intrinsic vs optimized"** の選択問題と捉えられる。
本研究では intrinsic 側 (probe-free, average-over-subspace leak) が勝った。

### 後追い: d_SVD 案の smoke test (2026-05-21 追記)

#### 背景

§9 の結論 (`d_H` を主指標として推奨) を確定したあとに、`q` を共通土俵として残す
**別系統の追加候補** が提案された (v4.md §8.8 案 P)。本案は、v3.md で raw `d_resp`
として棄却された「`S_C^(q)` 同士を Pred と GT で直接比較する」発想を、

- `S_C^(q)` を観測空間に投影した `T_C = J_phi S_C^(q)` を SVD して
- **主要な singular component のみを抽出** して比較する (= parametrization artifact
  と思われる small-σ 成分を捨てる)

という形でリハビリさせるものである。「重要な edit response の挙動を機械的に抽出
できないか」というアイデアに対応する。詳細な定式化は v4.md §8.8 参照。

本節は、v4 実装フェーズ (案 D / H / K / M / I / J の本格評価) の **後** に行った、
案 P の **smoke test 結果** を記録する。フル評価 (`d_H` と並列の paired Wilcoxon /
dropout sweep / R2 / model ranking) は本セッションでは実行せず、別セッションで予定。

#### Smoke test 1: 単一 sketch (21803_8a36dcda_0011_sketch01) での model 比較

`notebooks/test_dH.py` を用いて、`d_feas` / `d_H` / `d_SVD_*` を **同じ 1 件の
sketch** に対して per-pair で計算した。

- GT          : Ryotaro の annotation (15 entities, 3 dim + 37 rel)
- heldout_Sagar : 同 sketch の Sagar による annotation (intent-equivalent reference)
- solver_rl   : Vitruvion + solver-feedback RL 出力 (14 entities, 14 dim + 25 rel)
- dfeas_rl    : Vitruvion + d_feas-reward RL 出力 (14 entities, 22 dim + 17 rel)

このケースは v4.md / v4_results.md 本文では使われていないが、**`d_feas` が誤判定
する典型例** として注目している。すなわち model 出力の品質を定性的に並べると
`heldout_Sagar < solver_rl < dfeas_rl` (= solver_rl の方が高品質) であるのに対し、
`d_feas` は `dfeas_rl` を solver_rl より良いとランク付けする (probe-conditioned leak
が dfeas_rl の方が小さく見えるため)。

各指標の値 (lower = better):

| metric | heldout_Sagar | solver_rl | dfeas_rl | 正解 ranking? |
|---|---:|---:|---:|---|
| d_feas | 0.0000 | 0.3052 | **0.1806** | ✗ (dfeas_rl < solver_rl) |
| d_H | 0.0000 | **0.9884** | 1.0251 | ✓ |
| d_SVD_resp | 0.0000 | **0.3751** | 0.4437 | ✓ |
| d_SVD_sigma | 0.0001 | **3.4866** | 11.3126 | ✓ |
| d_SVD_sigma_rel | 0.0000 | **0.3082** | 1.0000 | ✓ |
| d_SVD_sub (untrunc) | 0.8159 | 1.1681 | 1.3530 | ranking 正 / heldout で大きく外れる |
| d_SVD_sub_trunc (tau=0.01) | 0.0000 | **0.9139** | 1.0000 | ✓ |
| d_SVD_top1 | 0.0000 | **0.6883** | 1.2615 | ✓ |

特に重要なのは特異値スペクトルの内訳:

```text
heldout_Sagar : sigma_P = [11.418, 8.411, 0.000]   sigma_G = [11.417, 8.411, 0.000]   (一致)
solver_rl     : sigma_P = [14.799, 0.000]          sigma_G = [11.313, 0.000]
dfeas_rl      : sigma_P = [0.000,  0.000]          sigma_G = [11.313, 0.000]          (←完全縮退)
```

dfeas_rl の `T_P` は **全成分が numerical zero まで縮退** しており、これは「dim
constraint を過剰に与えた結果、GT が呼ぶ edit (Q_G の Diameter/Radius) を flow
できなくなった (= 観測空間で response が消えた)」状態を意味する。`d_SVD_sigma_rel`
はこの縮退を「GT スケールの 100% の差」として直接捕捉する。

一方、`d_H` の `leak_prime` 成分はこのケースで solver_rl=0.36 / dfeas_rl=0.39 と
ほぼ同水準で、縮退を強く penalise しない。`d_H` 全体としても 0.988 vs 1.025 と 3.7%
差で solver_rl を選ぶに留まる。**d_SVD_sigma_rel は overconstrained pred の応答縮退
という固有の failure mode に対して d_H より敏感** という主張は、この 1 ケースでは
裏付けられる。

注: `d_SVD_sub` を **truncation なしで** 使うと heldout_Sagar 上でも 0.8159 と高い
値が出てしまう (実質 σ=0 の不安定特異ベクトルが principal angle に混ざるため)。
**`tau` truncation が無いと d_resp と同じ失敗モードに陥る** ことを直接示している。
truncation 付き (`d_SVD_sub_trunc`) では heldout_Sagar = 0.0000 で吸収される。

#### Smoke test 2: 複数 sketch × 複数 heldout pair での annotator variance 吸収

`notebooks/test_dSVD_cohort.py` で、batch 0 のうち Ryotaro / Sagar / Johnny 3 名が
共通に annotate した 12 件の sketch に対し、2 つの heldout pair

- Pred = Sagar,  GT = Ryotaro
- Pred = Johnny, GT = Ryotaro

を計算して、**24 件 (12 sketch × 2 heldout) の intent-equivalent ペアでの分布** を
取った。狙いは「`d_SVD_*` が heldout (annotator variance) を吸収するか = v3 で
`d_resp` が落ちた `p = 0.495` の現象が再現しないか」を見ること。

24 件の集計 (heldout なので **どの metric も median ≈ 0 が望ましい**):

| metric | n | median | Q90 | max | frac<1e-3 |
|---|---:|---:|---:|---:|---:|
| d_feas | 24 | 0.0000 | 0.1103 | 0.1895 | **0.792** |
| d_H | 24 | 0.0073 | 0.6139 | 0.8214 | 0.375 |
| d_sub | 24 | 0.0068 | 0.5774 | 0.7987 | 0.375 |
| leak_prime | 24 | 0.0000 | 0.1918 | 0.2490 | 0.750 |
| d_SVD_resp | 24 | 0.0012 | 0.3115 | 0.4534 | 0.458 |
| d_SVD_sigma (abs) | 24 | 0.0067 | 5.5806 | 10.4584 | 0.417 |
| **d_SVD_sigma_rel** | 24 | 0.0003 | 0.3708 | 0.6381 | **0.583** |
| d_SVD_sub (untrunc) | 24 | 0.6898 | 0.9816 | 1.0255 | **0.208** |
| d_SVD_sub_trunc | 24 | 0.0043 | 0.7346 | 0.7563 | 0.417 |
| d_SVD_top1 | 24 | 0.0011 | 0.6462 | 1.0468 | 0.458 |

`frac<1e-3` は「heldout pair のうち何割が事実上 0 と判定されるか」を表す annotator-
variance 吸収力の代表値である (本来は §7 の `frac<1e-4` と直接比較したいが、サンプル
数が 24 件と少ないので 1e-3 で代用)。

#### Smoke test 2 からの読み取り

1. **`d_feas` は heldout 吸収では依然最強** (0.792)。新候補のどれも超えなかった。
   §7 で報告された v3 fullset の `frac<1e-4 = 0.730` と整合的。
2. **`d_H` は 0.375** で、§7 の主結果と一致 (`d_H` は subspace 構造を直接見るので、
   annotator が dim 数を変えると非ゼロ tail が立つ)。これは v4_results.md §9 で既知の
   tradeoff で、本案 P でも改善しなかった。
3. **`d_SVD_sigma_rel = 0.583`** は `d_SVD` variant の中では最も heldout 吸収が良く、
   `d_feas` と `d_H` の中間に位置する。だが `d_feas` には及ばない。
4. **`d_SVD_sub` (untrunc) = 0.208** は最悪。**truncation 無しでは raw d_resp と同じ
   性能** であることを cohort レベルでも追認した。
5. `d_SVD_sub_trunc = 0.417` は untrunc 版より遥かに良いが `d_H` と同水準。
   tau=0.01 で principal angle のうち numeric noise 寄りの成分を除いた効果として
   理にかなっている。

#### 暫定結論

- **案 P は `d_H` を置き換える main metric 候補ではない**。heldout 吸収で `d_feas` を
  超えず、`d_H` と同水準。case D / H の本筋 (subspace 比較 + leak operator) を覆す
  証拠は出ていない。
- **しかし `d_SVD_sigma_rel` には `d_H` が捕捉しない独自シグナル** がある:
  21803 sketch のように Pred の応答が numerically 縮退する overconstrained failure
  に対し、`d_H` の `leak_prime` よりも 1 桁強い signal を出す。これは
  v4_results.md §7 の「solver_rl は gain を改善するが leak を閉じない」とは別の軸の
  failure mode で、**`d_H` の diagnostic を補強する補助指標** として価値がある。
- **truncation (`tau`) は必須**。raw `d_SVD_sub` は v3 d_resp の失敗を完全に再現する。
- フル評価 (`d_H` と並列の paired Wilcoxon / dropout sweep / R2 / model ranking) を
  行うかは §10 残作業に追加した。優先度は `d_H` 関連の 5 項目より低い。

#### Smoke test に使った artifact

- 単一 sketch script : [notebooks/test_dH.py](test_dH.py)
- cohort script     : [notebooks/test_dSVD_cohort.py](test_dSVD_cohort.py)
- 21803 元データ    : `/home/sturk/Downloads/fiverr/{Ryotaro,Sagar,Johnny}/0/21803_*.FCStd`,
  `/home/sturk/Downloads/fiverr/Vitruvion_{,dfeas_}*/0/21803_*.FCStd`
- cohort 元データ   : `/home/sturk/Downloads/fiverr/{Ryotaro,Sagar,Johnny}/0/*.FCStd`
  のうち 3 名共通の 12 sketch

`d_SVD` 自体は smoke test のための per-sketch 実装で、`metrics_v4.py` には**未登録**
である (`test_dH.py::compute_d_svd` 内に local 定義)。フル評価フェーズで
`metrics_v4.py` に移植し、`exp_v4_controls.py` の出力 CSV に column を足して
`analyze_v4.py` 系の分析パイプラインに通す予定。

## 10. 残作業

1. ✓ **`d_H` を新 main metric として v3 paper の §6.1 / §6.2 / §7 を再計算** —
   今回のデータで paired stats、CDF、divergent cases を `d_H` で出し直す
   → §13.1 で実施
2. ✓ **`eps`, `bbox-normalize-dst-length`, `delta-scale` の sensitivity sweep を
   `d_H` で実施** — v3 で `d_feas` について行った robustness check の `d_H` 版
   → §13.2 で実施
3. ✓ **R3 (subset cohort discrimination) を `d_H` で評価** — heldout cohort を
   `n_dim_G ≤ 3` の subset に絞って fully-constrained regime での識別力を確認
   → §13.3 で実施
4. ✗ **R10 (finite consistency)** — `d_H` の linear approximation が finite edit
   solver の挙動と整合するかを `exp_linear_vs_finite.py` で確認
   → ユーザー指示により skip (別 PR で対応)
5. ✓ **PP-Coincident / smatch / gate sensitivity の 3 ablation を `d_H` で再 run** —
   v3 supplement で行った ablation の `d_H` 版
   → §13.4 / §13.5 / §13.6 で実施
6. ✓ **案 P (d_SVD) のフル評価** —
   smoke test (本 doc §9 後追い) で `d_SVD_sigma_rel` に独自 signal を確認したため、
   `metrics_v4.py` に case P を実装し、`exp_v4_controls.py` の出力 column に追加して
   `d_H` と並列に paired Wilcoxon / tau sensitivity / R3 cohort / PP-Coincident /
   strict-gate を実施 (smatch correlation のみ skip)。
   → §14 で実施

## 11. 実行コマンド

```bash
# Smoke (batch 0+9, dim_drop 25,50,75,100 + rel_drop 50,100)
conda run -p /home/sturk/miniforge3/envs/freecad python notebooks/v4_exp/exp_v4_controls.py \
  --batches 0 9 \
  --dim-drop-rates 25,50,75,100 \
  --rel-drop-rates 50,100 \
  --output-csv artifacts/v4_exp/v4_controls_scores_smoke09.csv \
  --details-csv artifacts/v4_exp/v4_controls_details_smoke09.csv

# Full (all 6 batches)
/home/sturk/miniforge3/envs/freecad/bin/python -u notebooks/v4_exp/exp_v4_controls.py \
  --dim-drop-rates 25,50,75,100 \
  --rel-drop-rates 50,100 \
  --output-csv artifacts/v4_exp/v4_controls_scores_full.csv \
  --details-csv artifacts/v4_exp/v4_controls_details_full.csv

# R2 continuity sweep (fine-grained dim dropout, batch 0+9)
/home/sturk/miniforge3/envs/freecad/bin/python -u notebooks/v4_exp/exp_v4_controls.py \
  --batches 0 9 \
  --dim-drop-rates 10,20,30,40,50,60,70,80,90,100 \
  --rel-drop-rates 50 \
  --no-eval-models \
  --output-csv artifacts/v4_exp/v4_controls_scores_sweep.csv \
  --details-csv artifacts/v4_exp/v4_controls_details_sweep.csv

# Analysis
python notebooks/v4_exp/analyze_v4.py --input-csv artifacts/v4_exp/v4_controls_scores_full.csv
python notebooks/v4_exp/select_best_metric.py
python notebooks/v4_exp/analyze_r2_continuity.py --input-csv artifacts/v4_exp/v4_controls_scores_sweep.csv
```

## 12. 主要 artifact

- 実装: [notebooks/v4_exp/metrics_v4.py](v4_exp/metrics_v4.py)
- 実験 driver: [notebooks/v4_exp/exp_v4_controls.py](v4_exp/exp_v4_controls.py)
- 分析: [notebooks/v4_exp/analyze_v4.py](v4_exp/analyze_v4.py),
        [select_best_metric.py](v4_exp/select_best_metric.py),
        [analyze_r2_continuity.py](v4_exp/analyze_r2_continuity.py)
- データ: [artifacts/v4_exp/v4_controls_scores_full.csv](../artifacts/v4_exp/v4_controls_scores_full.csv)
          (主表), [v4_controls_scores_sweep.csv](../artifacts/v4_exp/v4_controls_scores_sweep.csv)
          (R2 sweep)
- 集計: [v4_quantile_table.csv](../artifacts/v4_exp/v4_quantile_table.csv),
        [v4_paired_stats.csv](../artifacts/v4_exp/v4_paired_stats.csv),
        [v4_scorecard.csv](../artifacts/v4_exp/v4_scorecard.csv),
        [v4_r2_continuity.csv](../artifacts/v4_exp/v4_r2_continuity.csv)

## 13. `d_H` 徹底テスト (§10 残作業 1, 2, 3, 5 の実施結果, 2026-05-21 追記)

§10 の Task 1, 2, 3, 5 をすべて実施 (Task 4 はユーザー指示で skip)。`d_H` は
sensitivity sweep / fully-constrained cohort / PP-Coincident 除外 / strict-gate
cohort / smatch correlation のすべてで主結果 (人間 > model, p << 1e-10) を
維持し、当時は paper main metric として確定できると判断した。
2026-05-24 の Hθ 追試後は、この節は旧 `d_H` の robustness record と位置付ける。

実装:

- 解析: [analyze_dH_paper.py](v4_exp/analyze_dH_paper.py) (Task 1, 3),
  [analyze_phase_b.py](v4_exp/analyze_phase_b.py) (Task 2, 5(1)),
  [ablation_smatch_gate.py](v4_exp/ablation_smatch_gate.py) (Task 5(2), 5(3))
- driver: [run_parallel_remaining.sh](v4_exp/run_parallel_remaining.sh)
  (5 並列 full pipeline run, ~15 min)
- code patch: `exp_v4_controls.py` に `--exclude-pp-coincident` flag 追加

### 13.1 Task 1 — `d_H` で §6 / §7 / §9 を再計算

§4 (paired Wilcoxon) と §3 (variant 別 median) に含まれる base/solver_rl の
`d_H` 数値が v3 paper §6.1 / §6.2 / §7 の `d_feas` 版を直接置き換える形に
なる。本セクションでは §3 / §4 で出していない 2 表 — **`d_H` decomposition
table per drop rate** と **OES-`d_H` divergent cases per model** — のみ追加する。

#### 13.1.1 §6.2 dropout sweep の連続値分解 (n=396) — X6 後

| variant | `d_feas` | `d_H` | `d_sub` (gain') | `leak_prime` (leak') |
|---|---:|---:|---:|---:|
| source_self | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heldout | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dim_drop25 | 0.0002 | 0.5424 | 0.5348 | 0.0000 |
| dim_drop50 | 0.1547 | 0.7071 | 0.7071 | 0.0000 |
| dim_drop75 | 0.4430 | 0.8660 | 0.8660 | 0.0000 |
| dim_drop100 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| rel_drop50 | 0.2975 | 0.9031 | 0.8783 | 0.1935 |
| rel_drop100 | 0.4284 | 1.0834 | 1.0409 | 0.2530 |
| **base** | 0.1213 | 0.7354 | 0.7188 | 0.0212 |
| **solver_rl** | 0.0421 | 0.5321 | 0.5195 | 0.0732 |

`d_sub` / `leak_prime` 分解は v3 の `gain`/`leak` (binary 的) と等価な
診断軸を **連続・微分可能** な形で提供する。base→solver_rl で
`d_sub` 0.72→0.52 (subspace は人間に接近) / `leak_prime` 0.021→0.073
(relational mismatch は悪化) という、v3 §7 の「RL は gain を直すが leak は
閉じない」構造を維持。

#### 13.1.2 §9 OES vs `d_H` divergent cases (base, top-5)

| OES | `d_H` | `d_sub` | leak' | `d_feas` | file (gt) |
|---:|---:|---:|---:|---:|---|
| 0.729 | 1.334 | 1.329 | 0.123 | 0.811 | 24603_a4021250_0000_sketch00 (Ryotaro) |
| 0.753 | 1.237 | 1.231 | 0.123 | 0.414 | 24603_a4021250_0000_sketch00 (Sagar) |
| 0.694 | 1.237 | 1.231 | 0.126 | 0.414 | 24603_a4021250_0000_sketch00 (Eko) |
| 0.941 | 1.138 | 1.133 | 0.088 | 0.492 | 21734_7cf58bd0_0004_sketch04 (Ryotaro) |
| 0.934 | 1.138 | 1.133 | 0.103 | 0.587 | 21734_7cf58bd0_0004_sketch04 (Johnny) |

`d_H` は分散の上限が ~1.4 程度に収まるため (`leak_prime` が `1/sqrt(n_R * dim_P)`
で正規化されるため)、v3 paper の主役だった `127646` の extreme leak (41-100) は
top に出ない。新 figure 候補主役は **`24603_a4021250`**。すべての case で
OES は 0.69-0.94 と高評価で、`d_H` ≥ 1.1 → "OES が見落とす design intent failure を
`d_H` が捕捉" は再現。

Artifacts: [dH_paired_paper.csv](../artifacts/v4_exp/dH_paired_paper.csv),
[dH_dropout_sweep.csv](../artifacts/v4_exp/dH_dropout_sweep.csv),
[dH_model_ranking.csv](../artifacts/v4_exp/dH_model_ranking.csv),
[dH_cdf_long.csv](../artifacts/v4_exp/dH_cdf_long.csv),
[dH_oes_divergent_cases.csv](../artifacts/v4_exp/dH_oes_divergent_cases.csv)

### 13.2 Task 2 — Sensitivity sweep

`exp_v4_controls.py` を center (eps=1e-9, delta-scale=0.01,
bbox-normalize-dst-length=1.0) に加え, eps ∈ {1e-7, 1e-11}, delta-scale ∈ {0.005,
0.05}, bbox ∈ {0.5, 2.0} で再走 (6 full runs)。

#### 13.2.1 median `d_H` の swing

| axis | value | heldout | base | solver_rl | rel_drop50 | dim_drop50 |
|---|---:|---:|---:|---:|---:|---:|
| (center) | 1e-9 / 0.01 / 1.0 | 0.0000 | 0.7749 | 0.5195 | 0.9405 | 0.7071 |
| eps | 1e-7 | 0.0000 | 0.7667 | 0.5130 | 0.9402 | 0.7071 |
| eps | 1e-11 | 0.0000 | 0.7667 | 0.5130 | 0.9402 | 0.7071 |
| delta_scale | 0.005 | 0.0000 | 0.7708 | 0.5130 | 0.9396 | 0.7071 |
| delta_scale | 0.05 | 0.0000 | 0.7749 | 0.5195 | 0.9394 | 0.7071 |
| bbox | 0.5 | 0.0000 | 0.7603 | 0.5130 | 0.9366 | 0.7071 |
| bbox | 2.0 | 0.0000 | 0.7902 | 0.5314 | 0.9584 | 0.7071 |

最大 swing は `bbox=2.0` の base で 0.77 → 0.79 (+2.0%) / solver_rl で
0.52 → 0.53 (+2.3%)。**heldout / dim_drop50 の median はすべての設定で完全に
不変** (annotator absorption と dim drop signal は設定 non-sensitive)。

#### 13.2.2 paired z (`d_H`, model vs heldout)

| axis | value | base z (p) | solver_rl z (p) |
|---|---:|---:|---:|
| (center) | 1e-9 | +13.62 (3.0e-42) | +11.23 (3.0e-29) |
| eps | 1e-7 | +13.59 (4.5e-42) | +11.25 (2.4e-29) |
| eps | 1e-11 | +13.58 (5.3e-42) | +11.23 (2.9e-29) |
| delta_scale | 0.005 | +13.58 (5.3e-42) | +11.14 (8.0e-29) |
| delta_scale | 0.05 | +13.61 (3.3e-42) | +11.19 (4.7e-29) |
| bbox | 0.5 | +13.64 (2.3e-42) | +11.23 (2.9e-29) |
| bbox | 2.0 | +13.77 (4.0e-43) | +11.19 (4.7e-29) |

**全 7 設定で paired z swing は ±0.15 以内**、p 値はすべて 1e-29 以下。
v3 §6.3 / Appendix 3 の `d_feas` eps robustness と同等の安定性を `d_H` でも示す。
v4_results.md §1 「`d_H` は probe 設計に依存しない」と整合。

Artifacts: [sensitivity_summary.csv](../artifacts/v4_exp/sensitivity_summary.csv),
[sensitivity_paired.csv](../artifacts/v4_exp/sensitivity_paired.csv),
[sensitivity/scores_*.csv](../artifacts/v4_exp/sensitivity/)

### 13.3 Task 3 — R3 subset cohort (`n_dim_G ≤ 3`)

heldout の `min_dim_G ≤ 3` を満たす rotation のみで paired Wilcoxon。

| Model | n | Metric | h | m | t | z | p |
|---|---:|---|---:|---:|---:|---:|---:|
| base | 125 | `d_feas` | 80 | 13 | 32 | +7.32 | 2.4e-13 |
| base | 125 | **`d_H`**    | 95 | 22 |  8 | **+8.02** | **1.1e-15** |
| base | 125 | `d_sub`      | 95 | 19 | 11 | +8.10 | 5.4e-16 |
| base | 125 | `leak_prime` | 77 | 15 | 33 | +6.54 | 6.1e-11 |
| base | 125 | `d_feas_gain`| 28 |  1 | 96 | +4.38 | 1.2e-05 |
| base | 125 | `d_feas_leak`| 63 | 16 | 46 | +5.57 | 2.6e-08 |
| solver_rl | 152 | `d_feas` | 74 | 13 | 65 | +6.23 | 4.5e-10 |
| solver_rl | 152 | **`d_H`**    | 108 | 28 | 16 | **+8.09** | **6.1e-16** |
| solver_rl | 152 | `d_sub`      | 106 | 24 | 22 | +8.16 | 3.5e-16 |
| solver_rl | 152 | `leak_prime` | 91 |  9 | 52 | **+8.08** | 6.5e-16 |
| solver_rl | 152 | `d_feas_gain`|  3 |  4 | 145 | -0.59 | **0.55 (n.s.)** |
| solver_rl | 152 | `d_feas_leak`| 76 | 11 | 65 | +6.75 | 1.5e-11 |

**重要観察**: fully-constrained 領域 (heldout 側が dim を絞った highly constrained
状態) では、v3 `d_feas_gain` は solver_rl に対して **n.s. (z=-0.59)** に縮退する
(gain residual は dim 不足を測る構成のため当然)。しかしその regime でも **`d_H`
の subspace 成分 (`d_sub`) と operator leak (`leak_prime`) は両方とも z>+8 で
強く識別**。すなわち `d_H` は under/over-constrained 両 regime で well-defined
であり、`d_feas` の構成上のブラインドスポットを補完する。

Artifact: [dH_subset_cohort.csv](../artifacts/v4_exp/dH_subset_cohort.csv)

### 13.4 Task 5(1) — PP-Coincident 除外 ablation

`source_probes(gt_sketch, exclude_point_point_coincident=True)` を全 evaluation
に適用 (full pipeline 1 run)。

| Model | Metric | setting | n | h | m | t | z | p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| base | `d_H` | include_pp | 307 | 257 | 35 | 15 | +13.62 | 3.0e-42 |
| base | `d_H` | exclude_pp | 306 | 256 | 33 | 17 | **+13.64** | **2.3e-42** |
| base | `leak_prime` | include_pp | 307 | 167 | 44 | 96 | +9.39 | 6.1e-21 |
| base | `leak_prime` | exclude_pp | 306 | 167 | 41 | 98 | **+9.85** | **7.1e-23** |
| solver_rl | `d_H` | include_pp | 366 | 256 | 74 | 36 | +11.23 | 3.0e-29 |
| solver_rl | `d_H` | exclude_pp | 365 | 260 | 78 | 27 | **+11.48** | **1.7e-30** |
| solver_rl | `leak_prime` | include_pp | 366 | 230 | 26 | 110 | +12.68 | 7.8e-37 |
| solver_rl | `leak_prime` | exclude_pp | 365 | 233 | 20 | 112 | **+13.11** | **2.8e-39** |

per-rotation diagnostic (n_changed = PP 除外で値が変わった rotation 数):

| metric | variant | n | changed | med abs Δ | med signed Δ |
|---|---|---:|---:|---:|---:|
| `d_H` | solver_rl | 365 | 233 | 0.0016 | +0.0014 |
| `d_H` | rel_drop50 | 392 | 381 | 0.0085 | **-0.0051** |
| `leak_prime` | solver_rl | 365 | 232 | 0.013 | +0.010 |
| `leak_prime` | rel_drop50 | 392 | 367 | 0.049 | **-0.038** |
| `d_sub` | (all variants) | — | — | ~0 | ~0 |

**読み**: PP-Coincident 除外で `d_H` paired z は **わずかに強化** (+13.62 → +13.64,
+11.23 → +11.48); `leak_prime` も同様 (+12.68 → +13.11, +9.39 → +9.85)。
モデル leak の主成分は PP-Coincident **ではない**。rel_drop の signed Δ が負
なのは「除外された PP-Coincident は元から GT 側からも消えるので penalty が縮む」
という整合的挙動。v3 で HistCAD OES の confound として知られた PP-Coincident は、
**`d_H` の判定にはほぼ影響しない**。

Artifacts:
[v4_controls_scores_no_ppcoincident.csv](../artifacts/v4_exp/v4_controls_scores_no_ppcoincident.csv),
[pp_coincident_paired.csv](../artifacts/v4_exp/pp_coincident_paired.csv),
[pp_coincident_diagnostic.csv](../artifacts/v4_exp/pp_coincident_diagnostic.csv)

### 13.5 Task 5(2) — smatch correlation

`exp_smatch_metric.py` の出力 `d_smatch_{rel, sym, rms}` と v4 details の
`d_H` / `d_sub` / `leak_prime` / `d_feas` を `(rotation_id, variant, gt)` で
join (n=522 per variant)。

| variant | smatch metric | v4 metric | n | Spearman ρ | Pearson r |
|---|---|---|---:|---:|---:|
| heldout | `d_smatch_sym` | `d_H` | 522 | **+0.534** | +0.546 |
| heldout | `d_smatch_sym` | `d_feas` | 522 | +0.516 | +0.369 |
| solver_rl | `d_smatch_sym` | `d_H` | 522 | +0.538 | +0.562 |
| solver_rl | `d_smatch_rms` | `d_feas` | 522 | **+0.665** | +0.319 |
| solver_rl | `d_smatch_sym` | `leak_prime` | 522 | +0.343 | +0.273 |
| dim_drop50 | `d_smatch_sym` | `d_H` | 522 | +0.348 | +0.465 |

heldout と solver_rl で `d_smatch_sym` と `d_H` の Spearman 相関は +0.53 / +0.54
で **適度に一貫**。dim_drop 系では smatch metric が rotation 内でほぼ一定値を
取るため相関が弱い。これは「smatch が token-level の表現差、`d_H` が
response-level の design intent 差」という別軸性と整合し、v3 の "smatch では
intent-equivalence は判定できない" 主張を `d_H` 側でも踏襲できる。

Artifact: [ablation_smatch_dH_correlation.csv](../artifacts/v4_exp/ablation_smatch_dH_correlation.csv)

### 13.6 Task 5(3) — Strict-gate cohort (DoF==2 ∧ Converged)

旧 OES CSV の strict-gate admit cohort (base n=60, solver_rl n=321) に v4 結果を
restrict して paired Wilcoxon。

| Model | n_strict | Metric | n_paired | h | m | t | z | p |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 60 | `d_feas` | 60 | 20 | 6 | 34 | +2.58 | 9.9e-3 |
| base | 60 | **`d_H`** | 60 | 27 | 20 | 13 | **+1.68** | 9.2e-2 (n.s.) |
| base | 60 | `d_sub` | 60 | 29 | 16 | 15 | +2.30 | 2.2e-2 |
| base | 60 | `leak_prime` | 60 | 20 | 8 | 32 | +2.09 | 3.6e-2 |
| solver_rl | 321 | `d_feas` | 318 | 155 | 33 | 130 | +7.85 | 4.3e-15 |
| solver_rl | 321 | **`d_H`** | 318 | 208 | 74 | 36 | **+9.39** | **6.2e-21** |
| solver_rl | 321 | `d_sub` | 318 | 196 | 75 | 47 | +8.78 | 1.7e-18 |
| solver_rl | 321 | `leak_prime` | 318 | 186 | 24 | 108 | **+11.32** | **1.1e-29** |

- solver_rl: strict gate でも `d_H` z=+9.39 / `leak_prime` z=+11.32、新 gate
  (+11.23 / +12.68) からほぼ変動なし。
- base: cohort 縮小 (309→60) で `d_H` z=+1.68 (p=0.09, n.s.) に弱化。方向は維持
  されるが、strict gate で生き残った 60 件は under-constrained failure を含まない
  ため signal が薄まる (v3 §7.1 で観察された "loadable subset bias" と同じ構造)。
  これは `d_H` 固有の問題でなく v3 `d_feas` でも同じ。

Artifact: [ablation_gate_strict.csv](../artifacts/v4_exp/ablation_gate_strict.csv)

### 13.7 まとめ

§10 残作業 (Task 4 を除く) のすべてで `d_H` 主結果は維持。追加で

- hyperparameter sensitivity 絶対保証 (eps / delta / bbox を ±1 桁振っても
  paired z swing は 0.15 以内)
- fully-constrained regime での識別力保持 (`d_feas_gain` が縮退する `n_dim_G ≤ 3`
  cohort で `d_H` / `leak_prime` は z>+8)
- PP-Coincident への non-dependence (除外しても結論不変、むしろ強化)
- gate change への robustness (strict / loose 両 gate で同方向結論)

を示し、当時は `d_H` を **paper main metric として確定**できる状態にした。
ただし 2026-05-24 の Hθ 追試 (§14) により、main scalar は `d_H_angle`、
主診断は `d_sub_angle` / `leak_angle` に更新する。

残課題は §10 task 4 (R10 finite consistency) と §10 task 6 (案 P d_SVD フル評価)
のみで、いずれも別 PR で対応予定。

(2026-05-24 追記) §10 task 6 は §14 で実施。残課題は task 4 のみとなった。

## 14. 案 P (d_SVD) 徹底テスト (§10 task 6 の実施結果, 2026-05-24 追記)

§10 task 6 を実施。推奨 3 variant (`d_SVD_sigma_rel`, `d_SVD_sub_trunc`,
`d_SVD_top1`) を `d_H` と並列にフル評価。Task 1 (paired Wilcoxon, dropout
sweep, OES divergent), Task 2 (tau sensitivity), Task 3 (R3 subset cohort),
Task 5-1 (PP-Coincident 除外), Task 5-3 (strict-gate cohort) を実施。
Task 5-2 (smatch correlation) は d_SVD が SVD 内部比較で token-level 表現と
独立軸なため skip。

**結論**: case P は `d_H` を置き換える main metric ではないが、`d_SVD_sub_trunc`
と `d_SVD_top1` は appendix supplement diagnostic として paper 採用に値する
強い識別力を示した。`d_SVD_sigma_rel` は cohort 全体では弱いが §9 後追い
smoke test で示した overconstrained pred (dfeas_rl 系) の応答縮退検出という
独自軸を持つ。

実装:

- 解析: [analyze_d_metric_paper.py](v4_exp/analyze_d_metric_paper.py)
  (analyze_dH_paper.py の汎化版, `--metric-main` / `--metric-related` / `--tag`),
  [analyze_phase_b_dsvd.py](v4_exp/analyze_phase_b_dsvd.py) (tau sweep),
  [ablation_smatch_gate.py](v4_exp/ablation_smatch_gate.py)
  (`--skip-smatch` / `--metric-list` / `--tag` flag 追加)
- metric: [metrics_v4.py](v4_exp/metrics_v4.py) の `_compute_case_p`,
  [_fast.py](v4_exp/_fast.py) の cached path にも同等実装を追加
- driver: [run_parallel_dsvd.sh](v4_exp/run_parallel_dsvd.sh)
  (7 並列 full pipeline run, ~35 min)
- code patch: `exp_v4_controls.py` に `--tau-p` flag 追加

### 14.1 Task 1 — paired Wilcoxon (full 6 batches)

| Model | Metric | n | h | m | t | z | p |
|---|---|---:|---:|---:|---:|---:|---:|
| base | d_feas | 307 | 218 | 26 | 63 | +11.68 | 1.6e-31 |
| base | **d_H** | 307 | 256 | 36 | 15 | **+13.62** | **3.2e-42** |
| base | d_SVD_sigma_rel | 274 | 202 | 32 | 40 | +10.63 | 2.2e-26 |
| base | **d_SVD_sub_trunc** | 274 | 215 | 32 | 27 | **+11.93** | **8.1e-33** |
| base | d_SVD_top1 | 274 | 207 | 40 | 27 | +11.18 | 4.9e-29 |
| solver_rl | d_feas | 366 | 195 | 39 | 132 | +8.94 | 3.9e-19 |
| solver_rl | **d_H** | 366 | 256 | 74 | 36 | **+11.25** | **2.2e-29** |
| solver_rl | d_SVD_sigma_rel | 366 | 191 | 85 | 90 | +6.25 | 4.1e-10 |
| solver_rl | d_SVD_sub_trunc | 366 | 234 | 92 | 40 | +9.33 | 1.1e-20 |
| solver_rl | **d_SVD_top1** | 366 | 227 | 82 | 57 | **+10.08** | **6.5e-24** |

**読み**:

- 3 variant すべて両モデルで人間優位 (z > +6, p < 1e-9)
- `d_H` に対し `d_SVD_sub_trunc` は base で 88%、solver_rl で 83% の paired z
  を達成 (各 +11.93/+9.33 vs +13.62/+11.25)
- `d_SVD_top1` は solver_rl で **`d_H` の 90%** (z=+10.08) の識別力。
  cohort 全体では sub_trunc/top1 が sigma_rel より強い
- sigma_rel の n=274 は他 (n=307) より少ない: pred の sigma が完全 0
  (dim count 0 系の縮退) で sigma_G_norm ≈ 0 となるケースで分母発散を
  避けるために除外されているため。dim_drop100 で `min_d_SVD_sigma_rel = NA`
  になる挙動と整合

### 14.2 Task 1.2 — variant 別 median (full 6 batches)

| variant | d_feas | d_H | d_SVD_sigma_rel | d_SVD_sub_trunc | d_SVD_top1 |
|---|---:|---:|---:|---:|---:|
| source_self | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heldout | 0.0000 | 0.0000 | 0.0006 | 0.0015 | 0.0002 |
| dim_drop25 | 0.0002 | 0.5107 | 0.3097 | 0.5774 | 0.3473 |
| dim_drop50 | 0.1543 | 0.7071 | 0.5122 | 0.7071 | 0.7286 |
| dim_drop75 | 0.4444 | 0.8660 | 0.6718 | 0.8660 | 0.9792 |
| dim_drop100 | 1.0000 | 1.0000 | NA | NA | NA |
| rel_drop50 | 0.2977 | 0.9394 | 0.5579 | 0.9434 | 0.9695 |
| rel_drop100 | 0.4296 | 1.1047 | 0.6529 | 1.0000 | 1.1749 |
| **base** | 0.1213 | 0.7667 | 0.3071 | 0.7117 | 0.7213 |
| **solver_rl** | 0.0426 | 0.5195 | 0.1088 | 0.5495 | 0.3992 |

- `d_SVD_sub_trunc` は dim_drop sweep で 0 → 0.71 → 1.00 と滑らかに上昇
  (`d_H` の `d_sub` 成分とほぼ同じ shape)、dim_drop100 だけ NA
- `d_SVD_top1` は rel_drop100 で 1.17 と `d_H` (1.10) を超える: top-1
  singular direction が relation drop の影響を相対的に強く受ける
- solver_rl で sigma_rel/top1 は heldout/source_self に最も近い (0.11/0.40),
  sub_trunc は中間 (0.55) — case ごとに異なるシグナル軸を持つことを示唆

### 14.3 Task 2 — tau sensitivity (paired z swing)

3 variant について tau_p ∈ {0.001, 0.005, 0.01 (center), 0.05, 0.1} の
paired z を比較。center は §13 と同じ `v4_controls_scores_full_dsvd.csv`
(tau_p=0.01); 4 sweep CSV は `tau_sweep/scores_tau_{value}.csv`。

| Metric | tau_p | base z | solver_rl z |
|---|---:|---:|---:|
| d_SVD_sigma_rel | 0.001 | +10.54 | +6.24 |
| | 0.005 | +10.48 | +6.37 |
| | **0.01 (ctr)** | **+10.63** | **+6.25** |
| | 0.05 | +10.60 | +6.51 |
| | 0.1 | +10.62 | +6.23 |
| d_SVD_sub_trunc | 0.001 | +11.97 | +9.26 |
| | 0.005 | +12.00 | +9.34 |
| | **0.01 (ctr)** | **+11.93** | **+9.33** |
| | 0.05 | +11.89 | +9.40 |
| | 0.1 | +11.69 | +9.27 |
| d_SVD_top1 | 0.001 | +11.20 | +10.11 |
| | 0.005 | +11.16 | +10.04 |
| | **0.01 (ctr)** | **+11.18** | **+10.08** |
| | 0.05 | +11.24 | +10.25 |
| | 0.1 | +11.22 | +10.13 |

**最大 swing**: sigma_rel で base 0.15 / solver_rl 0.28、sub_trunc で base
0.31 / solver_rl 0.14、top1 で base 0.08 / solver_rl 0.21。**全 variant で
paired z swing ≤ 0.31** (§13.2 で `d_H` が示した ±0.15 と同等水準)。tau は
load-bearing でない。tau=0.01 を center とした smoke test の選定は妥当。

### 14.4 Task 3 — R3 subset cohort (n_dim_G ≤ 3)

heldout の `min_dim_G ≤ 3` cohort (base n=125, solver_rl n=152) で paired
Wilcoxon。fully-constrained regime で case P が機能するかを確認。

| Model | n | Metric | z | p |
|---|---:|---|---:|---:|
| base | 125 | d_feas | +7.27 | 3.5e-13 |
| base | 119 | d_SVD_sigma_rel | +6.43 | 1.3e-10 |
| base | 119 | **d_SVD_sub_trunc** | **+7.29** | **3.1e-13** |
| base | 119 | d_SVD_top1 | +6.82 | 8.8e-12 |
| solver_rl | 152 | d_feas | +6.13 | 8.9e-10 |
| solver_rl | 152 | d_SVD_sigma_rel | +5.63 | 1.8e-08 |
| solver_rl | 152 | d_SVD_sub_trunc | +7.65 | 2.0e-14 |
| solver_rl | 152 | **d_SVD_top1** | **+7.91** | **2.6e-15** |

**重要観察**: §13.3 で `d_feas_gain` が solver_rl に対し n.s.
(z=-0.59) に縮退した fully-constrained regime で、case P 3 variant
すべて z > +5.6 を維持。`d_SVD_top1` は solver_rl で d_feas を上回る
+7.91、`d_SVD_sub_trunc` は base で d_feas と同等 +7.29。`d_H` (§13.3 で
+8.02/+8.09) より少し弱いが、d_feas の弱点を補う diagnostic として機能。

### 14.5 Task 5-1 — PP-Coincident 除外

`source_probes(gt_sketch, exclude_point_point_coincident=True)` を全
evaluation に適用 (full pipeline 1 run, `v4_controls_scores_no_ppcoincident_dsvd.csv`)。
include vs exclude の paired z 差で PP-Coincident への依存度を測る。

| Metric | Model | include z | exclude z | diff |
|---|---|---:|---:|---:|
| d_SVD_sigma_rel | base | +10.63 | +10.34 | 0.293 |
| d_SVD_sigma_rel | solver_rl | +6.25 | +6.25 | 0.005 |
| d_SVD_sub_trunc | base | +11.93 | +11.78 | 0.158 |
| d_SVD_sub_trunc | solver_rl | +9.33 | +9.33 | 0.003 |
| d_SVD_top1 | base | +11.18 | +11.11 | 0.066 |
| d_SVD_top1 | solver_rl | +10.08 | +10.13 | 0.045 |

**全 variant で z 差 ≤ 0.30** (§13.4 で `d_H` が示した同等の non-dependence)。
solver_rl での swing は ≤ 0.05 とほぼ完全な非依存。`d_H` と同様、case P
の判定は PP-Coincident 制約に **structurally non-dependent**。

### 14.6 Task 5-3 — Strict-gate cohort (DoF==2 ∧ Converged)

旧 OES CSV の strict-gate admit cohort (base n=60, solver_rl n=321) に
v4 結果を restrict して paired Wilcoxon。

| Metric | Model | n_strict | n_paired | z | p |
|---|---|---:|---:|---:|---:|
| d_feas | base | 60 | 60 | +2.58 | 9.9e-03 |
| d_H | base | 60 | 60 | +1.61 | 1.1e-01 |
| d_SVD_sigma_rel | base | 60 | 60 | +1.82 | 6.8e-02 |
| d_SVD_sub_trunc | base | 60 | 60 | +2.68 | 7.5e-03 |
| d_SVD_top1 | base | 60 | 60 | +2.39 | 1.7e-02 |
| d_feas | solver_rl | 321 | 318 | +7.70 | 1.4e-14 |
| d_H | solver_rl | 321 | 318 | +9.42 | 4.5e-21 |
| d_SVD_sigma_rel | solver_rl | 321 | 318 | +3.90 | 9.5e-05 |
| **d_SVD_sub_trunc** | solver_rl | 321 | 318 | **+7.26** | 3.8e-13 |
| **d_SVD_top1** | solver_rl | 321 | 318 | **+8.14** | 3.9e-16 |

- solver_rl: `d_SVD_top1` (+8.14) と `d_SVD_sub_trunc` (+7.26) は strict
  gate でも `d_feas` (+7.70) と同等、`d_H` (+9.42) からはやや低下するが
  方向性は完全保持
- base: cohort 縮小 (309→60) で `d_H` が +1.61 (n.s.) に弱まる §13.6
  と同じ "loadable subset bias" 構造を共有。case P 3 variant のうち
  `d_SVD_sub_trunc` (+2.68) と `d_SVD_top1` (+2.39) は `d_feas` (+2.58)
  と同程度、`d_H` (+1.61) より僅かに強い
- **sigma_rel は solver_rl で +3.90 とやや弱化**: strict gate cohort は
  overconstrained pred 比率が低いため、sigma_rel の独自シグナル軸の出番が
  減ることと整合

### 14.7 v4_scorecard.csv (N1-N5 + R3 自動評価)

| metric | score_total |
|---|---:|
| **d_H** | **5.960** |
| d_sub | 5.943 |
| **d_SVD_sub_trunc** | **5.731** |
| **d_SVD_top1** | **5.673** |
| d_feas | 5.656 |
| d_SVD_sigma_rel | 5.465 |

case P 3 variant とも `d_feas` (5.656) と同等以上のスコア。`d_H` /
`d_sub` には及ばないが、appendix supplement metric として paper 採用の
資格を満たす (§4.2 N1-N5 / R3 criteria の合算)。

### 14.8 まとめと推奨

| 軸 | d_H (main) | d_SVD_sub_trunc | d_SVD_top1 | d_SVD_sigma_rel |
|---|---:|---:|---:|---:|
| paired z (base / solver_rl) | +13.62 / +11.25 | +11.93 / +9.33 | +11.18 / +10.08 | +10.63 / +6.25 |
| R3 cohort z (base / solver_rl) | +8.02 / +8.09 | +7.29 / +7.65 | +6.82 / +7.91 | +6.43 / +5.63 |
| tau swing (max) | — | 0.31 | 0.21 | 0.28 |
| PP non-dependence (max diff) | 0.02 (§13.4) | 0.16 | 0.07 | 0.29 |
| strict-gate solver_rl z | +9.42 | +7.26 | +8.14 | +3.90 |
| scorecard | 5.96 | 5.73 | 5.67 | 5.46 |

**最終推奨**:

1. **main metric**: `d_H` (§13.7 の結論を維持)。3 variant 全てが `d_H`
   より弱いため、置き換えは行わない
2. **supplement diagnostic A**: `d_SVD_sub_trunc` — cohort 全体 / R3 cohort /
   strict-gate のすべてで case P 3 variant 中最も `d_H` に近い。pred の
   観測空間 subspace 構造を SVD ベースで補強する diagnostic として
   `d_sub` の代替/補完
3. **supplement diagnostic B**: `d_SVD_sigma_rel` — cohort 全体では弱いが
   overconstrained pred の応答縮退 (smoke §9 後追いで示した 21803 sketch の
   dfeas_rl 完全縮退) という `d_H` が捕捉しない failure mode に対し 1 桁
   強い signal。`sigma_P_norm` ≈ 0 の縮退率を per-rotation で出せる diagnostic
4. **d_SVD_top1**: 概ね sub_trunc と同等。1 つに絞る場合は sub_trunc を
   優先 (top-1 のみで判定は principal subspace 全体の情報を捨てるため)

### 14.9 §9 案 P の予測との対比

§9 後追いで「`d_SVD_sigma_rel` には `d_H` が捕捉しない独自シグナル」
「`appendix diagnostic として補強する位置付け`」と書いた予測は、フル評価で
**部分的に裏付けられた**:

- 独自シグナル ✓: strict-gate cohort で sub_trunc/top1 が d_H に近い数値を
  示すのに対し、sigma_rel だけ +3.90 と独自に弱化する挙動が、sigma_rel が
  cohort 全体とは別の "縮退検出" 軸を測っていることを示唆
- main metric 置換ではない ✓: scorecard / paired z すべてで d_H が優位
- tau truncation 必須 ✓: smoke §9 の `d_SVD_sub` (untruncated) failure を
  本フル評価では tau ∈ {0.001..0.1} sweep で **paired z swing ≤ 0.31** を
  確認、tau 選択は load-bearing でない

§9 では「`d_SVD_sigma_rel` を主役」と暫定的に位置付けていたが、cohort 全体の
paired Wilcoxon では `d_SVD_sub_trunc` / `d_SVD_top1` が sigma_rel を上回る。
21803 sketch のような overconstrained failure はコホート内で稀で、cohort
全体での識別は sub_trunc / top1 が担当、sigma_rel は per-sketch failure mode
分析の diagnostic という二段構成が paper-ready の整理。

### 14.10 主要 artifact

- 解析スクリプト: [analyze_d_metric_paper.py](v4_exp/analyze_d_metric_paper.py),
  [analyze_phase_b_dsvd.py](v4_exp/analyze_phase_b_dsvd.py),
  [run_parallel_dsvd.sh](v4_exp/run_parallel_dsvd.sh)
- 主データ (3846 rows × 7 CSV): [v4_controls_scores_full_dsvd.csv](../artifacts/v4_exp/v4_controls_scores_full_dsvd.csv),
  [v4_controls_scores_no_ppcoincident_dsvd.csv](../artifacts/v4_exp/v4_controls_scores_no_ppcoincident_dsvd.csv),
  [tau_sweep/scores_tau_{0.001,0.005,0.05,0.1}.csv](../artifacts/v4_exp/tau_sweep/),
  [v4_controls_scores_sweep_dsvd.csv](../artifacts/v4_exp/v4_controls_scores_sweep_dsvd.csv)
- per-variant artifacts (各 3 variant × 各 task):
  `{dSVD_sigmarel, dSVD_subtrunc, dSVD_top1}_{paired_paper, dropout_sweep, model_ranking, cdf_long, oes_divergent_cases, subset_cohort, tau_sensitivity_summary, tau_sensitivity_paired, pp_coincident_paired, pp_coincident_diagnostic, ablation_gate_strict}.csv`
  + `*_paper_summary.txt`, `*_phase_b_dsvd_summary.txt`, `*_ablation_summary.txt`

### 14.11 残課題

- §10 task 4 (R10 finite consistency) のみ残存 (case P でも別 PR で対応)
- 21803-tier overconstrained failure (sigma_P_norm ≈ 0) の per-rotation
  inspection で `d_SVD_sigma_rel` の独立シグナル軸の qualitative case
  study を追加するかは paper revision の判断


## 15. Hθ angular leak 追試まとめ (2026-05-24, 2026-05-28 X6 反映)

`leak_prime` は単独では強いが、`d_H = sqrt(d_sub^2 + leak_prime^2)` の total score では
絶対値が小さく、実質的に `d_sub` が支配的になる。この scale mismatch を解消するため、
`orth(row(J_R))` と `U_P = orth(range(R_P))` の subspace overlap を測る
`leak_angle` を追加した。

実装:

- [metrics_v4.py](v4_exp/metrics_v4.py): `d_sub_angle`, `leak_angle`, `d_H_angle`
- [_fast.py](v4_exp/_fast.py): cached `compute_metrics_v4` に同じ追加
- [exp_v4_controls.py](v4_exp/exp_v4_controls.py): CSV 出力キー追加
- [analyze_v4.py](v4_exp/analyze_v4.py): quantile / paired / dropout sweep 集計に追加

Artifacts (X6 後の full re-run):

- [v4_controls_scores_full.csv](../artifacts/v4_exp_x6/v4_controls_scores_full.csv)
- [v4_controls_details_full.csv](../artifacts/v4_exp_x6/v4_controls_details_full.csv)
- [v4_quantile_table.csv](../artifacts/v4_exp_x6/v4_quantile_table.csv)
- [v4_paired_stats.csv](../artifacts/v4_exp_x6/v4_paired_stats.csv)
- [v4_dropout_sweep.csv](../artifacts/v4_exp_x6/v4_dropout_sweep.csv)

(pre-X6 の artifact は [v4_exp_angle/](../artifacts/v4_exp_angle/) に保持。)

結論 (X6 後の更新):

- `leak_angle` は `leak_prime` と違い、solver_rl / rel_drop で `d_sub_angle` と同程度の
  scale に乗る。X6 と独立 (primitive 空間) なため絶対値も完全保持。
- solver_rl paired discrimination は `d_sub` z=+11.91 (X6 で +10.70 → +11.91)、
  旧 `d_H` z=+11.95 (+11.24 → +11.95)、`d_H_angle` z=+12.59 (+12.36 → +12.59)、
  `leak_angle` z=+13.25 (不変)。
- X6 で `d_sub` が大きく強化された結果、`d_H_angle` の優位は縮まった
  (旧 1.66 差 → 0.68 差)。それでも solver_rl では `d_H_angle` > `d_sub` の順序は維持。
- base では `d_H_angle` は旧 `d_H` / `d_sub` とほぼ同水準で、わずかに弱い (X6 前と同じ)。
- heldout tail は X6 で僅かに縮小 (`d_H_angle` Q90=0.6364 vs X6 前 0.6705)。
- 推奨は維持: **`d_sub` だけで十分とは判断しない**。`leak_angle` を mandatory diagnostic
  として残し、単一 scalar が必要なら `d_H_angle` を使う。X6 後の `d_sub` 単体強化は
  scalar simplification のオプションを広げるが、failure mode 分解 (gain' / leak')
  の有用性は変わらない。

## 16. X6 以降の pending re-run (2026-05-28)

X6 で `pairwise_distance_observable` に Circle/Arc radius を append した結果、
J_phi に依存する metric (= `d_sub`, `d_H`, `d_H_angle`, `d_K`, `d_M`, `d_SVD_*`
全変種) の数値が変化する。本 doc §3, §4, §5, §5.1, §7, §13.1, §15 は再走後の
数値で更新済みだが、以下の ablation は別途追加データセットを生成する必要があるため
pre-X6 の数値のまま残してある:

| 節 | 内容 | 再走に必要な追加 run |
|---|---|---|
| §6 | R2 連続性 sweep (10% 刻み dim/rel drop) | `exp_v4_controls.py --dim-drop-rates 10,20,...,90,100 --rel-drop-rates 50 --no-eval-models` |
| §13.2 | Sensitivity sweep (eps / delta-scale / bbox を ±1 桁) | 6 full runs (eps={1e-7,1e-11} × delta-scale={0.005,0.05} × bbox={0.5,2.0}) |
| §13.3 | R3 subset cohort (`n_dim_G ≤ 3`) | `analyze_dH_paper.py` 系の再走 |
| §13.4 | PP-Coincident 除外 ablation | `exp_v4_controls.py --exclude-pp-coincident` 1 run |
| §13.5 | smatch correlation | `exp_smatch_metric.py` 出力との join |
| §13.6 | Strict-gate cohort | 旧 OES CSV cohort restrict |
| §14 | 案 P (d_SVD) フル評価 | tau sweep / R3 cohort / PP / strict-gate の各 dedicated run |

優先度:

1. **§13.3 (R3 subset cohort)** — fully-constrained regime での `d_sub` 強化が
   `d_H_angle` の優位を完全に消すか確認。X6 後の `d_sub` z 強化 (+10.70 → +11.91)
   が cohort restrict 後にも保持されるかが paper claim の根幹に影響する
2. **§13.4 (PP-Coincident 除外)** — X6 と直交する ablation だが、X6 後の z 上昇が
   PP-Coincident に支えられていないことを確認するために必要
3. §13.2 (sensitivity sweep) — X6 後の `d_sub` z 強化が hyperparameter 設定に
   依存しないことの確認。±0.15 程度の swing で収まるはず
4. §6 (R2 連続性) — radius observable で per-rotation 連続性がさらに改善するか確認
5. §14 (d_SVD) — Case P は appendix supplement 扱いなので優先度低
6. §13.5 (smatch correlation), §13.6 (Strict-gate) — paper supplement 用なので
   余力があれば実施

再走スクリプトは sequential で 1 run あたり ~4 min なので、全部やっても
1 時間以内で完了する。
