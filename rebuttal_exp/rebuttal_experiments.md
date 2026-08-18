# Rebuttal 検証実験手順書

## 0. この文書の目的と実行上の制約

本書は、査読コメントに対する追加検証を、別の実装担当者が曖昧性なく再現できる粒度で定義する。主眼は次の三点である。

1. 一次近似に基づく motion metrics が、有限編集後の実際の solver behavior を予測するか。
2. 指標が数値単位および primitive parameterization にどの程度依存するか。
3. under-constrained system における Moore–Penrose pseudoinverse の最小ノルム解選択に、結論がどの程度依存するか。

その後、multi-reference aggregation、human-human 分布、dataset scope、RL/downstream validation について補助実験を行う。

### 0.1 現在の厳守事項

- 本マシンでは学習が進行中である。ユーザーが明示的に解除するまで、**既存の Python コード、設定、データ、チェックポイントを一切変更しない**。
- GPU を使う処理、全データに対する FreeCAD solve、大規模な並列処理も実行しない。
- 本書作成時点では、実験設計と既存実装の参照だけを確定する。
- 実装開始が許可された後も、原則として既存 core code は変更せず、新規スクリプトを `rebuttal_exp/` 以下に追加する。
- 出力は `artifacts/rebuttal/` 以下に置き、既存の `ablation/out/` 等を上書きしない。

### 0.2 記号

- sketch を \(s\)、その human annotation（constraint set）を \(C\) とする。
- 比較する二つの annotation を \(P\) と \(G\) とする。有限編集実験では原則としてどちらも human annotation である。
- primitive state を \(x\)、driving/dimensional constraint value を \(p\) とする。
- constraint residual Jacobian を \(J_x=\partial f/\partial x\)、driving parameter Jacobian を \(J_p=\partial f/\partial p\) とする。
- 現行の最小ノルム応答を
  \[
  R_0=-J_x^\dagger J_p
  \]
  とする。
- observable map を \(\phi(x)\)、その Jacobian を \(J_\phi\) とする。
- 論文の通常の pair metrics を
  \[
  \theta_{\mathrm{full}},\quad
  \theta_{\mathrm{rel}},\quad
  \theta_{\mathrm{leak}},\quad
  d_{\mathrm{place}}=1-S_{\mathrm{place}}
  \]
  と表記する。

### 0.3 既存実装の参照先

実装担当者は、実装前に少なくとも以下を読むこと。

- bundle 構築: `ablation/bundle_loader.py` の `build_bundle`
- manifest: `src/eval/manifest.py` の `discover_manifest`
- entity ordering: `src/eval/file_bundle.py` の `build_base_eids`
- 評価設定: `src/eval/config.py` の `BASE_POLICY`
- motion metrics: `src/eval/motion_metrics.py` の `SketchMotion`, `pair_paper_metrics`
- FreeCAD solve と constraint 更新: `src/metrics/histcad_2d.py` の `_solve_via_freecad`, `_set_quantity`, `_evaluate_observable`, `_bbox_diag`
- dimensional constraint 判定・signature・observable: `src/metrics/edit_response_distance.py` の `_is_dimensional`, `constraint_signature`, `pairwise_distance_observable`
- native response 構築: 同ファイルの `NativeResponse` とその生成処理

関数名や引数が変更されている場合は、同等の現行実装を `rg` で特定する。既存評価との一致を優先し、独自解釈で metric 定義を変えない。

---

## 1. 実験 A: 一次近似 metric と finite solver behavior の対応

### 1.1 答える査読コメント

> The evaluation is based only on infinitesimal local behavior. ... The paper should validate how well the proposed distances predict actual solver behavior under edits of different magnitudes.

この実験では、線形化から有限変位を外挿した \(M_C^{\mathrm{lin}}(\alpha)\) や secant subspace を新たな正解として使わない。実際の CAD solver に同一の driving dimension edit を与え、solver が返した有限編集後 geometry から非線形 observable を再計算する。その finite discrepancy と、編集前に一度だけ計算した論文の通常 metric との順位相関を見る。

### 1.2 検証仮説

- H-A1: annotation pair の \(\theta_{\mathrm{full}}\)、\(\theta_{\mathrm{rel}}\)、\(\theta_{\mathrm{leak}}\)、\(1-S_{\mathrm{place}}\) が大きいほど、同一 finite edit 後の observable response discrepancy も大きい。
- H-A2: 編集量が小さい領域で相関が最も強く、編集量を大きくすると nonlinear effects、branch change、solver failure により相関が徐々に低下しうる。
- H-A3: 実用的な編集量まで正の相関が保たれれば、tangent-space metric が有限編集 behavior の有用な proxy である。

相関が全編集量で強いことを成功条件にはしない。重要なのは、編集量ごとの劣化曲線と failure count を透明に示すことである。

### 1.3 分析単位

主要分析の一行は、次の tuple で定義する。

```text
(sketch_id, annotation_P, annotation_G,
 matched_driver_signature, sign, magnitude)
```

同一 pair に複数の共通 driver がある場合も、それぞれを独立した有限編集 trial として保存する。pair 内で median を取らない。

ただし、同一 sketch・同一 annotation pair 由来の行は統計的に独立ではない。したがって confidence interval は sketch 単位の cluster bootstrap で計算する。通常の iid Spearman p-value は主要結果として報告しない。

### 1.4 対象 annotation pair

優先順位は次の通り。

1. **主要分析:** 同一 sketch に対する異なる human annotations の全 directed pair \(P\to G\)。
2. human-human の metric 値に十分な dynamic range がない場合の補助分析:
   - 共通 driver を保持した model-human pair。
   - 明示的に作った synthetic degraded constraint set と human annotation の pair。

補助分析は human-human と混ぜず、別表・別曲線として報告する。

directed pair とする理由は \(\theta_{\mathrm{leak}}(P\to G)\) が非対称だからである。対称 metric について同じ unordered pair が二重に現れることは、cluster bootstrap で依存性を扱う。必要なら補助表で unordered pair に限定した結果も示す。

### 1.5 共通 driving parameter の厳密な matching

各 annotation に含まれる dimensional/driving constraints を `_is_dimensional` で列挙し、`constraint_signature` と同じ canonicalization を用いて matching する。

共通 driver は、少なくとも以下が完全一致するものだけとする。

- constraint type
- ordered/canonicalized entity references
- subelement/endpoint position
- driving quantity の種類（length、radius、angle 等）
- orientation や sign convention に関係する追加 field が signature に含まれる場合はそれも含める

同じ signature が一つの annotation 内に複数存在して一意に対応付けられない場合、その signature は skip し、`skip_duplicate_signature` として記録する。曖昧な近似 matching はしない。

対応する baseline quantity \(q^P_0,q^G_0\) は物理的に同じ値であることを tolerance 内で確認する。推奨 tolerance は、length について normalized sketch 単位で `1e-6 * max(1, |q0|)`、angle について radian で同様とする。不一致なら `skip_baseline_quantity_mismatch` とする。

### 1.6 編集量

length-like driver と angle-like driver を別々に扱う。

#### Length-like driver

編集量は baseline sketch の bounding-box diagonal \(L\) に対する比率で定義する。

```text
alpha_len = {0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20}
delta = sign * alpha_len * L
sign ∈ {-1, +1}
```

bundle が長辺 1 に正規化されていても、実際に solver に渡す baseline geometry から `_bbox_diag` と同じ定義で \(L\) を計算する。

#### Angle-like driver

```text
delta_deg = {0.1, 0.5, 1, 2, 5, 10, 20}
sign ∈ {-1, +1}
```

内部 API が radian を要求する場合のみ境界で変換する。CSV には `delta_deg` と `delta_internal` の両方を保存する。

#### 無効 target

radius や length が非正になるなど、編集前に明らかに domain 外となる target は solve を試みず `invalid_target_domain` と記録する。ただし failure denominator から消さず、attempt design 上の invalid として件数を別掲する。

### 1.7 baseline solve の固定

JSON geometry と FreeCAD の round-trip drift を finite response に混ぜないため、各 `(sketch_id, annotation)` について次を一度だけ行う。

1. 元 geometry と constraint set を FreeCAD に投入する。
2. driving quantity を変更せず solve する。
3. solve 成功時の geometry を `x_C^base` としてキャッシュする。
4. `phi_C^base = phi(x_C^base)` を保存する。
5. 以後の各 finite edit は、必ずこの同じ solved baseline の複製から開始する。前の magnitude の解を warm start として引き継がない。

baseline solve が失敗した annotation は、その annotation を含む trial を `baseline_solve_failure` とし、有限編集成功集合には含めない。

### 1.8 同一 finite edit の適用

matched driver ごとに、\(P\) と \(G\) の対応する既存 constraint の quantity を `_set_quantity` で更新する。

- constraint を新規挿入してはならない。
- \(P\) と \(G\) に同じ物理増分 \(\delta\) を与える。
- target はそれぞれ `q0_C + delta` とする。
- solver tolerance、iteration limit、initial geometry、FreeCAD version は両者で同一にする。
- 各 trial は baseline solved geometry から独立に開始する。

各 side について以下を保存する。

- solver return status
- convergence flag
- residual norm（取得可能なら）
- iteration count（取得可能なら）
- elapsed time
- solved geometry が finite 値を持つか
- target quantity が tolerance 内で満たされたか

「solver API が成功を返した」だけでなく、geometry finite、constraint residual、target attainment を使って成功判定する。具体的 tolerance は既存 FreeCAD 評価のものを再利用し、スクリプト内に理由なく別値を作らない。

### 1.9 exact finite observable の定義

有限編集後の observable は、必ず solved geometry から非線形に再計算する。`J_phi @ delta_x` は使わない。

`pairwise_distance_observable` と完全に同じ意味・順序を持つ standalone helper を実装する。入力 sketch 間で observable coordinate を揃えるため、`build_base_eids` と同じ canonical ordered `base_eids` を使う。

landmark の順序は既存実装に合わせ、少なくとも次を含む。

- Line: start, end
- Circle: center
- Arc: start, end, center
- Point: point coordinate

その後、landmark の全 unordered pair \(i<j\) の Euclidean distance を canonical order で並べ、円・円弧の radius を既存定義と同じ順序で append する。

同じ `base_eids`、同じ primitive type、同じ landmark order を \(P\) と \(G\)、baseline と edited geometry の全てで使う。次元が一致しない場合は silently truncate せず error/skip reason を保存する。

各 annotation \(C\in\{P,G\}\) の finite response は

\[
r_C^{q,\delta}
=
\phi(x_{C,q,\delta}^{\mathrm{solved}})
-
\phi(x_C^{\mathrm{base}})
\]

とする。

### 1.10 finite discrepancy

両 side が成功した trial に対して

\[
D_{\mathrm{finite}}^{q,\delta}(P,G)
=
\frac{\lVert r_P^{q,\delta}-r_G^{q,\delta}\rVert_2}
{\lVert r_P^{q,\delta}\rVert_2+
 \lVert r_G^{q,\delta}\rVert_2+\varepsilon}
\]

を計算する。`epsilon = 1e-12` を normalized observable space で用いる。

性質:

- 同じ finite response なら 0。
- 一方だけが動く場合はほぼ 1。
- 両 response が同程度で逆方向なら最大 2 近くになりうる。
- 編集量による absolute displacement scale を分母で正規化するため、異なる driver/magnitude を一つの相関分析に入れやすい。

両 response norm がともに `1e-10` 未満の場合は metric が数値ノイズに支配されるため、`near_zero_both_response` として主要相関から除外し、件数を報告する。閾値感度として `1e-12, 1e-9, 1e-8` も確認する。

### 1.11 infinitesimal predictor

各 annotation pair について、finite edit を行う前の baseline state から論文の通常実装で一度だけ次を計算する。

```text
theta_full
theta_rel
theta_leak(P -> G)
placement_distance = 1 - S_place
```

ここで重要なのは以下である。

- `R_C[:, q]` の driver-specific slice を predictor にしない。
- driver ごとの局所 metric を新設しない。
- finite responses から subspace を再推定しない。
- common drivers が motion subspace 全体を span するかを条件にした coverage gate を主要分析に置かない。
- 同じ pair の複数 driver を median で集約しない。

この設計は、「通常の提案 metric が、実際の有限編集時の behavioral discrepancy を予測するか」という最も直接的な検証である。

### 1.12 統計解析

length と angle は別に解析する。各 magnitude \(|\delta|\) ごとに、成功 trial 全行を用いて以下を計算する。

\[
\rho_{*,|\delta|}
=
\operatorname{Spearman}
\left(
\theta_*(P,G),
D_{\mathrm{finite}}^{q,\delta}(P,G)
\right),
\]

ここで \(*\in\{\mathrm{full},\mathrm{rel},\mathrm{leak},\mathrm{place}\}\)、`place` は \(1-S_{\mathrm{place}}\) である。

正負の edit は主要解析では同じ magnitude bin に入れるが、行自体は分ける。補助解析で `sign=+/-` を別々に計算し、符号による非対称性を確認する。

#### Confidence interval

95% CI は sketch-cluster bootstrap で計算する。

1. unique sketch IDs を replacement ありで同数抽出する。
2. 抽出された sketch に属する全 trial 行を multiplicity ごと含める。
3. Spearman \(\rho\) を再計算する。
4. 2,000 replicate を行う。
5. percentile 2.5%, 97.5% を報告する。

有効な unique metric 値または finite discrepancy 値が少なすぎて Spearman が定義できない replicate は除外し、有効 replicate 数を記録する。1,500 未満なら CI を不安定として警告する。

#### Magnitude trend

各 metric について `rho vs edit magnitude` を描く。length は x 軸を `% bbox diagonal`、angle は degree とする。これが査読への中心図になる。

### 1.13 solver failure の扱い

solver failure に任意の `D_finite=1` 等を割り当てない。Spearman は両 side が成功した trial のみで計算する。

ただし failure を隠さないため、各 magnitude ごとに必ず以下を表にする。

```text
n_designed
n_invalid_target_domain
n_baseline_failure
n_both_success
n_P_only_failure
n_G_only_failure
n_both_failure
n_near_zero_both_response
both_success_rate
```

failure rate が無視できない場合（目安: 5%以上）、各 metric の quartile ごとに `both_success_rate` を補助集計する。これにより、metric の大きい pair だけが相関分析から選択的に落ちていないかを確認する。failure の独立した scalar score \(F\) は新設しない。

### 1.14 optional sensitivity checks

主要結果ができた後に限り、以下を補助資料用に確認する。

- 同一 `(sketch, P, G, magnitude)` 内で driver と sign を median 集約してから Spearman を計算する。これは pair ごとの重みを均一にした sensitivity check であり、主要分析ではない。
- unordered human pair のみを使い、非対称な leak を除く結果。
- Pearson correlation は線形性の参考のみ。主要指標は Spearman。
- 2D histogram または hexbin により、ties と外れ値を可視化する。

### 1.15 成功・不成功時の解釈

望ましい結果は、小～中編集量で正の相関と十分な success rate があり、大編集量で相関が滑らかに低下することである。この場合は次の趣旨で返答する。

> Although the metrics are defined from the tangent space, their values positively predict discrepancies between actual nonlinear solver outcomes under matched finite edits. The correlation remains positive through [range], and degrades only as edits become large and solver failures/branch effects increase.

相関が小編集でも弱い場合は、canonical equivalence metric と finite behavior predictor を分けて主張する。結果を隠さず、局所 metric の適用範囲を明示する。大編集量で failures が急増する場合は、相関だけでなく success-rate curve を中心にして solver branch/convergence limitation を認める。

### 1.16 出力仕様

推奨ファイル:

```text
artifacts/rebuttal/finite_edit/
  run_config.json
  environment.json
  pair_metrics.csv
  driver_matches.csv
  trials.csv
  correlations.csv
  bootstrap_replicates.parquet
  failure_summary.csv
  figures/
    rho_vs_length_magnitude.pdf
    rho_vs_angle_magnitude.pdf
    success_rate_vs_magnitude.pdf
    scatter_<metric>_<magnitude>.pdf
```

`trials.csv` の必須列:

```text
sketch_id, annotation_p, annotation_g, pair_id,
driver_signature, driver_type, driver_index_p, driver_index_g,
sign, magnitude_kind, magnitude_value, delta_internal,
q0_p, q0_g, target_p, target_g,
baseline_ok_p, baseline_ok_g, solve_ok_p, solve_ok_g,
status_p, status_g, residual_p, residual_g,
response_norm_p, response_norm_g, d_finite,
theta_full, theta_rel, theta_leak, placement_distance,
skip_reason
```

---

## 2. 実験 B: numerical unit の影響

### 2.1 論点の分離

「同じ幾何を mm、cm、m で表す」ことと、「物理的に sketch を拡大・縮小する」ことを混同しない。ここで検証する unit conversion は、数値座標と全 length-like constraint value を同じ換算率で変え、その後に既存の canonical size normalization を適用する操作である。

### 2.2 解析的期待

raw coordinate を \(x\)、中心を \(c\)、characteristic length を \(L\)、unit conversion factor を \(s>0\) とする。既存 normalization が

\[
\bar{x}=\frac{x-c}{L}
\]

なら、unit conversion 後は

\[
\frac{s x-s c}{s L}=\frac{x-c}{L}
\]

である。length-like target も同じ \(sL\) で正規化されるため、正しく実装された unit conversion は normalized state と metrics を変えないはずである。

したがって、この項目は大規模 ablation を必要としない。本文または rebuttal で unit convention と normalization order を明示し、必要なら小規模 smoke test を添える。

### 2.3 smoke test protocol

20～50 sketches を固定 seed で抽出し、同じ annotation pairs に対して次を比較する。

```text
mm representation: scale factor 1
cm representation: scale factor 0.1
m representation:  scale factor 0.001
```

換算は **raw input の normalization 前** に行う。

- 全 length coordinates
- radius
- length/distance dimensional values
- solver tolerance が物理単位を持つ場合は、その tolerance

angle、dimensionless flag、entity ID は変えない。degree/radian の切替は length-unit test ではなく、今回の primary parameterization test C1/C2 の対象にも含めない。

各 representation で通常 bundle を再構築し、同じ pair の

```text
theta_full, theta_rel, theta_leak, S_place
```

を計算する。各 metric について次を報告する。

- max absolute difference
- median absolute difference
- 99th percentile absolute difference
- `allclose(rtol=1e-7, atol=1e-9)` を満たす割合

### 2.4 禁止する誤ったテスト

- normalized sketch に対して後から `coordinate_scale` だけを掛ける。
- 座標だけを変えて dimensional constraint value を変えない。
- mm/cm/m と称して物理 geometry 自体を拡大・縮小する。

これらは unit conversion ではなく inconsistent representation または physical scaling である。

### 2.5 判断

- floating-point tolerance 内で同一なら、解析的説明と smoke test を提示する。
- 有意な差が出た場合、「mm に固定するから問題ない」と即座に処理せず、normalization の前後関係、constraint target の換算漏れ、absolute solver tolerance を調べる。これは metric の本質的 unit dependence ではなく実装上の不整合である可能性が高い。
- 最終的には dataset/API の external convention が mm で固定されていることも明記するが、それだけを invariance の根拠にはしない。

---

## 3. 実験 C: primitive parameterization の影響

### 3.1 答える査読コメント

> Principal-angle distances are computed in the raw sketch-state space, which mixes coordinates, radii, angles, and different primitive parameterizations. It is unclear whether the metrics remain invariant under equivalent reparameterizations ...

本実験では、人工的な座標回転・group scaling・log-radius・degree/radian 変換を primary stress test にはしない。実際の CAD primitive schema 間で起こりうる、次元も異なる二つの表現差を Jacobian level で再現する。

```text
C1: Line  [point(2), direction(2), start, end] (6 variables)
          <-> [start point(2), end point(2)]    (4 variables)

C2: Arc   [center(2), radius, start angle, end angle] (5 variables)
          <-> [start point(2), arc midpoint(2), end point(2)] (6 variables)
```

円弧の第3点は、baseline arc の向き付き sweep のちょうど中点に固定する。中点を sweep の 1/3 や 2/3 に移す感度分析は行わない。

CAD importer、constraint handler、solver を別 schema に全面改修するのではなく、代替 state から現行 native state への局所写像を constraint/observable Jacobian に合成する。これにより、constraint semantics と baseline geometry を保ったまま、現行 metric が primitive chart の変更にどの程度影響されるかを測る。

なお、`theta_full`、`theta_rel`、`S_place` の比較自体は rigid-motion-invariant observable space で行われるが、その入力 response \(R=-J_x^\dagger J_p\) は native state の parameterization に依存しうる。`theta_leak` は response と relation row space を raw primitive tangent space で直接比較する。したがって、四指標すべてを再計算する。

### 3.2 長方形 Jacobian による変換式

native state を \(x\in\mathbb{R}^{n_x}\)、代替 state を \(y\in\mathbb{R}^{n_y}\) とし、baseline の近傍で代替表現を native 表現へ戻す写像

\[
x=g(y),\qquad
B=\left.\frac{\partial g}{\partial y}\right|_{y_0}
\in\mathbb{R}^{n_x\times n_y}
\]

を構築する。C1/C2 は次元の異なる表現間の写像であり、\(B\) は正方行列ではない。したがって、旧案のような可逆行列 \(A\) と \(A^{-1}\) は使用しない。

annotation \(C\) の full constraint Jacobian、relation-only Jacobian、dimensional-parameter Jacobian をそれぞれ \(J_{x,C}\)、\(J_{R,C}\)、\(J_{p,C}\) とすると、代替表現での Jacobian は

\[
\widetilde J_{x,C}=J_{x,C}B,\qquad
\widetilde J_{R,C}=J_{R,C}B,\qquad
\widetilde J_\phi=J_\phi B,\qquad
\widetilde J_{p,C}=J_{p,C}
\]

である。新しい response は、現行実装と同じ absolute/relative singular-value cutoff を用いて

\[
\widetilde R_C
=-\widetilde J_{x,C}^{\dagger}J_{p,C}
\]

と最初から再計算する。既存の \(R_C\) に \(B\) またはその pseudoinverse を掛けて作ってはならない。後者では、Moore--Penrose solution の parameterization dependence を測れないためである。

その後、現行 metric と同じ定義で

```text
theta_full_tilde  = distance(range(J_phi_tilde R_P_tilde),
                             range(J_phi_tilde R_G_tilde))
theta_rel_tilde   = distance(range(J_phi_tilde ker(J_R,P_tilde)),
                             range(J_phi_tilde ker(J_R,G_tilde)))
S_place_tilde     = assignment over normalized columns of J_phi_tilde R_tilde
theta_leak_tilde  = raw-y-space overlap between range(R_P_tilde)
                    and row(J_R,G_tilde)
```

を計算する。`theta_leak_tilde` は従来どおり Pred→GT の非対称量であり、勝手に observable space へ移さない。Pred と GT は同じ baseline geometry/entity ordering を持つため、同じ \(B\) と \(\widetilde J_\phi\) を使用する。

### 3.3 現行 primitive state layout の確認

実装前に `NativeResponse.state` と各 Jacobian column の対応を assertion 付きで確認する。現時点の layout は次であるが、コードを source of truth とする。

```text
Line:   6 variables [point(2), direction(2), start, end]
Circle: 3 variables [center(2), radius]
Arc:    5 variables [center(2), radius, start_angle, end_angle]
Point:  2 variables [point(2)]
```

`B` は global state ordering に沿った block-diagonal の長方形行列として組み立てる。変換対象でない primitive は identity block とする。entity ID/order のずれ、Pred/GT 間の geometry mismatch、想定外の state slice size は即時 error にする。

### 3.4 必須 transformation scenarios

同じ raw pair records に対して、native baseline、C0、C1、C2 を別々に計算する。一つの scenario 内では、対象 type の全 primitive block を同時に置換する。

#### C0: Identity pipeline test

\(B=I\)。standalone pipeline が既存 metric の全 raw-pair 値、subspace rank、response shape と numerical tolerance 内で一致することを確認する。これは paper に載せる stress result ではなく、C1/C2 の前提となる integration test である。このテストに失敗した場合は C1/C2 を実行しない。

#### C1: Line 6-parameter representation → endpoint 4-parameter representation

native line block を

\[
x_\ell=(p_x,p_y,d_x,d_y,s,t),\qquad
a=p+s d,\quad b=p+t d
\]

とする。代替 block は \(y_\ell=(a_x,a_y,b_x,b_y)\) である。baseline の \(s_0,t_0\) を固定した局所 lift

\[
d(y_\ell)=\frac{b-a}{t_0-s_0},\qquad
p(y_\ell)=a-s_0d(y_\ell),\qquad
s(y_\ell)=s_0,\quad t(y_\ell)=t_0
\]

を用い、\(B_\ell=\partial x_\ell/\partial y_\ell\in\mathbb{R}^{6\times4}\) を autograd または解析式で求める。この写像は baseline native state を厳密に通り、任意の局所 endpoint perturbation を同じ endpoints を持つ native line に戻す。

実装では forward endpoint map \(h_\ell(x_\ell)=(a,b)\) の Jacobian \(H_\ell\in\mathbb{R}^{4\times6}\) も作り、各 line で

```text
rank(H_line) == 4
rank(B_line) == 4
H_line @ B_line ≈ I_4
g_line(h_line(x0)) ≈ x0
```

を確認する。`ker(H_line)` は endpoints を変えない native representation gauge である。lift の選び方が constraint semantics に影響していないことを確認するため、orthonormal basis \(K_\ell=\ker(H_\ell)\) に対して、annotation ごとに

```text
relative_gauge_residual(J) = ||J @ K_line||_F / max(||J||_F, 1e-12)
```

を full constraint \(J_x\)、relation-only \(J_R\)、observable \(J_\phi\) について保存する。最大値が `1e-8` を超える line/pair は `line_gauge_not_annihilated` として primary 集計から除外し、件数と値を別表にする。`abs(t0-s0) <= 1e-8 * max(bbox_diagonal, 1)` の zero-length/degenerate line も同様に skip する。

#### C2: Arc angle 5-parameter representation → three-point 6-parameter representation

native arc block は

\[
x_a=(c_x,c_y,r,\alpha_s,\alpha_e)
\]

である。固定 metadata の `xDir/yDir`、`clockwise` と、branch-unwrapped な \(\alpha_s,\alpha_e\) を用いて、baseline の三点を

\[
q_s=q(\alpha_s),\qquad
q_m=q\!\left(\frac{\alpha_s+\alpha_e}{2}\right),\qquad
q_e=q(\alpha_e)
\]

と定義する。すなわち第3点 \(q_m\) は、向き付き arc sweep の幾何学的中点であり、代替 block は

\[
y_a=(q_{s,x},q_{s,y},q_{m,x},q_{m,y},q_{e,x},q_{e,y})
\]

である。他の sweep fraction は試さない。

逆写像 \(g_a(y_a)\) は三点を通る円から native parameters を回復する。circumcenter \(c\) は次の \(2\times2\) system を解いて求める。

\[
2\begin{bmatrix}(q_m-q_s)^\top\\(q_e-q_s)^\top\end{bmatrix}c
=
\begin{bmatrix}\|q_m\|^2-\|q_s\|^2\\
                \|q_e\|^2-\|q_s\|^2\end{bmatrix},
\qquad r=\|q_s-c\|.
\]

start/end angle は `atan2` で求め、固定 `xDir/yDir` と `clockwise` の規約へ戻す。\(2\pi\) branch は baseline の \(\alpha_s,\alpha_e\) に最も近く、かつ局所的に middle point を通る向き付き arc になる整数 offset を baseline で一度だけ決め、その整数を Jacobian 計算中は固定する。branch selection 自体を微分してはならない。

\(B_a=\partial g_a/\partial y_a\in\mathbb{R}^{5\times6}\) を autograd で求める。forward midpoint map \(h_a(x_a)=(q_s,q_m,q_e)\) の Jacobianを \(H_a\in\mathbb{R}^{6\times5}\) とし、各 arc で

```text
rank(H_arc) == 5
rank(B_arc) == 5
B_arc @ H_arc ≈ I_5
g_arc(h_arc(x0)) ≈ x0  # angles are compared modulo the fixed 2pi branch
```

を確認する。three-point coordinates の1本の redundant tangent direction は \(\ker(B_a)\) として残るが、これは同じ円弧を表す middle-point representation gauge であり、削除して正方行列にしない。

三点がほぼ collinear で circumcircle が不安定な arc は除外する。上の linear system matrix を \(M_a\)、`scale=max(bbox_diagonal, 1)` とし、次のいずれかを満たさない場合は `ill_conditioned_arc_three_point` として skip する。

```text
radius > 1e-8 * scale
sigma_min(M_arc) >= max(1e-10 * scale, 1e-8 * sigma_max(M_arc))
baseline point reconstruction max error <= 1e-8 * scale
baseline angle reconstruction max wrapped error <= 1e-8
```

clockwise/counterclockwise、minor/major arc、\(2\pi\) 近傍を個別に unit test する。skip threshold は formal run 前に `run_config.json` へ固定し、結果を見て変更しない。

### 3.5 対象データと集計

paper の main evaluation に使う **reference aggregation 前の raw annotation pairs** を source とする。少なくとも held-out human、提案モデル、比較モデル、relational/dimensional dropout の各 group を含める。同じ pair ID の native/C1/C2 を保存した後、各 scenario に対して paper と同じ reference aggregation を独立に適用する。scenario 間で reference ID を流用しない。

C1 は line を1本以上含む sketch、C2 は arc を1本以上含む sketch を primary 対象とする。scenario 固有の invalid/skip があるため、native vs transformed の差は必ず両方が valid な paired records だけで計算する。同時に、全 group について designed/valid/skipped sketches と raw pairs の数を理由別に報告する。

各 transformed metric \(d'_m\) と baseline metric \(d_m\) について次を保存する。

```text
absolute_delta = |d'_m - d_m|
relative_delta = |d'_m - d_m| / max(|d_m|, 1e-12)
```

metric ごと・scenario ごとに以下を報告する。

- median / Q90 / Q95 / max absolute delta
- baseline vs transformed の Spearman ranking correlation
- model/variant ordering が変わった sketch の割合
- aggregate mean/median ranking が変わるか
- worst 20 cases とその primitive composition

`theta_leak` は asymmetric direction ごとに保存する。

加えて block-level diagnostics として次を保存する。

```text
line: rank(H), rank(B), ||H B - I||, gauge residuals, |t0-s0|
arc:  rank(H), rank(B), ||B H - I||, circumcircle condition number,
      reconstruction errors, chosen angle branch
global: native/alternative state dimensions and Jacobian/response ranks
```

主要な paper result は、各 group × metric × scenario の paired median absolute delta、Spearman rank correlation、method/dropout ordering の維持率である。worst-case 図は geometry と constraint set を併記し、数値不安定による外れ値と実質的な parameterization dependence を区別する。

### 3.6 解釈と論文上の主張範囲

Appendix A.2.1 末尾の rationale、すなわち `theta_leak` が ground-truth relations への violation を直接測るため raw primitive tangent space に置かれている、という説明は semantic choice として妥当である。しかし、これだけで arbitrary reparameterization invariance は証明されない。

したがって rebuttal では次のように限定する。

- canonicalization の対象は、**固定した SketchGraphs-compatible primitive schema の下での constraint syntax/design-intent equivalence** である。
- arbitrary CAD systems 間の primitive coordinate chart に対する完全な invariance は主張しない。
- `theta_leak` は relation-centric な fixed-schema diagnostic である。
- endpoint-based line chart と midpoint-three-point arc chart への Jacobian-level 変換により、現実的な primitive reparameterization で aggregate conclusions がどの程度安定かを実証する。

変動が小さければ robustness evidence とする。変動が大きければ claim を狭め、parameterization-invariant metric を将来課題とする。「CAD system 依存だから無関係」とだけ返すのは避ける。

### 3.7 実装工数の目安

学習停止後、既存 core を変更せず standalone script と test を作る前提:

```text
rectangular-Jacobian framework + C0 test:        0.5–1.0日
C1 line endpoint map + gauge diagnostics/tests: 0.5–1.0日
C2 arc three-point inverse + branch tests:       1.0–2.0日
aggregation, plots, reproducibility checks:      0.5日
合計:                                           2.5–4.5日
```

これは Jacobian-level の chart stress test の工数であり、別 CAD importer/solver による end-to-end 再実装は含まない。後者は constraint handler、serialization、branch semantics、solver integration まで変わる別実験であり、rebuttal の必須範囲には含めない。

### 3.8 出力仕様

```text
artifacts/rebuttal/parameterization/
  run_config.json
  transformed_pair_metrics.parquet
  transform_diagnostics.parquet
  skip_summary.csv
  summary.csv
  ranking_changes.csv
  worst_cases.csv
  figures/
    delta_by_transform.pdf
    rank_correlation_by_transform.pdf
```

---

## 4. 実験 D: pseudoinverse minimum-norm choice の感度

### 4.1 答える査読コメント

> For under-constrained systems, the pseudoinverse selects one minimum-norm response although multiple valid geometry updates exist. The influence of this choice should be analyzed.

under-constrained sketch では、線形化方程式

\[
J_xR=-J_p
\]

の解は一般に一意でない。最小ノルム解 \(R_0=-J_x^\dagger J_p\) に、constraint を一次で破らない nullspace motion を加えた

\[
R=R_0+NZ,\qquad N=\ker(J_x)
\]

も有効である。この自由度を系統的にサンプルし、評価値と model ranking がどれだけ変わるかを測る。

### 4.2 perturb する側

主要実験では prediction/evaluated side \(P\) の response のみを perturb し、ground truth side \(G\) は現行の定義で固定する。両者を同時にランダム化すると variance の出所が不明になるためである。

補助実験として human-human の両 side を独立 perturb してもよいが、主要表とは分ける。

### 4.3 nullspace の計算

`compute_native_response` が rank 判定に使う effective threshold と同じものを使う。現行想定は

\[
\epsilon_{\mathrm{eff}}
=\max(10^{-9},10^{-6}\sigma_{\max}(J_x))
\]

であるが、実装時に source を確認する。

full SVD を使って \(J_x\) の right nullspace の orthonormal basis \(N\) を得る。economy SVD だけでは wide matrix の全 nullspace を失う場合があるので注意する。

各 response について保存する。

```text
raw_null_dim = dim ker(J_x)
observable_free_rank = rank(J_phi N)
consistency_residual = ||J_x R0 + J_p||_F
```

`consistency_residual` が既存 tolerance を超える場合は inconsistent linear system として別カテゴリにし、主要 sensitivity から除外する。

### 4.4 なぜ raw nullspace にそのまま Gaussian noise を入れないか

`N` は raw state coordinate の basis であり、その係数 norm は line direction、radius、angle 等の parameterization scale に依存する。単純な `Z ~ N(0, sigma^2)` では、どの程度 geometry を動かしたかが sketch 間で比較できない。

そこで observable space における自由 motion の大きさを制御する。

### 4.5 observable-normalized perturbation

各 driving column \(j\) について、固定 seed の isotropic random vector から \(\ker(J_x)\) の単位球上の方向

\[
\hat n_j=N\frac{g_j}{\lVert g_j\rVert_2}
\]

を引き、baseline observable response \(M_0=J_\phi R_0\) を使って

\[
\Delta_j=\eta\lVert M_{0,:,j}\rVert_2\frac{\hat n_j}{\lVert J_\phi\hat n_j\rVert_2},
\qquad
R_\eta=R_0+\Delta
\]

とする。\(\Delta_j\in\ker(J_x)\) なので \(J_xR_\eta=-J_p\) が保たれ、かつ

\[
\lVert J_\phi\Delta_j\rVert_2=\eta\lVert M_{0,:,j}\rVert_2
\]

が厳密に成り立つ。すなわち \(\eta\) は observable space での大きさである。

**draw guard.** 唯一の脆弱な演算はこの除算である。\(\ker(J_x)\) の大部分は \(J_\phi\) から見えない（\(\phi\) は完全な剛体不変量なので global rigid motion、Line の `(pnt, dir, startParam, endParam)` が持つ 1 本あたり 2 次元の parameterization gauge、`base_eids` 外の entity の全座標）。そのため引いた方向の \(\lVert J_\phi\hat n_j\rVert_2\) が数値的にゼロになりうる。

\[
\lVert J_\phi\hat n_j\rVert_2<10^{-9}
\]

の draw は棄却する（その column は摂動せず、件数を記録する）。\(10^{-9}\) は \(\epsilon_{\mathrm{eff}}\) の絶対フロアそのものであり、新しい定数ではない。

閾値の恣意性は仮定せず測定する。全 292 sketch の本番 run では gain に明確なギャップは無く、\(10^{-10}\) から \(10^{-5}\) まで 1 桁ごとに数百本ずつ棄却数が増える（1,293,670 draw 中 \(10^{-9}\):28、\(10^{-7}\):491、\(10^{-5}\):697）。したがって「ギャップがある」ではなく「影響が無い」を示す必要があり、`floor_sensitivity.csv` に各候補フロアで報告統計を再計算した結果を出す。フロアを 4 桁上げて \(10^{-5}\) にしても、報告する median / Q90 の変化は相対 0.5% 未満である。post hoc に検証できるのは run 自身のフロア以上の値のみで、それ未満は draw が生成されていないため再現できない。

棄却された draw は seed を変えて引き直さない。引き直すと \(\ker(J_x)\) 上の一様分布が条件付き分布に置き換わるためで、実測の棄却率がそれを正当化しない: `exactly_invariant` を除いた集団での棄却は 1,293,670 draw 中 30 column だった。

**厳密不変性.** `sigma_max(J_phi N) < 1e-9`、すなわち \(J_\phi N=0\) の response では、任意の許容応答に対して \(J_\phi R=J_\phi R_0\) が恒等的に成立する。したがって \(\theta_{\mathrm{full}}\)、\(\theta_{\mathrm{rel}}\)、\(S_{\mathrm{place}}\) は最小ノルム選択に**厳密に不変**であり、サンプリングの必要がない。これらは `exactly_invariant` として記録し、`invariance_coverage.csv` で集団別に報告する。raw state space の `range(R)` を見る \(\theta_{\mathrm{leak}}\) のみがここでも response 依存で、かつ draw が運ぶ gauge 成分にも応答する。この事実は別 arm に切り出さず、そのまま記述する。

### 4.6 edge cases

- `raw_null_dim = 0`: 一意解なので perturb せず、`not_underconstrained` と記録する。
- `exactly_invariant`: 上記のとおり sampling せず `invariance_coverage.csv` に計上する。全 eta でゼロの行を sensitivity 表に混ぜると、動きようのない movement に向けて median が引っ張られるため。
- `||M0[:,j]|| < 1e-12`: その driver column には noise を加えない。これは guard による棄却ではない（スケールする大きさが存在しないだけ）ので別に数える。
- 全 driver column が上記の zero column である response は `no_nonzero_observable_response` として skip する。
- 各サンプル生成後にも
  `||J_x R_eta + J_p||_F / max(||J_p||_F, 1) <= 1e-6` を検査する。
  threshold を超えた response--draw は別集計とし、metric 集計から除外する。
- 乱数 seed は `(global_seed, sketch_id, annotation_id, eta, sample_index)` から決定的に生成する。

### 4.7 perturbation strength と sample 数

```text
eta = {0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00}
n_samples = 50 per response per eta (eta=0 は1回)
```

計算量が過大な場合は最初に stratified pilot 50 sketches で実行し、分散が安定する sample 数を確認する。最終報告では最低 30 samples を確保する。

### 4.8 metric の再計算

各 \(R_\eta\) から現行 pipeline と同じ方法で metric を再計算する。

- \(\theta_{\mathrm{full}}\)
- \(S_{\mathrm{place}}\)
- \(\theta_{\mathrm{leak}}\)
- \(\theta_{\mathrm{rel}}\)

`theta_rel` が response choice と独立な定義なら、全 \(\eta\) で不変になることを implementation sanity check とする。変化した場合は cache/mutation bug を疑う。

### 4.9 主要集計

単に \(\eta\) と metric の相関を見るだけでは不十分である。各 metric と \(\eta\) について以下を報告する。

1. \(|d_\eta-d_0|\) の median / Q90 / Q95 / max。
2. 同一 pair における 50 sample の range と standard deviation。
3. baseline score と perturbed score の Spearman ranking correlation。
4. model/variant の aggregate ordering が変わるか。
5. sketch 単位の winner flip rate。
6. `observable_free_rank` 別の stratified sensitivity。
7. worst 20 cases の geometry、null dimensions、metric trajectories。

confidence interval は sketch-cluster bootstrap を使う。

### 4.10 finite solver による realism anchor（推奨、ただし補助）

実験 A の最小 magnitude の成功 trial を再利用し、各 driver について

\[
r_{\mathrm{FC}}/\delta
=
\frac{\phi(x^{\mathrm{solved}}_{q,\delta})-\phi(x^{\mathrm{base}})}{\delta}
\]

と

\[
r_{\mathrm{MP}}=J_\phi R_{0,:,q}
\]

の cosine similarity と normalized error を測る。これは別の「coordinate-weighted pseudoinverse」を仮定するものではなく、実 solver の微小 finite response が minimum-norm tangent prediction に近いかを直接見る補助検証である。

FreeCAD の内部 solver が同じ最小ノルム規約を明示的に保証しているとは主張しない。初期状態近傍からの incremental solve がどの branch を実際に選ぶかの empirical anchor として扱う。

### 4.11 rebuttal での説明

最小ノルム解は deterministic で、余計な unconstrained motion を加えない local minimum-displacement convention である。このため、エンジニアが現在の形状から連続的に drag/edit する状況の proxy として実務的に自然である。

ただし、「連続的な drag なら数学的に必ず Moore–Penrose 解になる」とは言わない。continuity だけでは nullspace component は一意に決まらない。

- sensitivity が小さい場合: 主結論は plausible な nullspace choices に対して安定であり、最小ノルムは再現可能な代表解であると主張する。
- sensitivity が一部で大きい場合: observable free rank が大きい sketch に限定して注意点を示し、指標の適用範囲を狭める。
- sensitivity が全般に大きい場合: response-selection dependence を limitation として認め、set-valued comparison や interaction model を将来課題とする。

### 4.12 出力仕様

```text
artifacts/rebuttal/pseudoinverse/
  run_config.json
  response_metadata.csv
  perturbation_metrics.parquet
  eta0_gate.csv
  invariance_coverage.csv
  guard_summary.csv
  floor_sensitivity.csv
  sensitivity_summary.csv
  sensitivity_by_observable_free_rank.csv
  ranking_changes.csv
  worst_cases.csv
  finite_anchor.csv
  figures/
    metric_delta_vs_eta.pdf
    rank_correlation_vs_eta.pdf
    sensitivity_by_observable_free_rank.pdf
```

---

## 5. 実験 E: multi-reference aggregation の整合性

### 5.1 答える査読コメント

> Each metric independently selects its most favorable reference. Consequently, a prediction may match the relational behavior of one annotation and the dimensional placement of another without being close to any complete human annotation.

現行の component-wise best-of-reference は、各 design-intent facet に対して同等に妥当な human answer を許す一方、異なる facet が別々の annotation を選ぶ「合成された理想参照」を作りうる。そこで、同一 human reference を全 component に共同選択する conservative score を追加する。

### 5.2 入力

既存の per-prediction/per-reference metric を再利用する。可能なら `ablation/out/motion_eval.csv` を読み、再 solve や response 再計算を避ける。必要列がなければ、新規 standalone script で同じ pair metrics を再計算する。

prediction \(P\) と human reference \(G_g\) の metric vector を

\[
d_g(P)=
\left[
\theta_{\mathrm{full}},
\theta_{\mathrm{rel}},
\theta_{\mathrm{leak}},
1-S_{\mathrm{place}}
\right]
\]

とする。全成分が「小さいほど良い」向きであることを assertion で確認する。

### 5.3 現行 independent-reference score

各 component を独立に最小化した vector

\[
d_{\mathrm{ind},k}(P)=\min_g d_{g,k}(P)
\]

を作る。既存 table が component ごとの値だけを報告している場合も、この vector を明示的に保存する。

比較用 scalar は

\[
D_{\mathrm{ind}}(P)=
\sqrt{\frac{1}{4}\sum_{k=1}^4 d_{\mathrm{ind},k}(P)^2}
\]

とする。

### 5.4 joint-reference score

一つの human reference を全 component に共通して選ぶ。

\[
D_{\mathrm{joint}}(P)
=
\min_g
\sqrt{\frac{1}{4}\sum_{k=1}^4 d_{g,k}(P)^2}
\]

最小 reference ID を `joint_reference_id` として保存する。同点は tolerance `1e-12` 内で判定し、annotation ID の lexicographic order で決定的に tie-break する。

この定義では metric scales が異なる可能性がある。主結果は論文の natural ranges を尊重した上式とし、感度分析として以下も計算する。

- L1: \(\min_g \frac14\sum_k d_{g,k}\)
- L-infinity: \(\min_g \max_k d_{g,k}\)
- dataset 内 human-human distribution の robust scale（IQR、0の場合は除外）で各 component を割った standardized L2

standardized score は主結果にせず、weighting choice に結論が依存しないかを見る。

### 5.5 optimism gap

\[
\Delta_{\mathrm{optimism}}(P)
=D_{\mathrm{joint}}(P)-D_{\mathrm{ind}}(P)\ge 0
\]

を計算する。また、independent minimum を与えた reference ID が4 component で何種類に分かれたかを `n_distinct_selected_refs` として保存する。

報告項目:

- model variant ごとの `D_ind`, `D_joint`, optimism gap の mean/median/Q90
- `n_distinct_selected_refs > 1` の割合
- optimism gap が 0、(0, 0.01]、(0.01, 0.05]、>0.05 の割合
- model ranking の Spearman correlation
- pairwise model ordering の flip 数
- worst optimism-gap examples
- sketch-cluster bootstrap 95% CI

### 5.6 human ceiling の計算

held-out human を prediction とみなし、残りの humans を reference pool とする leave-one-annotator-out evaluation を同じ joint protocol で行う。評価対象自身を reference pool に含めない。

人間の annotation 数が sketch ごとに異なる場合、最低2個の reference が残る sketch を主要対象とし、reference pool size 別に集計する。

### 5.7 事前 sanity result と再確認

過去の探索では、joint aggregation による aggregate deterioration は小さく、例として held-out human が概ね `0.112 → 0.121`、Base が `0.726 → 0.726`、Solver が `0.437 → 0.445`、optimism gap の Q90 最大が約 `0.034` であった。これは正式結果ではない。実装担当者は元 CSV の column definition、filter、sample count を再確認し、再現できた値だけを使う。

### 5.8 rebuttal の返答方針

joint score でも model ordering と主要結論が維持されるなら、指摘を受け入れて joint-reference aggregate を追加し、component-wise score は facet ごとの診断として残す。

推奨文意:

> We agree that independently minimizing each component can construct an optimistic composite reference. We therefore added a conservative joint-reference aggregate that selects one complete human annotation for all four components. The conclusions and method ordering remain [stable/result], while the optimism gap is [summary].

ordering が変わる場合は joint score を優先して本文結論を修正する。

### 5.9 出力仕様

```text
artifacts/rebuttal/multireference/
  per_reference_metrics.parquet
  aggregate_scores.csv
  human_ceiling.csv
  optimism_gap_summary.csv
  worst_cases.csv
  figures/
    joint_vs_independent.pdf
    optimism_gap_distribution.pdf
```

---

## 6. 実験 F: human-human full distributions と annotator variation

### 6.1 答える査読コメント

> Median motion distances of zero for held-out humans are encouraging but may hide substantial variation. Full distributions and pairwise annotator results would be more informative.

best-of-multiple-reference により 0 が増える集計だけでなく、raw human-human pair の全分布、annotator pair matrix、leave-one-annotator-out variation を示す。

### 6.2 raw pair construction

同一 sketch に annotation した全 human pair を列挙する。

- symmetric metrics (`theta_full`, `theta_rel`, `S_place`) は unordered pair を主要集計に使う。
- asymmetric metric `theta_leak` は両 directed pair を保存する。
- 同一 annotation 自身との比較は除外する。
- annotation が1個しかない sketch は raw pair analysis から除外し、除外 sketch 数を記録する。

`pair_paper_metrics` を変更せず使い、metric failure/skips も記録する。

### 6.3 必須集計

各 metric について:

- N sketches、N annotation pairs
- mean、standard deviation
- min、Q10、Q25、median、Q75、Q90、Q95、max
- exact/numerical zero の割合。zero tolerance を明記する。
- ECDF
- violin または boxen plot
- sketch ごとの median/IQR の分布

過去の raw 集計の目安は以下だが、正式値は再計算する。

```text
theta_full: mean≈0.249, median≈0.316, IQR≈[0, 0.447]
theta_rel:  mean≈0.257, median≈0.333, IQR≈[0, 0.463]
theta_leak: mean≈0.125, median≈0,     Q75≈0.302
S_place:    mean≈0.760, median≈0.759, IQR≈[0.608, 0.973]
```

### 6.4 annotator-pair matrix

6 annotatorsを行列の行・列に取り、共通に annotation した sketches 上で各 metric の median と sample count を表示する。

- symmetric metrics は対称 matrix。
- leak は directed matrix とし、`annotator_i prediction -> annotator_j ground truth` の向きを軸ラベルに明示する。
- cell ごとの sketch count が少ない場合は灰色表示し、閾値（例: N<10）を凡例に書く。
- raw value と、sketch difficulty を差し引くため sketch-fixed-effect centered value の両方を補助的に確認する。

### 6.5 leave-one-annotator-out

各 annotator \(a\) を一度 held-out prediction とし、同じ sketch の他 annotators を reference pool とする。

- independent-reference protocol
- 実験 E の joint-reference protocol

の両方を計算する。annotator ごとの median/IQR、N、bootstrap CI を報告する。これにより median 0 が特定 annotator や reference pool size に依存しないかを見る。

### 6.6 uncertainty

CI は sketch-cluster bootstrap 2,000 replicates。annotator pair cell は common sketch を resample する。多数の pairwise significance test は主目的ではないため、p-value の表を大量に作らず effect size と CI を示す。

### 6.7 図の構成

推奨 main/appendix figure:

1. 4 metrics の human-human ECDF。
2. raw pair distribution と best-of-reference distribution を重ねた図。
3. annotator-pair heatmap（metric ごと）。
4. leave-one-annotator-out forest plot。

Appendix Fig. 8 等の既存図が best-of-two reference なら、raw pair distribution とは別物であることを caption で明示する。

### 6.8 出力仕様

```text
artifacts/rebuttal/human_variation/
  raw_pairs.parquet
  distribution_summary.csv
  annotator_pair_matrix.csv
  leave_one_annotator_out.csv
  figures/
    human_ecdf.pdf
    raw_vs_best_reference.pdf
    annotator_pair_<metric>.pdf
    leave_one_annotator_out.pdf
```

---

## 7. 実験 G: dataset scope と downstream/RL validation

### 7.1 答える査読コメント

> The benchmark contains only 292 sketches, six annotators, four primitive types, and one solver environment. Moreover, the reinforcement-learning user study shows only a small preference margin over Solver-RL, with many ties, while the proposed rewards are evaluated partly using closely related metrics.

この懸念は一つの実験だけでは解消できない。rebuttal 期間内にできる robustness analysis と、追加データ/solver を要する limitation を分離する。

### 7.2 benchmark confidence intervals

Table 1 等の主要 aggregate に sketch-cluster bootstrap 95% CI を追加する。

- resampling unit は sketch。
- 同一 sketch の全 human references、全 variants を一緒に resample する。
- 2,000 replicates。
- method difference は各 replicate 内で paired に計算する。
- mean/median の両方を報告する。

method A vs B の差について、CI と bootstrap probability `Pr(A better than B)` を示す。単純な unpaired test はしない。

### 7.3 complexity stratification

292 sketches の範囲内で一般化の範囲を可視化するため、以下の sketch-level attributes で層別する。

- number of primitives
- number of constraints
- number of dimensional constraints
- primitive composition（line-only、circle/arc含む、point含む等）
- annotation disagreement（human-human metric quantile）
- under-constraint/nullspace dimension

continuous attributes は、各 bin の sample 数が極端に少なくならないよう tertile を基本とする。bin boundary は全 dataset から一度だけ決め、全 method に共通使用する。

各 stratum で method metrics、paired difference、sketch-bootstrap CI を報告する。層別後の N を必ず示し、小標本 stratum の優劣を強く主張しない。

### 7.4 annotator generalization

実験 F の leave-one-annotator-out を使い、特定の annotator syntax/style に過適合していないかを見る。training/evaluation split に annotator leakage がある場合は、可能な範囲で held-out annotator evaluation を明示する。

### 7.5 RL training seed robustness

現在進行中の学習を中断・変更しない。学習完了後に checkpoint と config を固定し、可能なら各 RL method について最低3 seeds、望ましくは5 seeds を用意する。

各 seed について:

- training steps
- best/final checkpoint selection rule
- reward normalization
- environment version
- evaluation sketch IDs
- generation/solver random seed

を保存する。checkpoint は validation criterion で選び、test metric を見て選ばない。

method 比較は sketch と seed の両方の変動を示す。最低限、seed ごとの aggregate と mean±SD を報告する。可能なら hierarchical/bootstrap resampling（seedを抽出し、その中で sketch を抽出）を使う。

### 7.6 circularity を避ける downstream endpoint

提案 reward と近縁な motion metrics だけで RL を評価すると循環的に見える。以下を明確に分ける。

#### Mechanism checks

- `theta_full`, `theta_rel`, `theta_leak`, `S_place`
- constraint validity / solver success

これらは reward が意図した behavior を改善したかを見る。

#### Independent endpoint

- 実験 A の actual FreeCAD finite-edit discrepancy と failure rate。
- constraint count/error や syntactic exact match（reward に直接含めていない場合）。
- user preference study。

有限編集実験に RL outputs を追加する場合は、human-human primary validation 完了後に行う。human と model output に共通 driver がある trial のみを対象とし、coverage/count を method ごとに必ず報告する。method により共通 driver の出現率が違うため、intersection set 上の paired analysis と、各 method の available set の両方を分ける。

### 7.7 user study の再解析

現行集計が概ね `69 win / 63 loss / 48 tie` であるなら、tie を捨てた勝率だけで強い superior claim をしない。

必須報告:

- win / loss / tie の raw counts と割合
- tie を除いた conditional win rate と exact/binomial CI
- tie を half-win とした preference score `(win + 0.5*ties)/N` と bootstrap CI
- participant-level または sketch-level cluster bootstrap（study design に合わせる）
- side/order randomization がある場合の position bias check
- sketch complexity 別の exploratory breakdown

study が paired comparison で同一 participant が複数回答している場合、回答を iid として扱わない。participant と sketch の両 cluster が必要なら two-way cluster bootstrap または mixed-effects logistic model を補助的に使う。

### 7.8 solver environment scope

FreeCAD の別 solver backend が利用可能で、実装コストが小さい場合のみ cross-solver replication を検討する。同じ FreeCAD solve を内部 Newton 実装と比較して「solver-independent」と呼ばない。

rebuttal 期間に独立 solver（例: SolveSpace 等）を正しく接続できない場合は、無理に追加せず limitation とする。

推奨文意:

> Our current benchmark targets one SketchGraphs-compatible representation and one solver environment. We add bootstrap uncertainty, complexity- and annotator-stratified analyses, and an independent finite-edit endpoint, but cross-CAD and cross-solver generalization remains future work.

### 7.9 claim calibration

- 292 sketches、6 annotators、4 primitive typesを「broadly representative of all CAD」とは言わない。
- user study の差が小さく CI が0を跨ぐ場合、「preferred」ではなく「comparable/slight observed preference」と記述する。
- ties が多いことを隠さない。
- reward metric improvement と user preference を別の evidence として記述する。

### 7.10 出力仕様

```text
artifacts/rebuttal/scope_downstream/
  benchmark_bootstrap.csv
  complexity_strata.csv
  annotator_generalization.csv
  rl_seed_summary.csv
  finite_endpoint_by_method.csv
  user_study_reanalysis.csv
  figures/
    method_difference_forest.pdf
    complexity_strata.pdf
    rl_seed_variation.pdf
    user_study_outcomes.pdf
```

---

## 8. 実験の優先順位と依存関係

rebuttal への寄与、実装コスト、計算資源を考慮し、次の順で進める。

### Priority 0: 学習終了までの read-only 準備

1. dataset/manifest と annotation schema の確認。
2. 過去 CSV・notebook の列定義と sample 数の確認。
3. 実験 A の common-driver coverage を read-only で数える。
4. 実験 E/F を既存 CSV だけで再計算できるか確認する。
5. 実行環境、FreeCAD version、commit hash を記録する。

この段階ではファイルを生成するスクリプトも実行しない。必要なら shell の read-only query のみとする。

### Priority 1: 査読の核心

1. **実験 A: finite solver behavior**
2. **実験 C: alternative primitive-chart Jacobian stress test**
3. **実験 D: pseudoinverse sensitivity**

実験 A は査読の major weakness に直接答え、実験 D の finite realism anchor も供給する。実験 C と D は FreeCAD を大量に回さず実行可能である。

### Priority 2: 低コストで強い補強

4. **実験 E: joint multi-reference**
5. **実験 F: human distributions/annotator pairs**
6. **実験 B: unit smoke test**

E/F は既存 pair metric CSV が十分なら再集計中心であり、優先度に対してコストが低い。

### Priority 3: 学習完了・計算資源解放後

7. **実験 G: RL seeds、finite downstream endpoint、user-study reanalysis**
8. 任意の cross-solver/task study。

### 8.1 最小 rebuttal package

時間が不足した場合でも、最低限以下を揃える。

- 実験 A の magnitude-wise Spearman + success/failure counts。
- 実験 C の identity test、line endpoint representation、arc midpoint-three-point representation。
- 実験 D の nullspace sensitivity curve と ranking stability。
- 実験 E の joint-reference score。
- 実験 F の raw human-human ECDF と annotator matrix。
- Table 1 の sketch-bootstrap CI。

---

## 9. 実装構成

ユーザーが実装開始を明示的に許可した後、次のような新規構成を推奨する。

```text
rebuttal_exp/
  README.md
  common.py
  exact_observable.py
  finite_edit_validation.py
  parameterization_stress.py
  pseudoinverse_sensitivity.py
  multireference_joint.py
  human_variation.py
  scope_downstream.py
  plot_results.py
  tests/
    test_exact_observable.py
    test_parameterization_identity.py
    test_nullspace_perturbation.py
    test_joint_reference.py
```

### 9.1 `common.py`

以下だけを共通化する。

- manifest/bundle loading
- deterministic seeding
- sketch-cluster bootstrap
- run metadata serialization
- stable pair/trial ID generation
- output directory creation guard

metric 本体をコピーして改変せず、既存関数を import する。

### 9.2 immutable input と mutation guard

FreeCAD/edit 処理は input object を mutation しうる。各 trial は serialized baseline から deep copy/reload して開始する。実験前後で input file hash を比較し、元 dataset が変わっていないことを確認する。

### 9.3 resume

FreeCAD 実験は長時間になりうるため、trial 単位で append-safe な shard に保存する。

- shard は worker ID ではなく deterministic sketch range/hash prefix で分ける。
- 完了 trial ID を読み、同じ config hash の trial のみ resume する。
- config hash が違う結果を同じ directory に混ぜない。
- 最終 merge 時に trial ID uniqueness を assertion する。

### 9.4 並列実行

学習終了後でも FreeCAD process safety を確認するまで並列度1で pilot する。FreeCAD が process-local state を持つ場合は thread parallelism を使わず、独立 process と独立 temp directory を使う。

### 9.5 run metadata

全実験で最低限次を `environment.json` または `run_config.json` に保存する。

```text
git commit hash
git dirty status（ファイル一覧のみでもよい）
Python version
FreeCAD version/build
OS
CPU/GPU model
package lock/hash
dataset manifest hash
evaluation config
random seeds
start/end timestamps
command line
script hash
```

dirty worktree に既存ユーザー変更がある場合、それを消したり reset したりしない。実験用新規ファイル以外の差分を結果に含めない。

---

## 10. 必須 unit/integration tests

### 10.1 exact observable test

小さな synthetic sketch を用意し、既存 `pairwise_distance_observable` と新しい exact nonlinear `phi` の baseline 出力が同一 ordering・値になることを確認する。

- line endpoints
- point
- circle center/radius
- arc start/end/center/radius
- entity order を入れ替えた input に canonical `base_eids` を与えた場合

finite edit 後の `phi` は geometry から再計算され、baseline Jacobian を使っていないことを test する。

### 10.2 common-driver matching test

- 同じ distance constraint が異なる list index にある場合も match。
- entity order が canonicalization で等価な場合は既存 signature の仕様通り match。
- endpoint location が違う場合は不一致。
- duplicate signature は曖昧として skip。
- angle と length は不一致。

### 10.3 parameterization identity test

`B=I` で transformed pipeline が baseline の全 metric、rank、response shape と `rtol=1e-8, atol=1e-10` 以内で一致する。

### 10.4 line endpoint transformation test

horizontal、vertical、oblique の非退化 line fixture について、次を確認する。

- \(g_\ell(h_\ell(x_0))=x_0\)。
- \(H_\ell B_\ell=I_4\)、`rank(H_line)=rank(B_line)=4`。
- analytic \(B_\ell\) と centered finite difference/autograd Jacobian が一致する。
- endpoint perturbation 後も native reconstruction の start/end が入力 endpoints と一致する。
- \(J_x\ker(H_\ell)\)、\(J_R\ker(H_\ell)\)、\(J_\phi\ker(H_\ell)\) が tolerance 内でゼロになる synthetic constraints を用意する。
- zero-length line が指定の reason code で skip される。

### 10.5 arc three-point transformation test

clockwise/counterclockwise × minor/major arc、start/end angle が \(2\pi\) branch を跨ぐ fixture について、次を確認する。

- 第3点が向き付き sweep の厳密な中点である。
- \(g_a(h_a(x_0))=x_0\)（angle は固定 branch で比較）。
- \(B_aH_a=I_5\)、`rank(H_arc)=rank(B_arc)=5`。
- autograd \(B_a\) と centered finite difference が一致する。
- 三点から復元した center/radius/start/end point が baseline と一致する。
- middle point を元の円周に沿って微小に動かす gauge direction が \(\ker(B_a)\) に入る。
- nearly-collinear points が `ill_conditioned_arc_three_point` で skip される。

### 10.6 nullspace test

各 sampled \(R_\eta\) に対して

```text
||J_x R_eta + J_p|| <= tolerance
||J_x N|| <= tolerance
```

を確認する。さらに、observable target noise が attainable range にある場合

```text
J_phi (R_eta - R0) ≈ E
```

であることを確認する。

### 10.7 joint-reference test

手計算可能な2 references × 4 components の fixture で:

- independent component minima が異なる references を選ぶ。
- `D_joint >= D_ind`。
- joint が一つの reference の全 component を使う。
- tie-break が deterministic。

### 10.8 bootstrap test

同一 seed で CI が再現し、同一 sketch 内の複数行が一緒に resample されることを test する。

---

## 11. 共通 quality-control checklist

各実験を正式実行する前に以下を満たすこと。

### Data integrity

- [ ] manifest の sketch 数と paper 記載数が一致する。
- [ ] annotation ID、annotator ID、sketch ID の mapping を確認した。
- [ ] train/validation/test または held-out split を記録した。
- [ ] input file hashes が実験前後で不変である。

### Metric consistency

- [ ] `BASE_POLICY` と paper table の設定が一致する。
- [ ] angle unit、length normalization、entity ordering を記録した。
- [ ] `theta_leak` の向きを列名と図に明示した。
- [ ] `S_place` を distance に混ぜるときは必ず `1-S_place` にした。
- [ ] numerical failure を 0 や worst score に silently 置換していない。

### Statistical integrity

- [ ] analysis unit を明記した。
- [ ] sketch-level dependence を cluster bootstrap で扱った。
- [ ] N sketches と N trials/pairs を両方示した。
- [ ] missing/failure/skip counts を示した。
- [ ] exploratory sensitivity と preregistered primary analysis を区別した。
- [ ] result を見て magnitude grid や threshold を都合よく変更していない。変更時は理由と全履歴を残した。

### Reproducibility

- [ ] seed と config hash を保存した。
- [ ] pilot と formal run の output directory を分けた。
- [ ] plot が raw/summary artifacts から再生成できる。
- [ ] worst-case example selection rule が決定的である。
- [ ] core code を変更していない、または必要な変更が明示的に承認・レビューされている。

---

## 12. 各査読コメントへの最終回答テンプレート

数値は formal run 後に置換する。結果が支持しない文は使わない。

### 12.1 First-order approximation

> We agree that the proposed motion metrics are defined locally. We added a finite-edit validation using matched dimensional constraints shared by pairs of human annotations. For each shared driver, we applied the same signed edit to both annotations, solved both sketches with FreeCAD from independently solved baselines, and compared the exact nonlinear observable changes. Across successful edits, the original pre-edit metrics—not driver-specific variants—showed Spearman correlations of [values] with finite response discrepancy at [magnitudes]. We also report all convergence failures rather than assigning them an artificial distance. Correlation [behavior] as edits become large, clarifying the practical range of the local approximation.

### 12.2 Unit and parameterization

> Length-unit changes are removed by the existing canonical-size normalization when coordinates and all dimensional quantities are converted consistently before normalization; we now state this convention explicitly and verify it with mm/cm/m representations (maximum numerical difference [value]). We also recomputed all four metrics after Jacobian-level changes from the native six-parameter line chart to a four-coordinate endpoint chart, and from the native center/radius/angle arc chart to a start/midpoint/end three-point chart. Native-to-transformed metric rankings were [correlations], and the model/ablation ordering [was/was not] preserved. These tests use rectangular chart Jacobians and recompute the Moore--Penrose response rather than transforming the old response. We have narrowed “canonical” to equivalence across constraint syntax within the fixed SketchGraphs-compatible primitive schema, rather than claiming invariance across arbitrary CAD primitive charts.

### 12.3 Multi-reference aggregation

> We agree that independently choosing a reference for each component can be optimistic. We added a joint-reference aggregate that selects one complete human annotation for all metric components. The optimism gap is [summary], and the method ordering [does/does not] remain unchanged. Component-wise minima are retained only as facet-specific diagnostics.

### 12.4 Dataset and downstream validation

> We added sketch-bootstrap confidence intervals, complexity-stratified results, leave-one-annotator-out analyses, and [multi-seed RL/finite endpoint results]. The finite-edit endpoint uses actual nonlinear solver outcomes and is distinct from the reward metrics. We have also moderated the user-study claim to reflect [win/loss/tie and CI]. Cross-CAD and cross-solver generalization remains a limitation.

### 12.5 Human distributions

> We now report the full raw human-human distributions, pairwise annotator matrices, and leave-one-annotator-out results, in addition to best-of-reference medians. These show [summary of variation] and clarify when zero medians arise from equivalent annotations versus reference aggregation.

### 12.6 Pseudoinverse

> The minimum-norm response is a deterministic local minimum-displacement convention, but we agree that an under-constrained linear system admits nullspace components. We therefore sampled constraint-preserving nullspace responses at controlled observable-space magnitudes and recomputed all metrics. Score/ranking changes were [summary], stratified by observable free rank. We additionally compare the minimum-norm tangent response with small actual FreeCAD edits as an empirical anchor. We do not claim that continuity uniquely implies the Moore–Penrose solution.

---

## 13. 実装担当者が最初に行う具体的な手順

ユーザーが学習終了と実装開始を明示した後、以下の順で進める。

1. `git status --short` を保存し、既存のユーザー変更を把握する。既存差分には触れない。
2. current commit、manifest、`BASE_POLICY`、FreeCAD version を read-only で記録する。
3. `rebuttal_exp/README.md` に実行コマンドと output path を定義する。
4. exact observable と common-driver matcher を実装し、Section 10.1/10.2 の tests を通す。
5. human-human 10 sketches、最小/中/最大 magnitude 各1点、並列度1で実験 A pilot を行う。
6. pilot で `trials.csv` の全 status category、observable dimensions、response norms、failure reasons を目視する。
7. magnitude grid や tolerance を変える必要があれば formal run 前に本書相当の run config に理由を固定する。
8. 実験 C の identity test を通し、line gauge diagnostics と arc branch/reconstruction tests を確認してから C1/C2 を実行する。
9. 実験 D の algebraic assertions を synthetic fixture と10 sketchesで確認してから全体を実行する。
10. 既存 CSV を使って実験 E/F を実行し、過去の探索値との差を調べる。
11. formal 実験 A を shard/resume 対応で実行する。進行中の学習や別ジョブと資源競合しないことを確認する。
12. formal artifacts を一度 freeze し、その後は raw results を変更せず plot/table を生成する。
13. Section 11 の checklist を別担当者が確認する。
14. 数値を Section 12 の回答テンプレートへ反映し、claim が実験結果を越えていないか確認する。

### 完了条件

本実験計画の完了は、単にスクリプトが終了したことではなく、以下を全て満たす状態とする。

- 各 major concern に直接対応する artifact、figure/table、解釈がある。
- solver failures と missing data が完全に集計されている。
- main analysis と sensitivity analysis が区別されている。
- sketch-cluster uncertainty が報告されている。
- raw artifact から全 table/figure を再生成できる。
- rebuttal の主張が結果に応じて適切に限定されている。
