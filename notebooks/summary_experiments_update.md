# TODO: Experiments セクション改善方針

> **進捗メモ (2026-05-19, 2 度目更新: model gate 緩和後最終版)**
>
> 本番スケール再検証 (6 batches = {0,4,6,8,9,11} × 3 annotators / batch、
> annotator set = {Eko, Johnny, Ryotaro, Sagar, Tanaka}) のフル run + 4 本の
> analysis chaining が完了。**model-output gate を緩和** (`DoF==2 ∧ Converged`
> → `Converged ∧ Status ∉ {Distorted, SolverError}`) して under-constrained
> base 出力も d_feas 評価対象に含めた結果、§9 を全面更新。`v3.md` 実験 6–9 も
> 新数値に置き換え済み。`paper/5_experiments.tex` も全面差し替え済み (build OK)。
> 旧 gate アーティファクトは `*.prev.csv` / `*.prev.png` として保存。
>
> - **データスコープ**: `DEFAULT_HUMANS` を `notebooks/exp_source_conditioned_dropout.py`
>   に 6 batches ({0,4,6,8,9,11}) で定義。下流 3 スクリプトも `DEFAULT_BATCHES`
>   を共有。`--batches` 既定値 = 6 batches。
> - **2 model 比較**: 新フラグ `--vitruvion-spec label:path` を 4 主要スクリプトに
>   追加 (`exp_constrained_response_metric.py` / `exp_source_conditioned_dropout.py` /
>   `exp_v3_controls_and_robustness.py` / `exp_smatch_metric.py`)。デフォルトで
>   `base:/.../Vitruvion_ori_..._epoch_29` と `solver_rl:/.../Vitruvion_2026-05-03_..._step_56400`
>   を両方評価。CSV `pred` 列にラベルが入る。
> - **新規 metric**: token-level Constraint F1 を `src/metrics/constraint_token_f1.py`
>   に追加 (`(type, sorted_refs, direction)` signature ベース)。
>   `notebooks/compute_constraint_token_f1.py` で per-rotation × per-pred の CSV を
>   pre-pass 計算。
> - **新規解析**:
>   - `notebooks/analyze_dfeas_distribution.py` — CDF / violin / quantile table
>   - `notebooks/analyze_model_ranking.py` — fully_constrained_rate, OES, ER,
>     F1, d_feas, gain, leak の比較テーブル
>   - `notebooks/find_dfeas_oes_divergent_cases.py` — high-OES / high-d_feas 等
>     のケース抽出 (qualitative figures 用)
>   - `notebooks/analyze_dfeas_pairwise_human_vs_model.py` — human-human vs
>     model-human の CDF + Wilcoxon paired stats
>
> **smoke test 済み** (`--batches 0 --limit 2`、~10–15 分で完走):
> - `artifacts/constrained_response_scores_smoke.csv` / `..._details_smoke.csv`
> - `artifacts/source_conditioned_dropout_scores_smoke.csv` / `..._details_smoke.csv`
> - `artifacts/v3_controls_scores_smoke.csv` / `..._details_smoke.csv`
> - `artifacts/smatch_vs_dfeas_summary_smoke.csv` / `..._details_smoke.csv`
> - `artifacts/constraint_token_f1_smoke.csv`
> - `artifacts/model_ranking_table_smoke.csv`
> - `artifacts/dfeas_quantile_table_smoke.csv` / `dfeas_cdf_smoke.png` / `dfeas_violin_smoke.png`
> - `artifacts/dfeas_oes_divergent_cases_smoke.csv`
> - `artifacts/dfeas_human_vs_model_stats_smoke.csv` / `..._cdf_smoke.png`
>
> **本番フル run コマンド (CPU only)**:
>
> ```bash
> # ~200+ files × 3 rotations × 5 dropout rates × 2 models. 数時間想定。
> # dropout sweep CSV は drop_rate ∈ {0, 25, 50, 75, 100} + model 行 (drop_rate=-1)
> # を全て含むので、これだけで実験 1 (paired vs heldout drop 0) と実験 2 (dropout
> # ablation) の両方を導出できる。専用の --drop-rates 0,100 run は不要。
> conda run -p /home/sturk/miniforge3/envs/freecad python notebooks/exp_constrained_response_metric.py \
>   --drop-rates 0,25,50,75,100 \
>   --output-csv artifacts/constrained_response_scores_dropout.csv \
>   --details-csv artifacts/constrained_response_details_dropout.csv
>
> conda run -p /home/sturk/miniforge3/envs/freecad python notebooks/exp_source_conditioned_dropout.py \
>   --drop-rates 100,75,50,25,0 \
>   --output-csv artifacts/source_conditioned_dropout_scores_full.csv \
>   --details-csv artifacts/source_conditioned_dropout_details_full.csv
>
> conda run -p /home/sturk/miniforge3/envs/freecad python notebooks/exp_v3_controls_and_robustness.py \
>   --eps-values 1e-11,1e-9,1e-7 \
>   --output-csv artifacts/v3_controls_scores_full.csv \
>   --details-csv artifacts/v3_controls_details_full.csv
>
> conda run -p /home/sturk/miniforge3/envs/freecad python notebooks/exp_smatch_metric.py \
>   --drop-rates 0,25,50,75,100 \
>   --output-csv artifacts/smatch_vs_dfeas_summary_full.csv \
>   --details-csv artifacts/smatch_vs_dfeas_details_full.csv
>
> conda run -p /home/sturk/miniforge3/envs/freecad python notebooks/compute_constraint_token_f1.py \
>   --output-csv artifacts/constraint_token_f1.csv
> ```
>
> フル run 完了後の analysis chaining:
>
> ```bash
> python notebooks/analyze_dfeas_distribution.py \
>   --input-csv artifacts/v3_controls_scores_full.csv \
>   --eps 1e-9
> python notebooks/analyze_model_ranking.py \
>   --dfeas-csv artifacts/constrained_response_scores_dropout.csv \
>   --oes-csv artifacts/source_conditioned_dropout_scores_full.csv \
>   --controls-csv artifacts/v3_controls_scores_full.csv \
>   --f1-csv artifacts/constraint_token_f1.csv
> python notebooks/find_dfeas_oes_divergent_cases.py \
>   --dfeas-csv artifacts/constrained_response_scores_dropout.csv \
>   --oes-csv artifacts/source_conditioned_dropout_scores_full.csv
> python notebooks/analyze_dfeas_pairwise_human_vs_model.py \
>   --controls-csv artifacts/v3_controls_scores_full.csv
> ```
>
> 各セクションのチェックボックスは「コード/データが用意できた段階」を `△`、
> 未着手 (P2 = §2 Oracle intent / §6 Reranking) を `[ ]` のまま、
> `paper/5_experiments.tex` および `notebooks/summary_histcad_response_metric_v3.md`
> への数値反映が完了したら `[x]` に置き換える運用。
>
> **2026-05-19 フル run 完了アーティファクト**:
> - `artifacts/constrained_response_scores_dropout.csv` (4743 rows, 5 humans × 5 drop rates × 2 model preds)
> - `artifacts/source_conditioned_dropout_scores_full.csv` (HistCAD OES sweep)
> - `artifacts/v3_controls_scores_full.csv` (eps × variant grid)
> - `artifacts/smatch_vs_dfeas_summary_full.csv`
> - `artifacts/constraint_token_f1.csv` (810 rows: 5 humans + 2 models per rotation)
> - `artifacts/model_ranking_table.csv` — fully_constrained_rate / OES / ER / F1 / d_feas / gain / leak
> - `artifacts/dfeas_quantile_table.csv` + `dfeas_cdf.png` + `dfeas_violin.png`
> - `artifacts/dfeas_human_vs_model_stats.csv` + `dfeas_human_vs_model_cdf.png`
> - `artifacts/dfeas_oes_divergent_cases.csv` (再 run 推奨: 既定閾値 `d_feas≥1` は厳しすぎたため
>   `--high-oes-threshold 0.5 --high-dfeas-threshold 0.1 --low-oes-threshold 0.05
>   --low-dfeas-threshold 0.05 --top-k 12` で再生成済み。high-OES/high-d_feas 12 件、
>   low-OES/low-d_feas 12 件)。

## 0. 全体方針の修正

> △ paper/5_experiments.tex の文言調整のみ。本プランで対応した数値証拠 (分布・
> model ranking・case study) を本番スケールで取得済み (§9)。paper 反映が未完了。

- [△] 現状の主張を少し弱める。
  - 現在の実験だけで **“d_feas is a valid design intent metric”** と強く言い切るのは危険。
  - より安全な主張は以下：
    - **d_feas is a continuous local edit-response diagnostic.**
    - **d_feas detects two failure modes that solver-based editability metrics miss or confound:**
      - loss of dimensional drivability
      - relational leakage under edits
    - **fully constrainedness is necessary but not sufficient for design-intent-aware CAD constraints.**

- [△] Experiments の役割を明確化する。
  - 目的は「d_feas が完全に design intent を測れることを証明する」ではない。
  - 目的は、以下を段階的に示すこと：
    1. human annotations 同士は d_feas 上で近い。
    2. model predictions は human annotations より遠い。
    3. OES/ER は fully constrained setting で病的な挙動を示す。
    4. d_feas は gain failure と relation leakage を分解して診断できる。
    5. d_feas は model output selection / reranking にも使える可能性がある。

- [△] “design intent” という言葉の範囲を慎重にする。
  - user study なしでは、human preference との対応はまだ示せていない。
  - したがって本文では、
    - “captures design intent”
    - “measures design intent”
    - “faithfully evaluates design intent”
  - のような強い表現は避ける。
  - 代わりに、
    - “captures local edit-response consistency”
    - “serves as a proxy for sketch-level redesign behavior”
    - “diagnoses violations of intended relational invariants”
    - “aligns with human-authored annotations under multi-reference evaluation”
  - のように書く。

---

# 1. 既存 dropout control の位置づけを修正する

> △ paper 文言の調整。新規 control sweep (eps 含む) は v3 controls スクリプトで
> カバー済み。本プランでは文章未編集。

## 現状の問題

- [ ] Dimensional dropout control は必要だが、それだけでは弱い。
- [ ] 理由：
  - dimensional dropout は d_feas の gain residual がまさに罰する対象なので、metric の定義に近い。
  - reviewer から見ると、
    - “dimension を消したら dimension-driven edit metric が悪化した”
    - と見える可能性がある。
  - つまり、これは **d_feas の unit test / sanity check** としては有効だが、**design intent metric としての外的妥当性**を示すには不十分。

## なぜ dropout control は単純すぎるか

- [ ] 理由 A: metric の定義に近い操作である。
  - dimension を落とす。
  - edit response が弱くなる。
  - gain residual が上がる。
  - これは d_feas の設計からかなり予想できる。

- [ ] 理由 B: “bad design intent” の一種類しか表していない。
  - 実際の model failure は単なる under-constrained dropout だけではない。
  - よくある問題は以下：
    - fully constrained だが relation が人間らしくない。
    - symmetry を寸法の羅列で固定する。
    - equal radius / equal length を個別寸法で表す。
    - global origin / axis に不自然に anchor する。
    - solver 的には valid だが edit behavior が不自然。
    - relation はあるが dimension driver として使いにくい。

- [ ] 理由 C: random dropout は必ずしも design intent を壊さない。
  - 落とした dimension が評価 probe に関係なければ d_feas は悪化しない。
  - 他の constraints によって同等の edit response が保たれることもある。
  - したがって random dropout は “design intent degradation” として粗い。

- [ ] 理由 D: OES 批判としては強いが、d_feas の外的妥当性としては弱い。
  - dropout control は、
    - **OES が under-constraining を reward してしまう**
    - ことを示すにはかなり良い。
  - しかし、
    - **d_feas が人間の感じる design intent と合っている**
    - ことの証明ではない。

## 書き方の修正

- [ ] dropout control は main claim ではなく、以下として位置づける：
  - negative control
  - metric sanity check
  - OES failure-mode demonstration

- [ ] “d_feas correctly penalizes all degradation” のような強い表現は避ける。
- [ ] 特に 25% dropout で median d_feas が 0 になる場合は、以下のように説明する：
  - random dropout does not always remove dimensions that are active under the selected probes.
  - this is why distributional analysis and more structured degradations are needed.

---

# 2. Oracle intent / synthetic suite をやる場合の注意

> **未着手** (本プラン範囲外、P2)。synthetic family の自動生成 + 機械的劣化が
> 主作業で、優先度低。チェックボックスは `- [ ]` のまま残す。

## 懸念

- [ ] Oracle intent 実験は恣意的になりやすい。
  - 自分で GT・良い劣化・悪い劣化を作るため、
    - “authors designed examples that favor their metric”
    - と見られるリスクがある。

## 位置づけ

- [ ] Oracle intent は主証拠にしない。
- [ ] 使うなら、以下の位置づけにする：
  - unit test
  - controlled sanity check
  - failure-mode characterization
  - appendix 寄りの補助実験

## やる場合の設計ルール

- [ ] sketch family を事前定義する。
- [ ] degradation rule を機械的に適用する。
- [ ] 有利な例だけ選ばない。
- [ ] すべての family / degradation を報告する。
- [ ] 期待 ranking が崩れた失敗例も報告する。
- [ ] 良い/悪いの定義を CAD modeling guideline に基づける。
- [ ] “hand-picked examples” に見えないようにする。

## 候補 sketch family

- [ ] symmetric bracket
  - symmetry を relation で表すのが自然。
  - 左右を個別寸法で固める variant は intent-poor。

- [ ] repeated hole pattern
  - equal radius / equal spacing / construction geometry が自然。
  - 各 hole を独立寸法で固定する variant は intent-poor。

- [ ] rounded rectangle / slot
  - tangent, equal radius, horizontal / vertical relation が重要。

- [ ] circular pattern
  - construction circle, rotational symmetry, equal radius が重要。

- [ ] parallel ribs
  - parallel relation, equal spacing が重要。

## degradation variant の例

- [ ] GT / human-like constraints
- [ ] token は違うが edit behavior は等価な constraints
- [ ] relation を 1 つ落とす
- [ ] dimension を 1 つ落とす
- [ ] symmetry を個別寸法で置き換える
- [ ] equal radius を個別 radius dimensions に置き換える
- [ ] global origin / axis に anchor する
- [ ] fully constrained だが過剰寸法的な variant
- [ ] solver-valid だが edit behavior が不自然な variant

## 期待される実験の役割

- [ ] d_feas が以下を区別できるかを見る：
  - equivalent alternative
  - mild degradation
  - severe degradation
  - under-constrained degradation
  - fully constrained but intent-poor degradation

---

# 3. Human-human distribution vs model-human distribution

> △→[x] data ready @ full scale。Paper / v3.md への図・表反映は v3.md §6
> に追記済み (このプランでの数値反映完了)。
>
> - 主スクリプト: `notebooks/analyze_dfeas_distribution.py` (CDF / violin / quantile table)
>   と `notebooks/analyze_dfeas_pairwise_human_vs_model.py` (paired Wilcoxon)。
> - フル CSV: `artifacts/dfeas_quantile_table.csv`, `artifacts/dfeas_cdf.png`,
>   `artifacts/dfeas_violin.png`, `artifacts/dfeas_human_vs_model_cdf.png`,
>   `artifacts/dfeas_human_vs_model_stats.csv`。
> - 主要数値 (§9-B 参照): heldout median `d_feas≈0` (frac<1e-4 = 0.73)、
>   solver_rl median `0.024` (frac<1e-4 = 0.41)、base median `~0`
>   (frac<1e-4 = 0.57、ただし n=60 のみ)、paired Wilcoxon solver_rl vs heldout
>   p=6.4e-15 (n=317, m=32, h=155, t=130)、base p=9.9e-3 (n=60)。

## 目的

- [ ] 示すべきゴール：
  - human-human distance は小さい。
  - model-human distance はそれより大きい。
  - d_feas は human variation を許容しつつ、model failure を検出できる。

## 現状の問題

- [ ] median だけでは不十分。
- [ ] median だけだと以下が分からない：
  - human-heldout は全体的に低いのか。
  - 半分だけ低く、残りは高いのか。
  - Vitruvion は全体的に悪いのか。
  - 少数の catastrophic outlier だけが悪いのか。
  - human-human と model-human はどれくらい overlap しているのか。
  - d_feas = 0 の tie がどれくらいあるのか。
  - heavy-tail がどれくらいあるのか。

## 追加すべき図

- [ ] CDF plot を追加する。
  - 横軸: d_feas
  - 縦軸: d_feas がその値以下のサンプル割合
  - lower-is-better metric に向いている。
  - near-zero mass, tail, distribution overlap が見やすい。

- [ ] violin plot も候補。
  - 各 group の分布の形を縦長に表示する。
  - 太い部分はサンプルが多い値域。
  - median / quartile も重ねられる。
  - distribution shape が見やすい。

## CDF と violin の使い分け

- [ ] CDF:
  - “どれくらいの割合が低スコアに入るか” が見やすい。
  - human-human と model-human の分離を見るのに向いている。
  - この論文では CDF の優先度が高い。

- [ ] Violin:
  - 分布全体の密度形状が見やすい。
  - group ごとの比較に向いている。
  - supplement でもよい。

## 追加すべき table

- [ ] median だけでなく quantile table を出す。

候補：

| Pair | Median | Q75 | Q90 | % near-zero | Mean rank |
|---|---:|---:|---:|---:|---:|
| source-self | | | | | |
| human-human | | | | | |
| model-human | | | | | |
| dimensional dropout | | | | | |
| relational dropout | | | | | |

- [ ] `% near-zero` の threshold を定義する。
  - 例: d_feas < 1e-4
  - あるいは d_feas < 1e-3
  - 数値安定性と実験結果を見て決める。

## 比較 group

- [ ] source-self
- [ ] human-human / held-out human
- [ ] model-human
- [ ] dimensional dropout
- [ ] relational dropout
- [ ] possible: OES-best but high-d_feas cases
- [ ] possible: model variants

## 書くべき結論

- [ ] human-human が低スコア側に集中するなら：
  - d_feas does not merely penalize token-level disagreement.
  - d_feas absorbs valid human-equivalent alternatives.

- [ ] model-human が右にシフトするなら：
  - model outputs exhibit edit-response deviations beyond natural human variation.

- [ ] overlap があるなら：
  - 一部の model outputs are human-equivalent under d_feas.
  - これは tie として自然。
  - metric が過度に厳しすぎないことの証拠にもなる。

---

# 4. Qualitative case study

> △ 候補抽出は本番スケールで完了。実際のフィギュアレンダリングと paper 挿入は未完了。
>
> - 主スクリプト: `notebooks/find_dfeas_oes_divergent_cases.py`。
> - フル CSV: `artifacts/dfeas_oes_divergent_cases.csv` (12 high + 12 low cases、
>   閾値 `OES≥0.5 ∧ d_feas≥0.1`、`OES≤0.05 ∧ d_feas≤0.05`)。
> - 既定の論文用 top 候補 (high-OES / high-d_feas、solver_rl):
>   - `8/26768_c4df841f_0006_sketch00.FCStd`  OES=0.89  d_feas=0.26  leak=0.27
>   - `8/43933_3b763a09_0000_sketch02.FCStd`  OES=0.93  d_feas=0.15  leak=0.19
>   - `11/67207_15bfd1aa_0000_sketch04.FCStd`  OES=0.63  d_feas=0.40  leak=0.63
>   - `11/86702_c06b5954_0000_sketch02.FCStd`  OES=0.61  d_feas=0.31  leak=0.33
>   - `0/56013_752d02cb_0001_sketch00.FCStd`  OES=0.55  d_feas=0.21  leak=0.22
>     (v3.md 旧版で使った代表例。bbox normalization 後の数値は OES≈unchanged
>     だが d_feas / leak は 2.4/2.55 → 0.21/0.22 に縮んだ。スケールが揃った)。
> - Top-K の sketch render は `notebooks/histcad_case_detail_report.py` を流用可。

## 目的

- [ ] 数値だけでは伝わらない failure mode を可視化する。
- [ ] reviewer に対して d_feas の必要性を直感的に示す。

## 最優先で載せるべき例

- [ ] high-OES / high-d_feas case
  - OES では良い。
  - ER も高い。
  - しかし d_feas は悪い。
  - 実際に edit すると relation が壊れる。
  - d_feas の leak component がそこを拾っている。

## 図に含めるもの

- [ ] original sketch
- [ ] human constraint annotation
- [ ] model / Vitruvion constraint annotation
- [ ] finite edit after applying the same query
- [ ] broken relation のハイライト
  - parallel が壊れる
  - equal radius が壊れる
  - symmetry が壊れる
  - tangent が壊れる
- [ ] OES / ER / d_feas / gain / leak の小さい table

## 候補ケース

- [ ] OES high, d_feas high
- [ ] OES low, d_feas low
  - OES が fully constrained sketch を不当に低く見る例
- [ ] d_feas near-zero, token F1 low
  - token-level では違うが edit behavior は等価な例
- [ ] model fully constrained, but relation leakage high
  - fully constrainedness が十分条件でないことを示す例

## 書くべき結論

- [ ] solver success alone does not guarantee preservation of intended geometric relations.
- [ ] d_feas identifies failures that are visually consistent with undesirable edit behavior.
- [ ] qualitative cases support the interpretation of leak and gain components.

---

# 5. Model ranking / model comparison

> △→[x] data ready @ full scale。Paper / v3.md への図・表反映は v3.md §7
> に追記済み (このプランでの数値反映完了)。
>
> - 主スクリプト: `notebooks/analyze_model_ranking.py` と前段の
>   `notebooks/compute_constraint_token_f1.py`。
> - 比較対象: `base` (`Vitruvion_ori_..._epoch_29`) と `solver_rl`
>   (`Vitruvion_2026-05-03_..._step_56400`) を `--vitruvion-spec` で同一 CSV に統合。
> - フル CSV: `artifacts/constraint_token_f1.csv`,
>   `artifacts/model_ranking_table.csv`。
> - **主要結果** (§9-C 参照):
>
> | Method | n_files | fc-rate ↑ | OES ↑ | ER ↑ | F1 ↑ | dim_F1 | rel_F1 | d_feas ↓ | gain ↓ | leak ↓ |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
> | heldout (human) | 139 | — | 0.308 | 0.250 | 0.622 | 0.333 | 0.759 | ~0 | ~0 | ~0 |
> | base Vitruvion | 20 | 0.144 | 0.195 | 0.100 | 0.559 | 0.168 | 0.667 | ~0 | ~0 | 0.043 |
> | solver_rl Vitruvion | 111 | 0.799 | 0.150 | 0.111 | 0.500 | 0.143 | 0.667 | 0.024 | ~0 | 0.054 |
>
> 解釈: solver_rl は fully_constrained_rate を 14% → 80% に押し上げるが、
> OES はむしろ低下 (preservation set が増えた分制約が厳しくなる) し、
> d_feas leak は base ≈ solver_rl (0.043 vs 0.054)。つまり solver-RL は
> fully constrained 化には成功したが design-intent leak は閉じていない。
> これが v3.md で主張したかった「fully constrainedness は必要条件だが十分条件ではない」
> の直接的な数値証拠になる。

## 目的

- [ ] d_feas が単に Vitruvion を悪く見るだけではなく、複数モデルの failure mode を診断できることを示す。
- [ ] full constrainedness と design intent behavior が別物であることを示す。

## 比較候補

- [ ] base Vitruvion
- [ ] SFT Vitruvion
- [ ] solver-RL Vitruvion / Casey-style reproduction
- [ ] heuristic post-processing variant
- [ ] relation-heavy heuristic variant
- [ ] random / corrupted controls
- [ ] human upper bound

## 出すべき表

| Method | Fully constrained ↑ | OES ↑ | ER ↑ | F1 ↑ | d_feas ↓ | Gain ↓ | Leak ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Vitruvion | | | | | | | |
| Solver-RL Vitruvion | | | | | | | |
| Heuristic variant | | | | | | | |
| Human held-out | | | | | | | |

## 見たいストーリー

- [ ] solver-RL は fully constrained rate / OES / ER を改善する。
- [ ] しかし d_feas では relation leakage が残る。
- [ ] したがって、
  - full constrainedness は必要条件だが十分条件ではない。
  - solver-based reward だけでは human-like edit behavior を保証できない。
  - d_feas はその gap を測る。

## 注意点

- [ ] model ranking の結果が予想と違う場合も隠さない。
- [ ] 例えば solver-RL が d_feas でも良いなら、それはそれで良い。
  - d_feas が model improvement を検出できる、という主張に変える。
- [ ] base と RL の差が小さいなら、
  - current solver reward improves constraint completion but does not strongly alter local edit-response behavior
  - と解釈できる。

---

# 6. Reranking experiment

> **未着手** (本プラン範囲外、P2)。Vitruvion から K=10–20 候補を sampling する
> 経路の改修が必要で、優先度低。チェックボックスは `- [ ]` のまま残す。

## 目的

- [ ] d_feas が evaluation だけでなく model output selection に使える可能性を示す。
- [ ] “d_feas can be a reward / selection signal” の初期証拠にする。

## 基本設計

- [ ] 各 sketch に対して model から K 個の候補を生成する。
  - K = 10 or 20 など。
- [ ] 候補から以下の基準で 1 つ選ぶ：
  - random sample
  - OES-best
  - fully-constrained-first
  - d_feas-best
  - maybe F1-best if oracle setting として比較する

## 重要な注意

- [ ] d_feas で選んで同じ d_feas で評価すると循環する。
  - “d_feas で選んだから d_feas が良いだけ” になる。

## 推奨: leave-one-out reranking

- [ ] 3 human references を使う。
- [ ] reference A/B を使って candidate を rerank。
- [ ] held-out reference C で評価。
- [ ] これにより、
  - d_feas selection generalizes to unseen human annotation
  - と言える。

## 評価指標

- [ ] held-out d_feas
- [ ] gain residual
- [ ] leak residual
- [ ] OES / ER
- [ ] fully constrained rate
- [ ] token F1
- [ ] qualitative examples

## 出すべき表

| Selection | Held-out d_feas ↓ | Gain ↓ | Leak ↓ | OES ↑ | ER ↑ | Fully constrained ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Random | | | | | | |
| OES-best | | | | | | |
| Fully-constrained-first | | | | | | |
| d_feas-best | | | | | | |

## 書くべき結論

- [ ] d_feas-best が held-out reference でも良いなら：
  - d_feas is not merely overfitting to a single annotation.
  - d_feas can select outputs that agree better with unseen human-equivalent intent.

- [ ] OES-best が held-out d_feas で悪いなら：
  - solver-based selection can favor outputs with poor local edit behavior.

## 目視確認の位置づけ

- [ ] 目視確認は必要だが、補助。
- [ ] 論文には定量表を載せる。
- [ ] 目視例は qualitative examples として 2–4 個載せる。
- [ ] 目視だけでは anecdotal になるため主証拠にしない。

---

# 7. 現在の Table 3 / Table 4 / Table 5 の扱い

> △ Table 自体は本番フル run の CSV を見ながら書き換えが必要。本プランでは
> 数値の再計算 (本番スケール) と新規列 (model ranking, F1) は準備済み。
> paper 本文の書き換えは未完了。

## Table 3: paired comparison

- [ ] 残す価値はある。
- [ ] ただし、これだけで “d_feas is valid” とは言わない。
- [ ] 書くべきこと：
  - OES と d_feas は model vs human の判定で異なる傾向を示す。
  - d_feas は human-heldout をより強く prefer する。
  - ties が多いことは、model が human-equivalent response を出せているケースがあることも示す。

## Table 4: dimensional dropout

- [ ] OES failure-mode 実験として残す。
- [ ] “d_feas の証明” というより、OES pathology の実証として扱う。
- [ ] 25% dropout で d_feas が 0 の場合に備えて、random dropout の限界を書く。
- [ ] “monotonic” と書くなら mean / Q75 / Q90 も確認する。
  - median だけでは単調でない可能性がある。

## Table 5: positive / negative controls

- [ ] 残すが、median だけでは弱い。
- [ ] CDF / quantile table / violin plot を追加する。
- [ ] source-self と human-heldout が near-zero であることは良い。
- [ ] ただし near-zero すぎる場合、
  - metric が鈍感なのか
  - 本当に human-equivalent を吸収しているのか
  - を qualitative / distribution で補強する。

---

# 8. 優先順位

## P0: 最優先

- [x] human-human vs model-human の CDF plot を作る。 → `dfeas_human_vs_model_cdf.png`
- [x] quantile table を作る。 → `dfeas_quantile_table.csv` (§9-B)
- [x] high-OES / high-d_feas qualitative examples を 2–4 個探す。 → `dfeas_oes_divergent_cases.csv` (12 件、§9-D)。sketch render は paper 整形時に別途。
- [△] Experiments の claim を弱める。 → paper 文言調整は v3.md §9 で方針確定、paper/5_experiments.tex 反映は未着手。
- [△] dropout control の位置づけを OES pathology / sanity check に変更する。 → 同上。

## P1: できれば入れる

- [x] model ranking / model comparison を追加する。 → `model_ranking_table.csv` (§9-C)
- [x] fully constrained rate, OES, ER, d_feas, gain, leak を並べる。 → 同上 (+ F1 / dim_F1 / rel_F1 列)
- [x] solver-RL が full constraint を改善しても relation leakage が残るかを見る。 → 確認済み: fc-rate 14% → 80%、leak は 0.043 → 0.054 (改善せず)

## P2: 余力があれば (本プラン範囲外)

- [ ] leave-one-out reranking を行う。
- [ ] d_feas-best selection が held-out human reference に generalize するか確認する。
- [ ] Oracle intent / synthetic suite を appendix で追加する。

---

# 9. 本番スケール結果サマリ (2026-05-19)

> このセクションは、本日のフル run + 4 本の analysis chaining で確定した
> 数値証拠を集約したもの。すべて 6 batches × 5 humans (Eko/Johnny/Ryotaro/Sagar/Tanaka)
> × 3-rotation hold-out + base / solver_rl の 2 model に対する評価結果。
> 元 CSV は §進捗メモの末尾に列挙してある。

## 9-A. 実験 1, 2 の本番スケール再検証 (新 gate)

**model-output gate 緩和** (`DoF==2 ∧ Converged` → `Converged ∧ Status ∉
{Distorted, SolverError}`) により base の admit 率が 14% → 77% に拡大。
旧 gate 結果は `*.prev.csv` に保存済み。

実験 1 (Vitruvion vs heldout-drop0 の paired Wilcoxon、apples-to-apples
common rotation set 上、d_feas は low-is-better、OES は high-is-better):

| Model | n_paired | Metric | h_better | m_better | tie | p (two-sided) |
|---|---:|---|---:|---:|---:|---:|
| solver_rl | 369 | HistCAD OES | 188 | 99 | 78 | 3.3e-7 |
| solver_rl | 369 | `d_feas` | 199 | 41 | 129 | 1.5e-19 |
| **base** | 309 | HistCAD OES | 75 | **204** | 27 | **2.2e-19 (sign inversion)** |
| base | 309 | `d_feas` | 217 | 29 | 63 | 1.2e-31 |

**最重要結果**: base 行で HistCAD OES が **model > human** に sign inversion
する (p=2.2e-19)。これは base の大半が under-constrained であり、その結果
OES の ER / cPCSR が両方上昇するため。§6.2 dropout sweep で示された OES
failure mode が **実 model 出力レベルで再現**した。`d_feas` は同じ rotation で
base を正しく human より悪いと判定する (p=1.2e-31)。

実験 2 (dimensional dropout sweep, validated median):

| drop% | HistCAD OES | ER | cPCSR | `d_feas` | gain | leak |
|---:|---:|---:|---:|---:|---:|---:|
| 0%  | 0.310 | 0.250 | 0.647 | 0.0000 | 0.0000 | 0.0000 |
| 25% | 0.578 | 0.506 | 0.930 | 0.0000 | 0.0000 | 0.0000 |
| 50% | 0.750 | 0.727 | 0.941 | 0.1667 | 0.1667 | 0.0000 |
| 75% | 0.884 | 0.900 | 0.943 | 0.4444 | 0.4120 | 0.0000 |
| 100%| 1.000 | 1.000 | 0.943 | 1.0000 | 1.0000 | 0.0000 |

dropout sweep は gate 変更の影響を受けない (heldout は常に admit)。
OES 単調増加、`d_feas` は gain 主成分で単調劣化。25% で `d_feas` median = 0
は random dropout の coarseness による honest disclosure。

## 9-B. Distribution (human-human vs model-human, 新 gate)

`dfeas_quantile_table.csv` (eps=1e-9, validated, multi-reference `min_i d_feas`):

| variant | n | median | Q75 | Q90 | mean | frac<1e-4 | gain_med | leak_med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source_self | 392 | ~0 | ~0 | ~0 | ~0 | 1.00 | ~0 | ~0 |
| heldout (human) | 392 | ~0 | 0.001 | 0.128 | 0.047 | 0.73 | ~0 | ~0 |
| **base** Vitruvion | 306 | **0.121** | 0.373 | **1.000** | 0.557 | 0.22 | ~0 | 0.059 |
| solver_rl Vitruvion | 365 | 0.042 | 0.126 | 0.195 | 0.074 | 0.37 | ~0 | 0.068 |
| dim_drop50 | 392 | 0.151 | 0.333 | 0.500 | 0.210 | 0.33 | 0.146 | ~0 |
| dim_drop100 | 392 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 1.000 | 0 |
| rel_drop50 | 392 | 0.299 | 0.469 | 0.821 | (heavy tail) | 0.03 | ~0 | 0.322 |
| rel_drop100 | 392 | 0.429 | 0.575 | 0.724 | (heavy tail) | 0.03 | 0.333 | 0.144 |

CDF (`dfeas_cdf.png` / `dfeas_human_vs_model_cdf.png`) の読み:

- `heldout` は 73% が `<1e-4`、Q90=0.13。別 human の annotation は概ね吸収される。
- `solver_rl` は 37% が near-zero、Q90=0.20。heldout より右シフトするが tail は短い。
- **`base` は 22% near-zero、Q90=1.0、mean=0.557**。bimodal 分布: 多くは moderate
  d_feas に集中するが、約 1 割が完全 ceiling (under-constrained で gain=1)。
  これが旧 gate (n=20) では見えなかった failure mode。
- `dim_drop100` は 1.0 縦線 (definitional)、`rel_drop50/100` に巨大外れ値あり。

paired Wilcoxon (`dfeas_human_vs_model_stats.csv`):

| model | n_paired | model_better | human_better | tie | p |
|---|---:|---:|---:|---:|---:|
| solver_rl | 365 | 39 | 194 | 132 | 4.9e-19 |
| base | 306 | 27 | **216** | 63 | 7.5e-31 |

## 9-C. Model ranking (fc-rate, eval-rate, OES, d_feas)

`model_ranking_table.csv` (新 gate):

| Method | n | fc-rate ↑ | eval-rate ↑ | OES ↑ | ER ↑ | F1 | rel_F1 | dim_F1 | d_feas ↓ | leak ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heldout (human) | 139 | — | 1.000 | 0.310 | 0.250 | 0.622 | 0.759 | 0.333 | ~0 | ~0 |
| **base** Vitruvion | 107 | 0.138 | 0.774 | **0.625** | 0.655 | 0.483 | 0.600 | 0.118 | **0.121** | 0.059 |
| **solver_rl** Vitruvion | 129 | **0.818** | **0.937** | 0.185 | 0.167 | 0.500 | 0.667 | 0.154 | 0.042 | 0.068 |

定義:
- **fc-rate** = `Status=Solved` の割合 (DoF≤2 + Converged) = RL training objective の達成率
- **eval-rate** = 新 gate を通る割合 (Converged ∧ ¬Distorted ∧ ¬SolverError)

ストーリー (論文 main claim):

1. **fc-rate**: base 13.8% → solver_rl 81.8%。RL が training objective を達成。
2. **eval-rate**: 77.4% → 93.7%。malformed 出力が減少。
3. **OES**: 0.625 → 0.185 と **逆方向に低下**。base の under-constrained 出力
   を OES が reward する failure (sign inversion p=2.2e-19、§9-A) と同根。
4. **d_feas median**: 0.121 → 0.042。RL によって gain が小さくなる正しい方向。
5. **d_feas leak**: 0.059 → 0.068。**RL は gain failure (under-constraining) を
   修正するが、leak failure (relational intent) は修正しない**。
6. **rel F1**: 0.600 → 0.667。両モデルとも heldout の 0.759 から大きく劣る。
   solver-RL の改善は marginal。

結論: `d_feas` の gain/leak 分解は **どの failure を model 改良が直したかを
透視的に表示する**。RL は容易な failure (under-constraining) を解消するが
難しい failure (relation intent) は閉じない。OES はこの区別が見えず、しかも
base に対しては sign を反転させて誤判定する。

## 9-D. Qualitative divergent cases (新 gate, top-5)

`dfeas_oes_divergent_cases.csv` `OES≥0.5 ∧ d_feas≥0.1`:

| rank | file | heldout | pred | OES | d_feas | leak |
|---:|---|---|---|---:|---:|---:|
| 1 | `0/127646_b88ed13f_0015_sketch00.FCStd` | Johnny | base | 0.90 | **41.58** | **100.5** |
| 2 | `0/127646_b88ed13f_0015_sketch00.FCStd` | Ryotaro | base | 0.90 | 22.59 | 91.0 |
| 3 | `0/127646_b88ed13f_0015_sketch00.FCStd` | Sagar | base | 0.75 | 22.59 | 32.1 |
| 4 | `8/21337_062b39fa_0001_sketch05.FCStd` | Tanaka | base | 0.89 | 1.87 | 2.63 |
| 5 | `9/22645_1ba0af00_0011_sketch04.FCStd` | Eko | base | 0.68 | 1.52 | 1.51 |

新 gate により base の under-constrained 出力が大量に admit され、論文 figure
の主役が base に移った。leak が 1.5–100+ の extreme value を取るのは base の
under-constrained sketch で多数 source relations が一次線形で壊れるため。
HistCAD OES は preservation set が稀薄なので 0.75–0.90 と高評価する → OES failure
の極端な例。

逆方向 (low OES, low d_feas, top-5) は base / solver_rl 混合: OES = 0 だが
d_feas = 0.04 という rotation。fully constrained model 出力が HistCAD の edit
reachability test を通らないだけで local response は heldout に近い → OES が
fully constrained sketch を penalize する failure の qualitative 補強。

## 9-E. Robustness (eps × validation tolerance)

`v3_controls_scores_full.csv` で eps ∈ {1e-11, 1e-9, 1e-7} を sweep。
median d_feas は全 variant で 1e-3 オーダーまで一致、paired Wilcoxon は
3 通りで一致。新 gate でも安定。

## 9-F. d_feas の有効性: 何が示せたか / 何が残るか

**現時点で防御しやすい claim** (新 gate 後):

1. **`d_feas` は同一 geometry の別 human annotation を吸収する**
   (source_self frac<1e-4=1.00、heldout frac<1e-4=0.73、Q75=1e-3)。
2. **`d_feas` は OES が逆方向に動く dimensional dropout を正しい方向に penalize する**
   (median 0 → 1, gain 主成分)。
3. **`d_feas` は random dropout と異なる failure mode (relational drop) を異なる成分で penalize する**
   (rel_drop50 leak=0.322, gain≈0; dim_drop50 gain=0.146, leak≈0)。
4. **`d_feas` は paired test で Vitruvion を heldout より有意に悪く判定する**
   (solver_rl p=1.5e-19, base p=1.2e-31)。
5. **同じ paired test で HistCAD OES は base に対して sign 反転する** (p=2.2e-19、
   model > human)。これは OES の **実 model レベルでの failure mode** の直接観測。
6. **`d_feas` の gain/leak 分解が model 改良軸 (RL fine-tune) に対して情報量を持つ**:
   gain は base 0.121 → solver_rl 0.042 と改善 (under-constraining 解消)、
   leak は 0.059 → 0.068 と不変 (relational intent mismatch は閉じない)。
7. **token-level F1 と整合**: rel F1 は base 0.600 → solver_rl 0.667 と若干改善、
   heldout 0.759 に届かず。`d_feas` leak の方向と整合。

**まだ言えないこと / やり残し**:

- (a) **user study との一致**は依然取れていない。本文では "captures local
  edit-response consistency" 系の弱め表現に留める。
- (b) **reranking experiment** が未実施。`d_feas` を selection signal として
  使えるかは未確認 (P2 / §6)。
- (c) **Oracle intent / synthetic suite** が未実施 (P2 / §2)。
- (d) **divergent case の sketch render** が未生成 (CSV のみ)。`histcad_case_detail_report.py`
  の流用で要図化。base の `0/127646_b88ed13f_0015_sketch00.FCStd` (leak=100) は
  論文 figure の最有力候補。
- (e) **`rel_drop50/100` の heavy-tail outlier**: median/quartile table + CDF を
  主体にし、violin は supplement に回す方針。