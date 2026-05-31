# ResponseMetric Reward RLOO 研究計画

作成日: 2026-05-29

## 0. この文書の目的

`d_H_angle` / `ResponseMetric` は、すでに
[summary_histcad_response_metric_v4_results.md](summary_histcad_response_metric_v4_results.md)
で **評価指標としての有効性** が示されている。本計画の目的は、それとは別に、
この指標を RLOO の報酬にしたときに、探索の足場を与えた上で追加効果が見えるかを
切り分けることである。

ここで重要なのは、RL の成功を `ResponseMetric` の唯一の証拠にしないこと。
RL は補助実験であり、失敗した場合でも `ResponseMetric` evaluator としての主張は
`v4_results` 側に残る。

## 1. 現状認識

現在の問題は、単に `ResponseIntentReward` の係数を上げ下げすれば解ける種類では
なさそうである。

- `gate=fc` では、序盤に FC サンプルが少ないため `ResponseIntentReward` が効く前に
  SolverReward が dimensional constraints 多めの方策を形作る。
- `gate=converged` にして強めると、設定によっては `[START], [STOP]` の短系列に逃げる。
- 現在の `motion_relational_angle` 単体報酬では FC rate の上昇を鈍らせる効果はあるが、
  qualitative にはまだ Distance 系 dimensional constraints が過剰。
- 特に `Distance=0` やそれに近い dimensional constraints が、Coincident 的な
  relational intent の代用品として出ている。
- Coincident 以外の基本 relational constraints、特に `Equal`, `Tangent`, `Mirror`
  (Symmetric 相当), `Parallel`, `Perpendicular` がほとんど出ない。

したがって、今後の実験では `ResponseMetric` 報酬そのものと、良い候補を探索空間に
出すための補助施策を分離する。

## 2. 中心仮説

今見るべき問いは次の形にする。

> relational constraints がそもそも出るような探索足場を置いたとき、
> `ResponseMetric` 由来の報酬は、同じ足場だけの場合よりも human-response-compatible な
> constraint set を選ぶか？

ここで「探索足場」とは、以下のような `ResponseMetric` ではない補助施策の総称である。

- `Distance≈0` や `Angle≈90/180°` を罰する hygiene reward
- rare relational token を出す class prior reward
- rel-only warm start
- sampling temperature / entropy / KL / SFT の調整

この足場そのものが design intent を証明するわけではない。足場は、`Equal` や
`Tangent` などの候補を方策分布に乗せるためのものと割り切る。

## 3. 実験の基本構造

比較は必ず同じ足場の上で行う。

| 条件 | 意味 | 目的 |
|---|---|---|
| `S` | scaffold only | 足場だけでどこまで良くなるか |
| `S + R` | scaffold + ResponseIntentReward | `ResponseMetric` 報酬の追加効果 |
| `S + R_shuffled` | scaffold + 対応関係を壊した ResponseIntentReward | 報酬スケールだけで効いていないかの負例 |
| `S + class_proxy` | scaffold + type/count 系 proxy | 単純な relational prior との差 |

最初から全条件を走らせる必要はない。第一段階では `S` と `S + R` を優先する。
差が見えた場合に `R_shuffled` と `class_proxy` を追加して主張を強化する。

`R_shuffled` の意味は「別 sketch の GT をそのまま使う」ことではない。ResponseMetric は
基本的に同じ primitive geometry 上の Pred/GT 比較を仮定するため、primitive 構造が違う GT を
渡すと metric 自体が壊れやすい。ここでの負例は、報酬値の分布とスケールは保ったまま、
「このサンプルにこの response reward が対応している」という対応だけを壊すことである。
具体的には、通常の ResponseIntentReward を各 sample で計算した後、batch または RLOO group 内で
sample reward scalar を permutation してから STOP 位置に置く方式を優先する。

## 4. 実装計画

実装は「学習に効く reward」と「validation に出す metric」を混同しないように分ける。
`RewardComposer` に入れた reward は train 側の score を作る。一方、
`validation_gen/*` に出したい値は `RewardValComposer.metrics` にも同じ module を入れる必要がある。
この点を config 作成時の必須条件にする。

### 4.1 まず見るための validation ログを増やす

目的は、定性観察を WandB 上で追えるようにすること。現状の solver / response metric
だけでは、何の constraint type が増減したかが見えにくい。
ただし type ごとの scalar をすべて `validation_gen/*` に並列で出すと dashboard が
読みにくくなる。したがって、scalar は集約値だけに絞り、type-wise distribution は
W&B の Table + bar chart として出す。

W&B は `wandb.Table` と `wandb.plot.bar()` で bar chart を script から直接 log できる。
また、Vega custom chart を `wandb.plot_table()` で作れば pie / donut 風の chart も可能だが、
type 数が多い場合は pie chart より bar chart の方が読みやすい。したがって本計画では
bar chart を標準にする。

参考:

- W&B custom charts: https://docs.wandb.ai/guides/app/features/custom-charts/
- W&B plots: https://docs.wandb.ai/guides/track/log/plots

編集候補:

- [src/metrics/constraint_stats.py](../src/metrics/constraint_stats.py)
- [src/models/rewards/composer.py](../src/models/rewards/composer.py)
- [src/models/constraint_ar_module_rloo.py](../src/models/constraint_ar_module_rloo.py)

scalar として残すログ:

- `validation_gen/eval_rel_ratio_mean` 既存
- `validation_gen/eval_key_rel_ratio_mean`
  - `(Coincident + Equal + Tangent + Parallel + Perpendicular + Mirror) / n_total`
- `validation_gen/eval_distance_family_ratio_mean`
  - `(Distance + DistanceX + DistanceY) / n_total`
- `validation_gen/eval_angle_family_ratio_mean`
  - `Angle / n_total`

bar chart / table として出すもの:

- `validation/constraint_type_mix_table`
- `validation/constraint_type_mix_bar`

table columns:

| column | 内容 |
|---|---|
| `step` | global step |
| `constraint_type` | `Coincident`, `Distance`, ... |
| `family` | `relational` or `dimensional` |
| `mean_count` | 1 sketch あたり平均 count |
| `fraction` | total constraints 中の割合 |

実装の具体案:

- `constraint_stats.py` に `constraint_head_counts(...)` を追加する。
- 入力は `cons_tokens`, `block_size`, `token_ids`, `prefix_len=1`。
- STOP 以降と PAD head は数えない。
- 戻り値は `(B, n_types)` tensor と token 名リスト、または `{name: (B,) tensor}` のどちらでもよい。
  `RewardValComposer` 側の呼び出しが簡単になるなら dict 形式を優先する。
- `RewardValComposer.compute()` では compact scalar と、chart 用の raw summary dict を返す。
- 現在の `validation_step()` は scalar を `self.log()` するだけなので、W&B Table / bar chart は
  [src/models/constraint_ar_module_rloo.py](../src/models/constraint_ar_module_rloo.py) 側で
  rank 0 のときだけ直接 `self.logger.experiment.log(...)` する。
- W&B 以外の logger では table/bar chart は skip し、scalar だけ出ればよい。

この段階で、既存 run の比較対象として次を見られるようになる。

- dimensional constraints が本当に増えているか
- `Distance`, `DistanceX`, `DistanceY` のどれが支配的か
- `Equal`, `Tangent`, `Mirror` がゼロ付近のままか
- `ResponseIntentReward` ありで relation count は増えているのか

### 4.2 SolverReward 内の dimensional shortcut hygiene penalty

目的は、`Distance=0` 付近を Coincident 代替として使う failure mode と、
`Angle=90/180°` を Perpendicular / Parallel 代替として使う failure mode を塞ぐこと。
これは design intent を直接注入する施策ではなく、明らかな dimensional hack の除去として扱う。

結論として、ここは新規 `DimensionalHygieneReward` class ではなく
[src/models/rewards/solver_reward.py](../src/models/rewards/solver_reward.py) 内の optional penalty
として実装する方を推奨する。

理由:

- `SolverReward` worker はすでに generated tokens を detokenize し、
  `complete_constraint_values` 済みの `sketch` を返している。
- 新規 reward class にすると同じ detokenize / value completion をもう一度行うため遅い。
- near-zero dimensional は solver-derived hygiene signal なので、SolverReward の optional
  penalty として扱う方が責務として自然。
- default を `0.0` にすれば既存 SolverReward の挙動は変わらない。

追加する SolverReward config:

```yaml
penalty_near_zero_distance: -0.30
near_zero_distance_threshold: 0.001
near_zero_distance_cap: 3
near_zero_distance_gate: converged
penalty_axis_angle: -0.30
axis_angle_tolerance_rad: 0.001
axis_angle_cap: 3
axis_angle_gate: converged
```

対象:

- `Distance`
- `DistanceX`
- `DistanceY`
- `Angle`

判定:

- `SolverReward.collect_results()` で worker から返ってくる `sketch` を使う。
- 対象 dimensional constraint の `QuantityParameter.value` を読む。
- `Distance` family は `abs(value) < near_zero_distance_threshold` なら 1 count。
- `DistanceX=0` / `DistanceY=0` も Coincident / alignment の dimensional 代替として扱い、
  意図的 CAD constraint としては許容しない方針にする。
- `Angle` は `90°` または `180°` に近いものを 1 count とする。
  これらは `Perpendicular` / `Parallel` の dimensional 代替として扱う。

reward:

```text
distance_penalty =
  penalty_near_zero_distance * min(near_zero_distance_count, near_zero_distance_cap)

angle_penalty =
  penalty_axis_angle * min(axis_angle_count, axis_angle_cap)
```

ログ:

- `solver_near_zero_distance_rate`
- `solver_axis_angle_rate`

rate の定義:

```text
near_zero_distance_rate
  = near_zero_distance_count / max(1, distance_family_count)

axis_angle_rate
  = axis_angle_count / max(1, angle_count)
```

`near_zero_count_mean` は出さない。count は sketch size と生成 constraint 数に引っ張られるため、
run 比較では rate の方が読みやすい。

`Distance=0` / `Angle=90/180°` を Malformed 扱いにする案もあるが、初期実装では
soft penalty を推奨する。理由は、これらはトークン列としては構文的に有効で、FreeCAD solver
上も解ける場合があるため、Malformed に混ぜると「構文失敗」と「design-intent 的 shortcut」が
同じ error bucket に入って診断しづらくなるからである。まずは soft penalty + rate logging で
挙動を見る。明らかに hack として残り続ける場合だけ、後段で hard invalid / Malformed 相当の
扱いに切り替える。

不明瞭だった点の整理:

- `Distance=0` を完全禁止するのではなく、near-zero を罰する。
- `DistanceX/Y=0` も罰する。
- 係数が強すぎて FC が崩れた場合は `penalty_near_zero_distance` の絶対値を下げるか
  `near_zero_distance_gate=fc` にする。
- `Angle=90/180°` も同様に罰する。ただし最初は hard invalid ではなく soft penalty にする。

### 4.3 RelationalClassPriorReward

目的は、rare relational token を探索空間に出すこと。これは `ResponseMetric` 由来の
報酬ではないため、必ず `ResponseIntentReward` あり/なし両方の条件に入れる。

新規ファイル候補:

- `src/models/rewards/relational_class_prior.py`

reward 名:

```python
name = "rel_prior"
```

初期仕様:

```yaml
_target_: src.models.rewards.relational_class_prior.RelationalClassPriorReward
context: ${data.context}
gate: converged
cap_total: 0.50
cap_per_type: 2
coeff_by_type:
  Coincident: 0.03
  Parallel: 0.05
  Perpendicular: 0.05
  Equal: 0.10
  Tangent: 0.10
  Mirror: 0.15
```

実装:

- detokenize は不要。
- token head を数えるだけでよい。
- `constraint_head_counts(...)` を再利用する。
- `gate=converged` の場合、SolverReward status が `FC` または `Under` のサンプルだけに効かせる。
- train 側だけでなく validation 側にも入れ、`validation_gen/rel_prior_*` を見る。

reward:

```text
bonus_raw = Σ_t coeff_by_type[t] * min(count_t, cap_per_type)
bonus = min(bonus_raw, cap_total)
```

ログ:

- `rel_prior_bonus_mean`
- `rel_prior_key_rel_bonus_mean`

type-wise counts は scalar として並べない。4.1 の `constraint_type_mix_bar` に集約する。

semantic spam について:

- training scaffold としては、多少の semantic spam は許容する。
- ただし最終主張では、この prior 自体を design intent の証拠にしない。
- 必ず `scaffold only` と `scaffold + ResponseIntentReward` の両方に同じ prior を入れる。
- `scaffold + ResponseIntentReward` だけに class prior を入れる比較はしない。

### 4.4 sampling temperature / entropy / SFT knob

目的は、RLOO 後や rel-only warm start 後に entropy が落ちすぎているとき、候補多様性を
少し戻すこと。

編集候補:

- [src/models/constraint_ar_module_rloo.py](../src/models/constraint_ar_module_rloo.py)

追加する引数:

```python
train_sample_temperature: float = 1.0
val_sample_temperature: float = 1.0
```

変更:

- `_compute_rloo_loss()` 内の `generate_sample(...)` に
  `temperature=self.train_sample_temperature` を渡す。
- `validation_step()` 内の `generate_sample(...)` に
  `temperature=self.val_sample_temperature` を渡す。

初期値:

- 通常 run: `train_sample_temperature=1.0`, `val_sample_temperature=1.0`
- rel-only warm start 後の full RLOO: `train_sample_temperature=1.2`

関連 hyperparameter:

- `entropy_bonus=0.001`
- `sft_coeff=0.01`
- `kl_coeff=0.005`

ログ方針:

- ここで新しい scalar を大量に増やさない。
- 基本は既存の `train/entropy`, `train/kl_loss`, `train/ratio` と、
  4.1 の `constraint_type_mix_bar` で挙動を見る。
- sampling diversity を追加で見たい場合だけ、`validation/sample_length_hist` や
  `validation/unique_sequence_rate` のような compact scalar / histogram を後から足す。

注意:

- temperature だけで解決しようとしない。
- rel-only warm start 後は dimensional token の確率が死んでいる可能性があるため、
  SFT や KL の参照先も併せて見る。

## 5. Config 計画

第一段階では 4 本に絞る。

全 config 共通で、`model.reward_composer.rewards` と `model.reward_val_composer.metrics` の
両方を確認する。dimensional shortcut hygiene は SolverReward 内の optional config として有効化する。
`RelationalClassPriorReward` は train reward として使わない対照条件でも、validation metric
としては入れてよい。

### 5.1 Hygiene-only scaffold

`configs/experiment/rloo_response_hygiene_no_intent.yaml`

train rewards:

- SolverReward
  - `penalty_near_zero_distance: -0.30`
  - `near_zero_distance_threshold: 0.001`
  - `penalty_axis_angle: -0.30`
  - `axis_angle_tolerance_rad: 0.001`
- DimParsimonyReward
- ResponseIntentReward disabled

validation metrics:

- SolverReward
  - `penalty_near_zero_distance: 0.0`
  - `penalty_axis_angle: 0.0`
  - `near_zero_distance_threshold` は train と同じ
  - `axis_angle_tolerance_rad` は train と同じ
- ResponseIntentReward (`gate=always`, `h_angle_reward_metric=response_full_angle`)
- RelationalClassPriorReward は metric-only として入れてもよい

目的:

- `Distance≈0` / `Angle≈90/180°` を減らしただけで、dimensional 過剰や response metric が
  どこまで改善するかを見る。

### 5.2 Hygiene + ResponseIntent

`configs/experiment/rloo_response_hygiene_intent.yaml`

train rewards:

- SolverReward
  - `penalty_near_zero_distance: -0.30`
  - `near_zero_distance_threshold: 0.001`
  - `penalty_axis_angle: -0.30`
  - `axis_angle_tolerance_rad: 0.001`
- DimParsimonyReward
- ResponseIntentReward enabled
- `h_angle_reward_metric: motion_relational_angle`
- `gate: converged`

validation metrics:

- Hygiene-only と同じ

目的:

- hygiene を入れた土俵で `ResponseIntentReward` の追加効果を見る。

### 5.3 Full scaffold without ResponseIntent

`configs/experiment/rloo_response_scaffold_no_intent.yaml`

train rewards:

- SolverReward
  - `penalty_near_zero_distance: -0.30`
  - `near_zero_distance_threshold: 0.001`
  - `penalty_axis_angle: -0.30`
  - `axis_angle_tolerance_rad: 0.001`
- DimParsimonyReward
- RelationalClassPriorReward
- ResponseIntentReward disabled

validation metrics:

- SolverReward
- ResponseIntentReward (`gate=always`, `h_angle_reward_metric=response_full_angle`)
- RelationalClassPriorReward

目的:

- rare relational token prior だけで qualitative がどこまで改善するかを見る。

### 5.4 Full scaffold with ResponseIntent

`configs/experiment/rloo_response_scaffold_intent.yaml`

train rewards:

- SolverReward
  - `penalty_near_zero_distance: -0.30`
  - `near_zero_distance_threshold: 0.001`
  - `penalty_axis_angle: -0.30`
  - `axis_angle_tolerance_rad: 0.001`
- DimParsimonyReward
- RelationalClassPriorReward
- ResponseIntentReward enabled

validation metrics:

- Full scaffold without ResponseIntent と同じ

目的:

- 同じ scaffold の上で、`ResponseMetric` reward が追加で効くかを見る。

## 6. 実験順序

### Phase 0: smoke test

CPU 2 step:

```bash
conda run -n freecad python src/train.py experiment=rloo_response_scaffold_intent \
  trainer.accelerator=cpu trainer.devices=1 trainer.strategy=auto trainer.precision=32 \
  data.batch_size=1 model.group_sample_size=2 \
  +trainer.max_steps=2 +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0
```

確認:

- 新規 reward が instantiate できる
- reward tensor shape が合う
- WandB / stdout に新規 metric が出る
- solver worker と reward gate が壊れていない

### Phase 1: short canary

各 config 300-500 step 程度。

見る指標:

- `train/reward_contrib_*_seq_mean`
- `train/entropy`
- `validation_gen/solver_fc_rate`
- `validation_gen/solver_under_rate`
- `validation_gen/eval_rel_ratio_mean`
- `validation_gen/eval_key_rel_ratio_mean`
- `validation_gen/eval_distance_family_ratio_mean`
- `validation_gen/eval_angle_family_ratio_mean`
- `validation_gen/solver_near_zero_distance_rate`
- `validation_gen/solver_axis_angle_rate`
- `validation/constraint_type_mix_bar`
- `validation/samples_0`

中止条件:

- `[START], [STOP]` collapse
- `solver_fc_rate` が 0 近くに張り付く
- `solver_redundant_rate` / `solver_conflicting_rate` が大きく悪化
- class prior の reward contribution が SolverReward を食いすぎる

推奨実行例:

```bash
conda run -n freecad python src/train.py experiment=rloo_response_scaffold_intent \
  trainer.devices=1 trainer.max_epochs=1 +trainer.max_steps=500 \
  trainer.val_check_interval=100
```

### Phase 2: full run

第一優先:

1. `rloo_response_hygiene_no_intent`
2. `rloo_response_hygiene_intent`
3. `rloo_response_scaffold_no_intent`
4. `rloo_response_scaffold_intent`

第二優先:

5. `rloo_response_scaffold_shuffled_intent`
6. `rloo_response_scaffold_class_proxy`

## 7. 評価指標

### 7.1 Solver validity

- `validation_gen/solver_fc_rate`
- `validation_gen/solver_under_rate`
- `validation_gen/solver_redundant_rate`
- `validation_gen/solver_conflicting_rate`
- `validation_gen/solver_malformed_rate`
- `validation_gen/solver_valid_dof_mean`

### 7.2 ResponseMetric family

- `validation_gen/intent_response_full_angle_mean`
- `validation_gen/intent_motion_full_angle_mean`
- `validation_gen/intent_motion_relational_angle_mean`
- `validation_gen/intent_leak_relational_angle_mean`
- `validation_gen/intent_response_relational_angle_mean`

解釈:

- `response_full_angle` は canonical `d_H_angle` 系の proxy。
- `motion_relational_angle` は reward target として使う場合がある。
- `leak_relational_angle` は solver_rl 型の relational mismatch を拾いやすいので重要。

### 7.3 Constraint mix

- `validation_gen/eval_rel_ratio_mean`
- `validation_gen/eval_key_rel_ratio_mean`
- `validation_gen/eval_distance_family_ratio_mean`
- `validation_gen/eval_angle_family_ratio_mean`
- `validation/constraint_type_mix_bar`

### 7.4 Hygiene

- `validation_gen/solver_near_zero_distance_rate`
- `validation_gen/solver_axis_angle_rate`

### 7.5 Qualitative

- `validation/samples_0` の Ground Truth / Generated
- `src/test.py` による FreeCAD script dump
- 重点観察:
  - Distance の連打が減っているか
  - `DistanceX/Y=0` が Coincident 相当として残っていないか
  - Equal/Tangent/Mirror が自然な場所に出ているか
  - relational class prior だけの spam に見えないか

## 8. 成功・失敗の解釈

### 8.1 強い成功

`scaffold_intent` が `scaffold_no_intent` と比べて:

- FC rate が同程度
- `response_full_angle` または `leak_relational_angle` が改善
- dimensional count が増えない
- near-zero distance が増えない
- qualitative に relational structure が自然

解釈:

- 探索足場の上で、`ResponseMetric` reward が候補選択に追加効果を持つ。
- 論文では「RL は補助的証拠」として使える。

次のアクション:

- `R_shuffled` を追加して、GT 対応の意味が効いているか確認する。
- `response_full_angle` reward や leak-safe `response_relational_angle` を検討する。

### 8.2 弱い成功

- FC は少し落ちる
- しかし response metric と qualitative は改善

解釈:

- `ResponseMetric` は design intent 側には効いているが、SolverReward との trade-off が残る。

次のアクション:

- SolverReward schedule を検討する。
- `reward_dof_reduction` / `reward_fully_constrained` の序盤係数を弱める。
- ただし config 数を増やしすぎない。

### 8.3 Scaffold-only で十分改善

`scaffold_no_intent` と `scaffold_intent` がほぼ同じ。

解釈:

- 改善は class prior / hygiene / sampling によるもので、`ResponseMetric` reward の追加効果は弱い。
- evaluator としての `ResponseMetric` の有効性は残るが、RL reward としての主張は弱める。

次のアクション:

- RL 補助実験は negative result として早めに整理する。
- `v4_results` を主軸に戻す。

### 8.4 Intent ありで collapse

`scaffold_intent` だけ `[START], [STOP]` や under-constrained に逃げる。

解釈:

- 報酬 gate / cap / selected metric がまだ hackable。

次のアクション:

- `h_angle_reward_metric=motion_relational_angle` を維持。
- `coeff_h_angle` を下げる。
- `gate=fc` と `gate=converged` の中間設計を検討する。
- empty / very-short sequence penalty を別途入れる。

### 8.5 Class prior で invalid が増える

解釈:

- rare relation を出す prior が強すぎる。

次のアクション:

- `cap_total` を下げる。
- `Equal/Tangent/Mirror` coeff を半分にする。
- `gate=fc` にする。

### 8.6 Class prior でも rare relations が出ない

解釈:

- 方策の token distribution がすでに硬すぎる。

次のアクション:

- `train_sample_temperature=1.2`
- `entropy_bonus=0.001` から試す。
- それでも無理なら rel-only warm start へ進む。

## 9. 第二段階: Rel-only warm start → high-entropy full RLOO

この section は、Sections 4–8 の第一段階実験を走らせた後の第二段階の追加施策である。
第一段階では、full constraint space のまま hygiene, class prior, ResponseIntentReward の
ablation を行う。その結果、`Equal`, `Tangent`, `Mirror`, `Parallel`, `Perpendicular`
などの rare relational constraints が依然としてほとんど出ない場合、方策分布そのものが
硬すぎると判断し、この rel-only warm start curriculum に進む。

逆に、第一段階の full-space scaffold だけで rare relational constraints が十分出るなら、
section 9 は実行しない。まず `scaffold only` と `scaffold + ResponseIntentReward` の比較を
優先し、`ResponseMetric` reward の追加効果を切り分ける。

この curriculum では、まず dimensional constraint token を禁止した relational-only RLOO を行い、
その checkpoint を初期重みとして、本来の full constraint space で RLOO を再開する。
Phase B では、rel-only training で落ちた探索 entropy を戻すために、通常より高い
`entropy_bonus` と sampling temperature を使う。

### Phase A: rel-only RLOO

既存:

- [configs/experiment/rloo_rel_only.yaml](../configs/experiment/rloo_rel_only.yaml)

目的:

- relational token の局所方策を作る。
- `Coincident` 以外の `Equal`, `Tangent`, `Mirror`, `Parallel`, `Perpendicular` が
  方策分布に乗る状態を作る。
- ここでは dimensional constraints を出せないため、最終性能を判断しない。

checkpoint 選択:

- FC rate だけで選ばない。
- `eval_rel_ratio_mean`
- `eval_key_rel_ratio_mean`
- `constraint_type_mix_bar`
- qualitative samples
- collapse していないこと
- entropy が完全に潰れていないこと。ただし rel-only で多少 entropy が下がるのは許容する。

### Phase B: high-entropy full RLOO

設定:

```yaml
pretrained_ckpt_path: <rel_only_ckpt>
model.relational_only: false
data.relational_only: false
model.train_sample_temperature: 1.2
model.entropy_bonus: 0.003  # canary: 0.001, 0.003, 0.005
model.sft_coeff: 0.01
model.kl_coeff: 0.005
```

意図:

- rel-only checkpoint から始めると、dimensional token の logits が弱くなりやすい。
- そのまま full RLOO をすると、relation-only 方策の低 entropy 状態から抜けられず、
  dimensional constraints が必要な場面で候補が出ない可能性がある。
- そこで `train_sample_temperature` と `entropy_bonus` を高めにし、必要なら `sft_coeff` で
  full constraint distribution を少し思い出させる。
- `entropy_bonus` は通常 run より明確に高めを試す。最初の grid は `0.001`, `0.003`,
  `0.005` 程度で十分。

比較:

- relwarm + scaffold only
- relwarm + scaffold + ResponseIntentReward

注意:

- rel-only 後は dimensional token の確率がかなり低い可能性がある。
- temperature だけではなく、SFT と KL も併用する。
- ここでも `ResponseMetric` の有無以外を揃える。
- `relwarm + scaffold only` でも十分改善する場合、改善は curriculum / prior 由来であり、
  `ResponseMetric` reward の追加効果とは言いにくい。

## 10. 実装順序

1. `constraint_head_counts` と validation type-count logging
2. SolverReward 内の dimensional shortcut penalty
3. `RelationalClassPriorReward`
4. `train_sample_temperature` / `val_sample_temperature`
5. 4 config 作成
6. CPU smoke test
7. 300-500 step canary
8. full run
9. 必要なら `R_shuffled`
10. rare relations が出ない場合は rel-only warm start → high-entropy full RLOO

各実装後に最低限確認すること:

- type-count logging 追加後:
  - `eval_rel_ratio_mean`, `eval_key_rel_ratio_mean`, `eval_distance_family_ratio_mean`,
    `eval_angle_family_ratio_mean` が出る。
  - `constraint_type_mix_bar` が出る。
- SolverReward dimensional shortcut penalty 追加後:
  - `validation_gen/solver_near_zero_distance_rate` が出る。
  - `validation_gen/solver_axis_angle_rate` が出る。
  - `penalty_near_zero_distance=0.0` で既存 SolverReward と reward が一致する。
  - `penalty_near_zero_distance<0` または `penalty_axis_angle<0` で
    `reward_contrib_solver_seq_mean` に反映される。
- `RelationalClassPriorReward` 追加後:
  - `reward_contrib_rel_prior_seq_mean` が train に出る。
  - validation に `rel_prior_bonus_mean` が出る。
  - type-wise distribution は `constraint_type_mix_bar` に出る。
- temperature knob 追加後:
  - default `1.0` で既存挙動が変わらない。

## 11. この計画でまだ不明瞭な箇所と推敲結果

### 11.1 dimensional shortcut thresholds

不明瞭:

- `Distance≈0` と `Angle≈90/180°` の正しい tolerance がデータスケールと数値誤差に依存する。

具体化:

- `near_zero_distance_threshold` の初期値は `0.001`。
- `axis_angle_tolerance_rad` の初期値も `0.001`。
- canary で `solver_near_zero_distance_rate` / `solver_axis_angle_rate` がほぼ変わらなければ、
  まず penalty の絶対値を上げる。threshold は安易に広げない。
- FC が崩れたら threshold ではなく penalty の絶対値を下げるか、gate を `fc` にする。
- 最初は Malformed 扱いにはしない。soft penalty でも残り続ける場合だけ hard invalid 化を検討する。

### 11.2 class prior の係数

不明瞭:

- `Tangent +1` のような大きい bonus は SolverReward とのスケール差が危険。

具体化:

- 初期値は 0.03-0.15 の小さい補助にする。
- `cap_total=0.5` で sequence-level に上限を置く。
- `reward_contrib_rel_prior_seq_mean` が SolverReward contribution に近づいたら強すぎる。

### 11.3 ResponseIntentReward の選択 metric

不明瞭:

- `response_relational_angle` は leak=0 問題で empty sequence に甘くなる可能性がある。

具体化:

- 第一段階は `motion_relational_angle` 固定。
- `leak_relational_angle` は diagnostic として見る。
- `response_relational_angle` に戻すのは、empty / no-dim hack を塞いだ後。

### 11.4 qualitative の扱い

不明瞭:

- qualitative は主観的になりやすい。

具体化:

- qualitative は以下の checklist で見る。
  - Distance 系が減っているか
  - zero Distance が減っているか
  - Coincident / Equal / Tangent / Mirror が増えているか
  - redundant/conflicting ではないか
  - source geometry に対して不自然な relation spam ではないか

### 11.5 「scaffold」の言葉

不明瞭:

- scaffold が何を含むのか曖昧だった。

具体化:

- 本文書では「探索足場」と呼び、具体的には hygiene, class prior, sampling knobs,
  rel-only warm start を指す。
- `ResponseMetric` reward は scaffold には含めない。

### 11.6 train reward と validation metric の混同

不明瞭:

- train に reward を入れれば validation にも自動で出るように見えがちだが、このコードでは
  `RewardComposer` と `RewardValComposer` は別である。

具体化:

- 学習に使う module は `model.reward_composer.rewards` に入れる。
- validation でログだけ見たい module は `model.reward_val_composer.metrics` に入れる。
- 対照条件で ResponseIntentReward を disabled にしても、validation metric としての
  ResponseIntentReward は残す。そうしないと `S` と `S + R` の比較指標が欠ける。
- SolverReward 内 near-zero metric は、train では penalty あり、validation では penalty 0 でも
  metric だけ出せるようにする。

### 11.7 `R_shuffled` の実装時期

不明瞭:

- `GT をシャッフル` という表現だと、別 sketch の GT constraints を ResponseMetric に
  渡すように読める。しかし ResponseMetric は基本的に同じ primitive geometry 上の比較を
  仮定するため、それは壊れやすい。

具体化:

- 第一段階では実装しない。
- `S + R` が `S` より良い結果を示した場合にだけ、主張補強として実装する。
- 実装する場合は GT sketch 自体を入れ替えるのではなく、通常の ResponseIntentReward を
  各 sample について計算した後、STOP に置く reward scalar だけを batch / RLOO group 内で
  permutation する。
- これにより、報酬の分布・スケール・計算コストは保ったまま、sample と reward の意味的対応を壊せる。
- `S + R_shuffled` で改善するなら、ResponseMetric の意味ではなく、報酬スケールや regularization が
  効いている疑いが強い。

## 12. 最終的な研究上の言い方

成功した場合:

> We first introduce solver- and exploration-oriented safeguards shared by all
> RL variants, preventing degenerate dimensional shortcuts and encouraging
> relational candidates to appear.  Under this controlled scaffold, adding the
> response-metric reward improves response compatibility relative to the same
> scaffold without the metric reward.

失敗した場合:

> The proposed response metric remains effective as an evaluator, but in the
> current RLOO setup it did not provide a robust enough optimization signal once
> the candidate distribution and solver reward dynamics were taken into account.
> This suggests that metric learning / reranking / supervised relational priors
> may be more appropriate than direct sparse RL optimization for this stage.

## 13. 実装ログと次セッションへの引き継ぎ (2026-05-29)

第一段階 (§4–§5) の infra と config を実装し、学習を開始した。本セクションは
数日後に結果が出たときに作業を再開するための自己完結な引き継ぎである。

### 13.1 実装の意図 (なぜこの形にしたか)

- **中心**: scaffold (hygiene + class prior + sampling temperature + 将来の
  rel-only warm start) と ResponseMetric 報酬を分離し、**同一 scaffold 上で
  `S` vs `S+R` を比較**する (§2–§3)。これが「改善は本当に我々の指標由来か」と
  いう帰属問題を解く唯一の clean な方法。RL はあくまで補助証拠で、evaluator
  としての主張は `v4_results` 側に残す (§0)。
- **hygiene を SolverReward 内 optional penalty にした理由** (§4.2): solver worker
  が既に detokenize + `complete_constraint_values` 済みの sketch を返すため、
  新規 reward class にすると同じ処理を二重に行う。near-zero は solver 由来の
  hygiene signal なので責務上も SolverReward が自然。default `0.0` で既存挙動不変。
- **train 報酬を `motion_relational_angle` にした理由** (§11.3, experiment_notes
  失敗例 sa2167a5): `response_relational_angle` は Pred に dimensional driver が
  無いと leak→0 となり `[START],[STOP]` 空系列を誤って報酬する。`motion_relational`
  は空系列を「relation 不足」方向に罰せる。leak は診断としてのみ残す。
- **class prior が必要な理由** (§4.3, 観察#2): 選択型報酬 (ResponseMetric) は
  方策がサンプルしない候補を浮上させられない。`Equal/Tangent/Mirror` は確率
  ~0 なので、まず探索分布に乗せてから metric 報酬に選ばせる、の2段構え。
- **train/val composer 分離** (§11.6): 本コードでは `RewardComposer` (学習) と
  `RewardValComposer` (検証) が別。報酬を `validation_gen/*` に出すには **両方**の
  list に入れる必要がある。これが最大の落とし穴。

ユーザー決定 (2026-05-29):

- 初回フル run は **5.3 (scaffold_no_intent) vs 5.4 (scaffold_intent)** を先行。
  canary は4本とも回す。
- 判定は**全指標を横並び**で見る (leak を特権化しない)。
- `class_proxy` 対照と `R_shuffled` は **Phase 2 に保留** (S+R が勝った場合のみ)。

### 13.2 実装済みの内容と検証状況

実装 (branch `histcad_refactor`, 2026-05-29 時点で未コミット):

| 項目 | 主な変更ファイル | 備考 |
|---|---|---|
| `constraint_head_counts` + token-id 群 | `src/metrics/constraint_stats.py`, `src/sketch_handlers/constants.py` | per-type head カウント。`KEY_RELATIONAL_*` 等 |
| `RelationalClassPriorReward` | `src/models/rewards/relational_class_prior.py` | name=`rel_prior`, cap_per_type/cap_total/gate |
| SolverReward hygiene penalty | `src/models/rewards/solver_reward.py` | near-zero Distance / axis Angle, default 0.0, rate 常時 log |
| temperature knob | `src/models/constraint_ar_module_rloo.py` | `train_/val_sample_temperature`, default 1.0 |
| val 指標 + chart | `src/models/rewards/composer.py`, `constraint_ar_module_rloo.py` | `eval_*_ratio_mean` + `validation/constraint_type_mix_{table,bar}` |
| 4 config | `configs/experiment/rloo_response_{hygiene,scaffold}_{no_intent,intent}.yaml` | train/val composer 両方を定義 |

検証: 3ヘルパーの単体テスト (カウント/cap/gate/Angle周期性で 0/90/180/270 検出)、
`scaffold_intent` の CPU smoke (train 2step + val、exit 0、reward失敗0/chart失敗0)、
全 val 新指標の実値ダンプ。残り3 config は compose 検証のみ。

**実装上の重要な性質 (再開時に忘れないこと)**:

- hygiene は **geometry 補完値**を読む。`Distance≈0` = 参照プリミティブが幾何的に
  一致 (= Coincident 代替) を正確に検出。Angle は **radians** を mod π で畳んで
  0/90/180 を判定 (270 も含む)。
- 閾値 `near_zero_distance_threshold=0.001` / `axis_angle_tolerance_rad=0.001` は
  **初期値**。効きが弱ければ §11.1 通り **penalty 絶対値を上げる (閾値は広げない)**。
- 新 val 指標は加算的なので既存 `rloo_response.yaml` にも自動で出る (後方互換)。

### 13.3 学習中に見る指標 (結果読み取りのチェックリスト)

最重要の比較は **5.3 (scaffold_no_intent) vs 5.4 (scaffold_intent)**。同一 scaffold
なので差分が ResponseMetric 報酬の純効果。5.1/5.2 は hygiene 単体の寄与を見る診断。

- 妥当性: `validation_gen/solver_fc_rate`, `solver_under_rate`,
  `solver_redundant_rate`, `solver_conflicting_rate`, `solver_valid_dof_mean`
- ResponseMetric family (全部横並び):
  `intent_response_full_angle_mean`, `intent_motion_full_angle_mean`,
  `intent_motion_relational_angle_mean`, `intent_leak_relational_angle_mean`
- constraint mix: `eval_rel_ratio_mean`, `eval_key_rel_ratio_mean`,
  `eval_distance_family_ratio_mean`, `eval_angle_family_ratio_mean`,
  `validation/constraint_type_mix_bar` (Equal/Tangent/Mirror がゼロ付近か)
- hygiene: `solver_near_zero_distance_rate`, `solver_axis_angle_rate`
- 探索/崩壊: `train/entropy`, `validation/samples_0` (空系列 collapse の有無)
- 寄与: `train/reward_contrib_{solver,dim_parsimony,rel_prior,intent}_seq_mean`
  (rel_prior が solver を食い過ぎていないか)

### 13.4 結果が出た後の次手 (分岐別の実装方策)

§8 の解釈を「次に何を実装するか」に落とした版。観測 → 実装アクションの形。

**(1) 強い成功** — 5.4 が 5.3 比で: FC 同等 / `response_full_angle` または
`leak_relational_angle` 改善 / dim count 増えず / near-zero 増えず。
→ 主張補強の対照を実装 (Phase 2):

  - **`R_shuffled`** (§11.7): GT を入替えるのではなく、通常の ResponseIntentReward
    を計算後、**RLOO group 内で STOP 位置の reward scalar を permutation** する。
    実装場所は `response_metric_reward.py`。`shuffle_within_group: bool` フラグを
    足し、`_score_parallel_from_arrays` / `_score_serial_from_arrays` が
    `placements` を作った後、同一 `id(sketch)` group 内で reward 値だけ並べ替える。
    `S+R_shuffled` で改善するなら指標の意味でなく報酬スケールが効いている疑い。
  - **`class_proxy`** (§3, 保留中): GT の relational type histogram に一致させる
    count-matching reward (ResponseMetric ではない単純 prior)。新規ファイル
    `src/models/rewards/relational_count_match.py` 想定。`constraint_head_counts`
    で Pred と GT(`ctx.gt_tokens`) の type 別 count を取り、overlap をボーナス化。
    これと S+R を比較し「単純 prior 超えの効果」を示す。

**(2) 弱い成功** — response/qualitative は改善するが FC が落ちる。
→ SolverReward の schedule を実装 (§8.2)。序盤の `reward_dof_reduction` /
  `reward_fully_constrained` を弱める step-based schedule を SolverReward に追加
  (現在は定数)。config 数は増やさない。

**(3) scaffold-only で十分** — 5.3 ≈ 5.4。
→ ResponseMetric 報酬の追加効果は弱い、と結論。ただし結論前に「intent 報酬が
  実際に効いていたか」を確認: `reward_contrib_intent_seq_mean` が非ゼロか、gate を
  通る FC/Under サンプルが十分あったか。本当に差が無ければ RL は negative result
  として早期に整理し (§8.3)、`v4_results` を主軸に戻す。

**(4) intent collapse** — 5.4 だけ `[START],[STOP]` / under に逃げる (§8.4)。
→ `coeff_h_angle` を下げる / **empty・very-short sequence penalty を実装**
  (新規 or SolverReward 内で n_total==0 を罰する) / gate を `fc` と `converged`
  の中間 (例: FC のみだが warmup 後に converged へ切替) 設計。

**(5) rare relations が依然出ない** — `eval_key_rel_ratio_mean` が低いまま /
  `constraint_type_mix_bar` で Equal/Tangent/Mirror がゼロ付近 (class prior でも不可)。
→ §9 の **rel-only warm start → high-entropy full RLOO** に進む。infra
  (`relational_only`, temperature knob) は実装済みなので主に config 作業:

  - Phase A: `rloo_rel_only.yaml` で rel-only RLOO。ckpt 選択は FC でなく
    `eval_key_rel_ratio_mean` / `constraint_type_mix_bar` / collapse していない /
    entropy が潰れていない で選ぶ。
  - Phase B: 新規 config (scaffold_intent ベース) に
    `pretrained_ckpt_path=<A の ckpt>`, `model.train_sample_temperature=1.2`,
    `model.entropy_bonus` を {0.001, 0.003, 0.005} で grid, `model.sft_coeff=0.01`,
    `model.kl_coeff=0.005`。`relwarm + scaffold only` と
    `relwarm + scaffold + intent` を比較 (ResponseMetric 有無以外を揃える)。

**(6) class prior で invalid 増加** (§8.5) → `cap_total` を下げる /
  `Equal/Tangent/Mirror` coeff を半分 / `gate=fc`。config のみ。

**(7) hygiene が効かない** — `solver_near_zero_distance_rate` /
  `solver_axis_angle_rate` が S と変わらない。
→ まず penalty 絶対値を上げる (config)。それでも near-zero が hack として残るなら
  §4.2 後段の **hard invalid / Malformed 相当**への切替を実装 (SolverReward で
  near-zero Distance を Malformed bucket に入れる)。最初は soft のまま様子見。

### 13.5 未実装の TODO (優先度順、結果次第で着手)

1. `R_shuffled` (permutation 対照) — 分岐(1)。`response_metric_reward.py` にフラグ追加。
2. Phase B warm-start config — 分岐(5)。config のみ (infra 済)。
3. `class_proxy` count-match reward — 分岐(1)。新規 reward ファイル。
4. empty/very-short sequence penalty — 分岐(4)。collapse 防御。
5. SolverReward の reward schedule — 分岐(2)。FC trade-off 緩和。
6. near-zero の hard-invalid 化 — 分岐(7)。soft で足りない場合のみ。

コミットはユーザー指示後に行う。学習完了後、本ファイルに run 別の結果と採用した
分岐を追記してから次の実装に進むこと。

## 14. 第一段階の結果と第二段階の計画 (2026-05-30)

この章は、これまでの経緯を知らない読者が単独で読んでも追えるように、前提・用語・
数値・解釈・次段階の計画を省略せずに記述する。

### 14.1 実験設定（前提）

- タスク: 2D CAD sketch の primitive（点・線・円・弧）が与えられたとき、その間に入る
  constraints を生成する。constraints は relational（Coincident, Horizontal, Vertical,
  Parallel, Perpendicular, Tangent, Equal, Midpoint, Mirror=対称, Concentric=同心 …）と
  dimensional（Distance, DistanceX, DistanceY, Radius, Diameter, Length, Angle）に大別。
- 学習法: RLOO（policy が constraint 系列を K=8 サンプルし、報酬の group 相対 advantage で
  方策勾配）。報酬は `RewardComposer` で合成。
- 比較する2条件（同一 scaffold 上の ablation。scaffold = SolverReward + hygiene +
  DimParsimony + RelationalClassPrior という「探索の足場」。詳細は §2–§5）:
  - **S (no_intent)**: scaffold のみ。run `7u6kwzhf`（config `rloo_response_scaffold_no_intent.yaml`）。
  - **S+R (intent)**: scaffold + ResponseIntentReward（`motion_relational_angle` を報酬化）。
    run `oiytlo2x`（config `rloo_response_scaffold_intent.yaml`）。
  - 両者とも開始重みは `logs/train/runs/2026-05-02_20-35-42/checkpoints/epoch_29.ckpt`
    （絶対位置エンコーディング版、データ `sg_filtered_unique_valid.npy` で pretrain）。
    RLOO の学習データ兼「報酬の比較対象 GT」は `sg_pseudo_rel_167k_unique_valid.npy`。
- 主要指標の意味:
  - `solver_fc_rate`: FreeCAD ソルバが fully-constrained と判定した生成の割合（高い=拘束完備）。
  - `solver_valid_dof_mean`: 残自由度の平均（低い=より拘束されている）。
  - `intent_response_full_angle_mean`: canonical な `d_H_angle`（motion-subspace 不一致と
    relation leak の RMS）。低い=人間 response により整合。**これが評価器としての本指標**。
  - `intent_leak_relational_angle_mean`: Pred の許す運動が GT の関係禁止方向へ漏れる量（低い=良）。
  - `eval_distance_family_ratio_mean`: 生成 constraints のうち Distance/DistanceX/DistanceY の
    割合（Radius/Length/Diameter は含まない）。
  - `eval_key_rel_ratio_mean`: Coincident+Equal+Tangent+Parallel+Perpendicular+Mirror+
    Concentric の割合。
- **重要な注意（比較の公平性）**: 2 run は学習途中で wandb step が一致していなかったため、
  途中の「最新値どうし」の比較は交絡する。本章の比較はすべて **同一 step** で行う。
  両 run は最終的に step 1037（最後の validation）で finished。

### 14.2 学習データ／生成物の constraint 分布（実測）

報酬がそもそも何を「正解」として比較しているかを確認するため、3つの分布を実測した
（% は全 constraints に対する割合）。

| type | 報酬GT `sg_pseudo_rel_167k`(後処理後) | pretrain `sg_filtered`(epoch_29が学習) | 生成 S+R intent @step1035 |
|---|---|---|---|
| Coincident | 40.8 | 44.6 | 42.3 |
| Horizontal | 16.5 | 9.0 | 14.7 |
| Vertical | 10.7 | 5.4 | 14.7 |
| **Equal** | **11.9** | **1.8** | **1.6** |
| Parallel | 8.7 | 8.0 | ~0 |
| Perpendicular | 5.0 | 5.4 | 1.1 |
| Tangent | 4.2 | 3.0 | 1.5 |
| Midpoint | 2.3 | 3.4 | 1.7 |
| dimensional 計 | **0** | ~19 | ~43 (DistX14.7/DistY14.1/Radius7.8/Dist6.6) |
| Mirror / Concentric | **0 / 0** | ~0 / 0.05 | 0 / 0 |

読み取り（ここが従来の認識を覆した点）:
- **報酬GT(`sg_pseudo_rel_167k`)は relational が「渋い」どころか Equal 11.9%（全体3位）・
  Parallel 8.7%・Perp 5.0%・Tangent 4.2% と豊富**。欠けているのは **Mirror/Concentric と
  全 dimensional のみ**。したがって「報酬GTに Equal/Tangent が無いから学習が伸びない」という
  仮説は誤り。
- **生成側で Equal/Parallel/Tangent がほぼ消える**のは報酬GTのせいではなく、**pretrain
  (`sg_filtered`) が Equal 1.8% と希少**で、その事前分布を引き継いだ policy が RLOO で
  Coincident へ mode-collapse するため（後述 §14.4）。
- **Mirror/Concentric は報酬GTに 0%** なので、`motion_relational_angle` 報酬は構造的に
  これらを「狙う」ことができない。ユーザーが「Symmetric をほぼ見ない」と感じる根本原因。

### 14.3 S vs S+R の結果（同一 step、最終）

学習ログ：
```
[rloo_response_scaffold_intent]
run = api.run("/kakuda40611-nanyang-technological-university-singapore/vitruvion/runs/oiytlo2x")

[rloo_response_scaffold_no_intent]
run = api.run("/kakuda40611-nanyang-technological-university-singapore/vitruvion/runs/7u6kwzhf")
```

scalar（step 1037、両 run finished）と、生成 constraint mix（`constraint_type_mix_table`、
同一 step 1035）:

| 指標 | S+R intent (oiytlo2x) | S no_intent (7u6kwzhf) | 向き |
|---|---|---|---|
| intent_response_full_angle ↓ | **0.495** | 0.526 | intent 良 |
| intent_leak_relational_angle ↓ | **0.487** | 0.519 | intent 良 |
| intent_motion_relational_angle ↓ | **0.446** | 0.464 | intent 良 |
| solver_fc_rate ↑ | 0.658 | **0.711** | no_intent 良(+5.3pt) |
| solver_valid_dof_mean ↓ | 3.85 | **3.57** | no_intent 良 |
| eval_distance_family_ratio ↓ | 0.284 | 0.287 | ほぼ同 |
| 生成 total dimensional(@1035) ↓ | **0.431** | 0.454 | intent 良(−2.3pt) |
| 生成 Radius(@1035) | 0.078 | 0.100 | intent 少 |
| 生成 Equal(@1035) | **0.016** | 0.000 | intent 復活/対照ゼロ |
| 生成 Perpendicular(@1035) | **0.011** | 0.003 | intent ~3.4× |
| 生成 rare rel 計(Eq+Tan+Par+Perp) | **0.042** | 0.019 | intent ~2.2× |
| eval_key_rel_ratio ↑ | **0.390** | 0.365 | intent 良 |
| near_zero_distance_rate | 0.029 | 0.025 | ほぼ同(共有hygiene) |

中間 step でも傾向は一貫（例 @867: total dim 0.395 vs 0.451、Equal 0.025 vs 0.000;
@517: 0.388 vs 0.393、Equal 0.019 vs 0.000）。response_full_angle の intent 優位は
step 302→1037 で単調に持続。

### 14.4 解釈: 「弱い成功」（§8.2、§8.1 寄り）

- **ResponseMetric 報酬は意図通りに効いている**: canonical な `response_full_angle` と
  （報酬対象でない）`leak_relational_angle` の両方を一貫して改善し、生成側で
  **dimensional（特に Radius）を抑え、Equal を復活させ Perpendicular を増やす**。
  これは報酬が Equal 豊富な pseudo-GT を参照しているため自然な帰結。
- **ただし FC は約5pt 低い**: no_intent は **Radius などの dimensional を足して
  fully-constrained を稼ぐ**（FC 0.711 / DoF 3.57）。S+R はそれを報酬で抑えるので、
  同じ拘束を relational で賄いきれない分 FC が下がる（0.658 / DoF 3.85）。
  → 典型的な §8.2「response 指標と質は改善するが FC が少し落ちる」。
- **絶対品質はまだ不十分**（ユーザーの定性確認と整合）: 生成は依然 dimensional 過多、
  Equal 1.6%（GT 12% に遠い）、Parallel/Tangent.edge ~0、Mirror/Concentric 皆無、
  かつ極小 Distance（例 1e-16 mm）が残存。相対改善は明確だが、単独の説得材料としては弱い。
- **律速の特定（3点）**:
  1. **探索が事実上オフ**: 当時の config は `entropy_bonus=0.0`・`train_sample_temperature=1.0`
     で探索正則化が無く、Coincident への mode-collapse が起きる。報酬GTに Equal/Tangent が
     あっても policy がサンプルしなければ選択型報酬は浮上させられない。
  2. **hygiene が弱く信用割当も拡散**: 極小 Distance ペナルティ `-0.30`（cap3）に対し、
     その極小 Distance が Under→FC を達成すると FC bonus `+5` を稼ぐため相殺できず、しかも
     ペナルティを STOP 位置にまとめ加算していた（違反トークンに局所化されていない）。
  3. **報酬GTに Mirror/Concentric が無い**: 報酬が構造的にこれらを狙えない（§14.2）。

### 14.5 本結果を受けて実装した是正（commit `594cf67`, branch `rloo_response_scaffold`）

- **探索の有効化**: scaffold-v2 config で `entropy_bonus 0→0.005`,
  `train_sample_temperature 1.0→1.2`。
- **hygiene escalation**（`solver_reward.py`）: 極小 Distance / axis-Angle ペナルティを
  `-0.30→-1.5` に強化し、**違反 constraint の4トークン末尾へ局所化**（redundant/malformed と
  同経路）。`abs()` は既存で正しく、DistanceX/Y の負値も検出済み。
- **class prior の頭打ち緩和**: `cap_total 0.5→1.0`, `cap_per_type 2→3`（係数は人間GT由来の
  ままで据え置き）。
- **検証ログの改善**: per-type mix を `val_cmix/<type>` の **時系列スカラー**として記録
  （スナップショット棒グラフは最新 validation しか描画しないため）。polymorphic な subtype
  カウント（`rel_prior_subtype_*`）は `validation_submix/` セクションへ分離。
- 新 config: `configs/experiment/rloo_response_scaffold2_{intent,no_intent}.yaml`
  （= scaffold-v2 / M1。abs-pos・現 pretrain から開始）。

### 14.6 第二段階の計画

判断: §8.6/§14.4 の通り「rare relation の絶対量不足」と「Mirror/Concentric の構造的欠落」が
残課題。前者は探索（scaffold 強化）、後者は報酬GTの充足で対処する。fiverr 人間データは
**ベンチマーク（評価専用）**で、学習・報酬GTには一切使わない（read-only の頻度照合のみ可）。

- **(M1) scaffold-v2 を実行中**（`rloo_response_scaffold2_intent`, GPU1 で起動済み）。
  狙いは「`entropy_bonus=0` を解除＋temperature↑＋hygiene 強化で rare relation の探索が
  回復し、FC を犠牲にせず relational に拘束できるか」。no_intent は別マシンで対照実行。
- **(規則付与器 + fiverr recall テスト)**: モデル不要・純幾何のデータ処理。Coincident/
  Horizontal/Vertical/Parallel/Perpendicular/Tangent/Midpoint/Concentric/Equal は幾何述語で
  決定的に検出可能（Mirror は対称軸探索が必要で困難→保留）。fiverr 人間 sketch の
  constraints を剥がして primitives だけ残し、規則付与器で復元 → 人間GTと照合して
  **type 別 recall/precision** を測り、tolerance 較正と「信頼できる type」を選別する
  （fiverr は評価専用の使い方）。recall/precision が高ければ、これで報酬GTを充足（特に
  Concentric の穴埋め）し、pseudo-GT v2 を生成できる。
- **(RoPE + rel-only RLOO 列挙器)**: 規則で取れない分と一般的な relational richness 用。
  `motion_relational_angle`/`leak` は部分空間/ランク量なので、**冗長な関係そのものは
  指標を強めない**——強めるのは**独立な関係が増えること**。現生成器は redundant ペナルティで
  関係を sparse に省くので、`penalty_redundant=0`（Conflicting/Malformed フィルタは維持）+
  強 class prior + 高 entropy にして独立関係を最大化した rel-only 列挙器を作り、richer な
  pseudo-GT v2 を生成する。列挙は系列が長くなるため、`max_token_length` 非依存の **RoPE**
  が有効。RoPE はベース差し替え（abs 版 ckpt と非互換）なので現スレッドの比較を壊さないよう、
  この段で初めて導入する。ブランチは `rloo_response_scaffold` から派生し RoPE の単一 commit
  `1a607af` を cherry-pick（reward/scaffold 基盤は tokenizer 共通でそのまま移植可能、
  `pretrained_ckpt_path` は RoPE 版へ差し替え）。規則付与器とこの列挙器は同一新ブランチで、
  規則付与器を先に実装・テスト→その後 RoPE 導入→rel-only RLOO の順で進める。
- **(任意) M2 warm-start**: rel-only ckpt から本丸 RLOO を始めると dimensional 出力が
  死んでいるため高 entropy/SFT で復活させる必要があり、`sg_filtered` 始点の M1 より不利。
  M1 の結果次第で要否を判断。

実行順: M1 起動済み → M1 完走後に本章へ最終結果を追記 → 規則付与器+fiverr test → RoPE 列挙器。
各段で本ファイルを更新する。

### 14.7 scaffold-v2（M1）の結果と考察 (2026-05-31)

§14.6 の M1（scaffold-v2 = 探索ノブ ON + hygiene 強化）が両 arm とも完走した（共に
`epoch=3`, `global_step=14898`、v1 と **step 一致**）。本節はその結果・考察・判定を、
§14 の他節と同じく単独で追える形で記す。データは wandb から実測（同一 step 比較）。

- 比較対象 run:
  - `base_solver` = SolverReward 単体（`s2ibmbil`、scaffold なし。FC/DoF の基準点）。
  - v1（scaffold-v1）: `v1_noint=7u6kwzhf`, `v1_int=oiytlo2x`（§14.3 の run）。
  - v2（scaffold-v2 = M1）: `v2_noint=uywjlizh`, `v2_int=0csbzpdm`。
- M1 の差分（§14.5）: `entropy_bonus 0→0.005`, `train_sample_temperature 1.0→1.2`,
  hygiene 極小Distance/axisAngle `-0.30→-1.5`＋違反トークン局所化, rel-prior `cap_total
  0.5→1.0 / cap_per_type 2→3`。GT・pretrained_ckpt は v1 と同一。

学習ログ：
```
[rloo_response_scaffold_v2_intent]
run = api.run("/kakuda40611-nanyang-technological-university-singapore/vitruvion/runs/0csbzpdm")

[rloo_response_scaffold_v2_no_intent]
run = api.run("/kakuda40611-nanyang-technological-university-singapore/vitruvion/runs/uywjlizh")
```

**14.7.1 主要数値（同一 step、最終 / 後期10val平均）**

スカラーは validation が step ごとに揺れるため、**後期10 validation の平均**を頑健値とし、
括弧内に最終1点を付す（最終1点は端点ノイズを含む——後述）。

| 指標 | base | v1_noint | v1_int | v2_noint | v2_int | 向き |
|---|---|---|---|---|---|---|
| response_full ↓（後期10平均） | – | 0.534 | 0.491 | 0.493 | **0.464** | v2_int 最良 |
| 〃（最終1点） | – | 0.526 | 0.495 | 0.482 | 0.477 | （端点ノイズ大） |
| leak_relational ↓（最終） | – | 0.519 | 0.487 | 0.454 | **0.447** | v2_int 最良 |
| motion_relational ↓（最終） | – | 0.464 | 0.446 | 0.434 | **0.426** | v2_int 最良 |
| solver_fc_rate ↑（後期10平均） | **0.806** | 0.691 | 0.651 | 0.623 | 0.580 | v2 で更に低下 |
| solver_valid_dof ↓（最終） | **3.375** | 3.570 | 3.852 | 4.698 | 4.685 | v2 で大幅悪化 |
| solver_under ↑（最終） | 0.111 | 0.248 | 0.282 | 0.309 | 0.322 | v2 で増 |
| near_zero_distance（最終val） | – | 0.025 | 0.029 | **0.006** | 0.013 | v2 で半減（hygiene 成功） |
| eval_key_rel_ratio ↑（最終） | – | 0.365 | 0.390 | 0.378 | **0.406** | v2_int 最良 |
| eval_distance_family ↓（最終） | – | 0.287 | 0.284 | 0.254 | **0.247** | v2 で減 |
| train/entropy（後期10平均） | 0.373 | 0.370 | 0.371 | 0.386 | 0.377 | ほぼ不変 |

生成 constraint mix（`val_cmix/*`、後期10val平均。**注: これは type 別 per-family 比率で
和は1にならない**。相対比較のみ意味を持つ）:

| type | v2_noint | v2_int | （参考）v1_int @1035 |
|---|---|---|---|
| Coincident | 0.433 | 0.436 | 0.423 |
| Horizontal / Vertical | 0.183 / 0.166 | 0.168 / 0.161 | 0.147 / 0.147 |
| dimensional 計 | 0.392 | 0.382 | ~0.43 |
| Radius | 0.079 | 0.074 | 0.078 |
| **Equal** | 0.0065 | **0.024** | 0.016 |
| **Perpendicular** | 0.0008 | 0.004 | 0.011 |
| **Parallel** | **0.000**（max 0.016） | **0.000**（max 0.032） | ~0 |
| **Tangent** | 0.0088 | 0.019 | 0.015 |
| rare rel 計（Eq+Perp+Par+Tan） | 0.018 | **0.047** | 0.042 |
| Mirror / Concentric | 0 / 0 | 0 / 0 | 0 / 0 |

**14.7.2 読み取り**

1. **2つのレバーはほぼ加算的に効く**。response_full は v1_noint 0.534 →（scaffold-v2 単独で）
   v2_noint 0.493（**−0.041**）→（intent 上乗せで）v2_int 0.464（さらに **−0.029**）。
   注目すべきは **v2_noint(0.493) ≈ v1_int(0.491)**——「scaffold-v2 の探索/hygiene 強化だけで、
   v1 で intent 報酬が稼いだのと同等の response 改善が得られる」。intent 報酬の限界寄与は
   強 scaffold 上で 0.043→0.029 と**減衰したが消えてはいない**。
2. **0.005 は端点ノイズ。頑健差は 0.029**。最終1点だけ見ると v2 の intent 差は 0.482−0.477=0.005
   に見えるが、これは v2_noint の最終 val がたまたま良値（後期平均 0.493 に対し 0.482）だった
   ため。後期10val の std は両 run とも ≈0.006 で、差 0.029 は **約5σ**。intent 効果は単一 seed
   ながら統計的に実在する（seed 分散は未測——§14.7.4 の検証課題）。
3. **FC/DoF の代償が拡大した**。base 0.806/DoF3.375 → v1_int 0.651/3.85 → **v2_int 0.580/DoF4.69**。
   探索（温度↑）と hygiene 強化（極小 Distance に −1.5）は、**FC を稼いでいた dimensional の
   水増しを抑える**ため、その分だけ拘束が緩み DoF が増える。near_zero_distance は実際に半減
   （val 0.029→0.013、train 末 0.0007）で hygiene は意図通り効いたが、副作用として FC が
   base 比 −22pt・v1 比 −7pt 低下した。
4. **これがユーザの「per-sketch 推論漏れ」の正体**。集計では Coincident 0.436・H 0.168・V 0.161 と
   豊富なのに、定性的には個別 sketch で Coincident/H/V/Tangent が抜けて見える——これは矛盾でなく、
   **DoF=4.69 が「各 sketch で必要な制約が平均4.7自由度ぶん欠落」**を意味するため。集計頻度（type
   分布）と per-sketch 完全性（その図に必要な制約が付いているか）は別問題で、後者が悪化している。
   per-sketch 完全性は §15.2 の律速 **#2**（規則付与器の担当）。
5. **探索ノブは定量的にほぼ no-op だった**。entropy は v1 後期 0.371 → v2 後期 0.377（**+0.006**）で
   ほとんど動かず（中盤 0.33 まで沈み末で 0.38 へ戻るのは両 version 共通）。response/dimensional の
   改善は entropy 増ではなく **hygiene 強化＋温度**が主因と見られる。`entropy_bonus=0.005` /
   `temp=1.2` は steady-state entropy を 0.01 も上げられていない＝探索は名目 ON・実質弱い。
6. **Parallel の決定的所見——サンプルされても報酬で淘汰される**。Parallel は中盤に max 0.032
   （intent）/0.016（no_intent）まで出るが、**収束時は両 arm とも 0**。つまり「探索不足で一度も
   出ない」のではなく、**出たものが RLOO advantage で消される**。一方 Equal/Tangent/Perp は intent で
   部分的に生存（Equal 0.024 vs no_intent 0.0065、Tangent 0.019 vs 0.0088、Perp 0.004 vs 0.0008）。
   この **Parallel だけ死に、Equal/Tangent は生き残る**非対称は §15.1–15.2 の予測そのもの:
   d_H_angle は **Parallel と H+H の運動部分空間をほぼ同一**と見るため両者を区別できず、SolverReward の
   DoF 選好（H+H は DoF−2 > Parallel の DoF−1）が勝つ。逆に Equal/Tangent は代替制約と部分空間が
   **異なる**ので d_H_angle が区別でき、intent 報酬が浮上させられる。**律速 #3（報酬バランス）の直接実証**。
   なお rel_prior の Parallel/Perp 係数は 0.053/0.046 で、SolverReward の `reward_dof_reduction=10` に
   対し桁違いに小さく、#3 を埋める力がない。

**14.7.3 M1 ゲート判定（§14.6 の問いへの回答）**

§14.6 M1 の問いは「`entropy_bonus=0` 解除＋温度↑＋hygiene 強化で **rare relation の探索が回復し、
FC を犠牲にせず relational に拘束できるか**」だった。判定:

- **rare relation 回復: 否**。rare rel 計は v1 0.042 → v2 0.047 とほぼ横ばい。Parallel は依然 0
  （サンプルされるが淘汰）。Mirror/Concentric は構造的に 0（報酬 GT に無い）。
- **FC 非犠牲: 否**。むしろ FC −7pt・DoF +0.83 と悪化。
- **唯一の成功**: hygiene による near-zero distance のほぼ消去（ユーザ定性と一致）、dimensional 比率の
  微減、response 系指標の単調改善。ただしこれらは「拘束を緩める」方向の改善で、rare relation の
  絶対量を増やしていない。
- **結論**: 現強度の探索ノブだけでは rare relation 問題は解けない。残る律速は **#3 報酬バランス**
  （Parallel に関しては「指標の非分離性 × SolverReward の DoF 選好」）と **#2 per-sketch 完全性**
  であり、いずれも探索ノブの担当外。これは §15 の「列挙器ではなく規則付与器＋報酬バランス」という
  判断を強める実証データになった。

**14.7.4 次アクション（規則付与器の pseudo-GT 生成・学習とは独立に検証可能な事項）**

規則付与器による pseudo-GT v2 生成＋再学習は長ポール（167k で ~5h、1.67M で ~53h）だが、その完成を
待たずに律速を確定でき、かつ v2 GT を活かす報酬設定を前もって決められる検証が複数ある。優先度順:

1. **［学習不要・既存 ckpt から数時間］サンプル vs 採択の分解**。`v2_int` ckpt で val sketch に
   K=8 サンプルし、**生サンプルの type 別カウント**（報酬前）と **advantage>0 で採択された側**を分けて
   記録。Parallel が「生では出るが advantage が負で消える」ことを直接確認すれば、#3（報酬バランス）か
   探索不足かが確定する（14.7.2-6 の状況証拠を一次証拠に格上げ）。
2. **［学習不要］SolverReward の H+H vs Parallel を実測**。真に平行な辺対を持つ sketch で
   `{H,H}` と `{Parallel}` の2候補を作り、RewardComposer 全体（DoF 削減・FC bonus・rel_prior・intent）の
   スカラーを測る。§15.2 #3 の「H+H が勝つ」差を**数値化**し、coeff をどれだけ動かせば選好が反転するかを
   決める。これにより v2 GT 投入時の報酬設定（intent coeff↑ / H,V downweight / Parallel,Perp prior↑ の
   いずれをどれだけ）が事前に決まる。
3. **［学習不要・fiverr eval-only］d_H_angle の分離能監査**。fiverr 人間 GT で、Parallel と H+H 代替、
   Equal とその代替、Tangent とその代替について d_H_angle を計算し、「Parallel/H+H は不分離・Equal/Tangent は
   分離」を定量確認。これは**どの rare type なら報酬で浮上させられるか**（=規則付与器で GT に入れても
   intent で選ばせられる type）を type 別に切り分け、規則付与器の type 別方針（§15.4）と接続する。
   d_H_angle が評価器として妥当かという本研究の基盤主張（`summary_histcad_response_metric_v4_results.md`）の
   独立再検証も兼ねる。
4. **［学習要・1日・v1 GT のまま config だけ］報酬バランス probe**。GT・ckpt を変えずに、
   (i) Parallel/Perp の rel_prior 係数を ×5–10（0.05→0.3–0.5）かつ H,V を downweight、(ii) `coeff_h_angle
   3→6–8`、(iii) `{H,H}` への redundant penalty もしくは H+H→Parallel canonicalization、のいずれかで
   **Parallel が収束時 >0 になるか**を見る。v1 GT のままでも Parallel が立てば「律速は GT 完全性 #2 でなく
   報酬バランス #3」が確定し、**規則付与器だけでは Parallel 生成は直らない**ことの裏取りになる。
   （augmenter の長時間生成に着手する前に、活かす報酬設定を確定できる＝de-risk）。
5. **［学習要・1日・v1 GT のまま］探索強度 probe**。`entropy_bonus 0.005→0.02–0.05`, `temp→1.5`。
   14.7.2-5 で entropy がほぼ動いていないため、本当に entropy を上げた時に rare rel が出るか／FC だけ
   崩れるかを見る（#1 と #3 の切り分け）。あわせて温度のみ 1.0 に戻した変種で v2 の DoF 悪化（4.69）が
   どのノブ由来かを切り分け、v2 GT 学習に不要な FC 欠損を持ち込まないようにする。
6. **［学習不要・eval］報酬 GT と人間 GT の整合**。`v2_int` 予測の response_full を pseudo-GT v1 と
   fiverr 人間 GT の両方で計算し、両者が予測の良し悪しを**同順位に並べるか**を確認。並ばなければ報酬
   ターゲット（pseudo-GT v1）が評価（人間意図）と別物を最適化していることになり、これは augmenter の
   有無に関わらず効く根本論点。

優先実行: **1→2→3 を先に（全て学習不要、数時間で律速 #3 を一次証拠で確定）**、次に **4**（活かす報酬設定を
確定して augmenter 生成を de-risk）。これらは規則付与器の生成・再学習パイプラインと完全に独立で、並行可能。

## 15. 規則付与器の実装と「列挙器は後回し」の判断 (2026-05-30)

この章は §14.6 の第二段階を実装に落とす過程で得た判断を、**この章だけで完結して読める**よう、
用語を逐一定義して記す。

### 15.0 結論（先に）と用語

- 狙い: 人間の CAD 図面には現れるのにモデルがほとんど出さない関係（Parallel=平行,
  Perpendicular=直交, Equal=等寸, Tangent=接, Mirror=対称 …）を、学習・報酬の「正解(GT)」に
  十分入れること。
- §14.6 では2つの道具が候補だった:
  - **(A) 規則付与器(rule augmenter)**: 座標から幾何関係を機械的に検出して付与する、
    **学習モデル不要**のプログラム。
  - **(B) 列挙器(enumerator)**: 関係のみを RLOO で学習させ、関係を多めに吐かせる**学習モデル**。
- **判断: (A) を作る。(B) は明示ゲートの後ろに後回し**（理由は §15.2）。
- 本章では (A) を実装・テストし（§15.3）、人間データ(fiverr)で精度を測った（§15.4）。
  (B) は今は実行しない前提で、必要時のみ §15.5 の最後に戻る。

**用語（既出だが再掲。この章で繰り返す）**

- **primitive**: スケッチの基本図形 = 点 / 線分 / 円 / 円弧。
- **relational constraint（関係制約）**: 数値を持たない関係（Coincident=一致, Horizontal/Vertical,
  Parallel, Perpendicular, Tangent, Equal, Concentric=同心, Midpoint=中点, Mirror=対称）。本章の対象。
- **dimensional constraint（寸法制約）**: 数値を持つ（Distance, Radius, Angle）。本章の対象外。
- **DoF（自由度）**: その図形がまだ自由に動ける独立な方向の数（点=2, 線分=4, 円=3）。制約はこれを
  減らす。全部 0 = **FC（完全拘束）**。
- **GT（正解データ）**: 学習・報酬が参照する制約集合。現状は `sg_pseudo_rel_167k`（モデル生成の
  関係集合 = pseudo-GT v1）。
- **SolverReward**: 主報酬。制約が DoF をどれだけ減らしたか・FC に達したかで点を付ける。
- **d_H_angle / ResponseIntentReward**: 本研究の指標兼報酬。「生成した制約が許す動き」と
  「人間 GT が許す動き」のズレを測る（小さいほど人間意図に整合）。
- **class prior**: 特定の関係タイプを出すとボーナスを与える補助報酬（出現頻度を底上げするだけ）。

### 15.1 解きたい問題: 「代替可能な」関係はモデルが出さない

ユーザの問い「列挙器(B)で Parallel / Perpendicular / Mirror を出せるのか」への答えは **No**。
鍵は **代替可能性(substitutability)** で、具体例で示す。

**代替可能とは**: ある関係が、別の（多くはより一般的な）制約の組合せで自動的に成り立ち、
わざわざ書く必要がない状態。

- 例1: 線 L1, L2 を**両方 Horizontal** にすると、2本は自動で **Parallel** になる
  → Horizontal を2回書くモデルは Parallel を書く動機がない。
- 例2: L1=Horizontal, L2=Vertical なら自動で **Perpendicular**。
- 例3: **Mirror(対称)** も、対応点の Coincident や辺の Equal を並べれば近いことが言える。

**なぜ DoF ベースの報酬では出ないか**: `SolverReward` は「DoF をどれだけ減らしたか」で点を付ける。
ところが Horizontal+Horizontal（DoF を2削る）は Parallel（1削る）の上位互換で、しかも事前学習で頻出。
DoF で測る限りモデルは常に H+H を選び、Parallel を選ぶ理由がない。class prior で頻度は水増しできるが、
それは「正しい場所に置く」ことではなく単なる出現増(spam)。

→ つまり **代替可能な関係は「DoF を減らすか」を見るどんな方法（列挙器を含む）でも出てこない**。
出すには「DoF と無関係に、幾何的に成り立っているかだけを見る」検出が要る = 規則付与器。

### 15.2 なぜ列挙器(B)でなく規則付与器(A)か（律速の切り分け）

「rare relation が出ない」原因は3つに分解でき、**列挙器(B)はどれにも効かない**。

**決定的な事実**: 報酬 GT (v1) は既に Parallel 8.7% / Perpendicular 5.0% を含む。にもかかわらず
生成側はほぼ0 → **「GT に無いから出ない」のではない**。残る律速と対処:

| # | 律速（原因） | 中身 | 効く対処 |
|---|---|---|---|
| 1 | 探索の崩壊 | v1 は探索のばらつき(entropy)=0 で安全な Coincident に固まる | scaffold-v2（entropy/温度↑） |
| 2 | GT の図ごとの不完全さ | 集計%はあっても、その図の平行な「その2本」に実際付いているとは限らない | **規則付与器**（幾何的に真な全ペアに付与） |
| 3 | 報酬バランス | `SolverReward` が H+H(DoF−2) を Parallel(DoF−1) より高評価。意図報酬がこれを上回らない限り、GT が完璧でも H+H が勝つ | 報酬係数の調整（規則でも列挙器でも直らない） |

列挙器(B) は新しい GT 生成器を学習で作るものだが、上の 1–3 を直さない（むしろ #3 の DoF 偏重を
抱えたまま）。よって**後回し**にし、#2 に効く規則付与器(A) を先に作る。

**規則付与器が効く道筋（修正版）**:

1. 規則付与器は、幾何的に真なら **Horizontal 2本と Parallel を両方出す**。同様に、
   Vertical/Horizontal と Perpendicular も同時に出る。これは実装として正しい。
2. 現行の `motion_relational_angle` はこの冗長性を吸収するので、H+H と H+H+Parallel はほぼ区別しない。
3. ただしGT が `Parallel` だけで Pred が `H1 + H2` なら motion subspace は違う。H+H は全体回転を
   余計に禁止するため、d_H_angle は「拘束しすぎ」として差を出せる。
4. よって規則付与器の直接の役割は、**rare relation を選ばせること**ではなく、
   **人間 GT の sparse annotation を relational closure に近づけ、motion metric が表記ゆれを吸収できる
   ターゲットを作ること**である。Parallel を H/H より選ばせたい場合は、別途
   `H/H -> Parallel` の canonicalization、H/V の downweight、class prior、または sequence-level reward が要る。

### 15.3 規則付与器の実装（branch `rule_augmenter`）

ファイル: `src/sketch_handlers/rule_augmenter.py`（本体）/ `tests/test_rule_augmenter.py`（15 tests, green）
/ `scripts/eval_rule_augmenter_recall.py`（fiverr 較正）。

**何をするか（1 文）**: スケッチの**実座標**を見て、「点 P は線 L 上か」「線 A と線 B は平行か」等の
候補を総当たりで作り、**実際に成り立つものだけ**を制約として残す。

**判定の仕組み — residual の再利用**: このコードには各関係タイプごとに **residual（残差）**
=「その関係が成り立つとき 0 になる数値」関数が既にある（例: Parallel の residual は2線方向の外積で
平行なら 0。Coincident(点-点) は2点の座標差で一致なら 0）。検出器はこれを**そのまま流用**するので、
ソルバや d_H_angle と幾何の定義が必ず一致する。residual を実座標で評価するために **`SketchState`**
（スケッチの実座標を保持し端点・半径を取り出す補助オブジェクト）を使う。

**生 residual → 解釈できる margin（正規化）**: 生 residual は型ごとに**単位がバラバラ**（外積っぽい量・
長さ・長さの差…）で1つの閾値では比較できない。そこで型を3グループ(family)に分け各々を無単位量に
正規化する。この正規化後の値を **margin** と呼ぶ（0=完璧に成立、大きいほど外れている）。

| family | 含む型 | 生 residual | ÷ 正規化スケール | margin の意味 |
|---|---|---|---|---|
| orientation | Horizontal, Vertical, Parallel, Perpendicular | `\|d1\|\|d2\|·sinθ`(or cosθ) | 線長の積 | `\|sinΔθ\|`（角度ズレの sin） |
| position | Coincident, Concentric, Midpoint, Tangent, Mirror | 長さ。Tangent.point の接線成分だけ長さ² | bbox 対角長。Tangent.point の接線成分は線長×半径など | 図全体サイズに対する割合、または接線角ズレ |
| size | Equal(長さ/半径) | 長さの差 | 大きい方の長さ/半径 | 相対差 |

（bbox = スケッチ全体の外接矩形。bbox 対角長 = 図の大きさの代表値。）

**tolerance（許容閾値）= margin の上限**。`margin < tolerance` の候補だけ採用する。family ごとの既定値:
orientation = sin(2°) ≈ 0.035, position = 0.01(=図サイズの1%), size = 0.02(=相対2%)。

**無限直線とガード**（PointOnObject=点が辺上 / Tangent=接）: これらの residual は線分を**無限延長
した直線**への距離で定義されている（コード元来の仕様）ので**外挿（線分の外側の延長線上）も有効**。
当初「線分内に限る」ガードを足したが仕様違反で**撤去**した。残すガードは1つ — 点が**辺の端点ぴったり**
に乗る場合のみ除外（それは頂点どうしの一致=角で、point-point Coincident として別に出るため二重計上を防ぐ）。

**Mirror の扱い**: 既存の Line entity を対称軸にする場合だけ検出する。まだ「データには軸が無いが、
点集合から潜在対称軸を発見する」ことはしない。

**対象外（意図的に検出しない）**: line-line Coincident（= Collinear/共線。ほぼ Parallel と重複・希少・
residual が混合単位）。

**公開 API**:
- `detect_candidates(sketch)` → 全候補（margin 付き、閾値適用前）。較正用。
- `filter_candidates(cands, tolerances)` → margin で間引いた採用集合。
- `augment_sketch(sketch)` → 採用候補を制約として書き戻したスケッチ（既存との重複は除外）。

### 15.4 fiverr 較正結果と pseudo-GT v2 への含意

再実行コマンド:
`conda run -n freecad python -m scripts.eval_rule_augmenter_recall --fiverr-root /home/ryotaro/Downloads/fiverr --annotators Eko Johnny Ryotaro Sagar Tanaka`

まず、ここでの **scale は座標スケールでも報酬係数でもない**。`scripts/eval_rule_augmenter_recall.py`
が `DEFAULT_TOLERANCES` に掛ける tolerance multiplier である。

| scale | orientation | position | size |
|---:|---:|---:|---:|
| 0.25 | `sin(0.5°)` 相当 | bbox 対角の 0.25% | 相対 0.5% |
| 1.0 | `sin(2°)` 相当 | bbox 対角の 1.0% | 相対 2.0% |
| 4.0 | `sin(8°)` 相当 | bbox 対角の 4.0% | 相対 8.0% |

したがって scale を下げるほど「より厳密に幾何が成立する候補だけを採る」。この scale は
`margin = |residual| / family_scale` の上限を動かすだけで、候補の選択方針や報酬の強さは変えない。

再検証で見つけた実装修正:
- `TangentHandler` の line-circle edge tangency が signed distance のまま `dist - r` を使っていたため、
  線の片側にある真の接円を落としていた。`abs(dist) - r` に修正し、
  `tests/test_rule_augmenter.py` に両側接線の回帰テストを追加した。
- `Tangent.point`（接点を明示した Tangent）を列挙に追加した。line endpoint が circle に接するケースなどを
  `tests/test_rule_augmenter.py` に追加。
- `Mirror` は、既存 Line を対称軸とする候補だけ追加した。未知の潜在軸探索は未実装。
- `Midpoint` の未検出 12件を調べると、すべて「Circle.center が Line の中点」だった。そこで midpoint の点候補に
  Circle/Arc center を追加した。
- 3-ref Midpoint は `Arc.start, Arc.end, midpoint` の構造的ケースだけ追加した。任意2点+中点の全列挙は
  空間ハッシュで計算量は抑えられるが、fiverr では TP を増やさず FP を約7,700件増やしたため採用しない。
- 修正後: `tests/test_rule_augmenter.py` は 15 passed、`tests/test_relational_class_prior.py` は 12 passed。
- この修正により `Tangent.edge` recall は scale 0.25 で 0.609 → **1.000** になった。
- `Midpoint` recall も scale 0.25 で 0.864 → **1.000** になった。

全件結果（771 sketches, 11,740 human constraints, 5 annotators, errors=0; Tangent.point/Mirror/Midpoint center+arc 3-ref 追加後）:

| label | support | recall@0.25 | precision@0.25 | 解釈 |
|---|---:|---:|---:|---|
| Coincident.point_point | 6008 | 0.954 | 0.883 | かなり信頼できる。角の stitching は全採用候補。 |
| Coincident.point_on_object | 1052 | 0.933 | 0.184 | 幾何検出は強いが人間よりかなり密。 |
| Horizontal | 634 | 0.987 | 0.283 | 幾何的 H は多く、人間 annotation は疎。 |
| Vertical | 437 | 0.993 | 0.213 | H と同じく密度調整が必要。 |
| Perpendicular | 529 | 1.000 | 0.067 | recall は完璧だが過剰列挙が大きい。 |
| Tangent.point | 434 | 1.000 | 0.420 | human support が大きく、追加実装の価値が高い。 |
| Parallel | 381 | 1.000 | 0.055 | 本目的には有用だが、全ペア投入は危険。 |
| Concentric | 358 | 1.000 | 0.754 | 高信頼。全採用候補。 |
| Midpoint | 88 | 1.000 | 0.116 | center-as-midpoint を足して recall 解決。任意3点 midpoint は入れない。 |
| Equal.radius | 61 | 1.000 | 0.057 | 幾何的には拾えるが、人間密度は非常に低い。 |
| Tangent.edge | 46 | 1.000 | 0.034 | 修正後 recall は解決。ただし大量に追加される。 |
| Mirror | 37 | 1.000 | 0.006 | 既存軸だけで recall は出るが、候補爆発が非常に大きい。 |
| Equal.length | 5 | 0.600 | 0.001 | support が小さく、等長ペアの全列挙はほぼ使えない。 |

構造的に拾えない、または意図的に拾っていないもの:
- `Midpoint` の任意3点形式: `A, B, M` が幾何的に中点になる全組合せは、座標ハッシュで計算量は
  O(P² + hits) にできる。しかし grid/対称図形では「幾何的には真だが人間 intent としては書かない」
  候補が大量に出る。fiverr では recall 改善なし・FP 大増だったため、default では検出しない。
- `Mirror` の潜在軸: 既存 Line を軸にするものは拾うが、軸 Line がデータに無い対称性は拾わない。
- `Coincident.collinear`: support 6。line-line Coincident は Parallel と重複しやすく、現状は除外。
- dimensional constraints: `Distance` 820 / `Radius` 568 / `Angle` 74。本章の対象外。

scale sweep の読み:
- detectable types の micro recall は 0.25→4.0 で **0.964→0.967** とほぼ動かない。
- 一方で micro precision は **0.210→0.133** に落ち、FP は **36,546→63,433** に増える。
- つまり tolerance は recall の律速ではない。`scale=0.25` で既にほぼ拾えており、緩めると
  ほぼ FP だけが増える。運用点は **scale=0.25** が自然。

重要な解釈:
- 低 precision は「幾何判定が間違っている」という意味ではない。評価スクリプトの precision は
  **人間が同じ subtype と同じ base entity set に制約を書いたか**で測る。規則付与器は
  幾何的に真な候補を最大列挙するので、人間 annotation が疎なら precision は必ず低くなる。
- したがって precision の低さだけで候補を切るのは早い。特に `Parallel/Perpendicular` は
  H/V で代用されているだけ、`PointOnObject` や `Tangent.edge` も他の intent constraints から
  結果的に導かれるだけ、という可能性がある。この場合、`motion_relational_angle` は row-space
  冗長性として吸収する。
- ただし **全ての幾何的真が冗長とは限らない**。`Equal.length`、遠い点同士の `Mirror`、
  偶然の `PointOnObject` は motion subspace を実際に狭めうる。低 precision は discard 理由ではなく、
  over-closure のリスク指標として読む。

**pseudo-GT v2 方針**:
- precision で候補を落とす前に、まず **full relational closure @ scale=0.25** を試す価値がある。
  これは「recall 最大化」仮説の検証であり、ユーザ指摘の通り、motion subspace が冗長性を吸収できるなら
  低 precision は実害にならない。
- ただし full closure の安全性は仮定しない。`full_all`, `full_no_mirror`, `high_precision_only`
  を分け、追加後の relation Jacobian rank / null-space 次元 / d_H_angle 対 human GT を見る。
- 特に `Mirror` は recall 1.0 だが precision 0.006 なので、full_all の中でも別スイッチにする。

### 15.5 pseudo-GT v2 npy生成・学習推論パイプラインのトレースと安全性判断

Q: 規則付与器(`src/sketch_handlers/rule_augmenter.py`)で pseudo-GT を作るのに
`scripts/generate_pseudo_rel_dataset.py` を**素朴に**使ってよいか。relational constraints は
大量に冗長で、その冗長性を持つ GT がパイプラインのどこかで想定外の挙動を起こさないか。

---

**0. 結論(TL;DR)**

- **素朴な流用は不可**。理由は2つ:
  1. `generate_pseudo_rel_dataset.py` は *rel-only RLOO モデル* を走らせて関係を予測する
     スクリプトで、規則付与器を一切呼ばない。そのままでは augmenter pseudo-GT を作れない。
  2. 仮に差し替えても、既定の solver 検証 (`:168 if status.Status != 'Converged'`) が
     **Redundant を却下**する。`get_solver_status` は Converged+redundant を `'Redundant'` に
     再分類する(`solver_feedback.py:299–305`)ため、規則付与器の **full closure はほぼ全件が
     Redundant 判定 → ほぼ全却下**になる(コメント「Redundant tolerated」は誤り。メモリ既知)。
- **一方、冗長 GT 自体はダウンストリームで安全**。これは静的トレース＋実データで確認した(§2–§3)。
  - 報酬は GT を **`datalib.Sketch` オブジェクト**(`batch['sketch']`)として受け取り、関係を
    **SVD ランクで motion subspace に縮約**する。冗長行は特異値≈0で**ランクに寄与せず吸収**される。
  - tokenizer の `max_token_length` 切り詰めは **GT subspace 経路に無関係**(後述の二経路)。
- よって判断は **「小変更で可」**。専用スクリプト `scripts/generate_pseudo_rel_augmented_dataset.py`
  を新規作成した(§4)。safety gate 付きで、生成 npy は学習時にクラッシュしないことを保証する。
- 動作確認の **canary (2000件→1990 valid, 0 error)** を `/tmp/sg_pseudo_rel_v2_canary.npy` に生成済み。
  `ConstraintDataset.__getitem__` 経由 800件ロードで **エラー 0** = 学習可能。rare relation が
  GT に豊富に入った(Mirror/Perp/Parallel/Equal/Tangent…)。本生成のスコープのみ要確認(§6)。

---


**1. 静的トレース: 冗長 GT が通る2つの経路**

ConstraintDataset.`__getitem__`(`constraint_dataset.py:68`)は1つの sketch から**2つ**を作る:

| 経路 | 中身 | 切り詰め | 冗長の扱い | 消費先 |
|---|---|---|---|---|
| **① GT sketch object** | `result["sketch"]`=full `datalib.Sketch` | **なし**(オブジェクトそのまま) | SVD で吸収 | **報酬**(`ResponseIntentReward` 等) |
| ② tokenized `c_val` | `tokenizer.tokenize()` 出力 | **あり**(`pad_or_truncate(…,192)`) | 末尾を黙って切る | CE/SFT・`gt_tokens` |

**1.1 報酬は経路①(full object)を使う — 冗長は SVD で吸収**

- RLOO: `_compute_rloo_loss` → `build_context(sketch_original_list=expanded['sketch'], …)`
  (`constraint_ar_module_rloo.py:185`)。報酬は `ctx.sketch_original_list[i]` を `gt_sketch` として使う。
- `get_or_compute_gt_motion_relational_subspace` → `motion_relational_subspace(gt_sketch, …)`
  は `gt_sketch.constraints.values()` の**全関係**を `stack_relation_jacobians_at` で J_R に積む
  (`geometry.py:146–156`)。`N_C = null(J_R)`、`W_C = range(J_phi @ N_C)`、basis は
  `orthonormal_range`(`geometry.py:36–40`)で **特異値 ≥ eps の本数だけ**を採る。
  → **冗長行(H+H が含意する Parallel 等)は特異値≈0でランク非寄与 = 吸収**。`dim_gt` は count では
  なく rank なので、冗長を何本足しても不変。ユーザの仮説どおり。
- **重要**: GT subspace は経路①(full object)なので、tokenizer の 192 切り詰めの影響を**受けない**。
  どれだけ長い closure でも全関係が J_R に入る。

**1.2 切り詰め(経路②)が効く先と、現 config での無害性**

- **CE/SFT**: `total_loss = rl_loss + ce_loss * sft_coeff`(`constraint_ar_module_rloo.py:357`)。
  `sft_coeff` 既定 **0.0**(`configs/model/constraint_ar_rloo.yaml:62`、scaffold config で未上書き)
  → CE は学習に**寄与せず**、切り詰めは無害。
- **DimParsimonyReward**(`gt_tokens` 消費): GT の**寸法**トークン数を数える。pseudo-GT は
  relational-only なので寸法=0 → 切り詰めと無関係。
- **GTMatchReward**: scaffold config の rewards に**不在**(SolverReward + DimParsimony +
  RelationalClassPrior + ResponseIntent のみ)。
- 結論: **現 response-metric RLOO 設定では 192 切り詰めは報酬・学習に実害なし**。
  （SFT/pretrain で v2 を教師に使う、または `gt_tokens` 系報酬を足す場合のみ要対応 → §5 caveat。）

**1.3 cleanse は制約「間」の冗長を消さない(=意図通り保存)**

`cleanse_sketch_constraints` の dedup は**単一制約内のref重複**のみ(`sketch_utils.py:721–744`、
例 Coincident(A,A))。制約**間**の幾何冗長(H+H+Parallel)は消さない。よって冗長は GT に残り、
報酬側 SVD で吸収される、という設計が成立。

**1.4 正規化は相似変換 = 検出述語は不変**

`normalize_sketch` = center + **等方** rescale(`sketch_utils.py:248–253`、単一 scale_factor)。
Horizontal/Vertical/Parallel/Perp/Coincident/Concentric/Equal/Tangent/Mirror は相似変換不変 →
**augment を正規化の前後どちらで行っても同じ集合**。geometry 不一致の懸念なし。

---

**2. 実データ検証(`/tmp/probe_augmenter_pipeline.py`、freecad env、60件)**

`sg_filtered_unique_valid.npy` 先頭60件で、base(関係のみ)→ augment(scale0.25)→
to/from_sequence 往復 → dataset 前処理 を実行:

| 検証項目 | 結果 | 意味 |
|---|---|---|
| **往復後 dim == augment 後 dim** | **59/59 一致** | npy 詰め込み＋ロードで GT subspace が**厳密保存** |
| **augment 後 dim ≤ base dim** | **常に True** | 冗長を平均61本(最大367本)足しても dim は blow up せず縮むだけ |
| augment 後の dim 平均 | base 8.54 → aug 5.49 | 関係が増え motion が締まる(rank 有界) |
| 往復で関係数が減る量 | 平均 −6.2(最大 −17) | cleanse が**冗長**関係を一部削除。**ただし dim は不変**=削られたのは inert な冗長 |
| **エラー** | 1/60 | sketch54。augment 前の**元データ** c_41(3-ref Midpoint 厳密性)。augmenter 由来ではない |
| augment 時間 | 平均 75ms / p95 282ms | 後述スコープ算定の根拠 |

→ 「冗長 GT が pipeline を壊す」懸念は**経路①(報酬)について実証的に否定**された。

---

**3. 必要だった「小変更」: 専用生成スクリプト**

`scripts/generate_pseudo_rel_augmented_dataset.py`(新規)。`generate_pseudo_rel_dataset.py` との差分:

1. **モデル不使用**。各 sketch を `sketch_from_sequence → normalize → cleanse_primitives →
   (既定)全制約strip → augment_sketch(scale0.25) → lenient cleanse/complete → sketch_to_sequence`。
   - 既定は **pure relational closure**(bare primitives から復元)。v1 が「モデル予測の関係で
     置換」だったのと整合。`--keep_original` で既存関係に上乗せも可。
2. **solver 検証は既定 OFF**。`--solver_validate` 指定時のみ、**緩和版**(Redundant 許容、
   Conflicting/Malformed/Distorting のみ却下)。§0 の Redundant 誤却下を回避。
3. **safety gate(常時)**: 生成系列を **dataset と同一の STRICT 前処理**
   (`cleanse_sketch_constraints` 厳密＋`complete_constraint_values` 厳密)で再ロードし、
   `motion_relational_subspace` まで計算。**例外を出す標本/関係0の標本は捨てる**。
   → 保存 npy は **学習時クラッシュしないことを構成的に保証**。
4. `--no_mirror`(Mirror 除外=full_no_mirror ablation)、`--shard i/n`(並列)、`--limit`、`--types`。

**3.1 canary 実行結果(先頭2000件)**
```
valid 1990 / no_rel 10 / load_reject 0 / augment_error 0 / solver_reject 0
avg relations added/sketch 67.9   8.8 sketch/s
```
- `load_reject 0` = safety gate を全件通過。pure closure で**元データの不良制約が消える**ため、
  §3 の sketch54 系エラーも再発しない。

**3.2 ConstraintDataset 経由ロード(`/tmp/verify_v2_dataset.py`、rloo context, 800件)**
```
__getitem__ ok=800  errors=0          ← align=4 の >3-ref guard でもクラッシュ無し = 学習可能
relations/sketch (strict cleanse後) mean 71.2 / max 464
c_val が 192 cap 到達(切り詰め): 430/800 (54%)   ← 経路②のみ。報酬には無関係(§2.2)
relation HEAD-type: Mirror 11617, Coincident 11358, Perpendicular 10461,
  Parallel 8798, Equal 6004, Tangent 2936, Horizontal 2556, Vertical 2536, Midpoint 661
```
→ 本来「ほぼ無い」だった **Parallel/Perp/Equal/Tangent/Mirror が GT に豊富**に入った
(動機 §obs.2 を直接充足)。

---

**4. 注意点・既知の caveat**

- **C5-a 切り詰め(経路②)**: full closure は 54% の sketch で c_val が 192 を超える(最大 1188 tok)。
  **現設定では無害**(§2.2)。ただし将来 v2 で **SFT/pretrain**(`sft_coeff>0`)や **GTMatch/gt_tokens
  系報酬**を使うなら、`max_token_length` を上げるか型別に密度を間引く必要がある。
- **C5-b Concentric → Coincident 正規化**: `cleanse_sketch_constraints` の
  `_convert_concentric_to_coincident`(`sketch_utils.py:770`)が **Concentric を中心同士の
  Coincident に変換**する。よって GT に "Concentric" **ラベルは出ない**(canary でも head-type 0)。
  - これは**脱落ではなく相対ラベル変換**で、**幾何・motion subspace は等価**(中心一致=同心)。
    report の検証で raw Concentric 候補は 3000件中850件 検出されており、cleanse 段で Coincident に
    畳まれているだけ。**§14.4 が言う「Concentric の構造的欠落」の正体はこれ**で、v1 も含む既存挙動。
  - 報酬(motion subspace)には無害。ただし **class prior の `Concentric` 係数や型別の質的集計**には
    "Concentric" が現れない点に注意(必要なら cleanse の変換を gate するか、Coincident-of-centers を
    Concentric として再分類する)。
- **C5-c Mirror は survive**(canary 11617件)。§14.4 の Mirror 欠落は v1(モデルが出せない)由来で、
  augmenter 経由なら GT に入りパイプラインを通る。ただし fiverr precision 0.006 なので
  **full_no_mirror ablation を必ず併走**(§15.5)。
- **C5-d 関係過多のリスク**: closure は dim を強く締める(8.5→5.5)。これは「人間 GT の sparse
  annotation を closure に近づける」狙い(§15.2-4)に沿うが、over-closure(`Equal.length` 全ペア、
  遠点 Mirror 等)が motion subspace を**過剰に**狭める可能性は残る。`full_all` / `full_no_mirror` /
  `high_precision_only` を分けて relation Jacobian rank / null-space / d_H_angle(対 human)で診断する
  (§15.4 方針)。本トレースは**安全性**を確認したもので、**有効性**(d_H_angle 改善)は別途要評価。

---

**5. 本生成: sharded parallel**

- **スコープ確定**: 入力 `sg_filtered_unique_valid.npy`(**1,670,450件**)の **先頭 167k**
  (v1 `sg_pseudo_rel_167k` と同スケール)。variant = **full_all** と **full_no_mirror** の2本。
  scale=0.25。全 1.67M(~53h)は当面の実験に過剰なので採らない。
- **オーケストレータ**: `scripts/run_pseudo_rel_v2_gen.sh`。各 variant を **NSHARD=12** に分割し、
  両 variant を**同時**に走らせる(=24 python worker)。`--shard i/12` は **limit 先・shard 後**
  なので 12 本の和集合がちょうど先頭 167k(各 ~13,917件)。完了後、各 variant の shard を
  `--merge_inputs` で1ファイルにマージ(行順保存で sequence↔metadata の整合維持)。
- **環境条件**(実測): 32 コア / 空き 107GB / 入力 npy は `mmap_mode='r'`(各 worker が 2GB を
  RSS で抱えない)。worker は `torch.set_num_threads(1)` + `OMP/MKL/OPENBLAS=1` で 1 スレッド化し、
  24 本を全コアにばらまく。実測 worker affinity=0–31(フル)、各 ~100% CPU。
  → **ETA ~30 分**(単一プロセス 5.3h を短縮)。
- **出力**: `data_processed/sg_pseudo_rel_v2_full_all.npy` / `data_processed/sg_pseudo_rel_v2_no_mirror.npy`。
- **完全デタッチ起動(セッションを閉じても継続)**:
  ```bash
  setsid bash scripts/run_pseudo_rel_v2_gen.sh < /dev/null \
      > /tmp/gen_v2_orchestrator.log 2>&1 & disown
  ```
  `setsid` で新セッションリーダ化 → 起動した CLI セッションのプロセスグループ signal の影響を
  受けず、**会話セッション/ターミナルを閉じても走り続ける**。
- **監視 / 完了確認**:
  ```bash
  tail -n 20 /tmp/gen_v2_orchestrator.status   # マイルストーン(launching / merged / ALL DONE)
  ls /tmp/gen_v2_DONE                           # 存在すれば全完了
  grep -h valid= /tmp/gen_v2_full_all.shard*.log | tail   # shard 進捗
  ```
- 完了後の検証(自動ではない、手動 or 再開時に実行): 両 npy を `ConstraintDataset` で全件ロードし
  `__getitem__` エラー 0 と relation 分布(canary 同様 Mirror/Perp/Parallel/Equal/Tangent が豊富)
  を確認する。

**5.1 本生成完了・sanity check (2026-05-31)**

本生成は完了。`/tmp/gen_v2_DONE` が存在し、orchestrator status は
`2026-05-31 11:04:00 ALL DONE`。出力は `.gitignore` 配下のため commit 対象ではなく、
ローカル実験データとして `data_processed/` に保存:

| variant | npy | size | dataset size | 生成時棄却 |
|---|---:|---:|---:|---:|
| full_all | `data_processed/sg_pseudo_rel_v2_full_all.npy` | 214M | 166,254 | no_rel 746 / load_reject 0 / augment_error 0 |
| full_no_mirror | `data_processed/sg_pseudo_rel_v2_no_mirror.npy` | 200M | 166,254 | no_rel 746 / load_reject 0 / augment_error 0 |

`ConstraintDataset` の RLOO 実経路(`max_token_length=192`, align=4,
`PrimitiveNoise(std=0.15)`)で **全件 `__getitem__` sanity check green**:

| variant | `__getitem__` | rel/sketch mean | rel/sketch max | c_val 192 cap |
|---|---:|---:|---:|---:|
| full_all | 166,254 ok / **0 errors** | 69.4 | 2,936 | 87,558 / 166,254 |
| full_no_mirror | 166,254 ok / **0 errors** | 55.2 | 1,232 | 74,234 / 166,254 |

主要 relation head-type (全件):

| type | full_all | full_no_mirror |
|---|---:|---:|
| Mirror | 2,366,079 | 0 |
| Coincident | 2,283,136 | 2,283,136 |
| Perpendicular | 2,088,540 | 2,088,540 |
| Parallel | 1,751,349 | 1,751,349 |
| Equal | 1,244,472 | 1,244,472 |
| Tangent | 638,475 | 638,475 |
| Horizontal / Vertical | 518,630 / 509,642 | 518,630 / 509,642 |
| Midpoint | 137,427 | 137,427 |

metadata 長も `sequences/sketch_ids/sequence_lengths = 166,254` で揃っている。
したがって v2 GT は、RLOO 報酬経路の入力としては training-ready と判断する。


### 15.6 次手

1. ✅ 全件 fiverr スイープ完了（§15.4）。運用点 = scale 0.25。型別方針は precision cut ではなく、
   full relational closure を第一 ablation として試し、rank/null-space で over-closure を診断する。
2. **検出器の穴埋め（任意・優先度順）**:
   - ✅ **Tangent.point**（human support 434）: 実装済み。scale 0.25 で recall 1.0 / precision 0.420。
   - **Tangent.edge**: signed distance bug は修正済みで recall 1.0。残る問題は recall でなく
     precision 0.034 の密度選択。
   - ✅ **Mirror(existing-axis)**: 実装済み。scale 0.25 で recall 1.0 / precision 0.006。
     precision が極端に低いので `full_no_mirror` ablation も必ず切る。
   - ✅ **Midpoint center / arc endpoint 3-ref**: 実装済み。scale 0.25 で recall 1.0 / precision 0.116。
     任意2点+中点の3-refは FP 爆発に近い over-closure だったため default では採用しない。
3. **pseudo-GT v2 生成**: augmenter を学習側 primitives（`sg_filtered` 等）に適用。
   ※ 現状 detect は候補ごとに handler.residual を呼ぶため 167k 件には遅い → numpy 高速 path
   へ最適化が要る（既知 TODO）。
   - ✅ パイプライン安全性: 確認済み(冗長 GT は報酬経路で安全)。
   - ✅ 小変更スクリプト + safety gate + merge: 実装・smoke 検証済み。
   - ✅ 本生成(full_all + no_mirror @167k): 完了。全件 `ConstraintDataset.__getitem__` sanity
      check 0 errors（§15.5.1）。
   - scaffold-v2 + intent を **v2 GT** で回し、§14.6 ゲートで判定(Parallel がサンプルに乗るか /
      leak・response_full_angle が v1 比改善か / 残差が報酬バランス #3 か)。**C5-d** の
      rank/null-space/d_H_angle(対 human)over-closure 診断を併走。`full_all` 対 `full_no_mirror` で
      Mirror over-completion の影響を切り分ける。
   - (config) `train_sequence_file` を v2 npy に差し替え。SFT/gt_tokens を使う場合のみ §C5-a の
      `max_token_length` 対応が要る(現 RLOO 設定では不要)。
4. scaffold-v2 + intent を **v2 GT** で回し、§14.6 のゲートで判定（Parallel がサンプルに
   乗るか / v1 比で response_full_angle・leak 改善か / 残差が報酬バランス #3 か）。
5. **必要時のみ** RoPE rel-only 列挙器（§15.1 の union 採取構成）。
