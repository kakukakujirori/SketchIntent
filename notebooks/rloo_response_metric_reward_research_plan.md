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

### 14.7.5 「なぜ非代替の rare relation が出ないのか」を学習なしで切り分ける (2026-05-31)

この節は §14.7.4 で挙げた「学習不要の独立検証」を実際に実装・実行した結果を**一本化**して記す
（旧 `notebooks/diag_independent_checks_results.md` は本節へ統合し廃止。スクリプトは `scripts/diag_*.py`）。
他節と同じく、初見でも追えるよう目的・方法・結果・読み方を省略せず書く。

**背景と目的（何がしたいか）**: scaffold-v2（§14.7.1–3）でも rare relation の絶対量は伸びず
（rare-rel 計 0.042→0.047）、とりわけ「他の制約で**代替できない**」Tangent / Perpendicular / Equal が
人間 GT 比で大きく不足したままだった。「なぜ伸びないか」には処方の異なる複数の原因がありうる:

- **H1 未サンプル**: そもそも方策がその関係を出さない（事前分布・探索の問題）。
- **H2 報酬淘汰**: 出すが RLOO の報酬が選ばない（advantage が負）＝報酬バランスの問題。
- **H3 指標の非分離**: 指標 d_H_angle がその関係を代替制約と区別できない＝指標の分解能の問題。
- **H4 GT 欠落**: 報酬 GT にその関係が無い（Mirror/Concentric/全 dimensional）＝GT 組成の問題。
- **H5 汎化失敗**: 学習分布では出るが人間データ(fiverr)に汎化しない＝OOD の問題。

処方は H1=探索強化 / H2=報酬係数 / H3=指標再設計 / H4=規則付与器 / H5=汎化、と全く違う。規則付与器に
よる pseudo-GT v2 の生成＋再学習は長時間（167k で ~5h、1.67M で ~53h）なので、**その完成を待たずに
「どの原因が効いているか」を学習なしで切り分ける**のが本節の狙い。材料は既存の v2_int チェックポイント
（run `0csbzpdm`）と fiverr 人間データと報酬コードだけ。効用は2つ — 律速を一次証拠で確定すること、そして
v2 GT 投入時に使う報酬設定を**事前に**決めて長時間生成を de-risk すること。

4本の診断と、それぞれが切り分ける仮説:

| 診断 | 何を切り分けるか | スクリプト |
|---|---|---|
| #1 サンプル vs 採択 | H1（出さない）か H2（出すが消される）か ＋ H5（汎化） | `diag_sample_vs_select.py` |
| #2 統制構成の因果測定 | H2/H3 を統制実験で因果的に（H+H vs Parallel、正解 vs 近似ミス Tangent） | `diag_reward_balance.py` |
| #3 d_H_angle 分離能 | H3 の type 別版＝「どの関係なら指標が見えるか」＝引き上げ可能な type の順位 | `diag_dhangle_separability.py` |
| #6 報酬 vs 評価 整合 | 報酬指標(motion_relational)と評価指標(response_full)の構造的ズレ | `diag_reward_target_gap.py` |

#### #1 サンプル vs 採択の分解 — 「出さない」のか「出すが消される」のか

**知りたいこと**: ある関係が生成に出ない時、原因が H1（方策がそもそもサンプルしない）か H2（サンプルは
するが報酬で淘汰される）かで打ち手が真逆になる。これを type 別に分ける。

**方法**: v2_int で各スケッチに K=8 サンプルし（学習時と同じ温度1.2・本物の training reward composer）、
各サンプルの総報酬から RLOO の **within-group advantage**（同一スケッチの K サンプル間で、その type を
**含む**サンプルの advantage − **含まない**サンプルの advantage、の平均）を出す。within-group にするのは
「図の難しさ」を相殺して**その type を入れたことの純効果**を見るため。併せて、その type を含むサンプルの
**ソルバ状態**（Conflicting/Malformed/Distorted = risky）と、**報酬項別の寄与差**（solver / intent /
rel_prior の、type を含む − 含まない の平均差）も出す。さらに学習データ(sg_pseudo_rel_167k)と人間データ
(fiverr)の両方で回し、H5（汎化）も見る。

**結果① 学習データで prompt（3072 sample / 384 group, FC 74.5%, 全体 risky 2.4%）**:

| type | samp%(出た割合) | within-grp advΔ | risky% | solver寄与差 | intent寄与差 |
|---|---|---|---|---|---|
| Coincident | 96.7 | +0.23 | 2.4 | −1.36 | +0.41 |
| Horizontal | 95.3 | **−3.61** | 2.3 | +2.36 | +0.11 |
| Vertical | 65.1 | **−2.32** | 2.9 | −1.77 | −0.22 |
| Parallel | 54.0 | +0.53 | 0.9 | +2.36 | +0.27 |
| Perpendicular | 56.1 | +1.25 | 0.9 | +2.48 | +0.28 |
| Tangent | 12.8 | +0.99 | **4.6** | **−1.88** | **+1.10** |
| Equal | 15.7 | +0.28 | 2.9 | +0.08 | +0.45 |
| Midpoint | 30.1 | +3.13 | 1.0 | – | – |

**結果② fiverr で prompt（1192 sample / 149 group, FC 60.9%, 全体 risky 2.4%）**:

| type | samp% | within-grp advΔ | risky% | solver寄与差 | intent寄与差 |
|---|---|---|---|---|---|
| Coincident | 98.7 | −0.75 | 1.9 | +7.36 | +0.94 |
| Horizontal | 91.7 | +0.05 | 2.6 | +1.83 | +0.87 |
| Vertical | 87.0 | −0.50 | 2.7 | +0.84 | +0.65 |
| Equal | 22.2 | **−1.04** | 3.8 | −1.23 | −0.19 |
| Tangent | **8.6** | +1.04 | 0.0 | **−2.79** | **−0.37** |
| Perpendicular | **7.7** | −0.30 | 0.0 | +0.78 | +0.49 |
| Parallel | **0.4** | −1.70 | 0.0 | −2.11 | −0.10 |

**どう読むか**:
- **H1 か H2 か**: 学習データでは Parallel が **54%** のサンプルに出ており（H1 ＝未サンプルではない）、within-group
  advΔ も **+0.53**（淘汰どころか正に選ばれている）。Tangent +0.99、Perp +1.25 も正。**負なのは
  Horizontal(−3.61) / Vertical(−2.32)** ——intent 報酬は「安易な H/V」を不利にし rare relation を後押し
  している。つまり**学習分布上では報酬は正しく働いており、H1 でも H2 でもない**。
- **では val でなぜ出ないか＝ H5（汎化）**: fiverr で prompt すると **Parallel 54%→0.4%**、**Perp 56%→7.7%**
  とサンプル率が崩壊する。人間 CAD 図は軸平行(H/V)が多く斜め平行が少ないため、方策がそもそも Parallel/Perp を
  **提案しなくなる**。よって「val の Parallel≈0」は報酬の淘汰ではなく**学習分布→人間図への汎化ギャップ**。
- **Tangent は例外的に汎化する**: 13%→8.6% としか落ちない。だから Tangent 過少は H1/H5 ではない。手掛かりは
  ソルバ寄与: Tangent を含むサンプルは **solver 寄与が −1.9(train)/−2.8(fiverr)** と一貫して低く（risky 率も
  train で 4.6%＝全体の約2倍）、intent は +1.1(train) で押し返すが **fiverr では intent が −0.4 と弱い**
  （gate=converged で FC の低い fiverr では発火が減る）。→ Tangent は「solver が嫌がる(ドラッグ)のを intent が
  押し返す綱引き」で、**人間図では intent が負けている** ＝ H2（報酬バランス）の一種。
- 注意: advΔ・寄与差は**相関**指標（その type を含むサンプルは他の点でも違いうる）。因果は #2 で統制して確かめる。

#### #2 統制構成での因果測定 — H+H vs Parallel と Tangent のリスク

**知りたいこと**: #1 は相関なので、「Parallel を H+H に替えたら報酬がどう動くか」「正しい接と僅かにズレた接で
報酬がどう違うか」を **同一 primitives・制約だけ差し替え**で因果的に測る。特に H3（指標が H+H と Parallel を
区別できるか）を白黒つける。

**方法**: 2本の軸平行線（H+H も Parallel も成立）や 線＋接円を手で構成し、制約セットだけ差し替えた candidate を
v2_int の本物の reward composer に通す。同時に token 非依存の `_compute_h_theta_metrics` で raw な d_H_angle
（motion_relational・leak 等）も直接計算する。

**結果 P（GT=Parallel, 軸平行2線）**:

| candidate | solver状態 | solver | intent | TOTAL | mrel | leak |
|---|---|---|---|---|---|---|
| empty（制約なし） | Under | 0.00 | 1.57 | 1.57 | 0.333 | 0.000 |
| Parallel | Under | **1.67** | **3.00** | 4.72 | 0.000 | 0.000 |
| H+H | Under | **3.33** | **3.00** | 6.42 | 0.000 | 0.000 |
| H+H+Parallel | Redundant | −2.00 | 0.00 | −2.00 | 0.000 | 0.000 |

**結果 T（GT=Tangent, 線＋接円）**:

| candidate | solver状態 | solver | intent | mrel | leak |
|---|---|---|---|---|---|
| empty（接なし） | Under | 0.00 | 1.38 | 0.378 | 0.000 |
| Tangent（正解） | Under | **2.00** | **3.00** | 0.000 | 0.000 |
| 中心-線 Coincident（代替） | **Distorted** | −0.25 | 0.00 | 0.272 | 0.000 |
| Tangent（近似ミス: 円を0.1ズラす） | **Distorted** | **−0.25** | – | – | – |

**どう読むか**:
- **H3 の決着＝「H+H と Parallel は指標上で同じ」（ユーザの主張が正しい）**: pred=H+H と pred=Parallel は、
  ソルバ報酬こそ **H+H 3.33 = 2×Parallel 1.67**（DoF を2減らす vs 1減らす）だが、**intent は両方 3.00 で
  完全一致**、raw でも `motion_relational=leak=0`。つまり **d_H_angle は H+H と Parallel を区別しない**。
  理由: 本指標は「ペア間距離で観測できる動き」を測るが、図**全体の回転**は全ペア距離を保存する
  **gauge（測れない基準自由度）**なので不可視。H+H が余分に全体回転を止めても指標は感知しない。
  → §15.2 step-3「H+H は全体回転を余計に止めるので d_H_angle が差を出せる」は**誤り**。
  - 帰結: intent は Parallel を H+H より高く評価する**手段がない**し、その**必要もない**（両者は"意図でない
    全体回転"だけが違う＝同じ設計意図）。ソルバが H+H を 2:1 で選ぶのは正しく、**Parallel が出ないのは無害**。
- **Tangent は対照的に「本物の意図」**: 正解 Tangent は solver +2.00 / intent +3.00 で最良。接を**省く**と
  intent が 3.00→1.38（−1.62）に落ちる＝指標はちゃんと「接の欠落」を罰する（leak ではなく
  motion_relational 0.378 経由）。代替の「中心を線上に Coincident」は **Distorted（solver −0.25）かつ
  mrel 0.272≠0** ＝指標も「接とは別物」と判定。→ **Tangent は幾何的に非代替で、指標もそれを見えている**。
- **ソルバ・リスクの実測**: 円を僅か(0.1)ずらして接を主張すると **Distorted → solver −0.25**。正解 +2.00 との
  **振れ幅 2.25**。これが #1 の「Tangent の solver 寄与 −1.9〜−2.8」の正体＝**不完全な人手幾何で接を主張すると
  ソルバ罰**。Coincident のような寛容な制約と違い、Tangent はハイリスク。

#### #3 d_H_angle 分離能監査 — どの関係なら指標が「見える」か

**知りたいこと**: #2 を全 type・実データへ一般化する。「ある type を入れる/外すで d_H_angle がどれだけ動くか」
＝**指標がその type を見分けられる度合い** ＝ intent 報酬がその type を引き上げ"られる"上限。これで
「追う価値のある type」を順位づける。

**方法**: fiverr 人間 GT 149図で、各 type の制約を**全部外し（残りは保持）**、外した版と full GT の d_H_angle を
測る（限界 removal-leak）。残りを保持するので「その図で他の制約が肩代わりしているか＝代替可能か」が自動的に
効く: H/V で足りる Parallel は低く出て、何も肩代わりできない Tangent は高く出る。token 非依存。

**結果（149図, rfull = response_full の平均, 高いほど load-bearing ＝ 非代替）**:

| type | 重要度 rfull | frac>.05 | mrel | 注 |
|---|---|---|---|---|
| Coincident | 0.664 | 1.00 | 0.728 | 骨格（既に豊富に生成） |
| Distance | 0.606 | 0.98 | **0.000** | dimensional（mrel=0 ＝報酬に不可視, →#6） |
| **Tangent** | **0.516** | 1.00 | 0.606 | rare-rel 最重要・非代替 |
| **Perpendicular** | **0.445** | 0.90 | 0.531 | 非代替（H+V で代替されていない） |
| Radius | 0.420 | 1.00 | 0.000 | dimensional |
| **Equal** | **0.388** | 0.94 | 0.450 | 非代替 |
| Midpoint | 0.273 | 0.90 | 0.513 | 中 |
| Parallel | 0.226 | 0.92 | 0.410 | **最下位＝代替可能＝無害** |
| Mirror | 0.205 | 0.79 | 0.414 | 中（報酬GTに無い→augmenter） |
| Vertical | 0.158 | 0.72 | 0.348 | 代替可能 |
| Horizontal | 0.118 | 0.54 | 0.247 | 最も代替可能 |

**どう読むか**:
- rare relation の重要度（＝intent が引き上げるべき優先度）は **Tangent 0.52 > Perpendicular 0.45 >
  Equal 0.39 ≫ Parallel 0.23（最下位）**。Tangent は人間 CAD で**最も load-bearing な rare relation**
  （外すと必ず悪化: frac>.05=1.00）。**ユーザの「Tangent が本丸」を定量裏付け**。
- **Parallel が最下位**＝人間の平行は多くが軸平行で H/V が肩代わり→外しても効かない＝**代替可能＝無害**
  （#2 と整合）。H/V はさらに低い。
- **Perpendicular は意外に高い(0.45)**＝人間図の直角ペアは両方が軸平行とは限らず H+V で代替されていない＝
  load-bearing。生成 0.4% vs GT 5% は実損失。→ Tangent/Equal と並ぶ本丸。
- **dimensional の伏線**: Distance 0.606 / Radius 0.420 と重要度は高いのに **mrel=0**＝これらは
  motion_relational に全く映らない（報酬が見られない）。これは #6 へ。

#### #6 報酬指標 vs 評価指標 — dimensional の盲点

**知りたいこと**: 報酬が最適化するのは `motion_relational_angle`（relational のみ）、評価の本指標は
`response_full_angle`（dimensional 込み）。この2つはどれだけズレているか＝「報酬を完璧にしても評価が
どこまで残るか」。

**方法**: fiverr 人間 GT から「dimensional を全部外した版（＝報酬が狙える最善）」と full GT の d_H_angle を
測る。token 非依存。

**結果（149図, 人間GTの dimensional 構成比 0.257）**:

| ablation | response_full | motion_full | motion_rel | leak |
|---|---|---|---|---|
| rel_only（報酬が狙える最善） | 0.707 | **1.000** | **0.000** | **0.000** |
| dim_only | 0.783 | 0.776 | 0.835 | 0.789 |

**どう読むか**:
- dimensional を全部外すと **motion_relational=0・leak=0** ＝ **報酬は「完璧」と判定**。ところが
  **motion_full=1.0（飽和=最大）・response_full=0.71**。つまり **motion_relational 報酬は dimensional の
  意図を一切見ていない**。全関係を完璧に再現する方策でも、評価指標 response_full は ≈0.71 が下限で、これ以上は
  寸法制約を入れないと縮まらないが intent はそれを誘導しない（寸法は SolverReward の DoF/FC 任せ）。
- 逆に関係を全部外す(dim_only)と motion_relational 0.835 ＝ 関係も load-bearing。両者は相補的。
- 含意: **報酬指標と評価指標が構造的にズレている**。§14.7 で intent が response_full を改善したのは間接効果。
  これは規則付与器（関係 GT の充足）では直らない設計論点で、寸法意図も狙うなら dimensional-aware な intent 項が
  別途要る（ただし空系列 hack を避ける設計が前提＝§実験ノート 失敗例2）。

#### 統合: 律速の再特定と §14.7/§15.2 の訂正

**訂正**:
- §14.7.2-6 と §15.2 step-3 は誤り。#2 の通り **d_H_angle は H+H と Parallel を区別しない**（gauge 不変）。
  「Parallel が死ぬのは指標の非分離性ゆえソルバの DoF 選好が勝つから」という因果づけは不要で、正しくは
  **Parallel 欠落は無害**（H+H と同じ意図）。
- §14.7 の「Parallel は sampled-but-selected-away」も誤り。#1 の通り **val では未サンプル（汎化崩壊）**で、
  学習分布では出て正に選ばれている。

**律速の再特定（すべて規則付与器 非依存）**:
1. **Parallel = 無害**（gauge 等価, #2/#3）。追わない。
2. **Perpendicular = 汎化(H5)**（重要だが fiverr で未サンプル, #1/#3）→ 探索強化（probe B）。
3. **Tangent / Equal = 報酬バランス(H2)**（ソルバ・ドラッグ > intent, 特に fiverr で intent が弱い,
   #1/#2/#3）→ intent 振幅増＋Tangent de-risk（probe A）。
4. **dimensional = 報酬の構造的盲点(#6)**。probe では直らない。将来 dimensional-aware な intent 項。
5. **Mirror/Concentric = GT 組成(H4)**（報酬 GT に無い）。規則付与器の担当（重要度は中: Mirror 0.21）。

**launch-ready な次実験（学習直前まで準備済・CPU smoke exit0・hydra 解決値確認済）**:
- **probe A** `configs/experiment/rloo_response_scaffold3_rewardbalance_intent.yaml`（律速3を叩く）:
  `coeff_h_angle 3→8`・`cap 0.70→0.85`、rel_prior **Tangent/Perpendicular/Equal を 0.30 へ**
  （Parallel は据え置き＝無害）、SolverReward `penalty_deformed −0.25→0`（Tangent de-risk）。
  狙い: 非代替 rare-rel を solver ドラッグを上回って引き上げられるか／FC を壊さないか。
- **probe B** `configs/experiment/rloo_response_scaffold3_explore_intent.yaml`（律速2を叩く）:
  `entropy_bonus 0.005→0.02`・`temp 1.2→1.5`（DoF 原因切り分けは `model.train_sample_temperature=1.0`
  の CLI override）。狙い: 真に entropy を上げれば Perp/Tangent の fiverr サンプリングが回復するか／FC だけ崩れるか。
- **probe A/B は v1 GT のまま据え置く**（現 pretrain・val composer は scaffold2 と同一＝eval 比較可能）。
  これは意図的: 診断より A/B の標的（H2 報酬バランス・H5 汎化）は GT 組成と独立なので、GT を同時に変えると
  「reward を変えた効果」と「GT を変えた効果」が交絡する（§11.6 の統制比較）。

**v2 GT（規則付与器 pseudo-rel）を別軸として追加 (2026-05-31)**: 規則付与器の v2 GT が完成
（`data_processed/sg_pseudo_rel_v2_no_mirror.npy` / `..._full_all.npy`、166k sketch、per-sketch 完全で
constr/sketch 19→57/70、Perp 5→23%・Parallel 9→19%・Tangent 4→6.6%・Mirror 0/19%）。診断を踏まえた判断:
- **採用する。ただし probe A/B を書き換えず、独立軸として足す**。v2 GT が直すのは A/B とは別の律速＝
  **H4（Mirror/Concentric は augmenter でしか入らない）**と **per-sketch 完全性**（Tangent/Perp/Equal の
  d_H_angle ターゲットを図ごとに正しくする＝probe A を補完）。redundant 多発（57 constr）はモーション部分空間が
  rank で吸収（inert; pipeline-safety trace で motion dim 8.5→5.5＝より完全に拘束）。Parallel 9→19% は #2 の
  gauge 等価より **inert**（害も益も無い）。→ Mirror は precision 0.006＝ほぼ FP なので **no_mirror を既定**、
  full_all は Mirror ablation。
- これで **{v1, v2 GT} × {base, reward-balance} の 2×2** が揃う（GT 効果・reward 効果・交互作用を分離可能）:

| GT \ reward | base（scaffold2 stack） | reward-balance（probe A stack） |
|---|---|---|
| **v1** | `scaffold2_intent`（実行済 `0csbzpdm`） | **probe A** `scaffold3_rewardbalance_intent` |
| **v2** | **probe C** `scaffold3_v2gt_intent` | **probe D** `scaffold3_rewardbalance_v2gt_intent` |

  - **probe C**（GT 単独効果）: scaffold2 と同一 reward stack＋v2 GT。scaffold2_intent(v1) との差＝GT 効果。
  - **probe D**（合わせ技 best-bet）: probe A stack＋v2 GT。2軸同時なので attribution 用でなく「到達可能な最良」を見る。
  - 両 config とも **val composer は scaffold2 と同一**（eval 比較可能）・CPU smoke/hydra 解決値確認済。
- 推奨実行順: まず **probe A（v1,rb）と probe C（v2,base）を並行**（reward 軸と GT 軸を各々単独で測る）→
  両方効くなら **probe D（合わせ技）**。probe B（探索）は H5 用に随時。
- 起動（各半日級・各自のタイミングで。`CUDA_VISIBLE_DEVICES` でGPU指定）:
  - probe A: `conda run -n freecad python src/train.py experiment=rloo_response_scaffold3_rewardbalance_intent`
  - probe B: `conda run -n freecad python src/train.py experiment=rloo_response_scaffold3_explore_intent`
  - probe C: `conda run -n freecad python src/train.py experiment=rloo_response_scaffold3_v2gt_intent`
  - probe D: `conda run -n freecad python src/train.py experiment=rloo_response_scaffold3_rewardbalance_v2gt_intent`
  - Mirror ablation は各 config の `data.train_sequence_file=data_processed/sg_pseudo_rel_v2_full_all.npy` 上書きで。

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

## 16. v1/v2 reward-ranking 監査と次アクションの再整理 (2026-06-01)

この章は、§15 までの計画・実装・実験を前提に、2026-06-01 時点で得られた追加監査を
自己完結にまとめる。目的は、会話履歴や暗黙知なしで次の担当者が作業を再開できるようにすること。

### 16.1 この章で扱う問い

このコードベースのタスクは、2D CAD sketch の primitives を入力し、その primitives 間に入る
constraints を推論することである。CAD では同じ形状を保つ constraint set が複数あり得るため、
単に GT と同じ constraint command を出すだけでは design intent を評価しにくい。

本研究では、constraint command の表記ゆれや冗長性を吸収するため、parameter motion subspace を
比較する `d_H_angle` / `ResponseIntentReward` を提案している。ここでの RL 補助実験の目標は、
次の形に限定する。

> relational constraints が作る関係性を、command type や冗長性に過度に依存せず
> parameter motion subspace として捉えることで、SolverReward 単体より強いrelational constraints選択信号を
> RLOO に与えられるか。

重要な注意:

- `ResponseIntentReward` は「人間がその constraint type を選ぶか」を直接誘導する報酬ではない。
  例えば `Parallel` と `Horizontal+Horizontal` が同じ motion subspace を作る場合、motion-subspace
  reward はそれらを区別しない。
- `ResponseIntentReward` を強くすると FC rate が下がること自体は、直ちに失敗ではない。
  簡単に DoF を削れる `Distance` 系 shortcut を使いにくくし、relational motion space を合わせるために
  relational constraints を多めに推論させる設定なので、問題設定は SolverReward 単体より難しい。
  また `leak_relational_angle` は雑な Distance を張る候補を罰するため、FC rate と intent 指標は
  trade-off し得る。FC rate は主指標だがこれを最重要視しないように。
- 一方で、RLOO の学習信号として使うには、同一 prompt 内の候補間で reward ranking が実際に
  変わる必要がある。以下の監査はこの点を確認するために行った。

### 16.2 参照するデータ、run、checkpoint、監査スクリプト

**GT / pseudo-GT データ**

| name | path | 内容 |
|---|---|---|
| pseudo-GT v1 | `data_processed/sg_pseudo_rel_167k_unique_valid.npy` | 旧 pseudo relational GT。モデル生成関係を元にしたもの。 |
| pseudo-GT v2 full_all | `data_processed/sg_pseudo_rel_v2_full_all.npy` | rule augmenter による full relational closure。Mirror を含む。 |
| pseudo-GT v2 no_mirror | `data_processed/sg_pseudo_rel_v2_no_mirror.npy` | full_all から Mirror を除いた variant。 |
| validation human | `data_processed/data_fiverr_ryotaro_unique_valid.npy` | validation/test に使う human annotation 側データ。 |

**主な W&B run**

| shorthand | run id | 意味 |
|---|---|---|
| `sc2_noint` | `uywjlizh` | scaffold2 without ResponseIntentReward。SolverReward 単体ではなく hygiene / dim / rel_prior を含む。 |
| `sc2_int` | `0csbzpdm` | scaffold2 + ResponseIntentReward。v1 pseudo-GT。 |
| `sc2_v2_no_mirror` | `6bshl57m` | scaffold2 + ResponseIntentReward。train GT を v2 no_mirror に差し替え。 |
| `sc2_v2_full_all` | `mkxnjtff` | scaffold2 + ResponseIntentReward。train GT を v2 full_all に差し替え。 |
| `A_rewardbalance` | `vp9z9w32` | Section 14 probe A。rewardbalance。`coeff_h_angle=8.0`, `cap_h_angle=0.85`。 |
| `B_explore` | `ypk0yp9t` | Section 14 probe B。exploration 寄り。 |
| `C_v2gt` | `4k3nn1m4` | Section 14 probe C。v2 GT 系。 |
| `D_rb_v2gt` | `wckc2b2w` | Section 14 probe D。rewardbalance + v2 GT 系。 |
| `solver_only` | `s2ibmbil` | old SolverReward-only rerun。response / cmix を十分ログしていないため直接比較には注意。 |

**ローカル checkpoint**

| checkpoint | run | 用途 |
|---|---|---|
| `logs/train/runs/2026-05-30_15-17-10/checkpoints/last.ckpt` | `sc2_int` | 旧 scaffold2 intent の ranking 監査。 |
| `logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt` | `A_rewardbalance` | 完了済み rewardbalance probe A の ranking / diversity / sample-vs-select 監査。 |

**追加した監査スクリプト**

- `scripts/diag_v1_v2_reward_ranking.py`

このスクリプトは、同一 prompt から同一 K samples を生成した後、ResponseIntentReward の
target sketch だけを v1 / v2 に差し替えて reward ranking を比較する。つまり、policy 出力の違いではなく
「target を v2 にしたとき、同じ候補群に対する RLOO ranking が変わるか」だけを見る。

主な出力:

- intent-only reward ranking の v1 vs v2 差
- full reward ranking の v1 vs v2 差
- group winner flip rate
- pairwise ranking disagreement
- within-group type-conditioned advantage
- K 内の exact sequence diversity / type-count diversity

### 16.3 W&B summary による最終点の粗い比較

2026-06-01 時点で取得した W&B summary。`resp` は
`validation_gen/intent_response_full_angle_mean`、`mrel` は
`validation_gen/intent_motion_relational_angle_mean`、`leak` は
`validation_gen/intent_leak_relational_angle_mean`。いずれも低いほどよい。
`distFam` は生成 constraint 中の distance family 比率、`keyrel` は key relational 比率。

| run | step | resp | mrel | leak | FC | DoF | distFam | keyrel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1_noint` | 14898 | 0.526 | 0.464 | 0.519 | 0.711 | 3.570 | 0.287 | 0.365 |
| `v1_int` | 14898 | 0.495 | 0.446 | 0.487 | 0.658 | 3.852 | 0.284 | 0.390 |
| `sc2_noint` | 14898 | 0.482 | 0.434 | 0.454 | 0.631 | 4.698 | 0.254 | 0.378 |
| `sc2_int` | 14898 | 0.477 | 0.426 | 0.447 | 0.604 | 4.685 | 0.247 | 0.406 |
| `sc2_v2_no_mirror` | 15588 | 0.464 | 0.421 | 0.427 | 0.638 | 4.779 | 0.246 | 0.401 |
| `sc2_v2_full_all` | 15588 | 0.465 | 0.431 | 0.444 | 0.591 | 4.732 | 0.253 | 0.410 |
| `A_rewardbalance` | 14898 | 0.469 | 0.428 | 0.435 | 0.523 | 5.584 | 0.230 | 0.448 |
| `B_explore` | 14898 | 0.535 | 0.502 | 0.281 | 0.403 | 12.987 | 0.140 | 0.550 |
| `C_v2gt` †crashed | 10799 | 0.464 | 0.427 | 0.428 | 0.497 | 5.013 | 0.242 | 0.410 |
| `D_rb_v2gt` †crashed | 11049 | 0.455 | 0.424 | 0.400 | 0.389 | 6.034 | 0.201 | 0.459 |
| `solver_only` | 16949 | n/a | n/a | n/a | 0.806 | 3.375 | n/a | n/a |

> **2026-06-01 更新**: W&B summary を再取得した。他 9 run は前回値と完全一致（変化なし）。
> `C_v2gt`(`4k3nn1m4`) と `D_rb_v2gt`(`wckc2b2w`) は †`crashed` 状態で、step 10799 / 11049 で停止。
> 他 run の 14898 に未到達のため、**この2行だけは step 非整合の暫定値**として読む。両者とも前回
> （10499 / 10749）から +300 step ほど進んだだけで crash しており、傾向（C ≈ sc2_int 系・FC 微減、
> D = reward-balance + v2 GT で FC が最低水準）は前回と不変。同一 step での確定比較が要るなら再開・完走が必要。

読み:

- `A_rewardbalance` は `sc2_int` と比べて distance family を減らし、key relational 比率を上げている。
  FC / DoF は悪化しているが、これは §16.1 の通り、relational intent を優先する問題設定では
  想定される trade-off として読む。
- `B_explore` は keyrel を大きく増やすが DoF が大きく残り、現時点では本命ではない。
- `sc2_v2_no_mirror` / `sc2_v2_full_all` は v1 scaffold2 intent より response 系がわずかに良いが、
  差は定性的主張として十分強くない。
- `solver_only` は FC / DoF は良いが、response / cmix を十分ログしていないため、
  design intent 側の直接比較としては使いにくい。

### 16.4 v1 vs v2 reward-ranking 監査

**目的**: v2 pseudo-GT が ResponseIntentReward の ranking を実際に変えるかを確認する。
ここで ranking が変わらなければ、v2 GT を使った学習で差分が出ないのは自然である。

**probe A / v2 no_mirror**

実行:

```bash
conda run -n freecad python scripts/diag_v1_v2_reward_ranking.py \
  --ckpt logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt \
  --v2-data data_processed/sg_pseudo_rel_v2_no_mirror.npy \
  --limit 128 --K 8 --batch-size 32 --device cuda
```

結果:

```text
GLOBAL: 1024 samples, 128 groups, K=8
solver status: FC 66.5%, Under 25.3%, Redundant 5.1%, Distorted 2.6%
diversity: mean unique seq/group=8.00, all-identical groups=0.0%
type-count diversity: mean unique type-count/group=3.66, all-identical groups=26.6%

INTENT ONLY:
  mean v1=2.9626, v2=3.5996, delta=+0.6370
  centered corr=0.7491
  pairwise ranking disagreement=22.47%
  winner flip rate=19.53%
  advantage std v1=0.6808, v2=0.7183

TOTAL reward:
  centered corr=0.9902
  pairwise ranking disagreement=10.11%
  winner flip rate=11.72%
  advantage std v1=3.5150, v2=3.5557
```

**probe A / v2 full_all**

実行:

```bash
conda run -n freecad python scripts/diag_v1_v2_reward_ranking.py \
  --ckpt logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt \
  --v2-data data_processed/sg_pseudo_rel_v2_full_all.npy \
  --limit 128 --K 8 --batch-size 32 --device cuda
```

結果:

```text
GLOBAL: 1024 samples, 128 groups, K=8
solver status: FC 66.5%, Under 25.4%, Redundant 5.1%, Distorted 2.5%
diversity: mean unique seq/group=8.00, all-identical groups=0.0%
type-count diversity: mean unique type-count/group=3.66, all-identical groups=26.6%

INTENT ONLY:
  mean v1=2.9661, v2=3.5416, delta=+0.5755
  centered corr=0.7842
  pairwise ranking disagreement=17.44%
  winner flip rate=19.53%
  advantage std v1=0.6734, v2=0.7195

TOTAL reward:
  centered corr=0.9916
  pairwise ranking disagreement=7.93%
  winner flip rate=10.94%
  advantage std v1=3.5078, v2=3.5481
```

解釈:

- v2 pseudo-GT は ResponseIntentReward 単体では効いている。intent-only winner flip が約 19.5% あり、
  v1/v2 target の違いは ranking に現れている。
- しかし full reward では winner flip が約 11% まで落ちる。SolverReward / dim / rel_prior を含む
  総報酬の分散が大きく、ResponseIntentReward の差分が RLOO の実際の選択圧として薄まっている。
- no_mirror と full_all の差はこの監査では大きくない。生成候補側に `Mirror` がほぼ出ていないため、
  full_all の Mirror closure は immediate な ranking 差としては見えにくい。

### 16.5 K 内 diversity 監査

ユーザ懸念: RLOO 後は entropy が潰れており、K=8 や K=128 でサンプルしても差分が出ないのではないか。

確認結果:

```bash
# train temperature 1.2
conda run -n freecad python scripts/diag_v1_v2_reward_ranking.py \
  --ckpt logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt \
  --limit 64 --K 128 --batch-size 8 --device cuda --diversity-only
```

```text
GLOBAL: 8192 samples, 64 groups, K=128
mean unique seq/group=127.94
all-identical groups=0.0%
mean top-duplicate frac=0.008
mean unique type-count/group=22.83
type-count all-identical groups=12.5%
```

```bash
# validation temperature 1.0
conda run -n freecad python scripts/diag_v1_v2_reward_ranking.py \
  --ckpt logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt \
  --limit 32 --K 128 --batch-size 8 --device cuda --temperature 1.0 --diversity-only
```

```text
GLOBAL: 4096 samples, 32 groups, K=128
mean unique seq/group=127.44
all-identical groups=0.0%
mean top-duplicate frac=0.009
mean unique type-count/group=12.22
type-count all-identical groups=9.4%
```

解釈:

- exact token sequence level では collapse は主要因ではない。
- ただし type-count level の多様性は exact sequence より狭い。つまり token-level ノイズは残っているが、
  semantic action diversity はそれほど広くない。
- したがって「K を増やしても全く同じ出力しか出ないから ranking 監査が無意味」という状態ではない。
  ただし、重要な relation type について同一 prompt 内の有無差が出るかは別問題で、§16.7 がそれを扱う。

### 16.6 統制候補での報酬分解: Parallel vs H+H

実行:

```bash
conda run -n freecad python scripts/diag_reward_balance.py \
  --ckpt logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt \
  --device cuda
```

`P:axis-parallel-pair (GT=Parallel)` の結果:

| candidate | solver status | solver | intent | rel_prior | total | mrel | leak |
|---|---|---:|---:|---:|---:|---:|---:|
| empty | Under | 0.00 | 4.86 | 0.00 | 4.86 | 0.333 | 0.000 |
| Parallel | Under | 1.67 | 8.00 | 0.05 | 9.72 | 0.000 | 0.000 |
| H+H | Under | 3.33 | 8.00 | 0.08 | 11.42 | 0.000 | 0.000 |
| H+H+Parallel | Redundant | -2.00 | 0.00 | 0.00 | -2.00 | 0.000 | 0.000 |

解釈:

- `Parallel` と `H+H` は ResponseIntentReward 上では同点である。これは `motion_relational_angle` が
  command type ではなく motion subspace を見ているためで、設計上自然な挙動である。
- SolverReward は DoF を多く削る `H+H` を `Parallel` より高く評価する。
- 現行 `rel_prior` も `Horizontal` / `Vertical` を key relation として扱うため、少なくともこの統制例では
  `H+H` をさらに少し有利にする。
- よって「Parallel command を H+H より選ばせる」ことは、motion-subspace reward 単体からは出てこない。
  これは `d_H_angle` の失敗ではなく、不変性を持つ metric と command-type preference を混同している。
  command-type 誘導は別報酬または canonicalization として分離する必要がある。

`T:line-circle-tangent (GT=Tangent)` の結果:

| candidate | solver status | solver | intent | rel_prior | total | mrel | leak |
|---|---|---:|---:|---:|---:|---:|---:|
| empty | Under | 0.00 | 4.44 | 0.00 | 4.44 | 0.378 | 0.000 |
| Tangent(correct) | Under | 2.00 | 8.00 | 0.30 | 10.30 | 0.000 | 0.000 |
| Coincident.center-on-line substitute | Distorted | 0.00 | 0.00 | 0.00 | 0.00 | 0.272 | 0.000 |

ここでは正しい Tangent は reward 上で明確に勝つ。つまり `ResponseIntentReward` が常に無力なのではない。
問題は、実 rollout でその比較候補が同一 prompt 内に現れるかどうかである。

### 16.7 actual rollout の sample-vs-select 監査

実行:

```bash
conda run -n freecad python scripts/diag_sample_vs_select.py \
  --ckpt logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt \
  --data data_processed/sg_pseudo_rel_167k_unique_valid.npy \
  --limit 128 --K 8 --batch_size 32 --device cuda
```

結果の要点:

```text
GLOBAL: 1024 samples, 128 groups, K=8
solver status: FC 66.5%, Under 25.4%, Redundant 5.1%, Distorted 2.5%

type            samp%   grp%   advΔ_wg  ngrp
Tangent         13.28   13.3       nan     0
Parallel        53.91   53.9       nan     0
Perpendicular   69.73   71.1      1.22     3
Equal           25.59   28.1      1.07     8
Coincident      95.41   96.1     -9.78     2
Horizontal      95.21   96.9      1.00     6
Vertical        59.67   61.7     -0.21     4
```

`ngrp` は「同一 RLOO group 内で、その type を含む候補と含まない候補の両方が存在した group 数」。

解釈:

- Tangent と Parallel はサンプル自体には出ている。しかし今回の 128 prompts では、同一 group 内で
  `あり` と `なし` が比較できる prompt が 0 件だった。
- つまり RLOO は Tangent/Parallel の有無を選ぶ advantage をほぼ受け取っていない。
  出る prompt では全候補に出る、出ない prompt では全候補に出ない、という構造に近い。
- `Perpendicular` / `Equal` は少数ながら within-group contrast があり、advantage も正。
  ここは reward が比較可能な形で効き得る。
- `Coincident` はほぼ全サンプルに出るため、少数の contrast group では強く負に見える。
  これは Coincident 自体が悪いというより、ほぼ常時出る token の group 内比較が局所的であることに注意。

この監査から見える主な律速は、reward 値の良し悪しだけでなく、**on-policy RLOO の候補集合に
意味のある counterfactual が入っていないこと**である。

### 16.8 更新された次手

この小節が §15.6 の更新版である。次の担当者はこの順番で進める。

1. **FC rate だけで打ち切り判断しない**
   - `ResponseIntentReward` を強めると、SolverReward 単体が使いやすい `Distance` shortcut を避けるため
     FC / DoF は悪くなり得る。
   - 判断の主指標は、`intent_response_full_angle_mean`, `intent_motion_relational_angle_mean`,
     `intent_leak_relational_angle_mean`, `eval_distance_family_ratio_mean`,
     `eval_key_rel_ratio_mean`, qualitative samples とする。
   - FC / DoF は「どれだけ難しい制約集合を作れているか」の指標として併記する。これを主指標にするとSolverReward単体が自明に勝つため、副指標とまで言うのは避けたいが重要度は若干落とす。

2. **v2 pseudo-GT 仮説は破棄しない**
   - §16.4 により、v2 target は intent-only reward ranking を約 18-20% の winner flip で変えている。
   - 学習ログで差が薄い理由は、v2 target が完全に無効だからではなく、full reward 内で選択圧が薄まることと、
     actual rollout 内で key relation の counterfactual が不足すること。
   - 今後の報酬変更後も、`scripts/diag_v1_v2_reward_ranking.py` を acceptance test として使う。
     目標は total reward 側の winner flip / pairwise disagreement を増やし、intent 差分が
     RLOO advantage に乗る状態を作ること。

3. **dense / block-level reward を別タスクとして実装する**
   - 担当: 別エージェント。
   - 主な参照先: `src/models/rewards/response_metric_reward.py`,
     `src/models/rewards/utils/geometry.py`, `src/models/constraint_ar_module_rloo.py`,
     `scripts/diag_sample_vs_select.py`。
   - 背景: sequence-level STOP reward だけでは、Tangent/Parallel のような relation type が
     group 内 counterfactual を作らず、RLOO が「どの成分が足りないか」を学びにくい。
   - 要求: constraint block 追加ごと、または prefix ごとに、motion subspace / relation rowspace の
     不足成分が改善したかを返す dense reward を検討する。
   - 実装後の監査: §16.7 と同じ sample-vs-select を走らせ、Tangent/Parallel/Perpendicular/Equal の
     `ngrp` と `advΔ_wg` が増えるかを見る。

4. **motion-subspace reward と command-type 誘導を分離する**
   - 担当: 別エージェント。
   - 主な参照先: `src/models/rewards/response_metric_reward.py`,
     `src/models/rewards/relational_class_prior.py`, `scripts/diag_reward_balance.py`。
   - 背景: §16.6 の通り、`motion_relational_angle` は `Parallel` と `H+H` が同じ motion subspace を
     作る場合に同点にする。これは metric の狙い通りだが、`Parallel` command を明示的に好む
     preference ではない。
   - 要求: command-type preference が必要なら、motion-subspace reward ではなく別の canonicalization /
     type prior / H/V downweight / relation-specific preference reward として扱う。
   - 注意: この追加報酬は `d_H_angle` の主張と混ぜない。`d_H_angle` は冗長性を吸収する evaluator /
     subspace reward、command-type 誘導は human-readable command prior として別名で評価する。
   - （追記コメント：これはRelationalClassPriorRewardですでに分離できているのでは？）

5. **`RelationalClassPriorReward` を見直す**
   - 担当: 別エージェント。
   - 主な参照先: `src/models/rewards/relational_class_prior.py`,
     `src/metrics/constraint_subtypes.py`, `scripts/derive_relational_class_prior_coeffs.py`,
     `scripts/diag_reward_balance.py`。
   - 背景: §16.6 の統制例では、現行 rel_prior が `Parallel` より `H+H` を少し有利にしている。
     `Horizontal` / `Vertical` は relation token ではあるが、`Parallel` / `Perpendicular` の
     command-level 代替として働き得る。
   - 要求: H/V と key binary relation を同じ向きに boost する現設計を見直す。少なくとも
     `Horizontal/Vertical` の係数、cap、subtype/key-rel 集計上の扱いを再検討する。
   - 実装後の監査: `diag_reward_balance.py` の `P:axis-parallel-pair` で `H+H` が rel_prior により
     さらに有利にならないことを確認する。

6. **短期的には blind hyperparameter sweep を増やさない**
   - 現在の結果は「coeff を少し変えれば解ける」というより、reward placement と candidate contrast の問題を示している。
   - 係数調整をする場合も、まず §16.4 / §16.7 の監査で total reward ranking と within-group contrast が
     どう変わるかを確認してから full training に回す。

7. **ResponseIntent-only run は低優先度の診断に留める**
   - 目的: SolverReward なしで `intent_response_full_angle` / `motion_relational_angle` がどこまで改善するかを見る。
   - 期待: FC rate はかなり低くなる可能性が高い。これは本命実験ではなく、Distance shortcut を抑えたときの
     intent reward の単独挙動を見る診断である。
   - 実行するなら短い canary に留め、定性的主張の中心に置かない。
   - 追記：これをwarm-upとして、RLOOの途中からSolverRewardを加える運用にするのはあり？

8. **次に比較する run の最小セット**
   - dense / block-level reward と rel_prior 見直しが入るまでは、大規模な新規 run を増やさない。
   - 変更後に走らせる最小セット:
     1. `scaffold revised` without ResponseIntentReward
     2. `scaffold revised + ResponseIntentReward`
     3. 必要なら `scaffold revised + shuffled ResponseIntentReward`
   - すべて同一 train data / checkpoint / temperature / entropy / KL / SFT で揃える。
   - step が揃っていない途中 run の比較では、必ず `trainer/global_step` で同一 step または近傍 step を取る。

9. **この章の監査を再実行するための最小コマンド**

```bash
# v1 vs v2 reward ranking
conda run -n freecad python scripts/diag_v1_v2_reward_ranking.py \
  --ckpt <checkpoint> \
  --v2-data data_processed/sg_pseudo_rel_v2_no_mirror.npy \
  --limit 128 --K 8 --batch-size 32 --device cuda

# K diversity only
conda run -n freecad python scripts/diag_v1_v2_reward_ranking.py \
  --ckpt <checkpoint> \
  --limit 64 --K 128 --batch-size 8 --device cuda --diversity-only

# controlled reward balance
conda run -n freecad python scripts/diag_reward_balance.py \
  --ckpt <checkpoint> --device cuda

# sample-vs-select
conda run -n freecad python scripts/diag_sample_vs_select.py \
  --ckpt <checkpoint> \
  --data data_processed/sg_pseudo_rel_167k_unique_valid.npy \
  --limit 128 --K 8 --batch_size 32 --device cuda
```

### 16.9 律速の確定: counterfactual 生成（flip）と報酬バランスは相補的 (2026-06-01)

§16.4–§16.7 は「intent 報酬の within-group advantage 分散が小さい」「同一 group 内に
key relation の counterfactual（あり/なし両方）が無い」を別々に観察していた。本節は両者が
**同一現象**であることを実証し、それを解く2レバー（入力 flip による counterfactual 生成と、
SolverReward 圧縮による報酬バランス）が**相補的**で、どちらも単独では不足することを、いずれも
**学習なし**の監査で確定する。checkpoint は `A_rewardbalance`(`vp9z9w32`,
`logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt`, coeff_h_angle=8)。

**新規実装**（反射の責務は2層に分離。`sketch_utils` = object 空間、`flip_augment` = token 空間）:
- `src.sketch_handlers.sketch_utils.reflect_sketch(sketch, axis)`: `datalib.Sketch` の軸平行反射
  （x/y 座標符号反転）。等長変換（pairwise 距離保存。arc 含む 600 sketch で誤差 2.7e-15・involution
  を検証、`scripts/verify_reflect_sketch.py`）。relational/H/V は反射不変、constraint トークンは座標
  フリー（entity index 参照）かつ entity 順序は反射で不変なので、**反射入力で条件付け生成した制約
  系列はそのまま元スケッチで有効・採点可能**。
- `src.sketch_handlers.flip_augment`: token 空間の flip 拡張（**訓練ループと診断の単一の真実**）。
  `reflect_primitive_value_tokens`（量子化グリッドが対称なので bin q→num_bins-1-q が厳密。座標
  VALUE トークンのみ・軸別に反転。noise 保存。arc は点エンコードなので frame/clockwise 不要。
  object 版と非arc で ≤1bin＝sub量子化一致を検証）、`flip_view_schedule`（K を view に均等分割）、
  `reflected_expanded_pval`（K 展開済み `[B*K]` に per-sample 反射。単体テスト `tests/test_flip_augment.py`）。
- `scripts/diag_sample_vs_select.py --flip-axes x,y`: 各 prompt を orig + 反射 view から K ずつ
  サンプルし（反射は上記 token 版＝訓練と同一機構、反射 view も**元 primitives/GT で採点**）、
  1 つの RLOO group に pool。orig-only slice と pooled slice の ngrp/advΔ を比較。
  （注: §16.9.1 の数値は当初 object 版反射で取得したが、token 版へ統一後も差は sub量子化（≤1bin の
  条件付け差）で ngrp の定性は不変。）
- `scripts/diag_reward_rescale.py`: 固定 ckpt の K rollout を再採点し、SolverReward 寄与を係数
  α で縮小（`total_α = total − (1−α)·solver`、再 solve 不要）。winner-flip・intent/solver advantage
  比・**winner サンプルの key-rel / distance-family 比**・type 別 within-group advantage を α 掃引。

**16.9.1 flip-augment counterfactual 診断（Phase 0）**

同一予算（24 sample/group）での **ngrp**（同一 group 内に「型あり」と「型なし」が共存した group 数
＝ RLOO が選好を学べる counterfactual の数）:

| type | fiverr orig K=8 | fiverr plain K=24 | **fiverr flip(orig+x+y) 24** | v1 orig K=8 | v1 plain K=24 | **v1 flip 24** |
|---|---:|---:|---:|---:|---:|---:|
| Tangent | 11 | 11 | **21** | 1 | 1 | **5** |
| Perpendicular | 15 | 17 | **56** | 3 | 7 | **18** |
| Equal | 16 | 21 | 25 | 5 | 12 | 19 |
| Parallel | 5 | 13 | 14 | 2 | 3 | 2 |

読み:
- **plain に K を増やしても Tangent/Perp の counterfactual はほぼ増えない**（fiverr Tangent 11→11,
  Perp 15→17; v1 Tangent 1→1, Perp 3→7）。これらの型は prompt の向きに対し方策がほぼ決定論的
  （all-or-none per prompt）なので、同じ向きで何度サンプルしても一致する。§16.7 の `ngrp=0` 構造の正体。
- **flip は同予算で counterfactual を 2〜6× に増やす**（fiverr Perp 15→56, Tangent 11→21; v1 Perp 3→18,
  Tangent 1→5）。反射が方策の向き依存バイアスを破り、view ごとに rare relation の有無が割れるため。
  §14.7.5 #3 の最重要・非代替型（Tangent 0.52 / Perp 0.45）こそ、flip でしか届かない。
- flip は **FC を壊さない**（v1 FC 68%, fiverr 51%, risky ~3%。B_explore の温度↑が DoF を 13 まで
  壊したのと対照的）。各 sample は正常温度・有効入力に対する通常生成なので品質が落ちない。
- counterfactual が出来た上での within-group advΔ: **Tangent は flip 上で正**（fiverr +1.26, v1 +4.06;
  orig-only は contrast 群僅少でノイズ）。**Perp は v1 flip で +1.00 に転じるが fiverr flip では −0.31**、
  **Equal は両者で ~0/負**。→ Tangent は現報酬で既に選好されるが、Perp/Equal は counterfactual だけでは
  選好に届かず**報酬バランスが要る**。

**16.9.2 報酬バランス（SolverReward 圧縮）診断（Phase 1a）**

mean seq reward は solver 支配（v1: total 15.7 / solver 12.1 / intent 3.0; fiverr: 15.3 / 10.9 / 3.8）。
α 掃引（intent advΔ std は固定、solver advΔ std が α 倍）:

| α | v1 winflip | v1 int/sol | v1 win_keyrel | fiverr winflip | fiverr int/sol | fiverr win_keyrel |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.000 | 0.23 | 0.503 | 0.000 | 0.29 | 0.435 |
| 0.5 | 0.023 | 0.46 | 0.504 | 0.047 | 0.58 | 0.438 |
| 0.3 | 0.023 | 0.77 | 0.506 | 0.060 | 0.97 | 0.437 |
| 0.2 | 0.023 | 1.16 | 0.506 | 0.060 | 1.46 | 0.437 |
| 0.1 | 0.102 | 2.32 | 0.510 | 0.128 | 2.92 | 0.439 |

type 別 within-group advΔ（α=1 → 0.1）: fiverr **Perpendicular 0.001 → +0.49**, Tangent 0.56→0.58(既正),
Equal 0.50→0.47(既正); v1 **Equal 0.064 → +0.73**, Tangent/Perp は contrast 群僅少(ngrp 2-3)でノイズ。

読み:
- solver 圧縮で intent/solver advantage 比は上がる（0.23→2.32 / 0.29→2.92）が、**winner-flip は α=0.1 で
  やっと 10–13%、winner サンプルの key-rel 比はほぼ不動**（v1 0.503→0.510, fiverr 0.435→0.439）。
  ＝ **群内に「より関係的な代替候補」が無ければ、報酬を再重み付けしても選ぶものが無い**。especially v1 は
  ngrp が小さく（Tangent 2 / Perp 3 / Equal 6）、rebalance 単独では動かしようがない。
- ただし **type 別 within-group advΔ は Perp(fiverr) +0.49 / Equal(v1) +0.73 と改善**。RLOO 勾配は winner
  だけでなく全 K 標本の advantage を使うので、これは「counterfactual が在る型」に対しては訓練に効く信号。
- ＝ **rebalance は counterfactual がある所でだけ効く**。fiverr で Perp が効いて v1 で効かないのは、
  fiverr の Perp counterfactual が多い（ngrp 15）から。

**16.9.3 統一結論と次実験**

- §16.4 の「希釈」と §16.7 の「counterfactual 欠如」は同一現象。intent が advantage を作れるのは group 内で
  relational 構造が割れる時だけで、rare relation はほぼ all-or-none なので intent 寄与は係数に依らず near-zero。
  → **coeff_h_angle 増（probe A の 8）も dense/block 報酬も、この near-zero 信号を scale/再配分するだけで
  根本（group 内 counterfactual の不在）を直さない**。
- **2レバーは相補的で、どちらも単独では不足**:
  - **flip-augment**（入力対称変換）＝ counterfactual を作る唯一の安価な手段（plain bigK も rebalance も作れない）。
    FC 非破壊。Tangent は flip だけで選好に届く。
  - **SolverReward 圧縮**＝ counterfactual の上で Perp/Equal を選好に変える。flip が無いと作用対象が無い。
- **提案する次の学習実験（Phase 1b/2、要 GPU・半日）**: flip-augment rollout（orig+x+y を 1 RLOO group に
  pool）＋ **中程度の solver 圧縮**（`reward_dof_reduction 10→3–5`, `reward_fully_constrained 5→1.5–2.5`
  ＝ α≈0.3–0.5。α=0.1 は FC 崩壊リスク）。intent は据え置き。GT/pretrain/temperature は scaffold2 と同一に固定
  （flip と solver スケール以外を動かさない統制）。判定: §16.3 指標 ＋ §16.9.1 の ngrp 再測（Tangent/Perp/Equal
  が収束時 >0 を維持するか）＋ FC を guardrail。RLOO の flip pool は対称性により baseline 不偏（reward も反射不変）。
- block 報酬は保留（§16.8 #3）。`coeff_h_angle` 増は単独レバーとしては非推奨（上記理由）。

**16.9.4 実行スクリプトと追跡（launch & track）**

§16.9.3 の次実験を回すための具体コマンド。config は実装済（`scaffold4`=rebalance単独 / `scaffold5`=flip+rebalance）。
flip は `model.flip_augment_axes`、view 分割は `group_sample_size`（scaffold5 は 9＝3view×3）で制御。GT/pretrain/temperature/
val composer は scaffold2 と同一固定（flip と solver スケール以外を動かさない統制）。

attribution の 2×2（flip 軸 × solver 圧縮軸。**neither = scaffold2_intent `0csbzpdm` が既存ベースライン**）:

| | solver 10/5（無圧縮） | solver 4/2（圧縮） |
|---|---|---|
| **flip 無し** | `scaffold2_intent`（実行済 `0csbzpdm`） | **`scaffold4`**（rebalance単独, Phase 1b） |
| **flip [x,y]** | **flip単独**（下記 CLI override） | **`scaffold5`**（flip+rebalance, Phase 2 本命） |

```bash
# --- CPU smoke（起動前の健全性確認。各 config）---
conda run -n freecad python src/train.py experiment=rloo_response_scaffold5_flip_rebalance_intent \
  trainer.accelerator=cpu trainer.devices=1 trainer.strategy=auto trainer.precision=32 \
  data.batch_size=1 model.group_sample_size=3 \
  +trainer.max_steps=2 +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0

# --- 本番起動（各半日級。GPU は CUDA_VISIBLE_DEVICES で指定）---
# Phase 1b: rebalance 単独（別マシン可・コード非依存）
CUDA_VISIBLE_DEVICES=0 conda run -n freecad python src/train.py \
  experiment=rloo_response_scaffold4_solvercompress_intent

# Phase 2: flip + rebalance（本命）
CUDA_VISIBLE_DEVICES=1 conda run -n freecad python src/train.py \
  experiment=rloo_response_scaffold5_flip_rebalance_intent

# flip 単独（attribution 用。scaffold5 の solver を無圧縮 10/5 に戻す）
CUDA_VISIBLE_DEVICES=1 conda run -n freecad python src/train.py \
  experiment=rloo_response_scaffold5_flip_rebalance_intent \
  model.reward_composer.rewards.0.reward_dof_reduction=10.0 \
  model.reward_composer.rewards.0.reward_fully_constrained=5.0

# セッションを閉じても継続したい場合（§15.5 と同じ detached 起動）
setsid bash -c 'CUDA_VISIBLE_DEVICES=1 conda run -n freecad python src/train.py \
  experiment=rloo_response_scaffold5_flip_rebalance_intent' < /dev/null \
  > /tmp/scaffold5.log 2>&1 & disown
```

完走後の判定・追跡（§16.9.3 の判定基準を実コマンド化）:

```bash
# 直近 run の last.ckpt を取得（必要なら手動でパス指定）
CKPT=$(ls -td logs/train/runs/*/checkpoints/last.ckpt | head -1); echo "$CKPT"

# (1) flip が収束時も counterfactual を維持しているか（ngrp 再測。§16.9.1 と同形式）
CUDA_VISIBLE_DEVICES=1 conda run -n freecad python scripts/diag_sample_vs_select.py \
  --ckpt "$CKPT" --data data_processed/data_fiverr_ryotaro_unique_valid.npy \
  --limit 0 --K 8 --batch_size 16 --flip-axes x,y --device cuda \
  --focus Tangent,Perpendicular,Equal,Parallel

# (2) solver 圧縮が intent の選択圧シェアを上げたか（§16.9.2 の α 掃引を ckpt 上で再確認）
CUDA_VISIBLE_DEVICES=1 conda run -n freecad python scripts/diag_reward_rescale.py \
  --ckpt "$CKPT" --data data_processed/sg_pseudo_rel_167k_unique_valid.npy --limit 128
```

W&B 主指標（**`scaffold2_intent`=`0csbzpdm` と同一 step で比較**。§16.3 と同じ列）:
`intent_response_full_angle_mean`/`mrel`/`leak` ↓、`eval_key_rel_ratio_mean` ↑、`eval_distance_family_ratio_mean` ↓。
**FC を guardrail に**: `solver_fc_rate` が ~0.4 を割る／`solver_valid_dof_mean` が暴れる場合は solver 係数を 6/3 へ戻す。
判定の優先は §16.8 #1（FC を主指標にしない）に従う。collapse 兆候（`[START][STOP]` 多発・FC 0 付近張り付き）が出たら中止。

### 16.10 scaffold4/5 完走後の再整理: Distance-family 削減を主目的に戻す (2026-06-02)

**16.10.1 現状観察**

本節では目的を明確化する。最重要目標は「relational diversity 一般」ではなく、
**もっと relational を出し、相対的に Distance-family（Distance/DistanceX/DistanceY）を減らすこと**。
この改善が `motion_relational_angle` 系の motion-subspace 報酬に由来すると示せることが、研究上の主張として重要。

完走済み run:

- `scaffold4` = `rloo_response_scaffold4_solvercompress_intent`
  (`logs/train/runs/2026-06-01_19-07-33/checkpoints/last.ckpt`)
- `scaffold5` = `rloo_response_scaffold5_flip_rebalance_intent`
  (`logs/train/runs/2026-06-01_19-04-12/checkpoints/last.ckpt`)

W&B summary 上の同一 step 比較では、`scaffold5` は `scaffold2_intent` より
`intent_response_full_angle_mean` / `motion_full` / `motion_relational` /
`leak` が低く、`eval_distance_family_ratio_mean` も低い
（例: `scaffold2_intent` distFam 0.247 → `scaffold5` 0.222）。
一方、`probeD` は distFam 0.197 / keyrel 0.460 と最も強く関係寄りだが、
FC が低く DoF が高い partial/crashed run なので、本命というより「関係寄りに振ると距離は減る」
ことを示す参考点として扱う。

定性的には、`scaffold5` checkpoint は従来より Distance 過剰が抑えられている。
ただし、まだ Distance-family は多く、残る Distance の張られ方が非直感的な場合がある。
また `Radius` は Distance-family には含めないが、circle/arc の寸法固定として残りやすく、
「relational で表したい構造を dimensional で閉じている」印象を強めることがある。

`diag_sample_vs_select.py` の full fiverr + flip pooling 診断では、`scaffold5` の
within-group advantage は次の傾向:

| type | `scaffold5` advΔ_wg | 読み |
|---|---:|---|
| Distance | -0.18 | 現報酬は Distance 本体を局所的に選びにくくしている |
| DistanceX | +0.03 | ほぼ中立 |
| DistanceY | -0.02 | ほぼ中立 |
| Radius | -0.06 | 弱い負、ただし distFam ではない |
| Equal | +0.22 | やや正 |
| Tangent | +1.33 | 強く正 |
| Perpendicular | +0.00 | 中立 |
| Parallel | -0.41 | 負、かつ稀 |

対照的に `scaffold4` は Distance が +0.14、DistanceX/Y が負、Radius が +0.17。
つまり `scaffold5` では **Distance 本体を減らす選択圧は見えるが、DistanceX/Y まで一貫して
減らす報酬にはなっていない**。この差の完全な原因は未確定だが、現時点では
policy prior、solver 上の軸方向距離の安定性、relationals 後に残る DoF の種類が混ざっている可能性が高い。
この原因追跡は有用だが、次手を止めるほどの最優先ではない。

`diag_reward_rescale.py` の α 掃引では、`scaffold5` に対して solver contribution を
post-hoc に強く縮めても winner の distFam は 0.248 → 0.236 程度、keyrel は
0.498 → 0.507 程度しか動かない。したがって **solver 係数をさらに縮めるだけでは低 ROI**。
counterfactual 候補に relationals が出ていない／出ても motion 報酬が十分に採択していない箇所へ
作用する必要がある。

**16.10.2 解釈**

`motion_relational_angle` は Pred/GT の relational-only motion subspace を比較する。
dimensionals の placement/value は直接見ないため、Distance-family を減らす作用は
「relational を増やすことで、solver が dimensionals に頼る必要を下げる」という間接経路になる。

現在の観察はこの経路と整合する:

- `probeD` と `scaffold5` は、関係寄りに振るほど distFam が下がる。
- `scaffold5` は qualitative に Distance 過剰を抑える。
- ただし DistanceX/Y にはほぼ中立で、Distance-family 全体を十分には押し切れていない。
- scalar の response/motion/leak 指標は改善しても、非直感的な Distance placement までは十分に反映しない。

したがって、次に検証すべき主張は
**「motion-subspace 報酬をより relation不足に敏感な形にすると、Distance-family 全体がさらに下がり、
key relational が増える」**である。

**16.10.3 次に試す施策**

> **注（2026-06-02 追記・重要）**: 本節の `mixed_subset` 中心の提案は、後続の §16.11 Phase 0 診断（学習なし）で
> **改訂された**。**最終決定は §16.11.4**（本線＝v2 GT × flip × rebalance × `symmetric`。`mixed_subset` 単独強化は
> 予備に降格）。本節は提案の記録として残す。`mixed_subset` と `dimensional row-space overlap` の定義（§16.11 でも参照）は
> ここに置く。

第一候補は `scaffold5` を土台にした `motion_relational_angle` の強化版。

- `coeff_h_angle: 3.0 → 8.0`
- `cap_h_angle: 0.70 → 0.85`
- `motion_relational_angle_mode: symmetric → mixed_subset`
- `motion_relational_reverse_weight: 0.1` または `0.2`

`mixed_subset` は実装済みで、

```text
angle^2 = pred_subset_gt^2 + reverse_weight * gt_subset_pred^2
```

を使う。ここで `pred_subset_gt` は「Pred の relational-only motion が GT に含まれない方向へ
はみ出すこと」を罰する項。目的は、GT の rule-based relation set と完全対称に一致させることではなく、
**Pred が relation 不足により余計な運動を許すことを主に罰する**こと。
`symmetric` よりこの目的に合う可能性が高い。

`response_relational_angle` への単純置換は本命にしない。
これは `motion_relational_angle` と `leak_relational_angle` の RMS だが、
`leak` は Pred の dimensional response map `R_pred` が GT relational row-space へ漏れるかを見る。
no-dimensional/強い dimensional closure で小さくなり得るため、relational を増やす圧としては
`motion_relational_angle` より危うい。試すなら短い canary か、小係数の補助項に限る。

補助候補として `relation-only Pred leak` は残す。
現行の `leak_relational_angle` は

```text
leak^2 = || orth(row(J_R_GT))^T orth(range(R_pred)) ||_F^2 / dim(range(R_pred))
```

で、`R_pred` は Pred の dimensional parameter を微小編集したときの primitive 応答。
これに対して `relation-only Pred leak` は、`R_pred` の代わりに Pred relationals だけが許す motion

```text
N_R_pred = null(J_R_pred)
```

を使う:

```text
rel_only_leak^2 =
  || orth(row(J_R_GT))^T orth(N_R_pred) ||_F^2 / dim(N_R_pred)
```

直感的には「Pred が入れた relationals だけでは、GT relation を壊す運動をまだ許していないか」を測る。
dimensionals を増やしても改善しないため、現行 leak より relational 増加圧として安全。
ただしこれは `motion_relational_angle` の `pred_subset_gt` / `mixed_subset` と近い一方向の
relation-deficiency 信号なので、まずは実装済みの `mixed_subset` を試す。
`mixed_subset` でも relation不足が十分に拾えない場合、または leak 型の補助項が必要になった場合に
`relation-only Pred leak` を実装候補に戻す。

最低限追加で見る診断:

- per-type head ratio:
  `Distance`, `DistanceX`, `DistanceY`, `Radius`, `Equal`, `Tangent`,
  `Parallel`, `Perpendicular` を baseline / probeD / scaffold4 / scaffold5 / 次 run で比較する。
- `diag_sample_vs_select.py`:
  focus を `Distance,DistanceX,DistanceY,Radius,Equal,Tangent,Perpendicular,Parallel`
  にして、type ごとの局所選択圧を確認する。
- qualitative:
  Distance 数だけでなく、遠距離 primitive 間の非直感的 Distance が減っているかを固定 validation subset で見る。

`dimensional row-space overlap` は保留だが、定義は次の通り。
GT relational のヤコビアンを `J_R_GT`、Pred の特定 dimensional family
（例: Distance/DistanceX/DistanceY、または Radius を含めた family）のヤコビアンを `J_D_pred` とする。
同じ geometry 上で

```text
U_R = orth(row(J_R_GT))
U_D = orth(row(J_D_pred))
```

を作り、例えば

```text
overlap_recall    = || U_R^T U_D ||_F^2 / rank(U_R)
overlap_precision = || U_R^T U_D ||_F^2 / rank(U_D)
```

を見る。`overlap_recall` は「GT relational の禁止方向を Pred dimensionals がどれだけ張っているか」、
`overlap_precision` は「Pred dimensionals がどれだけ relation-like な方向を張っているか」を表す。

これは `Distance=0` や `Angle=0/90` の検出が目的ではない。それらは既に
`solver_near_zero_distance_rate` / `penalty_near_zero_distance` / `penalty_axis_angle` が担当している。
この diagnostic の目的は、nonzero Distance/DistanceX/Y/Radius の row-space が relationals の
row-space を代替しているかを見ること。single constraint row の overlap は Tangent/Equal のような
複合 relation で解釈が弱いので、sketch-level / family-level overlap として実装する。
ただし、これは現時点の主実験を進めるための必須項目ではない。

**16.10.4 望ましい結論と分岐**

望ましい結果:

- `eval_distance_family_ratio_mean` が `scaffold5` よりさらに低下する。
- `eval_key_rel_ratio_mean` / `eval_rel_ratio_mean` が上昇する。
- `Distance`, `DistanceX`, `DistanceY` の per-type ratio が一貫して下がる。
- `solver_fc_rate` と `solver_valid_dof_mean` は guardrail 内に留まる。
- qualitative に非直感的な遠距離 Distance が減る。

この場合に言えること:

> relation-only motion subspace の不足を罰する報酬を強めると、モデルは dimensionals ではなく
> relational constraints で design intent を表す方向に動き、Distance-family 過剰が下がる。

これは単なる class prior や dimensional penalty ではなく、motion-subspace reward の設計に由来する改善として
主張できる。特に `scaffold4`/`scaffold5`/次 run の比較で、flip と solver 圧縮だけでは不十分で、
`motion_relational_angle` の選択圧強化によって distFam が追加で下がることを示せると嬉しい。

悪い結果と読み:

- distFam は下がるが FC/DoF が崩れる:
  solver 係数を 6/3 へ戻す、または `coeff_h_angle` を下げる。
- keyrel は増えず DistanceX/Y だけ残る:
  現在の motion 報酬は DistanceX/Y に中立であり、relation class prior か別の family-level regularizer が必要。
- response/motion/leak は改善するが qualitative が悪い:
  現 scalar 指標は placement 品質を十分に反映していない。per-type ratio と qualitative subset を主評価に追加する。
- `response_relational_angle` canary で dimensionals が増える:
  leak 系は主報酬にしない判断を確定する。

**16.10.5 TODO**

> **注（2026-06-02）**: この TODO は §16.11.4 の確定版に置き換わった。`mixed_subset` 単独強化はせず、第一 run の config は
> **scaffold6（v2 GT × flip × rebalance × `symmetric`）**。下記 3–5（per-type ratio 表・qualitative・診断再実行）は確定版でも有効。

1. ~~`scaffold5` 派生 config を作る: `coeff_h_angle=8.0`, `cap_h_angle=0.85`, `motion_relational_angle_mode=mixed_subset`, `motion_relational_reverse_weight=0.1`~~（→ §16.11.4 で **scaffold6** に変更）。
2. 短い canary で FC/DoF collapse が無いことを確認し、問題なければ full run。
3. 完走後、W&B scalar に加えて `diag_sample_vs_select.py` を dimensional + keyrel focus で再実行。
4. baseline / probeD / scaffold4 / scaffold5 / 次 run の per-type ratio 表を作る。
5. 固定 validation subset で qualitative comparison を行い、Distance-family 数と非直感的 placement を分けて記録する。

### 16.11 Phase 0 無料診断と次手の確定: substitution 軸 と deficiency 軸 (2026-06-02)

§16.10 の次手（`mixed_subset` 強化）に GPU 半日を賭ける前に、**学習なし**の診断で「『Distance 過剰／relational
不足』の律速は何か」を切り分けた。**結論を先に**: 「Distance 過剰」には混同しやすい **2 つの別軸**があり、打ち手が違う。

- **substitution 軸** = Pred が出す Distance が、**関係で代替できる方向**を pin しているか（関係の代用 Distance）。
- **deficiency 軸** = Pred が**そもそも関係を出し切れず DoF を残す**（関係そのものの不足＝未拘束）。

診断の結果、**substitution 軸の余地は中程度**にとどまり、**ユーザが定性的に見ている「relational 不足」の本体は
deficiency 軸（余地が大きい）**だった。よって本線は deficiency 軸を叩く実験（§16.11.4）になる。

使用 checkpoint は `scaffold5`（`logs/train/runs/2026-06-01_19-04-12/checkpoints/last.ckpt`）。新スクリプト
`scripts/diag_dim_relation_overlap.py`（診断1＋診断2 の mixed 部）と既存 `scripts/diag_v1_v2_reward_ranking.py`
（診断2 の v2 部）。生成はすべて素（flip pool なし＝val と同条件）の K=8。

**この節で使う測定量（用語）**
- `row(J)`（row space）: 制約集合のヤコビアン `J` の行が張る方向＝その制約が **pin する（動きを止める）方向**。
  `orth(·)` はその正規直交基底、`rank` はその次元、`n_dof` はスケッチ全体の自由度数。
- `overlap_precision = ‖U_Aᵀ U_B‖_F² / rank(U_B)`: 部分空間 `B` のうち `A` の中に入っている割合（principal cos² の
  和を rank で正規化、[0,1]）。`overlap_recall` は分母が `rank(U_A)`。
- **chance**: `A` と `B` が無相関（ランダム部分空間）なときの期待値。precision の chance ≈ `rank(U_A)/n_dof`、
  recall の chance ≈ `rank(U_B)/n_dof`。absolute 値でなく chance / control からの**超過**を読む。
- **control**: 解釈の基準に置く **人間 GT 自身の dimensional**（＝人間が意図的に置いた**正当な driver**）の同じ overlap。
- **winflip**: 報酬を変えたとき RLOO group 内で argmax（採択される候補）が変わった group の割合。
- `ngrp` / within-group advΔ / **counterfactual**: §16.7 と同義（群内に当該 type の「あり」「なし」が両方在る group 数／
  その advantage の差）。

#### 16.11.1 診断1（substitution 軸の大きさ）— dimensional↔relational row-space overlap

Pred の Distance-family が GT relations と**同じ方向**を pin しているか（関係の代用か、真の driver か）を、同一 state 上で
`U_R=orth(row(J_R_GT))`・`U_D=orth(row(J_D_pred))` の overlap で測る。`leak`（§7.2）は under-constraint（Pred が
許す運動）を見るが、ここは **substitution（Distance で関係の仕事を肩代わり）**を見る——Distance が方向を正しく pin
している限り `leak` はこれを見逃す。基準は control（人間 driver）と chance。

人間 GT は疎な annotation なので、**幾何的に成り立つ関係を全部入れた closure**（rule-augmenter `augment_sketch`）に
対しても測った（fiverr, N=700, `--augment-gt`）:

| subspace pair | recall | precision |
|---|---:|---:|
| Pred distfam vs **human GT** relations | 0.144 | 0.734 |
| GT dims (control) vs human GT rel | 0.124 | 0.620 |
| Pred distfam vs **CLOSURE** relations | 0.146 | **0.796** |
| GT dims (control) vs CLOSURE rel | 0.135 | **0.703** |

`rank(human GT rel)=29.0`、`rank(closure rel)=30.7`（**ほぼ同じ**）、`rank(Pred distfam)=4.9`、`n_dof=49.8`、
chance precision ≈ `29/50 ≈ 0.58`。別途 N=900 の human-only run も同値（Pred 0.728 / control 0.622）。
v1（train 分布, N=600）は Pred distfam precision **0.776**（v1 は rel-only GT で control 無し）。

読み:
- **絶対 precision（0.7〜0.8）は substitution の度合いとして読めない**。関係空間が rank 29/50 ＝全方向の ~58% を占める
  ため、ランダムな driver でも chance ≈0.58、人間の**正当な** driver でも 0.703（chance+0.12＝幾何由来の下駄）出る。
- 意味があるのは **control からの超過**: Pred 0.796 − 人間 driver 0.703 = **+0.09**（closure 基準）。＝scaffold5 の
  Distance は人間自身の driver より **~9pt だけ関係寄り**＝substitution は実在するが**中程度**。closure rank≈human rank
  なので「人間は既にほぼ closure まで関係を付けている」＝この軸の余地が小さいのは確か。
- Radius を含めると precision は下がる（dist+rad 0.677）＝**代替的なのは Distance/DistanceX/DistanceY、Radius は真の driver**。
- ＝ **substitution 軸の余地は中程度（+0.09）**。なお「**非直感的な Distance の置き場所**」はこの軸でも見えない: どの
  primitive 対で pin しても row space は同じ（motion-subspace 不変）なので、overlap も motion_relational 報酬も原理的に
  不可視（§14.7.5 #6）。これは token-level / human-prior 系でしか触れない別問題。

#### 16.11.2 診断2（報酬を再形成しても選択は変わらない — 候補集合が律速）

**mixed_subset 再採点（fiverr, 149 group）.** 学習済 symmetric と同一 coeff/cap/gate で
モードだけ `mixed_subset` に替えて同一 rollout を再採点:

| mode | winflip | win_distfam | win_keyrel |
|---|---:|---:|---:|
| symmetric (base) | – | 0.236 | 0.418 |
| mixed rw=0.2 | 0.000 | 0.236 | 0.418 |
| mixed rw=0.1 | 0.007 | 0.236 | 0.419 |

- **winner はほぼ動かない**（winflip ~0、winner の distfam/keyrel 不変）。§16.9.2 の solver 圧縮・§16.4 の
  v2 target と**同じ**＝ per-sample 報酬の再形成は選択を変えない。**律速は候補集合**。
- ただし type 別 within-group advΔ は mixed が **Equal +0.117 / Perp +0.017** と微増（Tangent −0.157、既正）。
  RLOO は winner でなく全 K の advantage を使うので、`mixed_subset` は**勾配としては Equal/Perp を僅かに後押し**する
  （害は無いが distFam を動かす主レバーにはならない）。

**v2 target 再採点（scaffold5, v1 prompts, 128 group）.** §16.4 を scaffold5（solver 圧縮 4/2）で再実行:
intent-only winflip **14.8%**、total winflip **7.8%**（§16.4 の A_rewardbalance: 19.5% / 11.7%）。
圧縮下でも total 選択は ~8% しか動かない＝**希釈は残る**。type 別 d_total: **Coincident +0.71**（v2 の
per-sketch 完全 Coincident が効く）、Perpendicular +0.12、Equal −0.25（v1 の Equal 過剰補正）、Tangent は
**ngrp=0（flip 無しで counterfactual 皆無、§16.7 再現）**。

**診断2 の読み**: `mixed_subset`・solver 圧縮（§16.9.2）・v2 target のいずれも **flip 無しでは winner を動かさない**
（候補集合が律速）。ただし within-group advΔ レベルでは効く所がある（mixed で Equal/Perp、v2 で Coincident/Perp）。
＝報酬の再形成は「**counterfactual がある所**でだけ効く」。v2 の価値は distFam でなく **per-sketch 完全性
（特に Coincident/Perp の per-sketch『推論漏れ』補完）**にある。

#### 16.11.3 診断3（deficiency 軸の大きさ）— under-constraint の実測

overlap（診断1）は substitution 軸しか測らない。deficiency 軸（関係を出し切れず DoF を残す）は solver 状態で測る:
`scaffold5` は **fiverr で Under 48.2% / FC 45.2%、`solver_valid_dof_mean` ≈ 5.0**（FC は DoF≤2）＝**約半数の
スケッチで関係が大量に不足し未拘束**。これがユーザの「relational 不足」の正体で、**余地は大きい**（v1 train prompt 上では
Under 25% / FC 68% で、人間図 fiverr ほどは不足しない＝§14.7.5 #1 の汎化ギャップと整合）。

#### 16.11.4 統合結論と次手の確定

- **2 軸の切り分け**: substitution 軸（診断1）は control 比 **+0.09 ＝中程度**。deficiency 軸（診断3）は
  **Under 48% / DoF~5 ＝大きい**。**ユーザの定性観測（relational 不足）は deficiency 軸として正しい。**
- **報酬の再形成は単独では効かない（診断2）**が、それは「無効」ではない: `mixed_subset` の `pred_subset_gt` 項は
  **deficiency 軸そのものを罰する**（U_pred が U_gt に無い方向＝Pred の関係不足）ので**標的は正しい**。winner が動かないのは
  **flip 無しで群内 counterfactual が無い**ため（§16.7/§16.9 と同根）。v2 target も deficiency 軸を締める（training 標的の
  motion dim 8.5→5.5, §15.5）。**v2 も mixed も「無効」ではなく「flip 待ち」。**
- **distFam が下がる経路は2つ**: (i) substitution 除去（中程度・overlap-penalty の担当）、(ii) **deficiency を関係で閉じる
  → solver が Distance に頼らなくなる間接経路（大きい・v2/flip の担当）**。**(ii) が本線。**

**確定した次手（ユーザ確認済）**:
- **本線 = v2 GT × flip × rebalance × `symmetric`**（deficiency 軸を、flip が作る counterfactual の上で関係方向に閉じる）。
  config 実装済・CPU smoke 通過: `configs/experiment/rloo_response_scaffold6_v2gt_flip_rebalance_intent.yaml`
  （`scaffold5` の GT だけ v2 no_mirror に差し替え、他は完全同一＝GT 効果のクリーンな attribution）。判定は `scaffold5` と
  同一 step で「`solver_under_rate`↓ / `solver_valid_dof_mean`↓（deficiency 解消）＋ rare-rel↑ &
  `eval_distance_family_ratio_mean`↓、FC を guardrail、収束時 `ngrp`>0」。
- **`mixed_subset` は第一 run では使わない**: mixed は v1 GT の per-sketch **不完全さ**の回避策（不完全な GT より完全な
  Pred を罰さない）。**v2 はその不完全さを元から直すので `symmetric` が筋**であり、GT(v1→v2) と mode(sym→mixed) の同時変更は
  attribution を交絡させる。mixed(rev_weight 0.1–0.2) は v2×symmetric が伸び悩んだ時の**予備**。
- **overlap（substitution 軸）は診断/val ログ止まり**: 余地 +0.09 と小さく、報酬化は per-sketch baseline 相対の較正が難しく
  flip 依存で低 ROI。`scripts/diag_dim_relation_overlap.py` は診断として保持。
- **best-of-N rerank は standalone タスクとして取り下げ**: 独立 yardstick が無く循環（d_H_angle を最終比較に使えない）、
  human-eval は弱く K 候補の差分も微小、評価器としての主張は `summary_histcad_response_metric_v4_results.md` で既出。
  唯一の有用部分（サンプル vs 選択の客観上限＝既存 K 候補で d_H_angle 選択時の distFam/keyrel/DoF 到達上限 vs solver 選択
  vs 方策平均）は既存 `scripts/diag_sample_vs_select.py` がほぼ充足＝新規不要。
- **caveat**: 診断1/2 は flip pool 無し生成での測定。flip 込みでは `ngrp` が増え v2 効果は本測定より大きくなり得る（上限未測）
  ＝学習実験前の足切りとしては十分だが、最終判定は `scaffold6` 実走で。

#### 16.11.4 実行コマンド

```bash
# scaffold6 with intent
conda run -n freecad python src/train.py experiment=rloo_response_scaffold6_v2gt_flip_rebalance_intent

# scaffold6 without intent
conda run -n freecad python src/train.py experiment=rloo_response_scaffold6_v2gt_flip_rebalance_no_intent
```

### 16.12 intent on/off/shuffled の3者比較 — shuffle 負例対照で「強い成功」を確定 (2026-06-04)

§16.11.4 の本線 run（scaffold6 = v2 GT × flip × rebalance × symmetric）が完走し、ResponseIntentReward の
有効性を切り分けるための**決定的 ablation 軸**が揃った。判定軸は scaffold5-vs-6（= GT v1/v2 の質 ablation）ではなく、
**GT=v2・flip・rebalance・足場を全固定して intent {ON / OFF / shuffled} だけを動かす**ことである（§16.1 / §16.8 #1）。
shuffle 負例対照は、§3 / §8.1 next-action / §16.8 #6 で「S vs S+R に gap が出てから回す」と保留していたもので、
gap が出た（下表）ため実行した。

**実装（負例対照）**: `ResponseIntentReward(shuffle_within_group=True)`。`RewardContext.group_sample_size`（=K、
`constraint_ar_module_rloo.build_context` で付与）を見て、`compute()` で per-sample の intent 報酬スカラを
**同一 RLOO group（K=9、flip view 込み）内で permutation** してから各サンプル自身の STOP に再配置する。
permutation は group の multiset を保存する＝**RLOO leave-one-out baseline の group 平均は不変**で、
**sample↔報酬（=sample↔GT）の対応だけを破壊**する。よって S+R̃ に残る効果があれば、それは GT 対応ではなく
**報酬の scale / 分散**に由来する。ログの `intent_*` は真値（shuffle 前の角度）を出すので eval 軸は3者で同一。
config は `rloo_response_scaffold6_v2gt_flip_rebalance_shuffled_intent.yaml`（= o0xgo75k の config と完全同一＝
commit d82a42b の pre-`DimensionalClassPrior` 版＋train 側 `shuffle_within_group: true` の1行のみ）。
ユニット検証で group 和不変・gated-out サンプルが他サンプルの値を受領・metrics 真値を確認済み。

**3者の同一 step（~15.6k）比較**（`validation_gen/*`。`resp`/`mrel`/`leak`/`distFam`/`DoF`/`under` は低いほど良い、
`keyrel`/`rel`/`FC` は高いほど良い、ただし FC は guardrail 扱い §16.8 #1）:

| run | step | resp | mrel | leak | FC | under | DoF | distFam | keyrel | rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S+R** intent `o0xgo75k` | 15654 | **0.428** | **0.405** | **0.391** | 0.503 | 0.423 | 5.36 | **0.217** | **0.428** | **0.712** |
| S no-intent `13f89l4i` | 15588 | 0.475 | 0.442 | 0.447 | 0.570 | 0.389 | 5.46 | 0.246 | 0.413 | 0.682 |
| **S+R̃** shuffled `buxh9c4w` | 15588 | 0.477 | 0.439 | 0.434 | 0.597 | 0.362 | 5.59 | 0.241 | 0.417 | 0.695 |

**読み（決定的）**:
- **S+R̃ は全 design-intent 軸で no-intent ベースライン S に潰れる**: resp 0.477≈0.475（intent 0.428 から遠い）、
  mrel 0.439≈0.442、distFam 0.241≈0.246、keyrel 0.417≈0.413、rel 0.695≈0.682。intent の改善（S 比 resp −0.047 /
  distFam −0.029 / keyrel +0.015）は **scale を揃えて対応だけ壊すと消える**。
- ＝ **改善は GT 対応（各サンプルを自分自身の motion-subspace 標的に向ける信号）に由来し、報酬の大きさ/分散ではない**。
  これは §8.1「強い成功」の署名で、補助 RL 実験として狙っていた切り分けがクリーンに取れた。
- むしろ S+R̃ は FC 0.597 / under 0.362 / DoF 5.59 と **S よりやや拘束過多側へ**動く（ランダムに注入された報酬ノイズの
  作用）。真の intent が relational/under 側（FC 0.503・distFam 最小）へ動かすのと**逆向き**で、対応信号の方向性を裏書きする。
- 唯一の小さな残差は **leak**: S+R̃ 0.434 は S 0.447 をわずかに下回る（intent は 0.391）。intent 効果（S 比 −0.056）の
  約 1/4 で、3者の FC/DoF 動作点が微妙に違う（S+R̃ が最も拘束過多）ことの副作用と読める範囲。leak も「ほぼ no-intent 側」で、
  対応由来でない残差は小さい。

**caveat / 限界**:
- 各条件 1 seed・同一 step（~15.6k）・点推定の平均（paired 検定ではない）。ただし3者の分離は全軸で一貫し符号も安定。
  有意性を足すなら最も安いのは wandb history で直近 step 群の gap 符号安定性を見ること（新規計算不要）。
- 動作点（FC/DoF）が3者で完全一致していない。ただし shuffle 対照が「同じ intent 報酬 scale で対応のみ除去」を担保し、
  それが intent 指標を再現しない（むしろ逆向き）ので、relational シフトは対応駆動と帰属できる。
- eval の headline `response_full_angle`（gate=always、canonical d_H_angle）は**訓練報酬（`motion_relational_angle`）とは
  別 metric の sibling** なので完全な自己評価ではない。加えて報酬が直接最適化しない非報酬指標（distFam/keyrel/FC）が
  同方向に動いている点が、循環でない裏付けになる（§16.8 #1 / トートロジー懸念への回答）。

**結論**: ResponseIntentReward は、同一足場（v2 GT × flip × rebalance）の上で **GT 対応に由来する design-intent 信号として
機能している**（§8.1 強い成功）。FC は intent ありで下がる（trade-off、§16.1 で想定済み）が、これは主指標ではない。
**補助 RL 実験の主張はここで確定**とし、以降は (1) 設定凍結 → (2) 全件 pseudo-GT（`sg_pseudo_rel_v2_no_mirror_full.npy` /
`sg_pseudo_rel_v2_full_all_full.npy`、生成済/生成中）で本命 intent と −intent をフル長で step-matched に回し最終定性 figure、
の production 流れに移る。全件長期学習は rare-rel を伸ばす施策ではなく（収束で頭打ち）、fiverr の under（分布ギャップ）も
埋めないので、新しい主張を出すためではなく凍結設定の最終 checkpoint・figure 取りとして位置づける。

## 17. motion_angle の placement 盲目性と dimensioning-placement 指標 participation (2026-06-03)

**結論（先出し）**: motion_angle (= d_H_angle) は relational correctness（正しい接空間への到達）を測る指標で、**dimensioning placement にだけ構造的に盲目**。これを叩くため、GT と Pred が relational set・幾何・最終 DoF を共有する**統制設定**を設計し、その下で placement だけを測る指標 **participation** を確立した。実データ（`dim_uglify` 合成統制）で「汚い dimensioning ほど participation が悪化する」を決定的に実証（paired Wilcoxon p=3.8e-59）。

**本章の唯一の新規軸は placement = participation**（§17.3）。motion_angle とも leak_angle とも独立した dimensioning 専門軸。coverage（under/over-dimensioning）と relational correctness は既存の motion_angle（d_sub_angle の `max(dim_P,dim_G)` rank 正規化）+ leak_angle がすでに捉えており、別途 coverage 指標（resid 系）は不要だった — むしろ motion_angle は solver_rl の over-dim を正しく human 優位に並べ（z≈+11.9, exploit 無し）、resid 系が解こうとした問題は既存指標には存在しなかった。途中で検討した棄却案（resid_sym を含む / raw l1 最小化 / deviation / rank-redundancy / square-only）は §17.7 に集約。

**重要な限界（先出し）**: participation は「**GT の慣習との一致**」を測る指標で、絶対的な dimensioning の良し悪しではない。chain vs baseline(datum) のような**正当な別慣習どうしでも participation は高く出る**（§17.6, toy 実測 1.7–3.5＝ランダム uglify と同程度）。thesis 実証には無害だが、model 評価には multi-reference（複数 human GT に min）+ human-human floor が必須。**この交絡が実 Fiverr データで稀な病理ではなく「典型」（human-human median 1.34、frac≤1.05=25%）であること、および任意 model への適用が count-mismatch で壊れることを §17.8 で実データ検証した**（human-human/model 比較・drop test 込み）。

### 17.1 motion_angle はなぜ placement に盲目か（動機）

motion_angle は `R_C = -pinv(J_x)J_p` が張る線形部分空間を比較する。dimensional を加える操作は部分空間のランクを上げて真の接空間に近づける操作、relational を加える操作はランクを保ったまま空間の傾きを変える操作。

例: 長方形 ABCD の AB と BC に Distance、残りを Horizontal/Vertical/Coincident で fully constrained（DoF=2、global translation を除く）。

- relational を全削除すると dx/dp_C は AB/BC の長さ変動で見るので接空間は依然 dim=2、ただし AB/BC 以外は不動＝接空間が軸平行に傾いた配置。
- BC の Distance を削除すると dim=1、接空間の1次元分に移動が制限される。

つまり relational をフルに入れた時点でゴールの接空間は定まり、dimensional を加える操作は「その接空間上で何次元動けるか」の自由度を増やすだけ。**fully constrained になればどんな dimensioning でも行き着く先は真の接空間全体**となり、motion_angle は dimensioning の入れ方を区別できない。

### 17.2 統制設定: placement だけを残す

GT と Pred が **relational set 同一・幾何同一・ともに fully constrained**（DoF=2）とする。この設定の構造的帰結:

- 同 rel set + 両 fully constrained ⟹ **dim 本数 m が一致**、かつ `range(R_GT)=range(R_Pred)= relational manifold の接空間`。
- ⟹ `motion_angle` は同一部分空間を比較 → **principal angle 全て 0**（盲目の最もクリーンな言明）。
- ⟹ `A := J_{Q_GT} R_Pred`（GT probe を Pred 幾何で評価した行 × Pred native response）は **正方・可逆** → 再現は常に可能で coverage 差はゼロ。

placement 情報は**部分空間（同一）でなく、その中の基底**に宿る: `A` はちょうど「Pred dims → GT dims」の基底変換。clean な dimensioning なら `A` は置換行列様（各 GT 編集が単一 Pred ツマミ）、ugly なら密。

### 17.3 採用指標 participation: 定義・お気持ち・gaming 不能性

各 GT 編集 i（「GT dim i を1動かし他を保つ」）を Pred のツマミで実現する変化:
```text
dp_i = A^{-1} e_i                              # 正方可逆なので一意
participation_i = ‖dp_i‖_1^2 / ‖dp_i‖_2^2      # GT 編集ごとに平均
```

**お気持ち（なぜ (L1/L2)^2 か、なぜ L2 で割るか）**: これは inverse participation ratio ＝「実効的に動くツマミの本数」。等しい k 本が動くとき `‖dp‖_1^2/‖dp‖_2^2 = (kc)^2/(kc^2) = k` と本数に一致、不均等なら soft count（例 `(1,0.01)`→≈1.02＝実質1本）。**L2 で割る意味は scale 不変化**: `participation(c·dp)=participation(dp)`。`‖dp‖_1` 単独は「本数 × 大きさ」を混ぜるが、その「大きさ」こそ gain/感度の軸で、§17.7 で deviation を汚した張本人。L2（＝最大成分が支配する大きさのスケール）で割ると magnitude が消え、**純粋に「いくつのツマミが必要か」だけ**が残る。`‖dp‖_1` = 総移動量、`‖dp‖_2` = 大きさスケール、その比（の二乗）が spread/count。

**基準と gaming 不能性**: GT-on-GT で `A=I` → participation=1。統制下で **participation=1 ⟺ A^{-1} が monomial（各列に非ゼロ1つ）⟺ 各 GT 編集が単一 Pred ツマミで実現 ⟺ Pred が GT と edit-equivalent**。「ugly かつ participation=1」は定義上ありえない（edit-equivalent-to-GT と not-ugly-relative-to-GT は同義）ので、**統制下では gaming failure mode が存在しない**。本数 m 固定なので over-dimensioning exploit も構造的に不可能。（datum/基準寸法は participation≈1 で正しく「良い実務」と判定、chain 寸法は GT の datum 編集を再現するのにツマミが連鎖して >1。）

**数値実証（toy: 共有2D接空間を R^6 に埋め込み、Pred 基底を GT から角度 mix で回転＝ugliness）**:

| ugliness | motion_angle(deg) | resid | participation | ‖dp‖_1 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0000 | 0.0 | 1.000 | 1.000 |
| 0.50 | 0.0000 | 0.0 | 1.707 | 1.307 |
| 1.00 | 0.0000 | 0.0 | 2.000 | 1.414 |

motion_angle・resid は 0 のまま、participation だけが 1→2 へ単調増加。

**補助・代替案**:
- `‖dp_i‖_0`（閾値超え本数）= 仮説の文字通りの離散版。participation はその滑らかな代理。
- 割当外エネルギー比（`|A^{-1}|` 上の Hungarian 最適割当、置換で0・密で→1）は **可逆な統制下では participation とほぼ一致**（A 可逆 → 列独立 → 1:1 割当が必ず存在）。標準的で解釈容易な participation を採用し、これは非可逆ケース用の頑健化候補として記録のみ（実装不要）。

### 17.4 dim_uglify 生成器の実装

[notebooks/v4_exp/dim_uglify.py](v4_exp/dim_uglify.py) `uglify_sketch(gt, alpha, ...)`:

1. relational + size dim（Radius/Diameter）を保持し、**uglifiable**（Distance/DistanceX/DistanceY/Angle）の α 割合を除去。
2. 貪欲に「GT 自身の dim **型**（point-point / point-line Distance, Angle…）を random ref に張り替えた候補」を追加し、`rank(J_x)` が上がるものだけ採択（冗長は棄却）。**型を再利用するのが要点**: point-point Distance だけでは line 長・point-to-line・向き DoF を張れない。
3. full-rank 復帰後 `complete_constraint_values`（[sketch_utils.py](../src/sketch_handlers/sketch_utils.py)）で各 dim 値を GT 幾何から確定（`VALUE_ASSIGNED:` watermark 付与）。SketchGraphs 原データの値は壊れていることが多く、これが正準経路。
4. FreeCAD native solver で最終検証（`converged` のみ採択）。

**速度・並列**: 内側ループは Jacobian-rank 判定のみ（rank は target 値に非依存なので候補は dummy 値、FreeCAD 不使用）、FreeCAD solve は build 毎に1回だけ。driver [gen_dim_uglify.py](v4_exp/gen_dim_uglify.py) は 1 file=1 task の ProcessPoolExecutor（spawn, FreeCAD は process-safe, 各 worker 自前 context, Vitruvion model はロードしない）。16 workers で 139 files が ~7 分。

**実装上の落とし穴（記録）**: 候補 dim の `QuantityParameter` は `value`=float かつ `expression` が `VALUE_ASSIGNED:` で始まる必要があり（[constraints/base.py](../src/sketch_handlers/constraints/base.py) `get_constraint_props`）、欠けると handler residual が raise → `compute_native_response` が黙ってその制約を落とし rank が上がらない。`complete_constraint_values` を使えば自動付与。

### 17.5 dim_uglify 実データ結果（決定的）

α∈{0,.25,.5,.75,1}, seed×3, GT=各 human が自分の dimensioning を uglify する self-comparison。artifact: `artifacts/v4_exp/dim_uglify_details.csv`、uglified sketch は `artifacts/v4_exp/dim_uglify_cache/` に pickle。集計は `uglifiable_dof>0` に限定。

**フル結果（6 batch, 139 files, 5330 builds, build 成功率 ~84%@α>0）**:

| alpha | participation (median) | resid | deviation | n |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.0000 | 0.000 | 371 |
| 0.25 | 1.160 | 0.0000 | 0.167 | 1044 |
| 0.50 | 1.333 | 0.0000 | 0.414 | 1035 |
| 0.75 | 1.500 | 0.0000 | 0.578 | 1039 |
| 1.00 | 1.600 | 0.0000 | 0.667 | 1031 |

- **participation は α（汚さ）とともに単調増加、resid は終始 0.0000**＝統制成立（同接空間なので motion_angle も 0）。
- per-sketch 単調性 96%（343/357）、end-to-end 厳密増加 91%（324/357）。
- paired Wilcoxon（α=1 vs α=0, n=355 sketches）: median 1.000→1.613、mean Δ=+0.725、**p=3.8e-59**。tie 31 は uglifiable 自由度が実質ゼロの sketch。

**結論**: §17.1 の主張（motion_angle は placement に盲目／participation がその軸を捉える）を実データで決定的に実証。GT は同一 rel set・幾何・最終 DoF を持つ「汚い」再 dimensioning より participation が低い＝**GT の dimensioning が「各編集を最小ツマミで達成できる自然な配置」であることを定量化できた**。

備考: `uglifiable_dof=0`（全 dim が Radius 等で placement 自由度ゼロ）や `=1`（単一 line 長等の強制 DoF）の sketch では participation は α によらず 1 のまま（正しい挙動＝置換する placement 自由が無い）。Angle 候補が Circle 同士を引く無効ケースは reject されるだけ（warning のみ、結果に影響なし）。

### 17.6 participation の射程と限界（GT 慣習依存・chain vs datum）

**(1) 統制域でのみ有効**: participation が placement 指標として信頼できるのは **count-matched な統制域のみ**。任意の model 出力（over-dimensioning しうる）に直接当てると `A` が非正方になり pinv の min-norm が余剰ツマミに分散して値が信用できない。よって participation は **thesis 実証用の instrument**（および count-matched subset での診断）であり、任意 model 評価の drop-in metric ではない。

**(2) GT 慣習依存 — chain vs datum 問題（重要）**: participation は「Pred が **GT の慣習** とどれだけ一致するか」を測るのであって、絶対的な良し悪しではない。基準は 1（GT-on-GT）で、GT と違う placement は全て >1。問題は、**正当な別慣習どうしでも大きく出る**こと。toy（直線上 n 点, datum=基準点から各点を寸法 / chain=隣接間を寸法）:

| | participation (n=3 / 4 / 6) |
|---|---:|
| GT=datum, Pred=chain | 1.67 / 1.75 / 1.83 |
| GT=chain, Pred=datum | 2.00 / 2.50 / 3.50 |
| GT=datum, Pred=**random ugly** | 1.67 / 1.75 / 1.83 |

すなわち **chain↔datum はランダム uglify と同程度のスコア**。participation だけでは「劣った dimensioning」と「正当な別慣習」を magnitude で区別できない。これは participation が placement 軸を測れている証拠の裏返し（chain と datum は tolerance stack-up vs 独立配置という**別の design intent** で、motion_angle は両者を 0=同一と見るが participation は区別する）でもあるが、**model を「正しい/間違い」で採点する用途では交絡**。

→ 対処: model 評価では **multi-reference**（複数の human GT に対して participation を計算し min を取る＝どれか1人の慣習に合致すれば低スコア）＋ **human-human participation を floor** として併置（人間どうしも慣習差で >1 になり、その分布が「正常範囲」）。v4 の source-conditioned multi-reference 枠組みがそのまま使える。絶対値 vs 単一 GT を品質判定に使ってはならない。

**(3) coverage は既存指標で足りる**: 「適切な DoF を被覆しているか（under/over-dimensioning）」という軸は、本章で一度 `resid_sym` として作ったが、**motion_angle（d_sub_angle の rank 正規化）+ leak_angle が既に捉えている**ため新規には不要（§17.7 参照）。よって最終的に追加すべき新軸は participation の1つのみ。

### 17.7 検討の経緯と棄却・派生案まとめ

placement 指標 participation に至るまでに検討し、棄却ないし別軸へ振り分けた案を一括記録（教訓つき）:

| 案 | 何を測ろうとしたか | 判定と理由 |
|---|---|---|
| **raw deform_gain `‖dp‖_1` 最小化** | 編集に要する Pred driving param 総変化（小さいほど良い、の初案） | **棄却**。dim_drop で劣化に対し `l1` が**減少**（1.27→0.0）＝逆向き。基準は 0 でなく **1**（GT-on-GT で `A=I`→1）。 |
| **deviation `\|1−l1\|` / logdev** | 1 からの乖離＝GT 同等労力か（過敏 `<1`・過結合 `>1` の両側を罰） | **headline から降格**。再現が綺麗な領域に限定しても human/model を分けず、むしろ over-dim model を僅か優遇（min-norm が多ツマミに分散し `l1≈1` に寄る）。effort 軸は intent と相関しない。operability 診断としてのみ残置。 |
| **redundancy `n_pP − rank(R_P)`** | over-dimensioning（冗長 dim）を rank 欠損で検出 | **棄却**。solver_rl の余剰 dim は独立（rank=n_pP）で、over-determined でなく「人間が空けた DoF まで拘束」型。rank 欠損が無く検出不能。 |
| **square-only 統制** | n_gt==n_pP に絞り placement を count から分離 | **棄却**。coverage 多様性（human diversity）を潰し、残るのは gamed な deviation のみ。統制は dim_uglify（rel 同一・幾何同一）で行うべき。 |
| **forward-only resid** | GT 編集を Pred が再現できるか（coverage） | **棄却**。単独では over-dim exploit（solver_rl 符号反転 z=−11.3）。backward を足した symmetric `resid_sym` で z=+12.1 に修正できたが（下記）、結局 coverage 自体が既存指標で足りるので不要。 |
| **resid_sym = max(resid_fwd, resid_bwd)** | coverage（under/over-dimensioning）を双方向で頑健に | **棄却（冗長）**。coverage は motion_angle（d_sub_angle の `max(dim_P,dim_G)` rank 項）+ leak_angle が既に捉え、しかも motion_angle は solver_rl の over-dim を exploit 無しで human 優位に並べる（z≈+11.9）。resid_sym が解いた exploit は forward-only resid 固有の自作問題で、既存指標には存在しなかった。新軸としては participation のみ残す。 |

**全体の教訓**: 当初は design-intent を「coverage + placement の2軸」と捉えたが、coverage は既存の motion_angle + leak_angle で足り、**本章が追加すべき真に新規な軸は placement = participation の1つだけ**だった。participation は motion_angle / leak_angle が構造的に盲目な「どの辺に dim を置くか」を、`A^{-1}` 列の実効ツマミ数で測る。L2 正規化により magnitude=gain を落として本数のみを抽出するのが核心（effort 軸 `l1`/deviation はこの汚染で intent を測れなかった）。ただし「GT 慣習との一致」を測る性質上、chain↔datum 等の別慣味交絡があり、評価には multi-reference + human floor が必須（§17.6）。

### 17.8 participation を model 評価に当てた実データ検証: human-human floor・drop test・chain/datum 頻度 (2026-06-04)

**結論（先出し）**: §17.5 の dim_uglify は **count-matched 統制域** での placement signal を実証したが、(a) human-human vs human-Vitruvion 比較、(b) relational/dimensional drop test、(c) §17.6 で警告した chain↔datum 交絡が実データで**どれくらい頻発するか**、の3点は未検証だった。これらを既存 artifact `artifacts/v4_exp/deform_gain_details.csv`（[deform_gain_ablation.py](v4_exp/deform_gain_ablation.py) 出力、6 batch・396 rotations・base/solver_rl 込み）から検証した。結果は **§17.6 の限界記述を実データで決定的に確証**: chain/datum 交絡は稀な病理ではなく**典型**（human-human participation 中央値 1.34、frac≤1.05 はわずか 25%）であり、participation を任意 model の drop-in 評価指標にしてはならないという §17.6 の position が裏付けられた。dim_uglify thesis（§17.5）は count-matched 統制内の主張なので無傷。

分析: [notebooks/v4_exp/analyze_participation_eval.py](v4_exp/analyze_participation_eval.py)（既存 details CSV を読むだけ、再生成不要）。各 detail 行は (rotation, variant, GT-ref)。1 rotation に GT-ref が2人いるので ref 上で **mean**（単一慣習一致）と **min**（multi-reference, §17.6: どれか1人の慣習に合えば低スコア）の両方で集約。participation は `A` が正方（`n_gt==n_p_pred`）のときのみ信頼でき（§17.6(1)）、SQUARE-ONLY view は count-matched 行のみに制限する。

#### 17.8.1 count 一致の実態（なぜ任意 model で participation が壊れるか）

| variant | n | square% | under% | over% | med_resid |
|---|---:|---:|---:|---:|---:|
| source_self | 792 | 100% | 0% | 0% | 0.0000 |
| heldout (human) | 792 | 48% | 26% | 26% | 0.0000 |
| dim_drop25 | 792 | 19% | 74% | 6% | 0.3476 |
| dim_drop50 | 792 | 6% | 93% | 1% | 0.6200 |
| dim_drop75 | 792 | 1% | 99% | 0% | 0.8089 |
| dim_drop100 | 792 | 0% | 100% | 0% | 1.0000 |
| rel_drop50 | 792 | 48% | 26% | 26% | 0.1256 |
| rel_drop100 | 792 | 48% | 26% | 26% | 0.3763 |
| base | 612 | 21% | 46% | 32% | 0.2500 |
| solver_rl | 732 | 21% | 2% | 77% | 0.0000 |

**human-human ですら dim 本数が一致するのは 48% のみ**（26% under / 26% over）。base は under-constrained（46% under）、solver_rl は **77% over-dim**（resid=0＝全 GT 編集を over-span で再現）。任意 model は count-matched 域をほぼ外れるため、participation の前提（`A` 正方可逆）が成立しない。

#### 17.8.2 drop test: participation は coverage でも relational でもない（直交性の確認）

| variant | med participation (mean / min agg, ALL) |
|---|---:|
| source_self | 1.000 / 1.000 |
| **heldout** | **1.337 / 1.167** |
| dim_drop25 | 1.425 / 1.324 |
| dim_drop50 | 1.348 / 1.219 |
| dim_drop75 | 1.000 / 1.000 |
| dim_drop100 | **0.000 / 0.000** |
| rel_drop50 | 1.390 / 1.231 |
| rel_drop100 | **0.917 / 0.779** |
| base | 1.269 / 1.142 |
| solver_rl | 1.473 / 1.250 |

- **dimensional drop は participation の妥当な負例にならない**: 1.425→1.348→1.000→**0.000** と非単調で、劣化が進むほど participation が**下がる**（dim_drop100 が「最良」に見える）。これは dim を抜くと `A` が non-square・under-dim になり resid が 0.35→1.0 と増大、pinv min-norm が `dp` を縮め participation が 1→0 に潰れるため。coverage（under-dimensioning）は participation の管轄外で、§17.6(3) の通り motion_angle/leak_angle が見る軸。
- **relational drop も off-axis**: rel_drop100 で participation=**0.917<1**（relation を全部壊しても placement スコアは「良い」）。participation は relational correctness と直交（motion_angle の管轄）。
- これらは「participation が placement 専用軸」という §17 主張の裏返しの確認（coverage/relational では動かない）であると同時に、**participation を単独で under/over-dimensioning の検出に使えない**ことの実証。

#### 17.8.3 human-human floor vs human-model 分布（§17.6(2) の頻度実測 + model 比較）

human-human（heldout）floor と human-model（base / solver_rl）の participation 分布を**同一形式で並置**。交絡は値が 1 を超えるたびに発火する。model 行が human floor と比較できるのは count-matched な SQUARE view のみ（§17.6(1)）:

| view | variant | agg | n | median | Q75 | Q90 | Q95 | max | frac≤1.05 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | heldout (human) | mean | 396 | 1.337 | 1.596 | 2.000 | 2.298 | 3.905 | **0.25** |
| ALL | heldout (human) | min | 396 | 1.167 | 1.415 | 1.753 | 2.019 | 3.690 | **0.43** |
| ALL | base | mean | 308 | 1.269 | 1.650 | 2.195 | 2.685 | 3.666 | 0.35 |
| ALL | base | min | 308 | 1.142 | 1.498 | 2.029 | 2.345 | 3.393 | 0.47 |
| ALL | solver_rl | mean | 368 | 1.473 | 1.966 | 2.724 | 3.451 | 5.347 | 0.19 |
| ALL | solver_rl | min | 368 | 1.250 | 1.500 | 2.277 | 2.734 | 4.944 | 0.38 |
| SQUARE | heldout (human) | mean | 257 | 1.217 | 1.500 | 1.780 | 2.192 | 3.690 | 0.38 |
| SQUARE | heldout (human) | min | 257 | 1.171 | 1.475 | 1.780 | 2.192 | 3.690 | 0.41 |
| SQUARE | base | mean | 85 | 1.222 | 1.447 | 1.733 | 1.867 | 2.460 | 0.40 |
| SQUARE | base | min | 85 | 1.200 | 1.447 | 1.693 | 1.867 | 2.460 | 0.45 |
| SQUARE | solver_rl | mean | 104 | 1.300 | 1.500 | 1.714 | 2.208 | 4.944 | 0.25 |
| SQUARE | solver_rl | min | 104 | 1.268 | 1.500 | 1.714 | 2.208 | 4.944 | 0.33 |

**(a) chain/datum 交絡は稀な病理ではなく常態**: human-human の participation 中央値は単一参照で 1.34、count-matched に絞っても 1.22 で、**1 近傍（≤1.05）に収まるのは 25–38% に過ぎない**。§17.6 の toy 実測（chain↔datum=1.7–3.5）が実 Fiverr データでそのまま再現し、人間どうしでも placement 慣習が日常的に食い違う。multi-reference min（ここでは ref 2人）は floor を 1.34→1.17 へ下げ frac≤1.05 を 0.25→0.43 に改善するが、**2参照では依然 1 に乗らない**（過半が >1.05）。参照数を増やすほど floor が 1 に近づくことを示唆する（より多くの慣習をカバー）。

**(b) 有効域では model が human floor 帯に埋もれる**: count-matched な SQUARE view では base=1.22/1.20（mean/min）が human=1.22/1.17 と**ほぼ同値**、solver_rl=1.30/1.27 も human floor の中央〜Q75 内（human Q75=1.50）に収まる。すなわち participation の有効域で model と human の placement 差はほぼ無く、§17.8.4 の paired test（SQUARE で marginal）と一致。**ALL view で solver_rl の分布が右に伸びて見える（median 1.47, Q95 3.45）のは over-dim 77% の count-mismatch inflation**（§17.8.4）であって placement 品質の劣化ではない。base は逆に under-dim で participation が 1 側に潰れ（ALL median 1.27 < human 1.34）、「under-constrained なのに participation 上は human より良く見える」という、participation を単独 model 指標にできないもう一つの証左になっている。

#### 17.8.4 human vs model: 単一/少数参照では分離しない、見かけの分離は count-mismatch artifact

paired Wilcoxon（model vs heldout, z>0 ⇒ human 優位＝participation 低い）:

| view | agg | model | n | z | p |
|---|---|---|---:|---:|---:|
| ALL | mean | base | 308 | −0.86 | 0.39 (n.s.) |
| ALL | mean | solver_rl | 368 | **+8.45** | 2.8e-17 |
| ALL | min | base | 308 | −0.10 | 0.92 (n.s.) |
| ALL | min | solver_rl | 368 | +6.11 | 9.8e-10 |
| SQUARE | mean | base | 67 | +1.15 | 0.25 (n.s.) |
| SQUARE | mean | solver_rl | 82 | **+2.54** | 0.011 |
| SQUARE | min | base | 67 | +0.98 | 0.33 (n.s.) |
| SQUARE | min | solver_rl | 82 | +2.23 | 0.026 |

- **base（under-constrained）は participation で human と分離しない**（全 view で n.s.）。under-dimensioning は participation の管轄外（§17.8.2）なので当然。
- **solver_rl の ALL-pairs での強い分離（z=+8.45）は placement 品質差ではなく over-dimensioning の count-mismatch artifact**。count カテゴリ別 participation を見ると solver_rl は square=1.27 / over=1.50（over-dim 77%）で、過剰ツマミに pinv min-norm が分散して participation が膨張している（heldout square=1.17）。**count-matched（SQUARE-ONLY）に絞ると z は +8.45→+2.54（min で +2.23, p≈0.03）と marginal に崩れる** = participation が有効な統制域では human/model の placement 差はほぼ無い。これは §17.6(1)（任意 model は `A` 非正方で値が信用できない）が予測した通りの挙動。

#### 17.8.5 統合結論

1. **§17.6 の限界記述を実データで確証**: participation は count-matched 統制（dim_uglify, §17.5）でのみ信頼できる thesis instrument であり、任意 model の drop-in 評価指標にしてはならない。
2. **chain/datum 交絡は頻発する**（human-human median 1.34, frac≤1.05=25%）。「正当な別慣習」が日常的に participation を 1 から押し上げるため、絶対値・単一 GT での品質判定は不可。multi-reference min + human-human floor の併置が必須で、参照数を増やすほど floor が締まる。
3. **drop test は participation の直交性を確認**: dimensional drop で非単調（count-mismatch で潰れる）・relational drop で <1。coverage と relational correctness は既存の motion_angle/leak_angle が管轄（§17.6(3) 再確認）。
4. **したがって §17 の最終形は維持**: 新規軸は placement=participation の1本、ただし用途は (i) dim_uglify 型の count-matched 統制実証、(ii) count-matched subset での診断、に限定。model ranking は motion_angle（coverage+relational）+ leak_angle が担い、participation はそれと独立な placement 軸の存在証明として paper に置く。

実行コマンド:
```bash
# details CSV 再生成（既存なら不要、~数分・FreeCAD solve 無し）
python notebooks/v4_exp/deform_gain_ablation.py \
  --vitruvion-spec base=<base_root> --vitruvion-spec solver_rl=<solver_rl_root>
# 本節の分析（CSV を読むだけ）
python notebooks/v4_exp/analyze_participation_eval.py
```

## 18. set_sim（motion-basis 一致）: encoding 不変な placement/diversity 軸とその射程 (2026-06-04)

**結論（先出し）**: participation（§17）は **native-knob 空間**の指標で、placement の符号的多様性（chain/datum 交絡）により model 評価に使えなかった（§17.8）。本章は同じ placement 軸を**幾何 motion 空間**で測る指標 `set_sim`（motion-basis 一致）を確立する。set_sim は (i) encoding 不変（「同じ隙間を point-point か point-line で寸法」が同一視される）、(ii) human-human 一致が高い（0.84, signature Jaccard 0.11 と対照的）、(iii) 実 model を正しい向きに強く分離する（z≈+11〜12, count 交絡なし）、(iv) toy で motion_angle が構造的に盲目な placement 差を捉える（正方形 side-vs-diagonal=1.000 / 長方形 right+top-vs-right+diagonal=0.86–0.95, いずれも motion_angle=0）。**ただし重大な留保**: 統制域（relation・幾何・DoF 同一 ⇒ motion_angle≡0）で隔離すると set_sim の純粋 placement 識別力は弱く（AUROC 0.62, chain/datum 交絡あり）、ablation の強い分離（z≈+11）の大半は relational+coverage チャネル＝**既存の motion_angle/leak_angle と冗長**な可能性が高い。set_sim が motion_angle に対し *増分* の識別力を持つかは §18.7 で未検証。また intrinsic な placement-**quality**（reference 不要で good/bad を測る）候補は全滅した（§18.6）。

### 18.1 定義

native response $R_C=-J_x^{+}J_p\in\mathbb{R}^{n_x\times m}$（列 $j$ = dim $j$ を駆動し他制約を保つ最小ノルム幾何 motion, [edit_response_distance.py:294](../src/metrics/edit_response_distance.py#L294)）、observable Jacobian $J_\phi$（pairwise 距離＋半径, [:599](../src/metrics/edit_response_distance.py#L599)）から:
$$M=J_\phi R_C\in\mathbb{R}^{m_\phi\times m},\qquad u_j=\frac{M_{:,j}}{\lVert M_{:,j}\rVert_2}\quad(\text{単位 motion 方向})$$
共有 `base_eids` 上で $J_\phi$ を組むため observable フレームは annotator 間で共通。2 つの単位列集合 $A=\{a_i\}, B=\{b_j\}$ の対称 mean max-cos:
$$\mathrm{set\_sim}(A,B)=\tfrac12\Big(\underset{j}{\mathrm{mean}}\,\underset{i}{\max}|\langle a_i,b_j\rangle| + \underset{i}{\mathrm{mean}}\,\underset{j}{\max}|\langle a_i,b_j\rangle|\Big)\in[0,1]$$
1=同一 motion 基底、低いほど基底が回転（placement が違う）。motion_angle は range$(R_C)$ の部分空間距離で、統制域では 0（盲目）。set_sim は**同一部分空間内の基底**を見る点が新規。実装: [diag_motion_basis_consistency.py](v4_exp/diag_motion_basis_consistency.py)。

### 18.2 annotator agreement: placement は多様、motion は一致

3 名の Fiverr annotator で寸法配置の合意を実測（[diag_annotator_agreement.py](v4_exp/diag_annotator_agreement.py)）:

| 指標 | 値 |
|---|---|
| 寸法 signature を 1 人だけが使用 | **79%** |
| 全 3 人が共有 | 3%（一致中央値 0.00）|
| pairwise Jaccard（placement 重なり）| **0.11** |
| **motion-basis 一致**（set_sim, human-human）| **0.83** |

→ **人間は「どこに寸法を置くか」では 8 割食い違うが、「どんな motion を駆動するか」では 8 割一致**。encoding 不変性が偽の多様性を潰し、kinematic な合意が現れる。motion 空間が diversity を測る正しい空間。

### 18.3 ablation（human-human / human-model / drop, n=396）

[diag_motion_consistency_ablation.py](v4_exp/diag_motion_consistency_ablation.py), ref 上で mean / max 集約:

| variant | n | med(mean) | med(max) |
|---|---:|---:|---:|
| source_self | 396 | 1.000 | 1.000 |
| **heldout (human)** | 396 | **0.843** | **0.904** |
| dim_drop25/50/75 | | 0.779 / 0.691 / 0.609 | 0.819 / 0.732 / 0.643 |
| rel_drop50/100 | | 0.649 / 0.520 | 0.678 / 0.536 |
| **base** | 276 | **0.739** | 0.770 |
| **solver_rl** | 369 | **0.749** | 0.790 |

paired Wilcoxon（human 優位 z>0）: **base z=+11.99, solver_rl z=+11.18**（both p<1e-28, mean/max 同様）。dim_drop / rel_drop とも単調減少（construct-valid）。**participation（AUROC 0.5, §17.8）と違い両 model を強く正しく分離し、count 交絡も無い**（[0,1] の similarity）。

### 18.4 強さの出どころ: relational+coverage チャネル vs 純粋 placement

ablation の z≈+11 が placement 純粋成分か検証。統制域（同一 relation・幾何・両 fully constrained ⇒ motion_angle≡0, 純粋 placement）で隔離:
- **dim_uglify 統制**: human-human 0.829 vs human-ugly 0.769、**AUROC 0.62**（[diag_motion_basis_consistency.py](v4_exp/diag_motion_basis_consistency.py)）。
- **chain/datum toy**（[diag_chain_datum_motion.py](v4_exp/diag_chain_datum_motion.py), n 個共線点）:

| n | datum vs chain | datum vs ugly(×20) |
|---|---:|---:|
| 4 | 0.678 | 0.763 |
| 8 | 0.561 | 0.723 |

→ set_sim は正当な別慣習 chain-vs-datum（0.56–0.68）を**ランダム ugly（0.72–0.76）より悪く採点**（cascade 系 motion の幾何差が大きいため）。participation と同じ convention 交絡を持つ。

**解釈**: ablation の z≈+11 と統制域 0.62 の差が **relational+coverage チャネルの寄与**。すなわち set_sim の強い model 分離は大部分が motion_angle/leak_angle が既に捉える軸で、真に新規な純粋 placement 成分は弱い。multi-reference（ref union への coverage）でも改善せず（AUROC 0.641 vs single 0.639, [diag_motion_multiref.py](v4_exp/diag_motion_multiref.py)）— ugly の motion は同一 $V$ 内にあり humans が $V$ を張るため union で覆えてしまう。

### 18.5 toy: set_sim は motion_angle が盲目な placement 軸を捉える

[diag_set_sim_toy.py](v4_exp/diag_set_sim_toy.py)、パイプライン数式 $M=J_\phi R_C$ で解析計算。両 toy とも relational 形状固定 ⇒ range$(R_C)=V$ 同一 ⇒ motion_angle≡0:

| toy | dimensioning 比較 | set_sim | motion_angle |
|---|---|---:|---:|
| (1) 正方形（1 DoF=scale）| 右辺 vs 対角 | **1.000** | 0.000 |
| (2) 長方形 w:h=2:1（2 DoF）| (右辺+上辺) vs (右辺+対角) | **0.908** | 0.000 |
| (2) 長方形 3:1 | 同上 | 0.952 | 0.000 |
| (2) 長方形 1.5:1 | 同上 | 0.864 | 0.000 |

- **(1)**: 正方形では side も diagonal も唯一の scale DoF を駆動 ⇒ motion 同一 ⇒ set_sim=1。**motion-equivalent な dimensioning を正しく同一視**（placement が違っても変形が同じなら差を出さない＝正しい挙動）。
- **(2)**: 対角を駆動すると（右辺保持で）w↔h が結合し、軸平行基底 {V,H} と異なる基底 {V−H, H} になる ⇒ set_sim<1。**motion_angle が盲目（同一 $V$）な placement 差を set_sim が捉える**。aspect ratio で値が動く（正方形に近いほど結合が強く低下）。

→ set_sim が motion_angle と**別軸**を見ること、かつ motion-equivalent なら 1 を返す健全性を toy で確認。実データでこの新規軸が motion_angle に**増分**の model 識別力を与えるかは §18.7（未検証）。

### 18.6 棄却した intrinsic placement-quality 候補（一括）

reference 不要で good/bad を測る intrinsic placement-quality 指標を多数試し、**全滅**した。dim_uglify の「ugly」が valid・full-rank ＝幾何的に alternative convention と区別不能（§17.6/17.8）なのが根本。一括記録:

| 候補 | 何を測るか | 結果（AUROC natural vs ugly / model 分離）|
|---|---|---|
| Euclidean 近接性 | dim が結ぶ entity の幾何距離 | **0.495**（無信号, natural 中央値 0.65 vs ugly 0.66）[diag_placement_locality.py](v4_exp/diag_placement_locality.py) |
| relational graph 距離 | 結ぶ entity のトポロジ距離 | 0.587（弱）|
| pooled per-dim 特徴密度 | learned P(placement\|geometry), 5-fold CV | **0.519**（無信号）[diag_placement_density.py](v4_exp/diag_placement_density.py) |
| 構造特徴（max-degree 等）| dimensioning graph 形状 | 0.62（弱）|
| per-primitive 局在 (IPR) | dim 駆動で何個 primitive が動くか | **0.531**（fully constrained では relational 結合で全体が動く）[diag_motion_localization.py](v4_exp/diag_motion_localization.py) |
| 条件数 κ(M) | motion 基底の decoupling | **失敗**: count 交絡（dim 減 ⇒ κ 低 ⇒ 劣化が「良く」見える, base AUROC 0.36 逆向き）[diag_motion_kappa_ablation.py](v4_exp/diag_motion_kappa_ablation.py) |
| frame 局在 eff_norm/aniso | 基底の実効方向数/異方性（Gram スペクトル）| dim_uglify 統制では **0.63**（natural が decoupled）だが ablation で **count 交絡**（drop/under-dim が「良く」見える）、count-matched human-model では**分離せず**（base タイ, solver_rl 逆向き）[diag_motion_localization_ablation.py](v4_exp/diag_motion_localization_ablation.py) |
| multi-reference coverage | Pred motion が human union に被覆されるか | 0.641（single 0.639 と同等, 改善なし）[diag_motion_multiref.py](v4_exp/diag_motion_multiref.py) |

**frame 局在の定義**（最有力だった intrinsic 候補）: $S=\sum_j u_ju_j^\top$ のスペクトル $\{\lambda\}$（= Gram $U^\top U$ の固有値, $\sum\lambda=m$）から $\mathrm{eff}=(\sum\lambda)^2/\sum\lambda^2=m^2/\sum_{ij}\cos^2\theta_{ij}\in[1,m]$, $\mathrm{eff\_norm}=\mathrm{eff}/m$, $\mathrm{aniso}=\lambda_{\max}/m$。直交基底で eff=m（decoupled）、平行で eff=1（冗長）。dim_uglify では natural がより decoupled（eff_norm 0.73 vs 0.64）だが、**count 統制が外れると交絡で逆転し実 model に転移しない**＝dim_uglify 固有のアーティファクト（random rewiring が冗長軸を作る）であって naturalness ではなかった。

**GL(m) 不能性論法の撤回**: 一時「統制域では任意の可逆 $A\in GL(m)$ が valid dimensioning ＝ good/bad の部分領域が存在しない」と論じたが、これは「群不変な関数が分離しない」ことしか言わず quality 関数の非存在の証明ではない（誤り、撤回）。frame 局在が弱いながら統制域で信号を出すこと自体がこの過剰主張への反証。正しい現状認識は経験的: 試した全 intrinsic 候補が 0.5–0.63 / count 交絡 / 合成 oracle 限定で、実 model 上の placement-quality 信号は見つからない。真の quality 評価には semantic/functional 入力か good/bad ラベルが要る。

### 18.7 暫定結論と未検証の問い

1. **diversity / convention-distance 軸としては set_sim が確立**: encoding 不変、human-human 一致 0.84、実 model を正しく強く分離（z≈+11〜12）、count 交絡なし、SVD のみ。試した全候補で最良。
2. **ただし純粋 placement-quality 軸としては不十分**: 統制域で 0.62、chain/datum 交絡、強い分離は relational+coverage（motion_angle 冗長）の疑い。
3. **intrinsic placement-quality は全滅**（§18.6）。
4. **増分検証（§18.7a, 実施済み）**: set_sim が motion_angle に対し *増分* の model 識別力を持つか。

### 18.7a set_sim の motion_angle に対する増分（[diag_set_sim_increment.py](v4_exp/diag_set_sim_increment.py)）

同一 bundle で per (rotation, variant, gt) に set_dist=1−set_sim と motion_angle(=d_sub_angle, range の部分空間距離) を計算し、model−heldout の paired 差で増分を測定（n: base 276, solver_rl 369 rotation）:

- **相関(set_dist, motion_angle) = 0.775**（全 pred-gt 集約）。**set_sim は motion_angle と 78% 冗長**。
- **motion_angle が分離しない subset**（|d_motion_angle| 最小 1/3）で set_sim がなお human を優位にするか:

| model | tie subset n | set_dist が human 優位 / model 優位 | z | p |
|---|---:|---:|---:|---:|
| base | 92 | **49 / 24** | +2.59 | **0.0096** |
| solver_rl | 123 | 52 / 39 | +1.36 | 0.17 (n.s.) |

→ **増分は実在するが弱く不安定**: base では motion_angle が盲目な所で set_sim がなお human 優位（p<0.01, placement 残差信号）だが、solver_rl では有意でない。

**§18 最終判定**: set_sim は実質「motion_angle を basis 敏感に測った版」（r=0.78）＋弱い placement 残差（base のみ有意）。toy（§18.5）が示す通り別軸を*見てはいる*が、実 model 上での*増分*識別力は小さく一貫しない。したがって:
- **placement を独立の model-eval 軸として強く正当化はできない** → model ランキングは既存の motion_angle + leak_angle に任せる（従来推奨を維持）。
- **set_sim の確立された価値**: (i) §17 placement 軸の motion 空間での*存在の可測化*（encoding 不変、human 一致 0.84、toy で motion_angle 盲目性を可視化）、(ii) **diversity / convention-distance 記述子**としての利用（human-human floor 0.84 を正常範囲とする）。品質採点ではなく多様性記述に用いる。
- intrinsic placement-quality は全滅（§18.6）。真の quality 評価には semantic/functional 入力か good/bad ラベルが必要、というのが §17–§18 を貫く確定結論。
