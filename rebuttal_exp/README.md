# Rebuttal experiments

Standalone scripts for the rebuttal. They import from `src/` and `ablation/` but never
modify them, and they write only under `artifacts/rebuttal/`.
The experiment definitions live in [`rebuttal_experiments.md`](rebuttal_experiments.md).

Everything runs in the `sketch_intent` environment, which provides FreeCAD 1.1.0:

```bash
CR="conda run -n sketch_intent"
```

---

## Experiment A — finite solver behaviour (`rebuttal_experiments.md` §1)

Answers *"the evaluation is based only on infinitesimal local behavior"*. For every ordered
pair of annotations (P, G) of the same sketch, each dimensional driver they both contain
receives the **same signed finite increment**, both sides are solved by FreeCAD from
independently solved baselines, and the exact nonlinear observable change of each side is
compared. The pre-edit paper metrics are computed once per pair and are never specialised
to a driver.

```bash
# unit tests (the FreeCAD ones skip themselves if FreeCAD is missing)
$CR python -m pytest rebuttal_exp/tests -q

# pilot
$CR python rebuttal_exp/finite_edit_validation.py --limit 10 \
    --length-alphas 0.001,0.02,0.20 --angle-degrees 0.1,2,20 \
    --out artifacts/rebuttal/finite_edit_pilot

# formal run over all 292 sketches
$CR python rebuttal_exp/finite_edit_validation.py --out artifacts/rebuttal/finite_edit

# correlations, failure accounting and figures
$CR python rebuttal_exp/analyze_finite_edit.py --in artifacts/rebuttal/finite_edit
```

### Outputs

| file | contents |
|---|---|
| `run_config.json` | config, its hash, counts, skip tally. Guards against mixing runs |
| `environment.json` | git commit + dirty files, FreeCAD/Python versions, dataset digest, command line, script digest |
| `pair_metrics.csv` | one row per directed annotation pair: θ_full, θ_rel, θ_leak, S_place, `placement_distance = 1 − S_place`, common-driver count, pair status |
| `driver_matches.csv` | every matched common driver with both baseline quantities |
| `baseline_solves.csv` | per-annotation baseline solve status, observable dimension, residual |
| `trials.csv` | one row per (pair, driver, sign, magnitude); the analysis unit |
| `correlations.csv` | Spearman ρ with sketch-cluster bootstrap CI, per analysis variant |
| `bootstrap_replicates.parquet` | raw bootstrap replicates of the primary analysis |
| `failure_summary.csv` | designed / invalid / success / failure counts per magnitude |
| `success_by_metric_quartile.csv` | written only when some magnitude drops below 95% success |
| `figures/` | ρ vs magnitude (length and angle), success rate vs magnitude, hexbin scatters |

### Protocol decisions

- **Edit magnitude.** Length drivers use `alpha × bbox_diag(canonical sketch)`; the canonical
  sketch is annotation-independent, so P and G receive the same physical increment and edit
  solves can be cached across pairs. Angle drivers use degrees, converted to the radians the
  sketch representation stores internally. Both are recorded in `trials.csv`.
- **Baseline.** Each annotation is solved once, unedited, and every finite edit starts from a
  fresh copy of that solved baseline — never warm-started from the previous magnitude.
- **Id space.** `_solve_via_freecad` returns a sketch whose entity ids are renumbered and whose
  constraint ids become `c_<i>`. `solved_sketch.rebase_geometry` writes the solved geometry back
  onto the input sketch, so canonical `base_eids`, `constraint_signature` and cached constraint
  ids stay valid. A round trip that does not preserve the entity sequence is rejected, not truncated.
- **Observable.** `exact_observable` recomputes φ from geometry with the same landmark order,
  pair enumeration and trailing radii block as `pairwise_distance_observable`. `J_phi @ delta_x`
  is never used. Layout disagreement between P and G is recorded, not silently compared.
- **Success.** The existing FreeCAD rule is reused unchanged: solver converged, geometry finite,
  and `|achieved − target| / bbox_diag ≤ HistCADConfig().er_target_tol` (1e-3). The residual norm
  is reported but does not gate, matching the existing FreeCAD path.
- **Failures are never imputed.** Solver failures get no substitute distance; they appear only in
  `failure_summary.csv`. `near_zero_both_response` is derived in the analyser from the recorded
  response norms, which is what makes the 1e-12 / 1e-9 / 1e-8 threshold sensitivity free.
- **Pair kinds.** `human` is the primary analysis; `model_human` is auxiliary and is always
  reported on separate rows, tables and curves. Model-model pairs are not generated.
- **Predictors** come from the bundle sketches, i.e. the same state the Sec. 5 table uses, and are
  the ordinary pair metrics — no driver-specific variant, no re-estimation from finite responses.

### Statistics

The analysis unit is one trial row. Rows of one sketch are not independent, so every confidence
interval is a **sketch-cluster bootstrap** (2,000 replicates; whole sketches are resampled with
replacement and all their rows are kept). Replicates where Spearman is undefined are counted, not
dropped silently, and `ci_unstable` flags any interval built on fewer than 1,500 valid replicates.
`correlations.csv` separates the primary analysis from the sensitivity variants
(`median_aggregated`, `unordered_only`, `near_zero_<threshold>`) via its `analysis` column.

### Sharding

FreeCAD solves are cheap here (~20 ms), so a single process is enough. If needed, shards split
deterministically by sketch key, not by worker:

```bash
for i in 0 1 2 3; do
  $CR python rebuttal_exp/finite_edit_validation.py --shard $i/4 \
      --out artifacts/rebuttal/finite_edit &
done; wait
```

Each shard writes to `<out>/shard_<i>_of_<n>/` and refuses to reuse a directory whose
`run_config.json` records a different config hash.

---

## Experiment C — primitive parameterization (`rebuttal_experiments.md` §3)

Answers *"it is unclear whether the metrics remain invariant under equivalent
reparameterizations"*. Two representations that real CAD systems actually use are
reproduced at Jacobian level. Both change the number of variables, so `B = ∂g/∂y` is
rectangular and is applied by right multiplication — no inverse of `B` is ever formed:

```
C0  identity, B = I                                       integration gate, not a result
C1  Line  [point, direction, start, end] (6) -> [start point, end point]        (4)
C2  Arc   [center, radius, start, end angle] (5) -> [start, sweep midpoint, end] (6)
```

The response is recomputed from scratch as `R~ = -pinv(J_x B) J_p` under the native
singular-value cutoff. Pushing the existing `R` through `B` would hide precisely what the
experiment measures: the parameterization dependence of the minimum-norm solution.

```bash
# C0 gate first; C1/C2 are not aggregated unless it passes
$CR python rebuttal_exp/parameterization_stress.py --scenarios native,C0 --limit 20 \
    --out artifacts/rebuttal/parameterization_c0
$CR python rebuttal_exp/analyze_parameterization.py --in artifacts/rebuttal/parameterization_c0

$CR python rebuttal_exp/parameterization_stress.py --out artifacts/rebuttal/parameterization
$CR python rebuttal_exp/analyze_parameterization.py --in artifacts/rebuttal/parameterization
```

### Outputs

| file | contents |
|---|---|
| `transformed_pair_metrics.parquet` | one row per (rid, variant, reference, scenario): the four metrics, their subspace ranks (including `rank(R)` for `theta_leak`), `n_y` |
| `transform_diagnostics.parquet` | per line/arc block: `rank_H`, `rank_B`, left-inverse error, dimensional/wrapped-angle/point round-trip errors, span, circumcircle condition, chosen 2π branch, gauge residuals |
| `skip_summary.csv` | designed / skipped counts by reason |
| `identity_gate.csv`, `identity_gate_ranks.csv` | the C0 gate |
| `summary.csv` | paired \|d'−d\| median/Q90/Q95/max, moved-record counts split by whether a subspace dimension also changed, Spearman with bootstrap CI |
| `ranking_changes.csv` | full and method-only variant ordering, per-sketch winner flip rate |
| `worst_cases.csv` | worst 20 per metric with the sketch's chart diagnostics alongside |

### Protocol decisions

- **Analysis unit** is the Sec. 5 row structure (`ablation/run_motion.py`): held-out human,
  dimensional/relational dropout controls, and each model run against two human references.
  Dropout variants use the same `stable_seed`, so all scenarios see identical variant
  sketches and the comparison is paired.
- **Reference aggregation** reuses `ablation.aggregate.aggregate` and is applied
  independently within each scenario; no reference id is carried across scenarios.
- **Paired populations.** A transformed scenario only covers sketches holding its target
  primitive, so every comparison — deltas *and* orderings — is restricted to the records
  both scenarios produced. Comparing an ordering over all sketches with one over the
  arc-bearing subset would measure the sample, not the chart.
- **Charts must earn their use.** Each block reports `rank(H)`, `rank(B)`, `H B ≈ I` (line)
  or `B H ≈ I` (arc), and a round-trip check; the arc's 2π branch integer is fixed at the
  baseline and never differentiated. Blocks failing the documented thresholds are skipped
  as `degenerate_line_span` / `ill_conditioned_arc_three_point` / `arc_reconstruction_failed`.
  Arc angle reconstruction is compared modulo 2π, and the arc's one-dimensional
  `ker(B)` (sliding the through-point along the same arc) is kept rather than squared away.
- **Gauge.** `ker(H_line)` is the native gauge the endpoint chart cannot see. An annotation
  whose `J_x`, `J_R` or `J_φ` still sees it above `1e-8` is excluded as
  `line_gauge_not_annihilated`; its residuals are still written to the diagnostics.
- **The C0 gate is split by magnitude.** `θ = sqrt(1 − cos²/max)` has a sqrt(machine eps)
  noise floor wherever two subspaces coincide, so resolved values must match to machine
  precision while near-zero values only have to stay inside that band. Both are reported.
- **Numerics vs dependence.** The null-space and range cutoffs are absolute, so a chart that
  rescales singular values can push a borderline direction across them. `summary.csv`
  therefore splits moved records by whether a subspace dimension also changed.

---

## Experiment D — pseudoinverse minimum-norm choice (`rebuttal_experiments.md` §4)

Answers *"the pseudoinverse selects one minimum-norm response although multiple valid
geometry updates exist"*. Whenever `ker(J_x)` is non-trivial, every `R0 + N Z` solves
`J_x R = -J_p` just as well. Only the **Pred** side is perturbed, so the variance has a
single source, and one scheme samples the freedom:

```
n_j        uniform on the unit sphere of ker(J_x), one draw per driver column
discard    if ||J_phi n_j|| < 1e-9   (below that floor the quotient below is roundoff)
Delta_j    = eta * ||(J_phi R0)_j|| * n_j / ||J_phi n_j||
           -> ||J_phi Delta_j|| = eta ||(J_phi R0)_j|| exactly; eta is an observable-space size
```

`eta` is normalised in observable space because raw state coordinates mix millimetres,
radians and Line direction components, so a Gaussian coefficient there would not mean the
same thing across sketches.

The experiment has **two halves that must be read together**:

- **Responses that cannot move.** `ker(J_x)` is largely invisible to `J_phi`: it contains
  global rigid motion (`phi` is a complete rigid invariant), the two-per-Line
  parameterisation gauge of `(pnt, dir, startParam, endParam)`, and every coordinate of an
  entity outside `base_eids`. When *all* of it is invisible — `sigma_max(J_phi N) < 1e-9`,
  reported as `exactly_invariant` — then `J_phi R = J_phi R0` for every admissible `R`, so
  `theta_full`, `theta_rel` and `S_place` are identical whichever response is chosen. Those
  responses are not sampled; `invariance_coverage.csv` counts them.
- **Responses that can.** Everything else is sampled over the `eta` sweep, and
  `sensitivity_summary.csv` reports how far each metric moves.

`theta_leak` is response-dependent in both halves: it compares `range(R)` in raw state
space, so unlike the other three it also responds to the gauge component a draw carries.
That is stated rather than split into a separate arm.

```bash
# eta = 0 gate first: the sampling path must reproduce pair_paper_metrics exactly
$CR python rebuttal_exp/pseudoinverse_sensitivity.py --etas 0 --limit 20 \
    --out artifacts/rebuttal/pseudoinverse_eta0
$CR python rebuttal_exp/analyze_pseudoinverse.py --in artifacts/rebuttal/pseudoinverse_eta0

$CR python rebuttal_exp/pseudoinverse_sensitivity.py --limit 50 \
    --out artifacts/rebuttal/pseudoinverse_pilot
$CR python rebuttal_exp/pseudoinverse_sensitivity.py --out artifacts/rebuttal/pseudoinverse
$CR python rebuttal_exp/analyze_pseudoinverse.py --in artifacts/rebuttal/pseudoinverse

# section 4.10 anchor, independent of the sampling run
$CR python rebuttal_exp/finite_anchor.py --out artifacts/rebuttal/pseudoinverse
```

### Outputs

| file | contents |
|---|---|
| `native_pair_metrics.parquet` | the unperturbed `pair_paper_metrics` row per (rid, variant, gt): the delta baseline and the gate |
| `perturbation_metrics.parquet` | one row per valid (rid, variant, gt, eta, sample): the four metrics, subspace ranks, `raw_perturbation_ratio`, `min_observable_gain`, `n_columns_guarded`, and post-perturbation constraint/nullspace residuals |
| `response_metadata.csv` | per annotation: `label_kind`, `raw_null_dim`, `observable_free_rank`, `observable_gain_max`, `exactly_invariant`, `consistency_residual`, driver count |
| `eta0_gate.csv` | the eta = 0 gate |
| `invariance_coverage.csv` | per population: how often `J_phi N = 0`, i.e. the response choice provably cannot move the observable metrics |
| `guard_summary.csv` | discarded draws per eta, and how many draws each candidate floor from `1e-12` to `1e-5` would have discarded |
| `floor_sensitivity.csv` | the reported median/Q90 recomputed under each candidate floor: what the threshold choice actually costs |
| `sensitivity_summary.csv` | \|d_eta − d_0\| median/Q90/Q95/max, signed quantiles, within-record sd and range, Spearman across samples and its bootstrap CI |
| `sensitivity_by_observable_free_rank.csv` | the same split by how many observable directions the response is free in |
| `ranking_changes.csv` | per eta: fraction of samples preserving the variant ordering, winner flip rate |
| `worst_cases.csv` | worst 20 per metric with the response's null dimensions and draw gain alongside |
| `finite_anchor.csv` | cosine and normalised error between the solver's small finite response and `J_phi R0[:, q]` |

### Protocol decisions

- **Pred only.** GT keeps the current definition (§4.2). Both-sides perturbation is not run.
- **One cutoff convention.** `ker(J_x)` and `rank(J_phi N)` use `compute_native_response`'s
  `eps_eff = max(1e-9, 1e-6 σ_max)`, and the draw guard is that rule's **absolute floor**
  applied to a single sampled direction — `sigma_max(J_phi N)` is literally the supremum of
  the quantity each draw is compared against, so the response-level and draw-level tests are
  one concept at two scales. The null space comes from a **full** SVD: the economy SVD of a
  wide matrix returns only `m` right singular vectors and would silently truncate it.
- **The floor is measured, not asserted.** There is no clean gap around it: over the full
  run the draw gains thin out continuously, and `guard_summary.csv`'s ladder shows each
  decade from `1e-10` to `1e-5` discarding a few hundred more draws of 1.29 million
  (`1e-9`: 28, `1e-7`: 491, `1e-5`: 697). What makes the choice harmless is the
  consequence rather than a gap, so `floor_sensitivity.csv` re-derives the reported
  statistics under each candidate floor — moving it four orders up to `1e-5` shifts every
  reported median and Q90 by under 0.5% relative. Only floors at or above the run's own
  floor can be checked this way; a lower one cannot be simulated from a finished run.
- **A guarded column is skipped, not resampled.** Retrying with a fresh seed would replace
  the uniform distribution on `ker(J_x)` with a conditioned one. The measured rejection rate
  does not justify that: once the exactly-invariant responses are set aside, the full run
  guarded 30 columns out of 1.29 million draws. The sampler stays a pure function and the
  counts are reported instead.
- **Exactly-invariant responses are excluded from the sensitivity tables, not hidden.** A row
  of zeros at every eta would pull the reported medians toward a movement the response cannot
  have; `invariance_coverage.csv` carries that population instead.
- **No metric formula is duplicated.** Experiment C had to copy `SketchMotion`'s assembly to
  inject `B`; here only `R` changes, so `motion_with_response` rebuilds exactly the
  `R`-dependent fields and `pair_paper_metrics` is reused. `PairEvaluator` caches the GT
  relation Jacobian at the Pred state (it does not depend on `R`) and `PredResponseState` is
  computed once per sample for both GT references; agreement with the paper path is asserted
  by unit tests and by the eta = 0 gate.
- **Mutation guard.** Each Pred's `native.R`, state vector and `U_rel` are fingerprinted
  before and after its sampling loop. §4.8 asks for theta_rel to be invariant across eta; the
  real content of that check is that no cache was mutated, and recomputing a quantity that
  does not depend on `R` would not test it.
- **Edge cases are categories, not silent drops.** `not_underconstrained`,
  `inconsistent_linear_system` (‖J_x R0 + J_p‖_F / max(‖J_p‖_F, 1) > 1e-6) and
  `no_nonzero_observable_response` are recorded per response; columns with
  ‖(J_phi R0)[:, j]‖ below 1e-12 receive no noise and are not counted as guarded draws
  (there is no size to scale to). Every sampled response is checked again after perturbation;
  a sample with ‖J_x R_eta + J_p‖_F / max(‖J_p‖_F, 1) > 1e-6 is counted and excluded.
- **Paired populations.** The run skips the responses it cannot perturb, so the ordering of
  each draw is compared against a native ordering rebuilt on exactly the records that draw
  produced.
- **Ties are ties.** An ordering is scored by counting inverted variant pairs *among the
  pairs the baseline separates by more than 1e-6*. theta_leak's aggregate median is exactly
  zero for many variants, and comparing sorted lists would report a reshuffle of ties as a
  broken ordering. The same tolerance gates the winner flip rate.
- **Single-threaded.** The per-sample matrices are small; intra-op threading cost more than
  it saved and a single core keeps this out of the training job's way.

### Statistics

The analysis unit is one (record, eta, sample) row. Spearman is reported as a distribution
over sample indices (median and minimum); the bootstrap CI is built on one fixed sample
index with a sketch-cluster bootstrap, and the spread across the other draws is reported
separately rather than folded into that interval.

## Module map

| module | responsibility |
|---|---|
| `common.py` | dataset iteration, sharding, run metadata, output-directory guard, sketch-cluster bootstrap, the padded-angle noise floors |
| `variants.py` | the Sec. 5 row structure: how the evaluated variants are built, ordered and classified |
| `solved_sketch.py` | FreeCAD solving that returns to the caller's id space; residual norm |
| `exact_observable.py` | the exact nonlinear φ and its layout |
| `driver_matching.py` | strict common-driver matching |
| `finite_edit_validation.py` | experiment A runner |
| `analyze_finite_edit.py` | correlations, failure accounting, figures |
| `primitive_charts.py` | the rectangular C1/C2 charts, their validation and gauge diagnostics |
| `transformed_motion.py` | the four metrics recomputed with `B` injected into every Jacobian |
| `parameterization_stress.py` | experiment C runner |
| `analyze_parameterization.py` | delta distributions, ordering stability, figures |
| `response_perturbation.py` | the admissible response family, its sampler and guard, per-response diagnostics |
| `perturbed_motion.py` | the paper metrics against a substituted `R`, with the `R`-independent parts cached |
| `pseudoinverse_sensitivity.py` | experiment D runner |
| `analyze_pseudoinverse.py` | invariance coverage, guard accounting, sensitivity, stratification, ordering stability, figures |
| `finite_anchor.py` | §4.10 solver anchor for the minimum-norm prediction |
| `deprecated/` | the superseded two-arm Experiment D implementation, archived for provenance |
