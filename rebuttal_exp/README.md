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

## Module map

| module | responsibility |
|---|---|
| `common.py` | dataset iteration, sharding, run metadata, output-directory guard, sketch-cluster bootstrap |
| `solved_sketch.py` | FreeCAD solving that returns to the caller's id space; residual norm |
| `exact_observable.py` | the exact nonlinear φ and its layout |
| `driver_matching.py` | strict common-driver matching |
| `finite_edit_validation.py` | experiment A runner |
| `analyze_finite_edit.py` | correlations, failure accounting, figures |
| `primitive_charts.py` | the rectangular C1/C2 charts, their validation and gauge diagnostics |
| `transformed_motion.py` | the four metrics recomputed with `B` injected into every Jacobian |
| `parameterization_stress.py` | experiment C runner |
| `analyze_parameterization.py` | delta distributions, ordering stability, figures |
