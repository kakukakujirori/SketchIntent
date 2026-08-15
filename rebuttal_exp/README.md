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

## Module map

| module | responsibility |
|---|---|
| `common.py` | dataset iteration, sharding, run metadata, output-directory guard, sketch-cluster bootstrap |
| `solved_sketch.py` | FreeCAD solving that returns to the caller's id space; residual norm |
| `exact_observable.py` | the exact nonlinear φ and its layout |
| `driver_matching.py` | strict common-driver matching |
| `finite_edit_validation.py` | experiment A runner |
| `analyze_finite_edit.py` | correlations, failure accounting, figures |
