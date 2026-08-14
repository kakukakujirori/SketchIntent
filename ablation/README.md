# Section 5 reproduction (experiments)

Reproduces the numbers in the paper's experiments section: the motion-subspace
metrics (`theta_full` / `theta_rel` / `S_place`, plus the dimensional-leak angle)
against the **OES** (HistCAD) and **token-F1** baselines, the **human-agreement
floor**, and the **synthetic-degradation controls** (dimensional / relational
dropout) that demonstrate the metrics absorb annotation diversity while detecting
constraint deficiencies.

It builds on the shared evaluation engine in [`src/eval`](../src/eval) and the
metric implementations in [`src/metrics`](../src/metrics); everything else it
needs is self-contained in this folder.

> Scope: this covers the **experiments table and controls only**. Dataset
> statistics and the Section 3 / appendix figures are out of scope here.

## Environment and inputs

Run everything **from the repository root** in the `sketch_intent` environment.

Inputs are fixed to two locations:

- `data/SketchIntent-292/` — the benchmark (human annotations + canonical inputs).
- `model_outputs/<run>/` — the model predictions to score as "model" rows.

The model runs to compare are configured in
[`src/eval/config.py`](../src/eval/config.py) via `MODEL_OUTPUT_ROOTS`
(label → `model_outputs/<run>` directory). Each configured root must contain
batch subdirectories `0`–`11` with per-sketch `<sketch_id>.py` files:

```python
MODEL_OUTPUT_ROOTS = {
    "base": REPO_ROOT / "model_outputs" / "base",
    "solver_rl": REPO_ROOT / "model_outputs" / "<your_solver_rl_run>",
    # add "ours": REPO_ROOT / "model_outputs" / "<ours_run>", etc.
}
```

If a configured root is missing, the run continues and that model's row is
reported as `NA`; the control / held-out / baseline rows (which come from the
human annotations) are unaffected.

All scripts write to `ablation/out/` by default; pass `--out` / `--out-dir` to
override. Pass `--limit N` to any script for a quick smoke run over the first
`N` sketches (the numbers will not match the paper at small `N`).

## Order of execution

The first four scripts each emit a CSV into `ablation/out/`; the fifth reads
them and prints the tables. They are independent and can run in any order before
`aggregate.py`.

### 1. Motion-subspace metrics

`theta_full`, `theta_rel`, `S_place` (`placement_sim`), and the dimensional-leak
angle over leave-one-out human rotations, the dropout controls, and the
configured model rows.

```bash
python ablation/run_motion.py
```

Output: `ablation/out/motion_eval*.csv` (+ `*_model_status.csv`).

### 2. OES (HistCAD) baseline

The HistCAD over-/under-specification baseline via the finite-edit solver, over
the same rotations / variants. **This is heavy** (the finite-edit solver is
called many times); shard by batch on a busy machine.

```bash
# all batches at once
python ablation/run_oes.py

# or shard two batches at a time (aggregate.py globs oes_eval*.csv, so no merge)
python ablation/run_oes.py --batches 0 1 --out ablation/out/oes_eval_b0_1.csv
python ablation/run_oes.py --batches 2 3 --out ablation/out/oes_eval_b2_3.csv
# ... batches 4 5 / 6 7 / 8 9 / 10 11
```

Output: `ablation/out/oes_eval*.csv`.

### 3. Token-F1 baseline

The constraint token-F1 baseline over the same rotations / variants.

```bash
python ablation/run_token_f1.py
```

Output: `ablation/out/token_f1.csv`.

### 4. Human agreement

Human–human dimensional-placement Jaccard and token-F1 (the agreement floor).

```bash
python ablation/run_agreement.py
```

Output: `ablation/out/agreement.csv`.

### 5. Aggregate the paper numbers

Reads the CSVs in `ablation/out/` and prints the tables: the main comparison
(Ours vs OES vs F1, matching `paper/sec/5_experiments.tex`), the dropout-control
medians, the paired Wilcoxon tests vs the held-out human, the placement-similarity
variants, and the human floors for `S_place` and the leak angle.

```bash
python ablation/aggregate.py
```

This script globs `motion_eval*.csv`, `oes_eval*.csv`, and `token_f1*.csv`, so
sharded OES outputs are picked up automatically.

## Layout

| file | role |
|---|---|
| `run_motion.py` | motion-subspace metrics (Ours) + controls + model rows |
| `run_oes.py` | HistCAD OES baseline |
| `run_token_f1.py` | constraint token-F1 baseline |
| `run_agreement.py` | human–human agreement floor |
| `aggregate.py` | reads `out/*.csv`, prints the paper tables |
| `controls.py` | synthetic relational/dimensional dropout variants |
| `bundle_loader.py` | `data/SketchIntent-292` → `FileBundle` adapter |
| `runtime_warnings.py` | model-root validation / warnings |
