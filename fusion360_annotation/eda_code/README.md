# eda_code

Code for producing the numbers in the experiments section of the paper.
The data input is fixed to `data/SketchIntent-292/` at the repo root.

`data/SketchIntent-292/<batch>/` is assumed to contain:

- `<sketch_id>.py`: the input sketch before annotation
- `<sketch_id>_<label>.FCStd`: the annotated sketch

## Responsibilities

| module | responsibility |
|---|---|
| `config.py` | the fixed `data/SketchIntent-292/` paths, annotation labels, shared constants. |
| `manifest.py` | scanning `data/SketchIntent-292/`, exclusion rules, complete-trio detection. |
| `canonical.py` | lookup of `data/SketchIntent-292/<batch>/<sketch_id>.py`, sketch loading, canonical-geometry remap. |
| `file_bundle.py` | builds a `FileBundle` that remaps the annotation `.FCStd` and the model `.py` onto a canonical geometry. |
| `bundle_loader.py` | a thin adapter that converts the `data/SketchIntent-292/` manifest format into `FileBundle` construction arguments. |
| `dataset_features.py` | extracts features for dataset statistics from `.FCStd`. |
| `controls.py` | synthetic degradation controls; currently relational dropout. |
| `motion_metrics.py` | implementation of the paper's `theta_full`, `theta_rel`, `asgn_sim`. |
| `scripts/` | CLI entrypoints, run in order from the top. |

## Data and environment

Run everything from the repo root in the `freecad` env.

There is no CLI option to switch the data root. It always uses `./data/SketchIntent-292`.

To evaluate model inference outputs, edit `MODEL_OUTPUT_ROOTS` in `config.py`. Like `data/SketchIntent-292/`, each root must have subdirectories `0` through `11`, with `<sketch_id>.py` files inside each batch directory.

```python
MODEL_OUTPUT_ROOTS = {
    "base": REPO_ROOT / "model_outputs" / "base",
    "solver_rl": REPO_ROOT / "model_outputs" / "solver_rl",
}
```

If a configured root does not exist, it emits a warning and continues without the model row. If a root exists but a `.py` for an evaluated batch/file is missing, it stops with an error (treated as a missing inference output).

The output location is unified at `fusion360_annotation/eda_code/out/`. Keeping intermediate JSON, figures, CSV shards, and aggregation inputs in the same place keeps the downstream-script arguments simple.

To clean out results from a previous run, delete `fusion360_annotation/eda_code/out/`, where all artifacts are collected. Each script recreates the output directories it needs.

```bash
rm -rf fusion360_annotation/eda_code/out
```

Long-running scripts show a progress bar if `tqdm` is installed. On environments without `tqdm`, they fall back to periodic progress logs.

## Order of execution (top to bottom)

### 0. manifest sanity check

Scans `data/SketchIntent-292/` and checks the complete-trio count, the number of exclusions, and any missing canonical `.py`.

```bash
python fusion360_annotation/eda_code/scripts/00_check_manifest.py
```

### 1. dataset statistics extraction

Reads all annotation `.FCStd` and extracts per-annotation statistics for section 3 / the appendix.

```bash
python fusion360_annotation/eda_code/scripts/01_extract_dataset_stats.py \
  --out fusion360_annotation/eda_code/out
```

Output:

- `out/records.json`
- `out/summary.json`

### 2. section 3 figure

Builds the section 3 constraint-type distribution figure and `ch3_stats.json` from `records.json`.

```bash
python fusion360_annotation/eda_code/scripts/02_plot_ch3_distributions.py \
  --out fusion360_annotation/eda_code/out
```

Output:

- `out/ch3_constraint_distribution.pdf`
- `out/ch3_constraint_distribution.png`
- `out/ch3_stats.json`

### 3. appendix dataset statistics

Builds the appendix dataset statistics from `records.json`.

```bash
python fusion360_annotation/eda_code/scripts/03_appendix_stats.py \
  --out fusion360_annotation/eda_code/out
```

Output:

- `out/appendix_stats.json`

### 4. section 5 token-vs-motion extraction

Extracts per-human-human-pair token-F1 and motion disagreement into `ch5_pairs.json`.

```bash
python fusion360_annotation/eda_code/scripts/04_extract_ch5_pairs.py \
  --out fusion360_annotation/eda_code/out
```

Output:

- `out/ch5_pairs.json`
- `out/ch5_summary.json`

### 5. section 5 token-vs-motion figure

Builds the section 5 token-vs-motion figure and `ch5_stats.json` from `ch5_pairs.json`.

```bash
python fusion360_annotation/eda_code/scripts/05_plot_ch5.py \
  --out fusion360_annotation/eda_code/out \
  --layout column
```

Output:

- `out/ch5_token_vs_motion_column.pdf`
- `out/ch5_token_vs_motion_column.png`
- `out/ch5_stats.json`

### 6. main motion metrics

Evaluates leave-one-out rotations and dropout controls with `theta_full` / `theta_rel` / `asgn_sim`.

```bash
python fusion360_annotation/eda_code/scripts/06_run_motion_eval.py \
  --out fusion360_annotation/eda_code/out/motion_eval.csv
```

Output:

- `out/motion_eval*.csv`
- `out/motion_eval*_model_status.csv`

### 7. HistCAD OES baseline

Evaluates the HistCAD OES baseline with the finite-edit solver over the same rotations / variants (`source_self`, `heldout`, dim-drop, rel-drop, model).
OES is heavy because it calls the finite-edit solver a great many times. Sharding by batch is recommended.

To run all batches in one go:

```bash
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --out fusion360_annotation/eda_code/out/oes_eval.csv
```

To run all batches sharded two batches at a time:

```bash
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 0 1 \
  --out fusion360_annotation/eda_code/out/oes_eval_b0_1.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 2 3 \
  --out fusion360_annotation/eda_code/out/oes_eval_b2_3.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 4 5 \
  --out fusion360_annotation/eda_code/out/oes_eval_b4_5.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 6 7 \
  --out fusion360_annotation/eda_code/out/oes_eval_b6_7.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 8 9 \
  --out fusion360_annotation/eda_code/out/oes_eval_b8_9.csv
python fusion360_annotation/eda_code/scripts/07_run_oes_eval.py \
  --batches 10 11 \
  --out fusion360_annotation/eda_code/out/oes_eval_b10_11.csv
```

To run them in parallel, append `&` to each command and `wait` at the end. Since OES is quite heavy, throttle the concurrency to suit your machine.

Multiple shards are read together by Step 11's `11_analyze_paper_numbers.py`, which globs `out/oes_eval*.csv`. No separate merge script is needed.

Output:

- `out/oes_eval*.csv`

### 8. token-F1 baseline

Evaluates the constraint token-F1 baseline over the same rotations / variants.

```bash
python fusion360_annotation/eda_code/scripts/08_run_token_f1.py \
  --out fusion360_annotation/eda_code/out/token_f1.csv
```

Output:

- `out/token_f1.csv`

### 9. human agreement

Computes human-human dimensional-placement Jaccard and token-F1.

```bash
python fusion360_annotation/eda_code/scripts/09_run_agreement.py \
  --out fusion360_annotation/eda_code/out/agreement.csv
```

Output:

- `out/agreement.csv`

### 10. toy placement demo

Confirms that `S_place` picks up the dimensional-placement axis on square / rectangle toys.

```bash
python fusion360_annotation/eda_code/scripts/10_run_toy_placement.py
```

### 11. aggregate paper numbers

Aggregates the CSV shards in `out/` and prints the numbers for the paper tables and appendix. This script reads `motion_eval*.csv`, `oes_eval*.csv`, `token_f1*.csv`, and `agreement.csv` directly. It first prints Ours (`theta_full` / `theta_rel` / `S_place`) and the Baselines (OES / F1) side by side, matching the table format of `paper/sec/5_experiments.tex`. It then generates the appendix dataset figures into `paper/figs/` from `records.json` and `appendix_stats.json`.

```bash
python fusion360_annotation/eda_code/scripts/11_analyze_paper_numbers.py \
  --out-dir fusion360_annotation/eda_code/out
```

Output:

- `paper/figs/appendix_annotator_summary.pdf`
- `paper/figs/appendix_dataset_overview.pdf`

## Notes

- This version is not about "changing the metric"; it is about "making the finished `data/SketchIntent-292/` the single dataset contract".
- The appendix dataset figures are generated directly into `paper/figs/` by Step 11.
