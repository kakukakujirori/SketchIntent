# Motion Subspace Canonicalization for Design-Intent Evaluation in CAD Sketches

## Abstract
Design intent has long been central to CAD because it enables parametric models to remain reusable under later edits. In 2D sketches, one observable carrier is the constraint set, which specifies the geometric relations to preserve and the dimensions that drive redesign. Yet design-intent-aware constraints are hard to evaluate: valid annotations are non-unique, and existing token-level metrics and single-reference datasets conflate syntactic variation with intent-level errors. We address this with motion-subspace canonicalization, representing a constraint set by the local deformations it permits under infinitesimal dimension edits. This factors out command-level variation and reveals the behaviorally meaningful differences that govern redesign. We further introduce a multi-reference benchmark of 292 sketches, each independently annotated by three CAD designers. Experiments show that our metrics absorb syntactic annotation diversity while distinguishing model-predicted constraints from human designs and detecting constraint deficiencies. Beyond evaluation, the same canonical form yields reinforcement-learning signals for relation-aware constraint generation. Together, these results lay practical groundwork for design-intent-aware CAD evaluation and generation.

## Install FreeCAD

We first need to build our custom FreeCAD for the CAD solver feedback in reinforcement learning (RL):

```bash
git clone https://github.com/FreeCAD/FreeCAD.git
cd FreeCAD
git checkout 1.0.2
git apply ../patches/freecad-1.0.2-get-solver-status.patch
git submodule update --init --recursive
```

We use a mamba environment, following the instruction of FreeCAD:

```bash
mamba env create -f conda/environment.devenv.yml
mamba activate freecad

# Remove non-existent HDF5 paths from the environment's compiler wrappers
sed -i 's|-I/home/conda/[^ ]*H5FDsubfiling||g' $CONDA_PREFIX/bin/h5cc $CONDA_PREFIX/bin/h5c++ $CONDA_PREFIX/lib/libhdf5.settings

cmake --preset conda-linux-release -DBUILD_FEM_NETGEN=OFF
cmake --build build/release -j $(nproc)
```

After the build completes, modify the default argument of `ensure_freecad_imported` in [`src/sketch_handlers/freecad_utils.py`](src/sketch_handlers/freecad_utils.py) to your FreeCAD library directory path.

### Usage

This custom version exposes the `getSolverStatus()` API, which returns the CAD solver's diagnosis results:
```python
doc = FreeCAD.newDocument(doc_name)
fc_sketch = doc.addObject("Sketcher::SketchObject", "Sketch")

fc_sketch.addGeometry(...)
fc_sketch.addConstraint(...)

doc.recompute()
fc_sketch.getSolverStatus()
# {
#   'DoF': xxx,
#   'Status': 'Converged', 'Redundant', 'Conflicting', 'Malformed', or 'Failed'
#   'RedundantConstraints': [c_id1, c_id2, ...],
#   'ConflictingConstraints': [c_id1, c_id2, ...],
#   'MalformedConstraints': [c_id1, c_id2, ...],
#   'ConflictGroups': [[c_id1, c_id2, ...], [c_id3, c_id4, ...], ...]
# }
```
Note that the constraint indices are 0-based.
Please refer to [src/models/components/solver_feedback.py](src/models/components/solver_feedback.py) for its usage.

## Install Dependencies

```bash
git clone https://github.com/kakukakujirori/My-Vitruvion.git
cd My-Vitruvion
mamba activate freecad
pip install -r requirements.txt
```

## Data

All datasets live under `data/`: the evaluation benchmark in `data/SketchIntent-292/`, the raw SketchGraphs shards in `data/SketchGraphs/`, and the converted / generated npy files in `data/processed/`.

### SketchIntent-292 (Evaluation Benchmark)

Our proposed benchmark dataset is available from the GitHub Releases page. Download it and extract it under `data/`.
```bash
wget "https://github.com/kakukakujirori/My-Vitruvion/releases/download/v.1.0.0/SketchIntent-292.zip"
unzip SketchIntent-292.zip -d data/
rm SketchIntent-292.zip
```
Read [data/SketchIntent-292/README.md](data/SketchIntent-292/README.md) for the details.

### Validation Set (Pretrain / RL)

The training-time validation set (`val_sequence_file` in the configs) is derived from SketchIntent-292: gather the `_R.FCStd` annotation (one of the six designers) from every batch and run it through the same FreeCAD→sequence pipeline as SketchGraphs, producing `data/processed/intent292_val_unique_valid.npy`:
```bash
mkdir -p /tmp/val_R
cp data/SketchIntent-292/*/*_R.FCStd /tmp/val_R/
python scripts/convert_freecad_to_seq.py --data_dir /tmp/val_R \
    --data_ext FCStd --out_path data/processed/intent292_val.npy
python scripts/cleanse_seq_for_freecad.py data/processed/intent292_val.npy
```

### SketchGraphs Dataset (Pretrain)

Downloads the raw SketchGraphs shards, converts them to our sequence format, and cleanses sketches that the FreeCAD solver cannot load, producing `data/processed/sg_filtered_unique_valid.npy`:
```bash
bash scripts/download_shards.sh data/SketchGraphs
python scripts/convert_sketchgraphs_to_seq.py
python scripts/cleanse_seq_for_freecad.py data/processed/sg_filtered.npy
```

### Rule-Augmented Data (RL)

Every geometric relation detected by the rule-based predicates is added as a relational constraint, producing `data/processed/sg_filtered_unique_augmented.npy` used as the RLOO relational pseudo-reference:
```bash
bash scripts/generate_rule_augmented_dataset.sh
```
The script runs 12 parallel shards (logging per-shard progress to `/tmp/rule_augmented.shard*.log`) and merges them into the output npy.

## Training

Training is a two-phase pipeline: supervised pretraining on SketchGraphs, then RLOO
fine-tuning with solver feedback. Configs live in [`configs/experiment/`](configs/experiment/) and are selected with Hydra's `experiment=` override. Checkpoints are written under `logs/train/runs/<timestamp>/checkpoints/`.

```bash
# 1. Supervised pretraining (autoregressive constraint generation)
python src/train.py

# 2. RLOO fine-tuning. Pick one of:

#    Solver-RL baseline: SolverReward + DimParsimony
python src/train.py experiment=rloo_solver_rl_coeff1.0

#    Ours: SolverReward + MotionReward + DimRelAlign
python src/train.py experiment=rloo_solver_motion_dimrelalign
```

**Notes:**

- The RLOO configs read the rule-augmented pseudo-reference (`data/processed/sg_filtered_unique_augmented.npy`) generated in the [Data](#data) section, so run that step first.
- Before fine-tuning, set `pretrained_ckpt_path` in the chosen experiment config yaml to the checkpoint produced by pretraining:
  ```yaml
  pretrained_ckpt_path: logs/train/runs/[yyyy-mm-dd_hh-mm-ss]/checkpoints/epoch_29.ckpt
  ```

## Inference

`src/test.py` generates a constraint set for each input sketch. The input `--data` can be a `.npy`, a single `.py`, or a directory of `.py` files (FreeCAD `.py` inputs are converted and cleansed automatically).

Run inference over the 12 benchmark batches, writing one output directory per batch:

```bash
CKPT=logs/train/runs/<timestamp>/checkpoints/<checkpoint>.ckpt
OUT=model_outputs/<my_run_name>

for split in $(seq 0 11); do
  python src/test.py --ckpt "$CKPT" --data data/SketchIntent-292/$split \
      --outdir "$OUT/$split" --n_iter 1
done
```

**Notes:**

- Each `$OUT/<split>/` holds the predicted `<id>.py` files plus a `results.json` with per-sketch solver status (DoF, fully-constrained / under-constrained / etc.).
- `--n_iter 1` is single-shot generation with no solver feedback (the evaluation protocol used in the paper); `--n_iter 5` runs up to five rounds of solver-in-the-loop self-correction, deleting malformed/conflicting/redundant constraints and rerun the model so it can refine its output.

## Evaluation

`src/eval/evaluate.py` scores a model-output directory against the SketchIntent-292 human annotations and reports three metric families: population solver metrics (FC / Under / Distorted rates, DoF), composition / over-dimensioning KPIs (dimensional ratio, relational ratios), and reference-geometry metrics (`motion_full` / `motion_relational` / `leak` angles and `asgn_sim` = S_place).

```bash
python src/eval/evaluate.py "$OUT" data/SketchIntent-292 --json "$OUT/metrics.json"
```

## Visualization

The generated python scripts can be loaded directly by FreeCAD GUI, by pasting them into its Python console.

To make the comparison of different model outputs easier, we provide a **Side-by-side interactive viewer**. It also displays the human-annotated references. You can also drag a vertex to watch an under-constrained sketch deform along its remaining degrees of freedom:
```bash
python src/eval/viz/compare.py model_outputs/run_a model_outputs/run_b
```

**Note:** It boots a headless FreeCAD GUI under `xvfb`, so it is **Linux-only**.

### Pretrained Weights

Reproducing the paper's evaluation does not require any checkpoint: the [Evaluation](#evaluation) and [Visualization](#visualization) tools score the provided model predictions (`model_outputs/`) against SketchIntent-292 directly. Pretrained and RLOO-fine-tuned checkpoints are needed only to run [Inference](#inference) from scratch or to continue [Training](#training), and will be released on publication. Once available, place a checkpoint anywhere and pass it via `--ckpt` (inference) or `pretrained_ckpt_path` in the experiment config (fine-tuning).