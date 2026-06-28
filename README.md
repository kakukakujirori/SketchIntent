# Motion Subspace Canonicalization for Design-Intent Evaluation in CAD Sketches

## Install FreeCAD

We first need to build our custom FreeCAD for the CAD solver feedback in reinforcement learning (RL):

```bash
git clone https://github.com/kakukakujirori/FreeCAD.git
cd FreeCAD
git checkout -b solver-api origin/solver-api
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

- SketchIntent-292 (Evaluation Benchmark)
  ```bash
  wget "https://github.com/kakukakujirori/My-Vitruvion/releases/download/v.1.0.0/SketchIntent-292.zip"
  unzip SketchIntent-292.zip
  rm SketchIntent-292.zip
  ```

- SketchGraphs Dataset (Training)
  ```bash
  bash scripts/download_shards.sh data_shards
  python scripts/convert_sketchgraphs_to_seq.py
  python scripts/cleanse_seq_for_freecad.py data_processed/sg_filtered.npy
  ```

- Rule-Augmented Data (RLOO)

  Every geometric relation detected by the rule-based predicates is added as a relational constraint, producing `data_processed/sg_filtered_unique_augmented.npy` used as the RLOO relational pseudo-reference:
  ```bash
  bash scripts/generate_rule_augmented_dataset.sh
  ```
  The script runs 12 parallel shards (logging per-shard progress to `/tmp/rule_augmented.shard*.log`) and merges them into the output npy.

## Training

```bash
# Pretraining
python src/train.py

# RLOO (Solver-RL)
python src/train.py experiment=rloo

# RLOO (SolverReward + MotionReward + DimRelAlign)
python src/train.py experiment=rloo_motion
```