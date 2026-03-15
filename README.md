# My-Vitruvion

## Install FreeCAD

We first need to build our custom FreeCAD for inference-time solver feedback:

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
This functionality is extensively used in [src/models/components/solver_feedback.py](src/models/components/solver_feedback.py).

## Install Dependencies

```bash
git clone https://github.com/kakukakujirori/My-Vitruvion.git
cd My-Vitruvion
mamba activate freecad
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

## Qwen3-VL Weight Download

```bash
mkdir -p qwen_weights
wget "https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-GGUF/resolve/main/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf?download=true" -O qwen_weights/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf
wget "https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-GGUF/resolve/main/mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf?download=true" -O qwen_weights/mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf
```

## Data

- SketchGraphs Dataset
  ```bash
  bash scripts/download_shards.sh data_shards
  python scripts/convert_sketchgraphs_to_seq.py input_folder=./data_shards output_folder=./data_processed
  ```

- Qwen3 Augmented Dataset
  ```bash
  python scripts/qwen3_refine_data_with_solver.py
  mkdir data_qwen && cp qwen3_refined_data_with_solver/*_refined.py data_qwen/
  python scripts/convert_freecad_to_seq.py ./data_qwen3
  ```

## Training

```bash
python src/train.py
```