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