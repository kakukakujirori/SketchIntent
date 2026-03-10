# My-Vitruvion

## Installation

```bash
conda create -n vitruvion python=3.12
conda activate vitruvion

# Install dependencies
pip install -r requirements.txt
```

We also need our customized FreeCAD for inference-time solver feedback.

[TODO: FreeCAD instrallation description.]

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