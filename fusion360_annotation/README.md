# Fusion360 Annotation Guide

## Usage

```bash
conda activate freecad

FUSION360_DIR="disk2/Fusion360Gallery/r1.0.1/reconstruction"
FREECAD_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands"
OUTSOURCE_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands_outsource"
OUTSOURCE_DIR_SPLIT="disk2/Fusion360Gallery/r1.0.1/freecad_commands_outsource_split"

python scripts/convert_fusion360_to_freecad.py --test_data_only --deduplicate \
    --data_dir $FUSION360_DIR \
    --out_dir $FREECAD_DIR
python fusion360_annotation/remove_dof2_sketches.py \
    --data_dir $FREECAD_DIR \
    --out_dir $OUTSOURCE_DIR
python fusion360_annotation/add_reference.py \
    --data_dir $OUTSOURCE_DIR
python fusion360_annotation/split_outsourcing.py \
    --data_dir $OUTSOURCE_DIR \
    --out_dir $OUTSOURCE_DIR_SPLIT \
    --num_sketches_per_annotator 25
```

- `scripts/convert_fusion360_to_freecad.py` converts the Fusion360 JSON data to FreeCAD files.
- `fusion360_annotation/remove_dof2_sketches.py` removes fully constrained (dof=2) sketches because no more annotation is necessary.
- `fusion360_annotation/add_reference.py` copies existing primitives in a different sketch so that annotators can refer to the primitive configulation before adding new constraints.
- `fusion360_annotation/split_outsourcing.py` splits the sketches into multiple folders for annotators while maintaining DoF distribution.