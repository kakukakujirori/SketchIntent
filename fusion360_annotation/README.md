# Fusion360 Annotation Guide

## Usage

```bash
conda activate freecad

FUSION360_DIR="disk2/Fusion360Gallery/r1.0.1/reconstruction"
FREECAD_CMD_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands"
FREECAD_CMD_CLEAN_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands_cleanup"
OUTSOURCE_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands_outsource"
OUTSOURCE_DIR_SPLIT="disk2/Fusion360Gallery/r1.0.1/freecad_commands_outsource_split"

python scripts/convert_fusion360_to_freecad.py --test_data_only --deduplicate \
    --data_dir $FUSION360_DIR \
    --out_dir $FREECAD_CMD_DIR
python fusion360_annotation/cleanup_sketch.py \
    --data_dir $FREECAD_CMD_DIR \
    --out_dir $FREECAD_CMD_CLEAN_DIR
python fusion360_annotation/remove_dof2_sketches.py \
    --data_dir $FREECAD_CMD_CLEAN_DIR \
    --out_dir $OUTSOURCE_DIR
python fusion360_annotation/add_reference.py \
    --data_dir $OUTSOURCE_DIR
python fusion360_annotation/split_outsourcing.py \
    --data_dir $OUTSOURCE_DIR \
    --out_dir $OUTSOURCE_DIR_SPLIT \
    --num_sketches_per_annotator 25
```

- `scripts/convert_fusion360_to_freecad.py` converts the Fusion360 JSON data to FreeCAD files.
- `fusion360_annotation/cleanup_sketch.py` cleans up garbage/duplicate primitives and add/remove heuristically obvious constraints.
- `fusion360_annotation/remove_dof2_sketches.py` removes fully constrained (dof=2) sketches because no more annotation is necessary.
- `fusion360_annotation/add_reference.py` copies existing primitives in a different sketch so that annotators can refer to the primitive configulation before adding new constraints.
- `fusion360_annotation/split_outsourcing.py` splits the sketches into multiple folders for annotators while maintaining DoF distribution.


## Annotators

| Filename | Spare Filename | Worker Name | Time (min) | Skip Count | Reject Count  |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.zip | 61.zip | Sam @abdusslaamajid | | | |
| 1.zip | 60.zip | sagar_poojary | 2h20m | 0 | MyFix |
| 2.zip | 59.zip | Eko Wahyudi | 5h | 0 | MyFix |
| 3.zip | 58.zip | av9525 | 20h | 0 | MANY |
| 4.zip | | | | | |
| 5.zip | 57.zip | Volodymyr R | | | |
| 6.zip | 56.zip | Johnny Web | | | |
| 7.zip | | | | | |
| 8.zip | | | | | |
| 9.zip | | | | | |
| 10.zip | | | | | |
| 11.zip | | | | | |
| 12.zip | | | | | |
| 13.zip | | | | | |
| 14.zip | | | | | |
| 15.zip | | | | | |
| 16.zip | | | | | |
| 17.zip | | | | | |
| 18.zip | | | | | |
| 19.zip | | | | | |
| 20.zip | | | | | |
| 21.zip | | | | | |
| 22.zip | | | | | |
| 23.zip | | | | | |
| 24.zip | | | | | |
| 25.zip | | | | | |
| 26.zip | | | | | |
| 27.zip | | | | | |
| 28.zip | | | | | |
| 29.zip | | | | | |
| 30.zip | | | | | |
| 31.zip | | | | | |
| 32.zip | | | | | |
| 33.zip | | | | | |
| 34.zip | | | | | |
| 35.zip | | | | | |
| 36.zip | | | | | |
| 37.zip | | | | | |
| 38.zip | | | | | |
| 39.zip | | | | | |
| 40.zip | | | | | |
| 41.zip | | | | | |
| 42.zip | | | | | |
| 43.zip | | | | | |
| 44.zip | | | | | |
| 45.zip | | | | | |
| 46.zip | | | | | |
| 47.zip | | | | | |
| 48.zip | | | | | |
| 49.zip | | | | | |
| 50.zip | | | | | |
| 51.zip | | | | | |
| 52.zip | | | | | |
| 53.zip | | | | | |
| 54.zip | | | | | |
| 55.zip | | | | | |
| 56.zip | | | | | |
| 57.zip | | | | | |
| 58.zip | | | | | |
| 59.zip | | | | | |
| 60.zip | | | | | |
| 61.zip | | | | | |