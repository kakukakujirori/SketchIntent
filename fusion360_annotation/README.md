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

## Annotation Check

The following checks whether:
- Equal(A,A) bugs are not present
- X/Y axes are not used
- Block constraint count = 1, and removing it results in DoF=2.
- All existing primitives are present

`xxxxxx_mid.FCStd` files are not checked.

```bash
python fusion360_annotation/check_submission.py --dir path_to_annotated_folder
```

## Annotators (New)

| Filename | Annotator1  |  Annotator2  |  Annotator3 |
| :--- | :--- | :--- | :--- |
| 0.zip |
| 1.zip |
| 2.zip |
| 3.zip |
| 4.zip |
| 5.zip |
| 6.zip | Eko |  |  |
| 7.zip | Eko |  |  |
| 8.zip | Eko |  |  |
| 9.zip | Eko |  |  |
| 10.zip | Eko |  |  |
| 11.zip | Eko |  |  |
| 12.zip | Eko |  |  |


## Annotators (Initial Test)

| Filename | Spare Filename | Worker Name | Time (min) | Skip Count | Reject Count  |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.zip | 61.zip | Sam @abdusslaamajid | | | |
| 1.zip | 60.zip | sagar_poojary | 2h20m | 0 | Moderate (Fixed) |
| 2.zip | 59.zip | Eko Wahyudi | 5h | 0 | Minor (Fixed) |
| 3.zip | 58.zip | av9525 | 20h | 0 | MANY (Fixed) |
| 4.zip | | | | | |
| 5.zip | 57.zip | Volodymyr R | 5h | ? | So-so (Not Yet Fixed) |
| 6.zip | 56.zip | Johnny Web | | | |