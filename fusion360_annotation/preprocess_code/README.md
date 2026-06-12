# Fusion360 Annotation Guide

## Usage

```bash
conda activate freecad

FUSION360_DIR="disk2/Fusion360Gallery/r1.0.1/reconstruction"
FREECAD_CMD_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands"
FREECAD_CMD_CLEAN_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands_cleanup"
OUTSOURCE_DIR="disk2/Fusion360Gallery/r1.0.1/freecad_commands_outsource"
OUTSOURCE_DIR_SPLIT="disk2/Fusion360Gallery/r1.0.1/freecad_commands_outsource_split"

python scripts/convert_fusion360_to_freecad.py --test_data_only --deduplicate_bins 8 \
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

| Filename | Eko  |  Ryotaro  | Johnny | Sagar | Dany | Tanaka
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.zip | | Ryotaro | Johnny | Sagar
| 1.zip | | | Johnny | Sagar | Dany
| 2.zip | | | | Sagar | Dany | Tanaka
| 3.zip | | Ryotaro | Johnny | | Dany
| 4.zip | | | Johnny | Sagar | | Tanaka
| 5.zip | | Ryotaro | | | Dany | Tanaka
| 6.zip | Eko | Ryotaro | Johnny
| 7.zip | Eko | | | | Dany | Tanaka
| 8.zip | Eko | Ryotaro | | | | Tanaka
| 9.zip | Eko | Ryotaro | | Sagar
| 10.zip | Eko | | | Sagar | Dany
| 11.zip | Eko | | Johnny | | | Tanaka


## Annotators (Initial Test)

| Filename | Spare Filename | Worker Name | Time (min) | Skip Count | Reject Count  |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.zip | 61.zip | Sam | | | |
| 1.zip | 60.zip | Sagar | 2h20m | 0 | Moderate (Fixed) |
| 2.zip | 59.zip | Eko | 5h | 0 | Minor (Fixed) |
| 3.zip | 58.zip | av9525 | 20h | 0 | MANY (Fixed) |
| 4.zip | | | | | |
| 5.zip | 57.zip | Volodymyr | 5h | ? | So-so (Not Yet Fixed) |
| 6.zip | 56.zip | Johnny | | | |