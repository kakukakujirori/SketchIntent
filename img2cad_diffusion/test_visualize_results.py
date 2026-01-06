import argparse
import torch
from torchmetrics.functional.classification import multiclass_accuracy

from img2cad_diffusion.constraint_models import ConstraintMDDModule
from img2cad_diffusion.constraint_models_ar import ConstraintARModule
from img2cad_diffusion.constraint_data import ConstraintDataModule, ConstraintTokenInput
from img2cad_diffusion.solver_feedback import tokens_to_freecad_sketch, get_solver_status, HAS_FREECAD, FreeCADConversionError
from img2cad_diffusion.tokens import ConstraintToken, CONSTRAINT_ACCEPTABLE_REF_COUNTS
from img2cad_diffusion.tokenization import CONSTRAINT_TOKEN_LEN

if HAS_FREECAD:
    import FreeCAD


def tokens_to_readable(tokens: torch.Tensor, vocab_size: int = 64) -> list[str]:
    """Convert token tensor to human-readable list."""
    assert len(tokens.shape) == 1, f"{tokens.shape=}"
    result = []
    for t in tokens.cpu().numpy():
        if t == int(ConstraintToken.Start):
            result.append("[START]")
        elif t == int(ConstraintToken.Stop):
            result.append("[STOP]")
        elif t == int(ConstraintToken.Pad):
            result.append("[PAD]")
        elif t == int(ConstraintToken.Mask):
            result.append("[MASK]")
        elif t == int(ConstraintToken.InnerPad):
            result.append("[IPAD]")
        elif t < len(ConstraintToken):
            result.append(ConstraintToken(t).name)
        else:
            # Reference token
            result.append(f"R{t - len(ConstraintToken)}")
    return result


def print_constraints_readable(tokens: torch.Tensor, prefix: str = "", blind: bool = False) -> set[tuple[str, ...]]:
    """Print constraint tokens in block format."""
    readable = tokens_to_readable(tokens)

    # Structure: [START, C0(4 tokens), C1(4 tokens), ...]
    if not blind: print(f"{prefix}Prefix: {readable[:1]}")  # START

    ret = set()

    idx = 1
    constraint_idx = 0
    while idx < len(readable):
        block = readable[idx:idx+4]
        # if not block or block[0] in ["[PAD]", "[STOP]"]:
        #     # Print remaining
        #     remaining = readable[idx:]
        #     if len(remaining) > 10:
        #         print(f"{prefix}  ... {remaining[:5]} ... ({len(remaining)} tokens)")
        #     else:
        #         print(f"{prefix}  {remaining}")
        #     break
        if not blind: print(f"{prefix}  c{constraint_idx}: {block}")
        constraint_idx += 1
        idx += 4

        # cleanse
        if block[0] in ["[PAD]", "[STOP]", "[MASK]", "[IPAD]"] or block[0].startswith("R"):
            continue
        while block[-1] in ["[IPAD]"]:
            block = block[:-1]
        if not all(b.startswith("R") for b in block[1:]):
            continue
        if len(block[1:]) not in CONSTRAINT_ACCEPTABLE_REF_COUNTS[ConstraintToken[block[0]]]:
            continue
        block = block[:1] + sorted(block[1:])
        ret.add(tuple(block))

    return ret


@torch.no_grad()
def qualitative_eval(
    model_module: ConstraintMDDModule | ConstraintARModule,
    batch: ConstraintTokenInput,
    solver_feedback: bool,
    seed: int = 42,
):
    # ========= Show ground truth ========
    gt_tokens = batch['c_val'][0]
    print("\n[GROUND TRUTH CONSTRAINTS]")
    gt_constraints = print_constraints_readable(gt_tokens, prefix="  ")

    # Evaluate baseline with FreeCAD
    doc, fc_sketch, history_commands, mapping = tokens_to_freecad_sketch(
        batch['val'][0],
        gt_tokens,
        config.data.num_position_bins,
        token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
    )
    gt_status = get_solver_status(fc_sketch, mapping)
    FreeCAD.closeDocument(doc.Name)

    print(f"\n  Status: {gt_status.Status}")
    print(f"  DoF: {gt_status.DoF}")
    print(f"  Malformed: {gt_status.MalformedConstraints}")
    print(f"  Conflicting: {gt_status.ConflictingConstraints}")
    print(f"  Redundant: {gt_status.RedundantConstraints}")
    print(f"  Mapped Conflicts: {gt_status.MappedConflictingIndices}")
    print(f"  Mapped Redundant: {gt_status.MappedRedundantIndices}")

    # ======== Run ========
    if isinstance(model_module, ConstraintMDDModule):
        if solver_feedback:
            model_module.ProbPath.method = 'tau_leaping+solver_feedback'
            model_module.ProbPath.noise_removal = False
        else:
            model_module.ProbPath.method = 'tau_leaping'
            # model_module.ProbPath.noise_removal = False

        torch.manual_seed(seed)
        pred_sample, pred_intermediates = model_module.ProbPath.generate_sample(
            batch.copy(),
            num_steps=64,
            start_tokens=[int(ConstraintToken.Start)],
            constraint_blockwise_masking=True
        )
    elif isinstance(model_module, ConstraintARModule):
        torch.manual_seed(seed)
        pred_sample = model_module.generate_sample(
            batch.copy(),
            max_len=batch['c_val'].shape[1]
        )

    print("\n[PREDICTION RESULT]")
    pred_constraints = print_constraints_readable(pred_sample[0], prefix="  ")

    # Evaluate prediction with FreeCAD
    doc, fc_sketch, history_commands,mapping = tokens_to_freecad_sketch(
        batch['val'][0],
        pred_sample[0],
        config.data.num_position_bins,
        token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
    )
    pred_status = get_solver_status(fc_sketch, mapping)
    FreeCAD.closeDocument(doc.Name)

    with open("history_without_solver.py", "w") as f:
        f.write("\n".join(history_commands))

    print(f"\n  Status: {pred_status.Status}")
    print(f"  DoF: {pred_status.DoF}")
    print(f"  Malformed: {pred_status.MalformedConstraints}")
    print(f"  Conflicting: {pred_status.ConflictingConstraints}")
    print(f"  Redundant: {pred_status.RedundantConstraints}")
    print(f"  Mapped Conflicts: {pred_status.MappedConflictingIndices}")
    print(f"  Mapped Redundant: {pred_status.MappedRedundantIndices}")

    # ======== Quantitative metrics ========
    relevant_idxs = gt_tokens != ConstraintToken.Pad
    gt_tokens_valid = gt_tokens[relevant_idxs]
    pred_tokens_valid = pred_sample[0][relevant_idxs]

    num_pointer_targets = len(ConstraintToken) + config.data.max_token_length
    acc = multiclass_accuracy(gt_tokens_valid, pred_tokens_valid, num_classes=num_pointer_targets)

    print("\n[QUANTITATIVE METRICS]")
    print(f"{gt_constraints=}")
    print(f"{pred_constraints=}")
    print(f"Accuracy: (On Masked Tokens):            {acc}")
    print(f"IoU       (Intersection / Union):        {len(gt_constraints & pred_constraints)} / {len(gt_constraints | pred_constraints)}")
    print(f"Precision (Intersection / Prediction):   {len(gt_constraints & pred_constraints)} / {len(pred_constraints)}")
    print(f"Recall    (Intersection / Ground Truth): {len(gt_constraints & pred_constraints)} / {len(gt_constraints)}")


@torch.no_grad()
def quantitative_eval(
    model_module: ConstraintMDDModule | ConstraintARModule,
    batch: ConstraintTokenInput,
    solver_feedback: bool,
    seed: int = 42,
):
    # ======== Run ========
    if isinstance(model_module, ConstraintMDDModule):
        if solver_feedback:
            model_module.ProbPath.method = 'tau_leaping+solver_feedback'
            model_module.ProbPath.noise_removal = False
        else:
            model_module.ProbPath.method = 'tau_leaping'
            # model_module.ProbPath.noise_removal = False

        torch.manual_seed(seed)
        pred_sample, pred_intermediates = model_module.ProbPath.generate_sample(
            batch.copy(),
            num_steps=64,
            start_tokens=[int(ConstraintToken.Start)],
            constraint_blockwise_masking=True
        )
    elif isinstance(model_module, ConstraintARModule):
        torch.manual_seed(seed)
        pred_sample = model_module.generate_sample(
            batch.copy(),
            max_len=batch['c_val'].shape[1]
        )

    # ======== Quantitative metrics ========
    gt_tokens = batch['c_val']

    relevant_idxs = gt_tokens != ConstraintToken.Pad
    gt_tokens_valid = gt_tokens[relevant_idxs]
    pred_tokens_valid = pred_sample[relevant_idxs]

    num_pointer_targets = len(ConstraintToken) + config.data.max_token_length
    acc_total = multiclass_accuracy(
        gt_tokens_valid,
        pred_tokens_valid,
        num_classes=num_pointer_targets,
    )

    iou_total = 0
    precision_total = 0
    recall_total = 0
    valid_total = 0

    for i, (gt, pred) in enumerate(zip(gt_tokens, pred_sample)):
        gt_constraints = print_constraints_readable(gt, prefix="  ", blind=True)
        pred_constraints = print_constraints_readable(pred, prefix="  ", blind=True)
        intersection = gt_constraints & pred_constraints
        # NOTE: gt_constraints can be empty!!!
        iou_total += len(intersection) / max(1, len(gt_constraints | pred_constraints))
        precision_total += len(intersection) / max(1, len(pred_constraints))
        recall_total += len(intersection) / max(1, len(gt_constraints))

        # Evaluate prediction with FreeCAD
        try:
            doc, fc_sketch, history_commands,mapping = tokens_to_freecad_sketch(
                batch['val'][i],
                pred,
                config.data.num_position_bins,
                token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
            )
            pred_status = get_solver_status(fc_sketch, mapping)
            FreeCAD.closeDocument(doc.Name)
            if not (pred_status.MalformedConstraints or pred_status.ConflictingConstraints or pred_status.RedundantConstraints):
                valid_total += 1
        except FreeCADConversionError as e:
            continue


    iou_total /= len(gt_tokens)
    precision_total /= len(gt_tokens)
    recall_total /= len(gt_tokens)
    valid_total /= len(gt_tokens)

    return acc_total, iou_total, precision_total, recall_total, valid_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_mdm', type=str)
    parser.add_argument('--checkpoint_ar', type=str)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--index', type=int, default=None, help="Data index for qualitative check. If None, calculate quantitative metrics for the entire batch.")
    parser.add_argument('--solver_feedback', action='store_true', help="Only applicable to MDM")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.checkpoint_ar and args.checkpoint_mdm:
        raise ValueError("Cannot specify both --checkpoint_ar and --checkpoint_mdm")

    if args.checkpoint_ar:
        module = ConstraintARModule.load_from_checkpoint(
            args.checkpoint_ar, map_location=args.device, weights_only=False
        )
        module.to(args.device)
        module.eval()
        config = module.hparams
        assert not args.solver_feedback, "Solver feedback is only applicable to MDM."

    if args.checkpoint_mdm:
        module = ConstraintMDDModule.load_from_checkpoint(
            args.checkpoint_mdm, map_location=args.device, weights_only=False
        )
        module.to(args.device)
        module.eval()
        config = module.hparams

    config.data.primitive_noise.enabled = False
    config.data.batch_size = args.batch_size
    config.data.num_data_workers = 8
    datamodule = ConstraintDataModule(config.data)
    datamodule.setup()

    dataset = datamodule._dataset
    assert dataset is not None

    if args.index is not None:
        data_item = dataset[args.index]
        batch = {}
        for k, v in data_item.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0).to(args.device)
            else:
                batch[k] = [v]

        print(f"\n{'='*80}")
        print(f"DATA SAMPLE: Index {args.index}")
        print(f"{'='*80}")

        qualitative_eval(module, batch, args.solver_feedback, args.seed)
    else:
        dataloader, fc_dataloader = datamodule.val_dataloader()
        acc_total = 0
        iou_total = 0
        precision_total = 0
        recall_total = 0
        valid_ratio_total = 0
        cnt = 0
        for batch in dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            acc, iou, precision, recall, valid_ratio = quantitative_eval(module, batch, args.solver_feedback, args.seed)
            acc_total += acc
            iou_total += iou
            precision_total += precision
            recall_total += recall
            valid_ratio_total += valid_ratio
            cnt += 1
            break  # NOTE: SINCE IT'S VERY TIME CONSUMING!!!
        print(f"{'='*80}")
        print(f"Accuracy  (On Masked Tokens):            {acc_total / cnt}")
        print(f"IoU       (Intersection / Union):        {iou_total / cnt}")
        print(f"Precision (Intersection / Prediction):   {precision_total / cnt}")
        print(f"Recall    (Intersection / Ground Truth): {recall_total / cnt}")
        print(f"Valid Ratio:                             {valid_ratio_total / cnt}")
        print(f"{'='*80}")
