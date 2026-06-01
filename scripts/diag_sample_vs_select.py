"""Diagnostic #1 — sample-vs-select decomposition for an RLOO checkpoint.

Question this answers (research plan §14.7.4 item 1): for each constraint type,
is it (a) never sampled, or (b) sampled by the policy but driven back down by the
RLOO reward (negative within-group advantage)?  This converts the circumstantial
evidence from the training logs ("Parallel hits 0.032 mid-training then 0",
"Tangent ends below its pretrain prior") into a direct, one-checkpoint
measurement — no training required.

For one checkpoint it reproduces the *training* reward path (not the val
composer): for every prompt it draws ``K`` samples at the training temperature,
runs the *training* ``reward_composer`` (SolverReward + ResponseIntentReward +
RelationalClassPrior + DimParsimony) to get each sample's total reward, forms the
RLOO leave-one-out advantage, and then, per constraint *type*, reports:

- whether the type is sampled at all (occurrence count, per-sample / per-group
  presence),
- the *within-group* advantage contrast — mean advantage of the samples in a
  prompt-group that contain the type minus those that don't, averaged over groups
  where both occur.  This controls for prompt difficulty (same baseline) and is
  the clean "is this type selected for or against" signal,
- the solver-status mix of the samples that contain the type (risky =
  Conflicting / Malformed / Distorted / Failed) — tests whether a type like
  Tangent is suppressed because a slightly-off placement turns the whole sample
  Conflicting,
- for a focus set of (rare) types, the per-reward contribution contrast
  (solver vs intent vs rel_prior vs dim_parsimony) for samples that do / don't
  contain the type — shows *which* reward term penalises or rewards it.

Usage (defaults target the scaffold-v2 intent run, on its training reward GT):
    conda run -n freecad python scripts/diag_sample_vs_select.py \
        --ckpt logs/train/runs/2026-05-30_15-17-10/checkpoints/last.ckpt \
        --data data_processed/sg_pseudo_rel_167k_unique_valid.npy \
        --limit 128 --batch_size 32
"""
from __future__ import annotations

import argparse
import os

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.constraint_dataset import ConstraintDataset
from src.data.components.primitive_noise import PrimitiveNoise
from src.data.constraint_datamodule import sketch_collate_fn
from src.metrics.constraint_stats import constraint_head_counts
from src.sketch_handlers.constants import (
    CONSTRAINT_TYPE_TOKEN_IDS,
    DIMENSIONAL_CONSTRAINT_TOKEN_IDS,
    ConstraintToken,
)
from src.sketch_handlers.flip_augment import reflect_primitive_value_tokens

RISKY_STATUS = {"Conflicting", "Malformed", "Distorted", "Failed", "Timeout"}
DEFAULT_FOCUS = ["Tangent", "Parallel", "Perpendicular", "Equal", "Coincident", "Horizontal", "Vertical"]


def load_module(ckpt_path: str, device: torch.device) -> L.LightningModule:
    """Instantiate the LightningModule from the checkpoint's saved Hydra config."""
    config_file = os.path.join(
        os.path.dirname(ckpt_path).replace("checkpoints", ".hydra"), "config.yaml"
    )
    cfg = OmegaConf.load(config_file)
    module = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    module.load_state_dict(state_dict)
    module.eval().to(device)
    return module


@torch.no_grad()
def collect_records(module, loader, device, K, temperature):
    """Run sampling + the training reward composer; return flat per-sample arrays."""
    block_size = int(module.hparams.context.align_constraint_token_length_to)
    type_ids = torch.tensor(CONSTRAINT_TYPE_TOKEN_IDS, dtype=torch.long)
    names = [ConstraintToken(int(t)).name for t in CONSTRAINT_TYPE_TOKEN_IDS]
    dim_set = {int(t) for t in DIMENSIONAL_CONSTRAINT_TOKEN_IDS}
    families = ["dim" if int(t) in dim_set else "rel" for t in CONSTRAINT_TYPE_TOKEN_IDS]
    reward_weights = {r.name: float(r.weight) for r in module.reward_composer.rewards if r.enabled}

    adv_all, counts_all, status_all, group_all = [], [], [], []
    contrib_all: dict[str, list] = {n: [] for n in reward_weights}
    group_offset = 0

    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        B = int(batch["p_val"].shape[0])
        max_len = int(batch["c_val"].shape[1])
        expanded = {
            k: (v.repeat_interleave(K, dim=0) if isinstance(v, torch.Tensor) else [x for x in v for _ in range(K)])
            for k, v in batch.items()
        }
        sample_memory = module._compute_group_memory(module.model, batch, expanded, K)
        generated, _ = module.generate_sample(
            expanded, max_len, dtype=batch["c_val"].dtype,
            temperature=temperature, memory=sample_memory,
        )
        ctx = module.reward_composer.build_context(
            prim_tokens=expanded["p_val"], cons_tokens=generated,
            sketch_original_list=expanded["sketch"], gt_tokens=expanded["c_val"],
        )
        handles = module.reward_composer.submit_async(ctx)
        total, outputs = module.reward_composer.collect(handles, ctx)  # total [B*K, L]

        seq_reward = total.sum(dim=1).float()                          # [B*K]
        r = seq_reward.view(B, K)
        baseline = (r.sum(dim=1, keepdim=True) - r) / max(1, K - 1)
        adv = (r - baseline).reshape(B * K)                            # [B*K]

        counts = constraint_head_counts(generated, block_size, type_ids.to(generated.device), prefix_len=1)
        status = outputs["solver"].per_sample_solver_status or ["?"] * (B * K)

        adv_all.append(adv.cpu())
        counts_all.append(counts.cpu())
        status_all.extend(status)
        group_all.append(torch.arange(B, device="cpu").repeat_interleave(K) + group_offset)
        group_offset += B
        for name in reward_weights:
            sc = outputs[name].scores
            contrib_all[name].append((reward_weights[name] * sc).sum(dim=1).float().cpu())

    out = {
        "adv": torch.cat(adv_all),                       # [N]
        "counts": torch.cat(counts_all),                 # [N, n_types]
        "status": status_all,                            # list[N]
        "group": torch.cat(group_all),                   # [N]
        "contrib": {n: torch.cat(v) for n, v in contrib_all.items()},
        "names": names, "families": families, "weights": reward_weights, "K": K,
    }
    return out


def _expand(batch: dict, K: int) -> dict:
    """Repeat each prompt K times (the same K-expansion the RLOO loop uses)."""
    return {
        k: (v.repeat_interleave(K, dim=0) if isinstance(v, torch.Tensor)
            else [x for x in v for _ in range(K)])
        for k, v in batch.items()
    }


@torch.no_grad()
def collect_records_flip(module, dataset, device, K, temperature, axes, batch_size):
    """sample-vs-select records with flip-augmented views pooled per prompt.

    For every prompt we draw K samples from each view in ``["orig", *axes]``
    (orig + the reflections requested, e.g. ["x", "y"]).  Each view conditions
    generation on the reflected primitives — reflected in TOKEN space via the
    same ``flip_augment.reflect_primitive_value_tokens`` the RLOO training loop
    uses (single source of truth) — but is SCORED against the original sketch/GT,
    and all views of a prompt share one RLOO group, so the leave-one-out
    advantage now sees the flip-induced structural diversity.  Returns the flat
    per-sample arrays (same schema as collect_records) plus a per-sample ``view``
    tag so the caller can compare orig-only vs pooled.
    """
    views = ["orig", *axes]
    num_bins = int(module.hparams.context.num_quantization_bins)
    block_size = int(module.hparams.context.align_constraint_token_length_to)
    type_ids = torch.tensor(CONSTRAINT_TYPE_TOKEN_IDS, dtype=torch.long)
    names = [ConstraintToken(int(t)).name for t in CONSTRAINT_TYPE_TOKEN_IDS]
    dim_set = {int(t) for t in DIMENSIONAL_CONSTRAINT_TOKEN_IDS}
    families = ["dim" if int(t) in dim_set else "rel" for t in CONSTRAINT_TYPE_TOKEN_IDS]
    reward_weights = {r.name: float(r.weight) for r in module.reward_composer.rewards if r.enabled}

    adv_all, counts_all, status_all, group_all, view_all = [], [], [], [], []
    contrib_all: dict[str, list] = {n: [] for n in reward_weights}
    group_offset = 0
    n = len(dataset)
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        items = [dataset[i] for i in idxs]
        orig_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in sketch_collate_fn(items).items()}
        B = len(idxs)
        max_len = int(orig_batch["c_val"].shape[1])
        exp_orig = _expand(orig_batch, K)

        # group winner / advantage are pooled across views, so collect each
        # view's per-sample reward first, then form one baseline per prompt.
        view_rewards, view_counts, view_status, view_contrib = [], [], [], []
        for view in views:
            if view == "orig":
                cond_batch = orig_batch
            else:
                cond_batch = {**orig_batch, "p_val": reflect_primitive_value_tokens(
                    orig_batch["p_val"], orig_batch["p_attr"], view, num_bins)}
            exp_cond = _expand(cond_batch, K)
            memory = module._compute_group_memory(module.model, cond_batch, exp_cond, K)
            generated, _ = module.generate_sample(
                exp_cond, max_len, dtype=orig_batch["c_val"].dtype,
                temperature=temperature, memory=memory,
            )
            # Score on the ORIGINAL primitives / sketch / GT (coordinate-free
            # constraints transfer back exactly).
            ctx = module.reward_composer.build_context(
                prim_tokens=exp_orig["p_val"], cons_tokens=generated,
                sketch_original_list=exp_orig["sketch"], gt_tokens=exp_orig["c_val"],
            )
            handles = module.reward_composer.submit_async(ctx)
            total, outputs = module.reward_composer.collect(handles, ctx)
            view_rewards.append(total.sum(dim=1).float().view(B, K))     # [B,K]
            view_counts.append(
                constraint_head_counts(generated, block_size, type_ids.to(generated.device), prefix_len=1).cpu()
            )
            view_status.append(outputs["solver"].per_sample_solver_status or ["?"] * (B * K))
            view_contrib.append({
                nm: (reward_weights[nm] * outputs[nm].scores).sum(dim=1).float().cpu()
                for nm in reward_weights
            })

        # Pooled RLOO baseline over all views*K samples of each prompt.
        r_pool = torch.cat(view_rewards, dim=1)                          # [B, V*K]
        Ktot = r_pool.shape[1]
        baseline = (r_pool.sum(dim=1, keepdim=True) - r_pool) / max(1, Ktot - 1)
        adv_pool = (r_pool - baseline)                                   # [B, V*K]
        base_groups = torch.arange(B) + group_offset
        for vi, view in enumerate(views):
            adv_all.append(adv_pool[:, vi * K:(vi + 1) * K].reshape(B * K).cpu())
            counts_all.append(view_counts[vi])
            status_all.extend(view_status[vi])
            group_all.append(base_groups.repeat_interleave(K))
            view_all.extend([view] * (B * K))
            for nm in reward_weights:
                contrib_all[nm].append(view_contrib[vi][nm])
        group_offset += B

    return {
        "adv": torch.cat(adv_all), "counts": torch.cat(counts_all),
        "status": status_all, "group": torch.cat(group_all),
        "view": view_all,
        "contrib": {n: torch.cat(v) for n, v in contrib_all.items()},
        "names": names, "families": families, "weights": reward_weights, "K": K * len(views),
    }


def slice_rec(rec: dict, keep_views: set[str]) -> dict:
    """Return a copy of a flip rec restricted to the given views (for orig-only
    vs pooled comparison).  Group ids are preserved so ngrp stays meaningful."""
    mask = torch.tensor([v in keep_views for v in rec["view"]])
    group = rec["group"][mask]
    n_groups = max(1, int(group.unique().numel()))
    out = {
        "adv": rec["adv"][mask], "counts": rec["counts"][mask],
        "status": [s for s, m in zip(rec["status"], mask.tolist()) if m],
        "group": group,
        "contrib": {n: c[mask] for n, c in rec["contrib"].items()},
        "names": rec["names"], "families": rec["families"],
        "weights": rec["weights"], "K": int(group.numel()) // n_groups,
    }
    return out


def within_group_adv_delta(adv: torch.Tensor, has_t: torch.Tensor, group: torch.Tensor) -> tuple[float, int]:
    """Mean over groups (with both has/not) of mean adv(has) - mean adv(not)."""
    deltas = []
    for g in group.unique():
        m = group == g
        a, h = adv[m], has_t[m]
        if h.any() and (~h).any():
            deltas.append(float(a[h].mean() - a[~h].mean()))
    if not deltas:
        return float("nan"), 0
    return float(sum(deltas) / len(deltas)), len(deltas)


def report(rec, focus):
    adv, counts, status, group = rec["adv"], rec["counts"], rec["status"], rec["group"]
    names, families = rec["names"], rec["families"]
    N = adv.numel()
    n_groups = int(group.unique().numel())
    risky = torch.tensor([s in RISKY_STATUS for s in status])
    from collections import Counter
    sc = Counter(status)

    print(f"\n{'='*78}\nGLOBAL: {N} samples, {n_groups} groups, K={rec['K']}")
    print(f"  reward weights: " + ", ".join(f"{k}={v:g}" for k, v in rec["weights"].items()))
    print(f"  solver status: " + ", ".join(f"{k} {100*v/N:.1f}%" for k, v in sc.most_common()))
    print(f"  global risky rate: {100*float(risky.float().mean()):.1f}%")

    # Per-type table
    print(f"\n{'type':<16}{'fam':<5}{'n_occ':>7}{'samp%':>8}{'grp%':>7}{'advΔ_wg':>10}{'ngrp':>6}{'risky%_with':>12}")
    print("-" * 78)
    order = sorted(range(len(names)), key=lambda j: -int(counts[:, j].sum()))
    for j in order:
        n_occ = int(counts[:, j].sum())
        if n_occ == 0 and names[j] in ("Mask", "Pad", "Start", "Stop"):
            continue
        has = counts[:, j] > 0
        samp = 100 * float(has.float().mean())
        grp = 100 * len({int(g) for g, h in zip(group.tolist(), has.tolist()) if h}) / max(1, n_groups)
        d, ng = within_group_adv_delta(adv, has, group)
        risky_with = 100 * float(risky[has].float().mean()) if has.any() else float("nan")
        flag = "  <-- sampled-but-down" if (d == d and d < 0 and samp > 0.5) else ""
        print(f"{names[j]:<16}{families[j]:<5}{n_occ:>7}{samp:>8.2f}{grp:>7.1f}{d:>10.2f}{ng:>6}{risky_with:>12.1f}{flag}")

    # Per-reward contribution contrast for focus types
    print(f"\nPer-reward contribution contrast (mean over samples WITH the type minus WITHOUT):")
    print(f"  (negative solver => the type makes the solver reward worse; positive intent => intent rewards it)")
    hdr = f"{'type':<14}{'samp%':>7}" + "".join(f"{n:>12}" for n in rec["contrib"])
    print(hdr); print("-" * len(hdr))
    name_to_j = {n: j for j, n in enumerate(names)}
    for t in focus:
        if t not in name_to_j:
            continue
        has = counts[:, name_to_j[t]] > 0
        if not has.any() or not (~has).any():
            print(f"{t:<14}{100*float(has.float().mean()):>7.2f}   (insufficient split)")
            continue
        cells = []
        for n in rec["contrib"]:
            c = rec["contrib"][n]
            cells.append(f"{float(c[has].mean() - c[~has].mean()):>12.2f}")
        print(f"{t:<14}{100*float(has.float().mean()):>7.2f}" + "".join(cells))
    print("=" * 78)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="logs/train/runs/2026-05-30_15-17-10/checkpoints/last.ckpt")
    p.add_argument("--data", default="data_processed/sg_pseudo_rel_167k_unique_valid.npy",
                   help="Sketches to prompt with (default: the training reward GT).")
    p.add_argument("--limit", type=int, default=128, help="Number of prompt sketches (0=all).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--K", type=int, default=0, help="Samples per prompt (0 = module.group_sample_size).")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = module.train_sample_temperature.")
    p.add_argument("--focus", default=",".join(DEFAULT_FOCUS))
    p.add_argument("--flip-axes", default="",
                   help="Comma list of reflection axes to pool per prompt (e.g. 'x,y'); "
                        "empty = original single-view behavior. Tests whether flip-augmented "
                        "views manufacture the within-group counterfactuals §16.7 found missing.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    L.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)
    module = load_module(args.ckpt, device)
    K = args.K or int(module.group_sample_size)
    temp = args.temperature or float(module.train_sample_temperature)
    print(f"ckpt={args.ckpt}\ndata={args.data}  limit={args.limit}  K={K}  temp={temp}  device={device}")

    context = module.hparams.context
    dataset = ConstraintDataset(args.data, context=context, primitive_noise=PrimitiveNoise(std=0.0, max_diff=0.0))
    if args.limit and args.limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(args.limit)))

    axes = [a for a in args.flip_axes.split(",") if a]
    if axes:
        rec = collect_records_flip(module, dataset, device, K, temp, axes, args.batch_size)
        print(f"\n### FLIP-AUGMENT: views = orig + {axes}  (per-view K={K}, pooled group = {K * (1 + len(axes))})")
        print("\n--- ORIG-ONLY slice (status quo: K orig samples / prompt) ---")
        report(slice_rec(rec, {"orig"}), args.focus.split(","))
        print(f"\n--- POOLED slice (orig + {axes} in one RLOO group) ---")
        report(rec, args.focus.split(","))
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sketch_collate_fn,
        )
        rec = collect_records(module, loader, device, K, temp)
        report(rec, args.focus.split(","))

    for comp in (module.reward_composer, module.reward_val_composer):
        for m in comp.rewards if hasattr(comp, "rewards") else comp.metrics:
            reset = getattr(m, "_reset_executor", None)
            if callable(reset):
                reset()


if __name__ == "__main__":
    main()
