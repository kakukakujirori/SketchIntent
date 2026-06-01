"""Phase 1a — does shrinking SolverReward change what RLOO selects? (no training)

Background (research plan §16.4): swapping the ResponseIntent target changes the
intent-only ranking, but in the TOTAL reward the change is diluted — even on the
coeff_h_angle=8 probe-A checkpoint, intent's within-group advantage std (~0.7) is
~1/5 of the total (~3.5).  The total variance is dominated by SolverReward's big
swings (reward_dof_reduction=10, reward_fully_constrained=5).  So the lever for
"let the relation-intent signal actually drive selection" is to compress
SolverReward's dynamic range, not to keep raising coeff_h_angle (which scales an
already-near-zero within-group signal).

This script tests that BEFORE spending a training run.  It samples K candidates
per prompt once, scores them with the real training composer, then re-weights the
SolverReward contribution by a factor ``alpha`` (alpha=1 is the status quo;
alpha<1 shrinks the solver).  ``total_alpha = total - (1-alpha) * solver_contrib``
(no re-solve needed — the solver STATUS/DoF is fixed; only the reward mapping
shrinks).  For each alpha it reports:

- winner-flip / pairwise-disagreement vs the status quo (does selection move?),
- within-group advantage std of solver vs intent (does intent's share rise?),
- the SELECTED winner's mean key-relational ratio and distance-family ratio
  (does the rebalance steer the chosen candidate toward more relational / less
  dimensional, which is the actual goal?),
- type-conditioned within-group advantage for Tangent/Perp/Equal at alpha=1 vs a
  shrunk alpha.

Usage:
    conda run -n freecad python scripts/diag_reward_rescale.py \
        --ckpt logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt \
        --data data_processed/sg_pseudo_rel_167k_unique_valid.npy \
        --limit 128 --K 8 --device cuda
"""
from __future__ import annotations

import argparse
import math
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
    DISTANCE_FAMILY_CONSTRAINT_TOKEN_IDS,
    KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS,
    ConstraintToken,
)

DEFAULT_FOCUS = ["Tangent", "Perpendicular", "Equal", "Parallel", "Coincident"]
DEFAULT_ALPHAS = [1.0, 0.5, 0.3, 0.2, 0.1]


def load_module(ckpt_path: str, device: torch.device) -> L.LightningModule:
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


def _loo_adv(x: torch.Tensor, B: int, K: int) -> torch.Tensor:
    grouped = x.view(B, K)
    baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / max(1, K - 1)
    return (grouped - baseline).reshape(B * K)


def _winner_flip(a: torch.Tensor, b: torch.Tensor, B: int, K: int) -> float:
    wa = a.view(B, K).argmax(dim=1)
    wb = b.view(B, K).argmax(dim=1)
    return float((wa != wb).float().mean())


def _pairwise_disagreement(a: torch.Tensor, b: torch.Tensor, B: int, K: int) -> float:
    ga, gb = a.view(B, K), b.view(B, K)
    dis = comp = 0
    for g in range(B):
        for i in range(K):
            for j in range(i + 1, K):
                sa = math.copysign(1, (ga[g, i] - ga[g, j]).item()) if ga[g, i] != ga[g, j] else 0
                sb = math.copysign(1, (gb[g, i] - gb[g, j]).item()) if gb[g, i] != gb[g, j] else 0
                if sa == 0 or sb == 0:
                    continue
                comp += 1
                dis += int(sa != sb)
    return float(dis / comp) if comp else float("nan")


def _winner_mean(metric: torch.Tensor, total: torch.Tensor, B: int, K: int) -> float:
    w = total.view(B, K).argmax(dim=1)
    sel = metric.view(B, K)[torch.arange(B), w]
    return float(sel.mean())


def within_group_delta(score, has_t, group):
    deltas = []
    for g in group.unique():
        m = group == g
        h = has_t[m]
        if h.any() and (~h).any():
            v = score[m]
            deltas.append(float(v[h].mean() - v[~h].mean()))
    return (float(sum(deltas) / len(deltas)), len(deltas)) if deltas else (float("nan"), 0)


@torch.no_grad()
def collect(module, loader, device, K, temperature):
    block_size = int(module.hparams.context.align_constraint_token_length_to)
    all_types = torch.tensor(CONSTRAINT_TYPE_TOKEN_IDS, dtype=torch.long)
    key_rel = torch.tensor(KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS, dtype=torch.long)
    dist_fam = torch.tensor(DISTANCE_FAMILY_CONSTRAINT_TOKEN_IDS, dtype=torch.long)
    type_names = [ConstraintToken(int(t)).name for t in CONSTRAINT_TYPE_TOKEN_IDS]
    weights = {r.name: float(r.weight) for r in module.reward_composer.rewards if r.enabled}

    rows = {k: [] for k in ("total", "solver", "intent", "keyrel", "distfam", "group", "counts")}
    statuses = []
    goff = 0
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        B = int(batch["p_val"].shape[0])
        max_len = int(batch["c_val"].shape[1])
        exp = {k: (v.repeat_interleave(K, 0) if isinstance(v, torch.Tensor) else [x for x in v for _ in range(K)])
               for k, v in batch.items()}
        mem = module._compute_group_memory(module.model, batch, exp, K)
        gen, _ = module.generate_sample(exp, max_len, dtype=batch["c_val"].dtype, temperature=temperature, memory=mem)
        ctx = module.reward_composer.build_context(
            prim_tokens=exp["p_val"], cons_tokens=gen,
            sketch_original_list=exp["sketch"], gt_tokens=exp["c_val"],
        )
        handles = module.reward_composer.submit_async(ctx)
        total, outputs = module.reward_composer.collect(handles, ctx)

        c_all = constraint_head_counts(gen, block_size, all_types.to(gen.device), prefix_len=1).cpu().float()
        c_key = constraint_head_counts(gen, block_size, key_rel.to(gen.device), prefix_len=1).cpu().float()
        c_dst = constraint_head_counts(gen, block_size, dist_fam.to(gen.device), prefix_len=1).cpu().float()
        ntot = c_all.sum(dim=1).clamp(min=1)

        rows["total"].append(total.sum(dim=1).float().cpu())
        rows["solver"].append((weights.get("solver", 1.0) * outputs["solver"].scores).sum(dim=1).float().cpu())
        rows["intent"].append((weights.get("intent", 1.0) * outputs["intent"].scores).sum(dim=1).float().cpu())
        rows["keyrel"].append(c_key.sum(dim=1) / ntot)
        rows["distfam"].append(c_dst.sum(dim=1) / ntot)
        rows["counts"].append(constraint_head_counts(gen, block_size, all_types.to(gen.device), prefix_len=1).cpu())
        rows["group"].append(torch.arange(B).repeat_interleave(K) + goff)
        statuses.extend(outputs["solver"].per_sample_solver_status or ["?"] * (B * K))
        goff += B

    rec = {k: torch.cat(v) for k, v in rows.items()}
    rec.update(status=statuses, type_names=type_names, K=K, weights=weights)
    return rec


def report(rec, alphas, focus):
    total, solver, intent = rec["total"], rec["solver"], rec["intent"]
    group, K = rec["group"], rec["K"]
    B = int(group.unique().numel())
    counts, names = rec["counts"], rec["type_names"]
    name_to_j = {n: j for j, n in enumerate(names)}

    print(f"\n{'='*92}\nGLOBAL: {total.numel()} samples, {B} groups, K={K}")
    print(f"  reward weights: " + ", ".join(f"{k}={v:g}" for k, v in rec["weights"].items()))
    print(f"  status: " + ", ".join(f"{k} {100*c/len(rec['status']):.1f}%"
                                     for k, c in __import__("collections").Counter(rec["status"]).most_common()))
    print(f"  mean seq reward: total={float(total.mean()):.3f} solver={float(solver.mean()):.3f} "
          f"intent={float(intent.mean()):.3f}")

    base = total
    adv_solver = _loo_adv(solver, B, K)
    adv_intent = _loo_adv(intent, B, K)
    print(f"\n{'alpha':>6}{'winflip':>9}{'pair_dis':>10}{'advσ_solver':>13}{'advσ_intent':>13}"
          f"{'int/sol':>9}{'win_keyrel':>12}{'win_distfam':>13}{'win_intent':>12}")
    print("-" * 92)
    for a in alphas:
        total_a = total - (1.0 - a) * solver
        wf = _winner_flip(base, total_a, B, K)
        pd = _pairwise_disagreement(base, total_a, B, K)
        adv_sol_a = _loo_adv(a * solver, B, K)
        s_sol = float(adv_sol_a.std(unbiased=False))
        s_int = float(adv_intent.std(unbiased=False))
        ratio = s_int / s_sol if s_sol > 1e-9 else float("inf")
        wk = _winner_mean(rec["keyrel"], total_a, B, K)
        wd = _winner_mean(rec["distfam"], total_a, B, K)
        wi = _winner_mean(intent, total_a, B, K)
        print(f"{a:>6.2f}{wf:>9.3f}{pd:>10.3f}{s_sol:>13.3f}{s_int:>13.3f}"
              f"{ratio:>9.2f}{wk:>12.3f}{wd:>13.3f}{wi:>12.3f}")

    # type-conditioned within-group advantage: status quo (alpha=1) vs a shrunk alpha
    a_lo = min(alphas)
    adv_t1 = _loo_adv(total, B, K)
    adv_ta = _loo_adv(total - (1.0 - a_lo) * solver, B, K)
    print(f"\nType-conditioned within-group advantage (does shrinking solver lift the rare types?)")
    print(f"{'type':<15}{'samp%':>8}{'ngrp':>6}{'adv_total@1':>14}{f'adv_total@{a_lo:g}':>16}{'Δ':>9}")
    print("-" * 70)
    for t in focus:
        j = name_to_j.get(t)
        if j is None:
            continue
        has = counts[:, j] > 0
        samp = 100 * float(has.float().mean())
        d1, ng = within_group_delta(adv_t1, has, group)
        da, _ = within_group_delta(adv_ta, has, group)
        f = lambda x: "nan" if math.isnan(x) else f"{x:.3f}"
        print(f"{t:<15}{samp:>8.2f}{ng:>6}{f(d1):>14}{f(da):>16}{f(da - d1):>9}")
    print("=" * 92)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="logs/train/runs/2026-05-31_16-20-28/checkpoints/last.ckpt")
    p.add_argument("--data", default="data_processed/sg_pseudo_rel_167k_unique_valid.npy")
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--K", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    p.add_argument("--focus", default=",".join(DEFAULT_FOCUS))
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sketch_collate_fn)

    rec = collect(module, loader, device, K, temp)
    report(rec, [float(a) for a in args.alphas.split(",")], args.focus.split(","))

    for comp in (module.reward_composer, module.reward_val_composer):
        for m in (comp.rewards if hasattr(comp, "rewards") else comp.metrics):
            reset = getattr(m, "_reset_executor", None)
            if callable(reset):
                reset()


if __name__ == "__main__":
    main()
