"""
exp_linear_vs_finite.py — 提案 metric (線形応答) と HistCAD-style (有限編集) の相関分析。

提案 d_resp は ∂x/∂q の Frobenius 距離 = Δ→0 線形応答比較。
有限編集での実際の Δx と比較し、

  (a) 線形予測がどの Δ scale まで実用的に保つか
  (b) 線形予測差と有限差はどの程度相関するか

を 1 ファイル × 数 probe で可視化。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sketchgraphs.data as datalib  # noqa: E402

from src.metrics.edit_response_distance import (  # noqa: E402
    collect_probes,
    compute_native_response,
    edit_response_for_probe,
    pairwise_distance_observable,
    constraint_signature,
)
from src.metrics.histcad_2d import (  # noqa: E402
    HistCADConfig,
    _apply_edit,
    _newton_solve,
    _bbox_diag,
)
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402
from src.sketch_handlers.sketch_solver import SketchState  # noqa: E402

from canonical_sketch import (  # noqa: E402
    build_entity_remap,
    find_canonical_py,
    load_sketch_from_fcstd,
    load_sketch_from_py,
    metric_base_eids,
    remap_sketch_to_canonical,
)


def load_three_gts(filename: str, base_policy: str = "canonical_solid"):
    data_dirs = {
        "Eko":     "/home/sturk/Downloads/fiverr/Eko/9",
        "Sagar":   "/home/sturk/Downloads/fiverr/Sagar/9",
        "Ryotaro": "/home/sturk/Downloads/fiverr/Ryotaro/9",
    }
    annotators = list(data_dirs)
    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    canon = load_sketch_from_py(
        find_canonical_py(filename, data_dirs, annotators, None), converter
    )
    sketches = {}
    for a in annotators:
        s = load_sketch_from_fcstd(data_dirs[a] + "/" + filename, converter)
        rmp, extras, _ = build_entity_remap(s, canon, 0.5)
        s2, _ = remap_sketch_to_canonical(s, canon, rmp, extras)
        sketches[a] = s2
    base_eids = (
        metric_base_eids(base_policy, canon)
        if base_policy != "none" else None
    )
    return canon, sketches, base_eids, context


def linear_response_dx(sketch, probe, context, delta: float):
    """δx_linear = S_C^(q) × Δq for given finite Δ on the same probe."""
    native = compute_native_response(sketch, context)
    S = edit_response_for_probe(native, probe, context)
    if S is None:
        return None
    return float(delta) * S.squeeze(-1)        # (n_x,)


def finite_response_dx(sketch, probe, context, delta: float, q0: float):
    """δx_finite = x* - x_0 obtained by Newton-solving the edited sketch."""
    edited = _apply_edit(sketch, probe, q0 + delta)
    state, status, residual, _it = _newton_solve(edited, HistCADConfig(), context)
    if status != "converged" or state is None:
        return None, status, residual
    base = SketchState(sketch)
    return (state.q.detach() - base.q.detach()), status, residual


def evaluate_probe_across_deltas(
    sketch_pred,
    probe,
    context,
    deltas: list[float],
    J_phi=None,
):
    """For one (sketch, probe) pair, sweep delta and compare linear vs finite.

    If `J_phi` is given (m_phi, n_x), additionally project both δx_lin and δx_fin
    through it to obtain observable-space (gauge-invariant) deltas, and report
    a separate rel_err on those.
    """
    base_state = SketchState(sketch_pred)
    handler = context.get_constraint_handler(probe.constraint_type.name)
    try:
        r0 = handler.residual(probe, base_state).reshape(-1)
        target0 = handler.get_constraint_props(probe)["value"]
        q0 = float(r0[0]) + float(target0)
    except Exception:
        return None

    results = []
    for d in deltas:
        dx_lin = linear_response_dx(sketch_pred, probe, context, d)
        dx_fin, status, _resid = finite_response_dx(sketch_pred, probe, context, d, q0)
        if dx_lin is None:
            results.append({"delta": d, "lin_norm": None, "fin_norm": None,
                            "diff_norm": None, "status": "no_R",
                            "lin_obs_norm": None, "fin_obs_norm": None,
                            "rel_err_obs": None})
            continue
        lin_norm = float(torch.linalg.norm(dx_lin))
        if dx_fin is None:
            results.append({"delta": d, "lin_norm": lin_norm, "fin_norm": None,
                            "diff_norm": None, "status": status,
                            "lin_obs_norm": None, "fin_obs_norm": None,
                            "rel_err_obs": None})
            continue
        fin_norm = float(torch.linalg.norm(dx_fin))
        diff_norm = float(torch.linalg.norm(dx_fin - dx_lin))
        rec = {
            "delta": d,
            "lin_norm": lin_norm, "fin_norm": fin_norm,
            "diff_norm": diff_norm,
            "rel_err": diff_norm / max(fin_norm, 1e-12),
            "status": status,
        }
        if J_phi is not None:
            d_phi_lin = J_phi @ dx_lin
            d_phi_fin = J_phi @ dx_fin
            rec["lin_obs_norm"] = float(torch.linalg.norm(d_phi_lin))
            rec["fin_obs_norm"] = float(torch.linalg.norm(d_phi_fin))
            rec["diff_obs_norm"] = float(torch.linalg.norm(d_phi_fin - d_phi_lin))
            rec["rel_err_obs"] = (
                rec["diff_obs_norm"] / max(rec["fin_obs_norm"], 1e-12)
            )
        else:
            rec.update({"lin_obs_norm": None, "fin_obs_norm": None,
                        "diff_obs_norm": None, "rel_err_obs": None})
        results.append(rec)
    return q0, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filename", default="122579_2663ded3_0000_sketch04.FCStd")
    ap.add_argument("--annotators", default="Sagar,Ryotaro")
    ap.add_argument("--n-probes", type=int, default=5,
                    help="Number of dim probes to sweep (default first N native to pred).")
    ap.add_argument("--out-png", default=os.path.join(REPO_ROOT, "artifacts",
                                                       "exp_linear_vs_finite.png"))
    args = ap.parse_args()

    canon, sketches, base_eids, context = load_three_gts(args.filename)
    annotators = args.annotators.split(",")
    print(f"Sketch: {args.filename}, comparing {annotators}")

    # Use the union of all GT dim constraints (same as histcad driver),
    # then split into native vs foreign per pred.
    probes, weights = collect_probes(list(sketches.values()), kind="dim")

    # Observable Jacobian (gauge-invariant pairwise distances) on the canonical
    # sketch. Same J_φ used by the proposed metric in observable space.
    J_phi = pairwise_distance_observable(canon, base_eids=base_eids)
    print(f"J_phi shape: {tuple(J_phi.shape)}")

    bbox = _bbox_diag(sketches[annotators[0]])
    deltas = [bbox * s for s in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)]
    print(f"bbox_diag={bbox:.3f}, deltas (signed scaling): {[f'{d/bbox:.3%}' for d in deltas]}")

    # 3 columns now: ||δx||, rel_err raw x, rel_err observable space
    fig, axes = plt.subplots(len(annotators), 3,
                             figsize=(16, 4 * len(annotators)),
                             squeeze=False)

    summary_x = defaultdict(list)
    summary_obs = defaultdict(list)

    for ai, ann in enumerate(annotators):
        sketch = sketches[ann]
        ann_sigs = {constraint_signature(c) for c in sketch.constraints.values()}
        native = [(i, p) for i, p in enumerate(probes)
                  if constraint_signature(p) in ann_sigs]
        chosen = native[:args.n_probes] if native else list(enumerate(probes))[:args.n_probes]
        print(f"\n[{ann}] sweeping {len(chosen)} native probes")

        ax_norm, ax_relx, ax_relobs = axes[ai]
        for pi, probe in chosen:
            res = evaluate_probe_across_deltas(sketch, probe, context, deltas, J_phi=J_phi)
            if res is None:
                continue
            q0, rows = res
            xs, lin_ys, fin_ys, rel_ys, rel_obs_ys = [], [], [], [], []
            for r in rows:
                if r["fin_norm"] is None:
                    continue
                xs.append(r["delta"] / bbox)
                lin_ys.append(r["lin_norm"])
                fin_ys.append(r["fin_norm"])
                rel_ys.append(r["rel_err"])
                rel_obs_ys.append(r["rel_err_obs"])
            if not xs:
                continue
            sig = constraint_signature(probe)
            label = f"q{pi} {sig[0]}"
            ax_norm.plot(xs, lin_ys, "o--", alpha=0.5, label=f"lin {label}")
            ax_norm.plot(xs, fin_ys, "s-", alpha=0.7, label=f"fin {label}")
            ax_relx.plot(xs, rel_ys, "o-", label=label)
            ax_relobs.plot(xs, rel_obs_ys, "o-", label=label)
            for x, rel, rel_obs in zip(xs, rel_ys, rel_obs_ys):
                summary_x[ann].append((x, rel))
                if rel_obs is not None:
                    summary_obs[ann].append((x, rel_obs))

        ax_norm.set_title(f"{ann}: ‖δx‖ vs Δ/bbox")
        ax_norm.set_xlabel("Δ / bbox_diag")
        ax_norm.set_ylabel("||δx||")
        ax_norm.set_xscale("log"); ax_norm.set_yscale("log")
        ax_norm.grid(True, which="both", alpha=0.3)
        ax_norm.legend(fontsize=7, loc="best")

        ax_relx.set_title(f"{ann}: rel err in raw x space")
        ax_relx.set_xlabel("Δ / bbox_diag")
        ax_relx.set_ylabel("‖fin−lin‖ / ‖fin‖   (x space)")
        ax_relx.set_xscale("log"); ax_relx.set_yscale("log")
        ax_relx.axhline(0.1, color="gray", linestyle=":", label="10% threshold")
        ax_relx.grid(True, which="both", alpha=0.3)
        ax_relx.legend(fontsize=7, loc="best")

        ax_relobs.set_title(f"{ann}: rel err in observable space (J_φ · δx)")
        ax_relobs.set_xlabel("Δ / bbox_diag")
        ax_relobs.set_ylabel("‖J_φ(fin−lin)‖ / ‖J_φ fin‖")
        ax_relobs.set_xscale("log"); ax_relobs.set_yscale("log")
        ax_relobs.axhline(0.1, color="gray", linestyle=":", label="10% threshold")
        ax_relobs.grid(True, which="both", alpha=0.3)
        ax_relobs.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Linear (proposed) vs Finite (HistCAD-solver) edit response — {args.filename}\n"
        f"Right column = same comparison projected through J_φ (gauge-invariant)"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=130)
    print(f"\nSaved: {args.out_png}")

    print("\n--- mean rel err: raw x  vs  observable space ---")
    print(f"{'annotator':<10} {'Δ/bbox':>10} {'rel(x)':>10} {'rel(obs)':>10} {'n':>4}")
    for ann in sorted(summary_x):
        by_d_x: dict[float, list[float]] = defaultdict(list)
        by_d_o: dict[float, list[float]] = defaultdict(list)
        for x, e in summary_x[ann]:
            by_d_x[round(x, 4)].append(e)
        for x, e in summary_obs[ann]:
            by_d_o[round(x, 4)].append(e)
        for d in sorted(by_d_x):
            ex = np.mean(by_d_x[d])
            eo = np.mean(by_d_o[d]) if by_d_o[d] else float("nan")
            print(f"{ann:<10} {d:>10.3%} {ex:>10.4f} {eo:>10.4f} {len(by_d_x[d]):>4d}")


if __name__ == "__main__":
    main()
