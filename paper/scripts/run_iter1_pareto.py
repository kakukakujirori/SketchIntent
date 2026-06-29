"""End-to-end single-shot (n_iter=1) Pareto pipeline for the Sec.6 RL-reward study.

ONE auditable script. For every variant checkpoint it:
  1. runs single-shot inference  -> src/test.py --n_iter 1 over data/SketchIntent-292/0..11
     into  model_outputs/Vitruvion_<variant>_step_<step>/<split>/
  2. evaluates                    -> src/eval/evaluate.py --json
     into  artifacts/fiverr_eval/iter1_<variant>_step<step>.json
     (composition incl. per-sketch dump + reference geometry metrics)
  3. draws the Pareto figure (FC vs dimensional-ratio / theta_motion_full / S_place)
     -> paper/figs/rl_reward_pareto_iter<N>.pdf  and  artifacts/fiverr_eval/iter<N>_pareto.png
  4. runs the FC-matched paired stats (Ours vs nearest-FC DP c1).

The ONLY evaluation operations performed are the two subprocess calls in `infer_split`
and `evaluate` below -- nothing else touches the metrics.

Idempotent / resumable (stepsave_*.ckpt grow over time):
  * eval JSON already exists                -> skip the checkpoint entirely
  * all 12 inference splits already present -> only (re)evaluate
  * otherwise                               -> generate just the missing splits, then eval

    conda run -n freecad python paper/scripts/run_iter1_pareto.py              # full run (single-shot)
    conda run -n freecad python paper/scripts/run_iter1_pareto.py --iter 5     # deployed solver-in-the-loop
    conda run -n freecad python paper/scripts/run_iter1_pareto.py --plot-only  # redraw only
    conda run -n freecad python paper/scripts/run_iter1_pareto.py --variants dpc1,ours

--iter namespaces all outputs (model_outputs/ + iter1_*.json for 1; model_outputs_iter<N>/ +
iter<N>_*.json otherwise), so the single-shot and deployed protocols never overwrite.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]  # paper/scripts/<file> -> repo root
MODEL_OUTPUTS = REPO / "model_outputs"
ART = REPO / "artifacts" / "fiverr_eval"
DATA_EVAL = REPO / "data/SketchIntent-292"

# Solver-intervention iterations at inference (1 = single-shot = the honest "pure policy"
# evaluation; 5 = the deployed solver-in-the-loop). Set from --iter in main(). Outputs are
# namespaced by ITER so the two protocols never overwrite each other:
#   iter==1 -> model_outputs/ + artifacts iter1_*.json ; else model_outputs_iter<N>/ + iter<N>_*.
ITER = 1


def mo_root() -> Path:
    return MODEL_OUTPUTS if ITER == 1 else REPO / f"model_outputs_iter{ITER}"


def jkey(variant: str, step: int) -> Path:
    return ART / f"iter{ITER}_{variant}_step{step}.json"
N_SPLITS = 12

# Only consider stepsave_*.ckpt within this step window: early checkpoints (<5k) are
# undertrained and late ones (>60k) drift past the comparison budget; restricting the
# range keeps every trajectory on the same training-budget footing.
STEP_MIN, STEP_MAX = 5000,70000 #60000

# --- variant registry -------------------------------------------------------
# kind="stepsave": use every stepsave_*.ckpt under run_dir/checkpoints.
# kind="ckpts":    explicit (step, filename) list under run_dir/checkpoints.
# Each entry also carries its plotting style (color, marker, connect-as-trajectory).
RUN = REPO / "logs" / "train" / "runs"
# `config` = experiment tail after `rloo_response_scaffold12_`; output dirs follow the
# project convention  model_outputs/Vitruvion_scaffold_v12_<config>_step_<step>.
# `ours` (28xstb2f) shares a config with `oursretry`, so its config carries the run
# timestamp to keep the two runs' folders distinct. `base` -> model_outputs/base.
VARIANTS = {
    "base":      dict(kind="ckpts", run=RUN/"2026-05-02_20-35-42", config=None,
                      ckpts=[(0, None)],  # resolved to epoch_29/last at runtime
                      color="#000000", marker="X", connect=False, label="Base (pretrained)"),
    "dpc1":      dict(kind="stepsave", run=RUN/"2026-06-16_22-29-57",
                      config="ablA_noML_dimprior_dimpars_c1",
                      color="#2171b5", marker="o", connect=True, label=r"Solver-RL ($\lambda_D{=}1$)"),
    "dpc03":     dict(kind="stepsave", run=RUN/"2026-06-16_22-40-00",
                      config="ablA_noML_dimprior_dimpars_c0.3",
                      color="#9ecae1", marker="o", connect=True, label=r"Solver-RL ($\lambda_D{=}0.3$)"),
    "dpc3":      dict(kind="stepsave", run=RUN/"2026-06-16_22-30-26",
                      config="ablA_noML_dimprior_dimpars_c3",
                      color="#08306b", marker="o", connect=True, label=r"Solver-RL ($\lambda_D{=}3$)"),
    "sm":        dict(kind="stepsave", run=RUN/"2026-06-15_14-37-56",
                      config="ablA_m4.5_dimprior_intent",
                      color="#fdae6b", marker="s", connect=True, label="SolverReward $+$ MotionReward"),
    # Solver-RL + M = Motion + DimParsimony (NOT Leak). The Leak<->DimParsimony
    # substitution control: same M as Ours, count-penalty instead of directional
    # Leak. Evaluated for the ablation TABLE only; plot=False keeps it out of the
    # main Pareto, where a 9th overlapping line would muddy the frontier read.
    "smdp":      dict(kind="stepsave", run=RUN/"2026-06-15_14-39-13",
                      config="ablA_m4.5_dimprior_dimpars_intent", plot=False,
                      color="#fdd0a2", marker="s", connect=True, label="SolverReward $+$ MotionReward $+$ DimParsimony"),
    "sl":        dict(kind="stepsave", run=RUN/"2026-06-18_20-36-15",
                      config="ablA_l1.5_leakdof_dimprior_intent",
                      color="#f16913", marker="s", connect=True, label="SolverReward $+$ DimRelAlign"),
    "oursretry": dict(kind="stepsave", run=RUN/"2026-06-18_21-03-15",
                      config="m4.5_l1.5_leakdof_dimprior_intent",
                      color="#a50f15", marker="s", connect=True, label="SolverReward $+$ MotionReward $+$ DimRelAlign (Ours)"),
    # Pruned C_G* ablation (App A2.2 tab:appendix_pseudo_ambiguous). Same config as Ours
    # but trained on the contact+axis pseudo-reference (Equal/Parallel/Perpendicular/
    # Mirror/Midpoint removed). Full arm of that table = oursretry above. plot=False:
    # table-only, kept off the main Pareto figure.
    "pruned":    dict(kind="stepsave", run=RUN/"2026-06-22_00-33-07",
                      config="m4.5_l1.5_leakdof_dimprior_intent_contactaxis", plot=False,
                      color="#6a51a3", marker="^", connect=True, label="Ours (pruned $C_G^\\star$)"),
    # Converged Ours: still evaluated (used by the FC-matched paired_stats), but kept
    # OFF the Pareto figure per request -- the trajectories tell the frontier story.
    "ours":      dict(kind="ckpts", run=RUN/"2026-06-13_19-03-56",
                      config="m4.5_l1.5_leakdof_dimprior_intent_2026-06-13_19-03-56",
                      ckpts=[(52200, "epoch_1_step_52200.ckpt"),
                             (59800, "step_59800_dof_5.107.ckpt")], plot=False,
                      color="#67000d", marker="s", connect=False, label="Ours (converged)"),
}

# Headline = dimensional ratio (independent over-dim axis) + theta_motion_full (proposed
# redesign-behavior metric). Third panel = S_place, reported honestly even though no variant
# separates on it: placement is not a target of any reward here, which we frame as an open
# axis / discriminative target metric for future design-intent-aware methods (see paper).
# First three = independent / proposed-evaluator headline axes. Last two are
# reward-family quantities (directly tied to MotionReward / DimRelAlign for
# the M/L variants), shown as MECHANISM checks -- they are not independent
# evaluation, but leak (theta_leak^dim) is over-dim sensitive, so Ours beating
# the baselines on it is indirect evidence of over-dimensioning suppression.
AXES = [
    ("dimensional_ratio", "Dimensional ratio (lower better)"),
    ("motion_full_angle", r"$\theta_{motion}^{full}$ (lower better)"),
    ("asgn_sim", r"$S_{place}$ (higher better)"),
    ("motion_relational_angle", r"$\theta_{motion}^{rel}$ (lower, mechanism)"),
    ("leak_relational_angle", r"$\theta_{leak}^{dim}$ (lower, mechanism)"),
]
MECHANISM_AXES = {"motion_relational_angle", "leak_relational_angle"}
MECHANISM_PANEL_FACE = "#f2f2f2"
# Human held-out level per axis: median multi-reference score of a held-out human
# annotation (leave-one-out), i.e. the bar a prediction must reach to be
# human-indistinguishable. Drawn as a horizontal reference so "model at/above (or below)
# human level" is readable at a glance. All five use the same median-multi-ref convention:
#   - motion_full / motion_relational / asgn_sim(S_place): Table 1 (tab:controls) human row.
#   - leak_relational_angle (theta_leak^dim): not in Table 1; computed alongside the
#     other angles in fusion360_annotation/eda_code (06_run_motion_eval + 11_analyze...),
#     human median = 0.000 (mean 0.090).
#   - dimensional_ratio: human-annotation dimensional share from eda_code/out/records.json
#     (median per-annotation 0.25; pooled 0.248 -- identical dimensional type set as the
#     figure's composition metric, so directly comparable).
HUMAN_HELDOUT = {
    "dimensional_ratio": 0.25,
    "motion_full_angle": 0.000,
    "motion_relational_angle": 0.000,
    "asgn_sim": 0.833,
    "leak_relational_angle": 0.000,
}

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 23, 21, 18, 20


# --- checkpoint discovery ---------------------------------------------------
def resolve_base_ckpt(run: Path) -> str | None:
    for name in ("epoch_29.ckpt", "last.ckpt"):
        p = run / "checkpoints" / name
        if p.exists():
            return str(p)
    cks = sorted(glob.glob(str(run / "checkpoints" / "epoch_*.ckpt")))
    return cks[-1] if cks else None


def jobs_for(variant: str, spec: dict) -> list[tuple[int, str]]:
    """Return [(step, ckpt_path)] for a variant, sorted by step."""
    out: list[tuple[int, str]] = []
    if spec["kind"] == "stepsave":
        for p in glob.glob(str(spec["run"] / "checkpoints" / "stepsave_*.ckpt")):
            step = int(Path(p).stem.split("_")[-1])
            if not (STEP_MIN <= step <= STEP_MAX):
                continue
            out.append((step, p))
    else:  # explicit ckpts
        for step, fname in spec["ckpts"]:
            if variant == "base":
                ck = resolve_base_ckpt(spec["run"])
            else:
                ck = str(spec["run"] / "checkpoints" / fname)
            if ck and Path(ck).exists():
                out.append((step, ck))
    return sorted(out)


# --- the two (and only two) evaluation operations ---------------------------
def infer_split(ckpt: str, split: int, outdir: Path) -> None:
    cmd = [sys.executable, "src/test.py", "--ckpt", ckpt,
           "--data", f"data/SketchIntent-292/{split}", "--n_iter", str(ITER),
           "--outdir", str(outdir / str(split))]
    subprocess.run(cmd, cwd=REPO, check=True,
                   stdout=open(outdir.parent / f".gen_{outdir.name}.log", "a"),
                   stderr=subprocess.STDOUT)


def evaluate(outdir: Path, json_path: Path) -> None:
    cmd = [sys.executable, "src/eval/evaluate.py",
           str(outdir), "data/SketchIntent-292", "--json", str(json_path)]
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    subprocess.run(cmd, cwd=REPO, check=True, env=env,
                   stdout=open(str(json_path) + ".log", "w"), stderr=subprocess.STDOUT)


def splits_complete(outdir: Path) -> bool:
    return all((outdir / str(i) / "results.json").exists() for i in range(N_SPLITS))


def outdir_for(variant: str, step: int) -> Path:
    """model_outputs/Vitruvion_scaffold_v12_<config>_step_<step>  (base -> model_outputs/base)."""
    config = VARIANTS[variant].get("config")
    if config is None:  # base (pretrained, not a scaffold12 run)
        return mo_root() / variant
    return mo_root() / f"Vitruvion_scaffold_v12_{config}_step_{step}"


def process(variant: str, step: int, ckpt: str) -> Path | None:
    outdir = outdir_for(variant, step)
    json_path = jkey(variant, step)
    if json_path.exists():
        print(f"[skip json]  {json_path.name}")
        return json_path
    outdir.mkdir(parents=True, exist_ok=True)
    if splits_complete(outdir):
        print(f"[eval only]  {variant} step {step}  (12 splits present)")
    else:
        print(f"[infer+eval] {variant} step {step}  ckpt={Path(ckpt).name}")
        for i in range(N_SPLITS):
            if (outdir / str(i) / "results.json").exists():
                continue
            infer_split(ckpt, i, outdir)
    evaluate(outdir, json_path)
    print(f"[done]       {json_path.name}")
    return json_path


# --- reading / plotting -----------------------------------------------------
def read_metrics(json_path: Path) -> dict:
    d = json.loads(json_path.read_text())
    s, c, r = d["solver"], d["composition"], (d.get("reference") or {})
    def rm(k):
        v = r.get(k)
        return float(v["mean"]) if isinstance(v, dict) and v.get("mean") is not None else float("nan")
    return {
        "fc_rate": s["fc_rate"], "valid_dof_mean": s.get("valid_dof_mean"),
        "dimensional_ratio": c.get("dimensional_ratio", float("nan")),
        "motion_full_angle": rm("motion_full_angle"),
        "leak_relational_angle": rm("leak_relational_angle"),
        "motion_relational_angle": rm("motion_relational_angle"),
        "asgn_sim": rm("asgn_sim"),
    }


def collect(active: list[str]) -> dict[str, list[tuple[int, dict]]]:
    rows: dict[str, list[tuple[int, dict]]] = {}
    for variant in active:
        pts = []
        for step, _ in jobs_for(variant, VARIANTS[variant]):
            jp = jkey(variant, step)
            if jp.exists():
                pts.append((step, read_metrics(jp)))
        if pts:
            rows[variant] = sorted(pts)
    return rows


def plot(rows: dict[str, list[tuple[int, dict]]]) -> None:
    fig, axes = plt.subplots(1, len(AXES), figsize=(5.5 * len(AXES), 6.2), dpi=180)
    for ax, (key, title) in zip(axes, AXES):
        for variant, pts in rows.items():
            spec = VARIANTS[variant]
            if not spec.get("plot", True):  # evaluated for the table, kept off the figure
                continue
            pp = sorted(pts, key=lambda sm: sm[1]["fc_rate"])
            xs = [m["fc_rate"] for _, m in pp]
            ys = [m[key] for _, m in pp]
            if spec["connect"] and len(pp) > 1:
                ax.plot(xs, ys, "-", color=spec["color"], alpha=0.6, lw=2.2, zorder=2)
            ax.scatter(xs, ys, s=spec.get("size", 70),
                       c=spec["color"], marker=spec["marker"], edgecolors="white",
                       linewidths=1.1, zorder=3,
                       label=spec["label"] if ax is axes[0] else None)
        hv = HUMAN_HELDOUT.get(key)
        if hv is not None:
            ax.axhline(hv, color="green", ls="--", lw=2.2, alpha=0.85, zorder=1)
        ax.set_title(title, fontsize=TITLE_FS)
        ax.set_xlabel("Fully constrained rate", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    hmap = dict(zip(labels, handles))
    human = Line2D([0], [0], color="green", ls="--", lw=2.2)
    # Legend = 2-row x 4-col grid. matplotlib fills ncol legends COLUMN-major (each
    # column top-to-bottom), so the flat order is interleaved (top, bottom, top, ...)
    # to land:
    #   top row    = Base, Solver-RL (blue circles) x3
    #   bottom row = Human held-out (directly under Base), proposed-reward (warm
    #                squares) x3
    # i.e. each consecutive (top, bottom) pair shares a column. Missing variants are
    # skipped. Labels spell out the reward names defined in Sec.6.2 (MotionReward /
    # DimRelAlign / DimParsimony weight lambda_D), not ad-hoc abbreviations.
    order = ["base", "__human__", "dpc03", "sm", "dpc1", "sl", "dpc3", "oursretry"]
    ord_h, ord_l = [], []
    for key in order:
        if key == "__human__":
            ord_h.append(human); ord_l.append("Human held-out")
        elif VARIANTS[key]["label"] in hmap:
            ord_h.append(hmap[VARIANTS[key]["label"]]); ord_l.append(VARIANTS[key]["label"])
    # Reserve a band below all panels for the shared legend so it never overlaps data.
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (key, _title) in zip(axes, AXES):
        if key not in MECHANISM_AXES:
            continue
        bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        pad = 0.008
        rect = Rectangle(
            (bbox.x0 - pad, bbox.y0 - pad),
            bbox.width + 2 * pad,
            bbox.height + 2 * pad,
            transform=fig.transFigure,
            facecolor=MECHANISM_PANEL_FACE,
            edgecolor="none",
            zorder=-10,
            clip_on=False,
        )
        fig.patches.append(rect)
    fig.legend(ord_h, ord_l, fontsize=LEGEND_FS, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.005), framealpha=0.95,
               handletextpad=0.4, columnspacing=1.8)
    ART.mkdir(parents=True, exist_ok=True)
    fig.savefig(ART / f"iter{ITER}_pareto.png")
    # iter-namespaced so iter1 and iter5 paper figures coexist and stay comparable.
    paper_fig = REPO / "paper" / "figs" / f"rl_reward_pareto_iter{ITER}.pdf"
    fig.savefig(paper_fig)
    print("wrote", ART / f"iter{ITER}_pareto.png", "and", paper_fig.relative_to(REPO))


def paired_stats(rows: dict) -> None:
    """Ours converged step52200 vs the DP c1 checkpoint nearest in FC."""
    if "ours" not in rows or "dpc1" not in rows:
        return
    ours_pt = next((m for st, m in rows["ours"] if st == 52200), None)
    if ours_pt is None:
        return
    dp = min(rows["dpc1"], key=lambda sm: abs(sm[1]["fc_rate"] - ours_pt["fc_rate"]))
    dp_step = dp[0]
    ours_json = jkey("ours", 52200)
    dp_json = jkey("dpc1", dp_step)
    print(f"\n[paired stats] Ours step52200 (FC {ours_pt['fc_rate']:.3f}) "
          f"vs DP c1 step{dp_step} (FC {dp[1]['fc_rate']:.3f})")
    subprocess.run([sys.executable, "paper/scripts/paired_stats_fc_matched.py",
                    "--ours", str(ours_json), "--base", str(dp_json)], cwd=REPO, check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma list of variants to process (default: all)")
    ap.add_argument("--iter", type=int, default=1,
                    help="solver-intervention iterations at inference (1=single-shot, 5=deployed). "
                         "Outputs are namespaced by this so iter1/iter5 never overwrite.")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    global ITER
    ITER = args.iter
    active = [v.strip() for v in args.variants.split(",") if v.strip() in VARIANTS]
    print(f"== n_iter = {ITER}  (outputs: {mo_root().name}/ , artifacts iter{ITER}_*.json) ==")

    if not args.plot_only:
        for variant in active:
            spec = VARIANTS[variant]
            js = jobs_for(variant, spec)
            if not js:
                print(f"[no ckpts]   {variant} ({spec['run']})")
                continue
            print(f"== {variant}: {len(js)} checkpoint(s) ==")
            for step, ckpt in js:
                try:
                    process(variant, step, ckpt)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] {variant} step {step}: {e}")

    rows = collect(active)
    if rows:
        plot(rows)
        paired_stats(rows)
    else:
        print("no JSONs yet; nothing to plot")


if __name__ == "__main__":
    main()
