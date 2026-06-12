from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fusion360_annotation.eda_code.config import DIMENSIONAL_DISPLAY_TYPES

HERE = Path(__file__).resolve().parents[1]


def summary(xs):
    return {"mean": st.mean(xs), "median": st.median(xs), "min": min(xs), "max": max(xs), "stdev": st.pstdev(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    recs = json.loads((out_dir / "records.json").read_text())
    n = len(recs)

    fc_total = collections.Counter()
    uses = collections.Counter()
    for r in recs:
        for k, v in r["fc_constraint_counts"].items():
            fc_total[k] += v
        for k in r["fc_constraint_counts"]:
            uses[k] += 1
    fc_grand = sum(v for k, v in fc_total.items() if k != "Block")
    interface_usage = {
        k: {
            "count": fc_total[k],
            "pct_of_constraints": 100 * fc_total[k] / fc_grand,
            "annotations_using": uses[k],
            "frac_annotations_using": uses[k] / n,
        }
        for k in sorted(fc_total, key=lambda k: -fc_total[k])
        if k != "Block"
    }

    n_constr = [r["n_construction"] for r in recs]
    constr_type = collections.Counter()
    sketches_with_constr = collections.defaultdict(bool)
    for r in recs:
        for k, v in r["construction_counts"].items():
            constr_type[k] += v
        if r["n_construction"] > 0:
            sketches_with_constr[r["sketch_id"]] = True
    construction = {
        "annotations_with_construction": sum(c > 0 for c in n_constr),
        "frac_annotations_with_construction": sum(c > 0 for c in n_constr) / n,
        "sketches_with_construction": sum(sketches_with_constr.values()),
        "mean_construction_per_annotation": st.mean(n_constr),
        "max_construction": max(n_constr),
        "construction_by_type": dict(constr_type.most_common()),
    }

    prim = collections.Counter()
    for r in recs:
        for k, v in r["primitive_counts"].items():
            prim[k] += v
    prim_tot = sum(prim.values())
    primitive_dist = {
        k: {"count": prim[k], "pct": 100 * prim[k] / prim_tot}
        for k in sorted(prim, key=lambda k: -prim[k])
    }

    counts = {
        "n_constraints": summary([r["n_constraints"] for r in recs]),
        "n_primitives": summary([r["n_primitives"] for r in recs]),
        "n_relational": summary([r["n_relational"] for r in recs]),
        "n_dimensional": summary([r["n_dimensional"] for r in recs]),
        "dof_all_zero": all(r["dof"] == 0 for r in recs),
    }

    by_ann = collections.defaultdict(list)
    for r in recs:
        by_ann[r["annotator_name"]].append(r)
    per_annotator = {}
    for ann, rs in sorted(by_ann.items()):
        rel = sum(x["n_relational"] for x in rs)
        dim = sum(x["n_dimensional"] for x in rs)
        per_annotator[ann] = {
            "n_annotations": len(rs),
            "relational_pct": 100 * rel / (rel + dim) if (rel + dim) else 0.0,
            "mean_constraints": st.mean([x["n_constraints"] for x in rs]),
            "mean_construction": st.mean([x["n_construction"] for x in rs]),
            "frac_using_angle": sum(x["fc_constraint_counts"].get("Angle", 0) > 0 for x in rs) / len(rs),
        }

    trio_of = collections.defaultdict(set)
    for r in recs:
        trio_of[r["sketch_id"]].add(r["annotator_name"])
    trio_counter = collections.Counter(tuple(sorted(t)) for t in trio_of.values())
    stats = {
        "n_annotations": n,
        "n_sketches": len({r["sketch_id"] for r in recs}),
        "interface_type_usage": interface_usage,
        "construction": construction,
        "primitive_distribution": primitive_dist,
        "count_distributions": counts,
        "per_annotator": per_annotator,
        "trio_coverage": {
            "n_unique_trios": len(trio_counter),
            "sketches_per_trio": {", ".join(k): v for k, v in trio_counter.most_common()},
        },
    }
    (out_dir / "appendix_stats.json").write_text(json.dumps(stats, indent=2))

    anns = list(per_annotator)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = range(len(anns))
    ax.bar([i - 0.2 for i in x], [per_annotator[a]["relational_pct"] for a in anns], width=0.4, color="#2c7fb8", label="relational %")
    ax2 = ax.twinx()
    ax2.bar([i + 0.2 for i in x], [per_annotator[a]["mean_construction"] for a in anns], width=0.4, color="#d95f0e", label="mean construction")
    ax.set_xticks(list(x))
    ax.set_xticklabels(anns, rotation=20, ha="right")
    ax.set_ylabel("relational %", color="#2c7fb8")
    ax2.set_ylabel("mean construction / sketch", color="#d95f0e")
    ax.set_title("Per-annotator bias")
    fig.tight_layout()
    fig.savefig(out_dir / "appendix_per_annotator.png", dpi=150)

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.0))
    a1.hist([r["n_constraints"] for r in recs], bins=20, color="#2c7fb8")
    a1.set_xlabel("constraints / annotation")
    a1.set_ylabel("count")
    a2.hist([r["n_primitives"] for r in recs], bins=range(3, 18), color="#41ab5d")
    a2.set_xlabel("primitives / annotation")
    fig2.tight_layout()
    fig2.savefig(out_dir / "appendix_count_hist.png", dpi=150)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
