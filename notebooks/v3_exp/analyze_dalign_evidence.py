"""Phase A-2 + A-3 + B-5 evidence aggregation for d_align / d_feas.

Reads:
  artifacts/constrained_response_scores_full.csv  (drop_rates=0,100; with Vitruvion)
  artifacts/constrained_response_scores_dropout.csv (drop_rates=0,25,50,75,100; with Vitruvion)
  artifacts/source_conditioned_dropout_scores.csv (HistCAD OES trajectory)

Outputs to stdout:
  A-2: Wilcoxon signed-rank p-values for Vitruvion vs heldout-drop0 paired difference
       across d_resp, d_cresp, d_leak, d_feas, d_feas_tilde,
       d_align, d_align_phi.
  A-3: Tie 49-case decomposition: split d_align ties by sign of d_resp(Vit) - d_resp(heldout0).
  B-5: drop-rate trajectory of median raw min_d_align / min_d_resp / median_attain_err
       min_d_feas / min_d_feas_tilde / median_attain_err and
       HistCAD OES / ER / cPCSR, side-by-side.
"""
from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_csv(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path)) as f:
        return list(csv.DictReader(f))


def finite_or_none(s):
    if s is None or s == "" or s == "None":
        return None
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return None
    except ValueError:
        return None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ---------- A-2: Wilcoxon signed-rank ----------

def wilcoxon_signed_rank(diffs, eps=1e-9):
    """Two-sided p-value via normal approx with tie correction.

    diffs: list of (value_vit - value_human) raw differences.
    Treats |d|<eps as ties (excluded).
    Returns (W, n_nonzero, z, p_two_sided, n_pos, n_neg, n_zero).
    """
    nz = []
    n_zero = 0
    for d in diffs:
        if d is None:
            continue
        if abs(d) <= eps:
            n_zero += 1
        else:
            nz.append(d)
    n = len(nz)
    if n == 0:
        return None
    abs_d = sorted(((abs(d), 1 if d > 0 else -1) for d in nz), key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1][0] == abs_d[i][0]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    W_pos = sum(r for r, (_, s) in zip(ranks, abs_d) if s > 0)
    W_neg = sum(r for r, (_, s) in zip(ranks, abs_d) if s < 0)
    W = min(W_pos, W_neg)
    mean_w = n * (n + 1) / 4.0
    tie_corr = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1][0] == abs_d[i][0]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_corr += (t**3 - t)
        i = j + 1
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
    if var_w <= 0:
        return None
    z = (W_pos - mean_w) / math.sqrt(var_w)
    # two-sided p via standard normal cdf approx
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    n_pos = sum(1 for _, s in abs_d if s > 0)
    n_neg = n - n_pos
    return dict(W=W, W_pos=W_pos, W_neg=W_neg, n=n, z=z, p=p,
                n_pos=n_pos, n_neg=n_neg, n_zero=n_zero, mean=mean_w,
                var=var_w)


def paired_pairs(rows, metric, value_key="min_"):
    """Return list of (rotation_id, vit_val, heldout0_val) for validated rows."""
    field = f"{value_key}{metric}"
    by_rot = {}
    for r in rows:
        if r.get("error") or r.get("source_validated") != "True":
            continue
        rotid = r["rotation_id"]
        d = by_rot.setdefault(rotid, {})
        if float(r["drop_rate"]) == -1.0 and r["pred"] == "Vitruvion":
            d["vit"] = finite_or_none(r.get(field))
        elif float(r["drop_rate"]) == 0.0:
            d["h0"] = finite_or_none(r.get(field))
    out = []
    for rotid, vals in by_rot.items():
        v = vals.get("vit")
        h = vals.get("h0")
        if v is not None and h is not None:
            out.append((rotid, v, h))
    return out


def a2_wilcoxon(rows):
    print("=" * 70)
    print("Phase A-2: Wilcoxon signed-rank — Vitruvion vs heldout-drop0 (validated)")
    print("=" * 70)
    print(f"{'metric':14s} {'n_pairs':>8s} {'n_pos(V<H?)':>12s} {'n_neg':>8s} "
          f"{'n_zero':>8s} {'z':>10s} {'p_two':>10s}")
    print(f"{'(V<H means Vitruvion better; we report W stats from sign of V-H)':>70s}")
    for metric in [
        "d_resp",
        "d_cresp",
        "d_leak",
        "d_feas",
        "d_feas_tilde",
        "d_align",
        "d_align_phi",
    ]:
        pairs = paired_pairs(rows, metric, "min_")
        diffs = [v - h for _, v, h in pairs]  # positive when Vit larger (worse)
        res = wilcoxon_signed_rank(diffs)
        if res is None:
            print(f"{metric:14s} (no valid pairs)")
            continue
        # n_pos here is count of pos diffs (V>H meaning human better)
        print(f"{metric:14s} {res['n']+res['n_zero']:8d} {res['n_pos']:12d} "
              f"{res['n_neg']:8d} {res['n_zero']:8d} {res['z']:10.4f} {res['p']:10.4g}")
    print()
    print("Note: n_pos = #(V>H) = #rotations where heldout-drop0 is better.")
    print("      n_neg = #(V<H) = #rotations where Vitruvion is better.")
    print("      Two-sided p-value via normal approx with tie correction.")


def a2_histcad_oes(rows):
    """Vitruvion vs heldout-drop0 paired comparison for HistCAD OES.

    Unlike distance metrics, larger OES is better, so the signed difference is
    heldout0 - Vitruvion: positive means the human heldout is better.
    """
    pairs = []
    by_rot = {}
    for r in rows:
        if r.get("error") or r.get("source_validated") != "True":
            continue
        rotid = r["rotation_id"]
        d = by_rot.setdefault(rotid, {})
        rate = finite_or_none(r.get("drop_rate"))
        if rate == -1.0:
            d["vit"] = finite_or_none(r.get("max_histcad_oes"))
        elif rate == 0.0:
            d["h0"] = finite_or_none(r.get("max_histcad_oes"))
    for rid, vals in by_rot.items():
        v = vals.get("vit")
        h = vals.get("h0")
        if v is not None and h is not None:
            pairs.append((rid, v, h))

    diffs = [h - v for _, v, h in pairs]
    res = wilcoxon_signed_rank(diffs)
    print()
    print("=" * 70)
    print("Phase A-2b: HistCAD OES — Vitruvion vs heldout-drop0 (validated)")
    print("=" * 70)
    if res is None:
        print("HistCAD OES: no valid pairs")
        return
    n_model = sum(1 for _, v, h in pairs if v > h + 1e-9)
    n_human = sum(1 for _, v, h in pairs if h > v + 1e-9)
    n_tie = len(pairs) - n_model - n_human
    print(f"HistCAD OES n={len(pairs)} "
          f"human-better={n_human} vit-better={n_model} tie={n_tie} "
          f"z={res['z']:.4f} p={res['p']:.4g}")


# ---------- A-3: Tie 49-case decomposition ----------

def a3_tie_decomposition(rows, eps=1e-9):
    print()
    print("=" * 70)
    print("Phase A-3: d_align tie 49-case decomposition vs d_resp sign")
    print("=" * 70)
    align_pairs = paired_pairs(rows, "d_align", "min_")
    resp_pairs = {rid: (v, h) for rid, v, h in paired_pairs(rows, "d_resp", "min_")}
    # Classify each rotation by d_align outcome and d_resp outcome
    counts = {
        ("align_tie", "resp_vit_better"): [],
        ("align_tie", "resp_hum_better"): [],
        ("align_tie", "resp_tie"): [],
        ("align_vit_better", "resp_vit_better"): [],
        ("align_vit_better", "resp_hum_better"): [],
        ("align_vit_better", "resp_tie"): [],
        ("align_hum_better", "resp_vit_better"): [],
        ("align_hum_better", "resp_hum_better"): [],
        ("align_hum_better", "resp_tie"): [],
    }
    for rid, va, ha in align_pairs:
        rresp = resp_pairs.get(rid)
        if rresp is None:
            continue
        vr, hr = rresp
        if abs(va - ha) <= eps:
            align_cls = "align_tie"
        elif va < ha:
            align_cls = "align_vit_better"
        else:
            align_cls = "align_hum_better"
        if abs(vr - hr) <= eps:
            resp_cls = "resp_tie"
        elif vr < hr:
            resp_cls = "resp_vit_better"
        else:
            resp_cls = "resp_hum_better"
        counts[(align_cls, resp_cls)].append(rid)

    print(f"{'':22s} {'resp_vit_better':>16s} {'resp_hum_better':>16s} "
          f"{'resp_tie':>10s} {'total':>8s}")
    total_align = {}
    total_resp = {"resp_vit_better": 0, "resp_hum_better": 0, "resp_tie": 0}
    for ac in ["align_vit_better", "align_hum_better", "align_tie"]:
        row_total = 0
        line = f"{ac:22s} "
        for rc in ["resp_vit_better", "resp_hum_better", "resp_tie"]:
            n = len(counts[(ac, rc)])
            line += f"{n:>16d} "
            row_total += n
            total_resp[rc] += n
        line += f"{row_total:>8d}"
        total_align[ac] = row_total
        print(line)
    line = f"{'TOTAL':22s} "
    for rc in ["resp_vit_better", "resp_hum_better", "resp_tie"]:
        line += f"{total_resp[rc]:>16d} "
    line += f"{sum(total_align.values()):>8d}"
    print(line)
    print()
    print("Interpretation of the d_align=tie row:")
    n_tie_v = len(counts[("align_tie", "resp_vit_better")])
    n_tie_h = len(counts[("align_tie", "resp_hum_better")])
    n_tie_t = len(counts[("align_tie", "resp_tie")])
    print(f"  - {n_tie_t} ties where d_resp also ties → both metrics see equivalence (true equivalence)")
    print(f"  - {n_tie_v} ties where d_resp says Vit-better → null-space drift the user worries about")
    print(f"  - {n_tie_h} ties where d_resp says human-better → d_resp catches what d_align misses")


# ---------- B-5: drop-rate trajectory ----------

def trajectory_constrained(rows, validated=True, raw_key="min_"):
    """Return drop_rate -> dict(metric -> list of raw values) for heldout-dropX."""
    out = {}
    for r in rows:
        if r.get("error") or (r.get("source_validated") == "True") != validated:
            continue
        rate = float(r["drop_rate"])
        if rate == -1:
            continue
        d = out.setdefault(rate, {m: [] for m in
            [
                "d_resp",
                "d_align",
                "d_align_phi",
                "d_leak",
                "d_cresp",
                "d_feas",
                "d_feas_tilde",
            ]})
        for m in d:
            v = finite_or_none(r.get(f"{raw_key}{m}"))
            if v is not None:
                d[m].append(v)
        d.setdefault("attain", []).append(
            finite_or_none(r.get("median_attain_err")))
        d.setdefault("feas_gain", []).append(
            finite_or_none(r.get("median_d_feas_gain")))
        d.setdefault("feas_leak", []).append(
            finite_or_none(r.get("median_d_feas_leak")))
        d.setdefault("feas_tilde_leak", []).append(
            finite_or_none(r.get("median_d_feas_tilde_leak")))
    return out


def trajectory_histcad(rows, validated=True):
    """Return drop_rate -> {oes, er, cpcsr}."""
    out = {}
    for r in rows:
        if (r.get("source_validated") == "True") != validated:
            continue
        rate = float(r["drop_rate"])
        if rate == -1:
            continue
        d = out.setdefault(rate, {"oes": [], "er": [], "cpcsr": []})
        d["oes"].append(finite_or_none(r.get("max_histcad_oes")))
        d["er"].append(finite_or_none(r.get("median_histcad_er")))
        d["cpcsr"].append(finite_or_none(r.get("median_histcad_cpcsr")))
    return out


def b5_trajectory(dropout_rows, histcad_rows):
    print()
    print("=" * 70)
    print("Phase B-5: heldout-dim-dropout trajectory")
    print("=" * 70)
    constr = trajectory_constrained(dropout_rows, validated=True, raw_key="min_")
    hist = trajectory_histcad(histcad_rows, validated=True)
    rates = sorted(set(constr) & set(hist))
    print(f"  validated rotations: per-rate n_constrained = "
          f"{[len(constr[r]['d_align']) for r in rates]}, "
          f"n_histcad = {[len(hist[r]['oes']) for r in rates]}")
    print()
    print(f"{'drop_rate':>10s} | "
          f"{'med_d_feas':>11s} {'med_feas~':>11s} {'med_d_align':>11s} "
          f"{'gain':>8s} {'leak':>8s} {'leak~':>8s} | "
          f"{'med_OES':>9s} {'med_ER':>9s} {'med_cPCSR':>10s}")
    print("-" * 112)
    for rate in rates:
        c = constr[rate]
        h = hist[rate]
        print(f"{rate:>10.1f} | "
              f"{median(c['d_feas']) or 0:>11.4f} "
              f"{median(c['d_feas_tilde']) or 0:>11.4f} "
              f"{median(c['d_align']) or 0:>11.4f} "
              f"{median(c['feas_gain']) or 0:>8.4f} "
              f"{median(c['feas_leak']) or 0:>8.4f} "
              f"{median(c['feas_tilde_leak']) or 0:>8.4f} | "
              f"{median(h['oes']) or 0:>9.4f} "
              f"{median(h['er']) or 0:>9.4f} "
              f"{median(h['cpcsr']) or 0:>10.4f}")
    print()
    print("Predicted pattern:")
    print("  d_feas ↑ with drop_rate via gain residual (correctly detects heldout degradation)")
    print("  d_align remains mostly invariant to dimensional dropout")
    print("  HistCAD OES ↑ with drop_rate (incorrectly rewards dim removal — known weakness)")
    print()

    # Also report d_align by rotation conditional on heldout same.
    # bootstrap-style monotonicity: how often does d_align(rate=k+1) > d_align(rate=k)
    # within the same rotation_id?
    by_rot = {}
    for r in dropout_rows:
        if r.get("error") or r.get("source_validated") != "True":
            continue
        rate = float(r["drop_rate"])
        if rate == -1:
            continue
        v = finite_or_none(r.get("min_d_align"))
        if v is None:
            continue
        by_rot.setdefault(r["rotation_id"], {})[rate] = v
    print("Within-rotation monotonicity (raw d_feas / d_align): "
          "fraction of rotations where metric(higher drop) ≥ metric(lower drop)")
    rates_sorted = sorted(by_rot[next(iter(by_rot))].keys())
    for r1, r2 in zip(rates_sorted, rates_sorted[1:]):
        cnt = up = down = 0
        for rid, vals in by_rot.items():
            if r1 in vals and r2 in vals:
                cnt += 1
                if vals[r2] > vals[r1] + 1e-12:
                    up += 1
                elif vals[r2] < vals[r1] - 1e-12:
                    down += 1
        ties = cnt - up - down
        print(f"  d_align {r1:>5.1f}% -> {r2:>5.1f}%: "
              f"up={up}/{cnt} ({100*up/max(cnt,1):.1f}%), "
              f"down={down}, tie={ties}")

    for metric in ("d_feas", "d_feas_tilde"):
        by_rot = {}
        for r in dropout_rows:
            if r.get("error") or r.get("source_validated") != "True":
                continue
            rate = float(r["drop_rate"])
            if rate == -1:
                continue
            v = finite_or_none(r.get(f"min_{metric}"))
            if v is None:
                continue
            by_rot.setdefault(r["rotation_id"], {})[rate] = v
        if not by_rot:
            continue
        rates_sorted = sorted(by_rot[next(iter(by_rot))].keys())
        for r1, r2 in zip(rates_sorted, rates_sorted[1:]):
            cnt = up = down = 0
            for rid, vals in by_rot.items():
                if r1 in vals and r2 in vals:
                    cnt += 1
                    if vals[r2] > vals[r1] + 1e-12:
                        up += 1
                    elif vals[r2] < vals[r1] - 1e-12:
                        down += 1
            ties = cnt - up - down
            print(f"  {metric:12s} {r1:>5.1f}% -> {r2:>5.1f}%: "
                  f"up={up}/{cnt} ({100*up/max(cnt,1):.1f}%), "
                  f"down={down}, tie={ties}")


def main():
    full_rows = load_csv("artifacts/constrained_response_scores_full.csv")
    dropout_rows = load_csv("artifacts/constrained_response_scores_dropout.csv")
    histcad_rows = load_csv("artifacts/source_conditioned_dropout_scores.csv")

    a2_wilcoxon(full_rows)
    a2_histcad_oes(histcad_rows)
    a3_tie_decomposition(full_rows)
    b5_trajectory(dropout_rows, histcad_rows)


if __name__ == "__main__":
    main()
