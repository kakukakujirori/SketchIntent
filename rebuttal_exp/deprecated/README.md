# Archived: the two-arm Experiment D implementation

Superseded on 2026-08-17 by the single-arm rewrite in `rebuttal_exp/`. Kept for provenance
only — these files are **not importable** (their `from rebuttal_exp.…` imports resolve to
the current modules) and `conftest.py` keeps pytest from collecting the archived tests.

The results they produced are unchanged and remain at
`artifacts/rebuttal/pseudoinverse_validated/`.

## What it did

Every `R = R0 + N Z` solves `J_x R = -J_p`, so the minimum-norm `R0 = -pinv(J_x) J_p` is one
member of a family. The freedom was sampled by **two arms**:

```
main             E_j = eta ||(J_phi R0)_j|| * unit vector in range(J_phi N)
                 R_eta = R0 + N pinv(J_phi N) E        -> eta sized in observable space
observable_null  R_eta[:,j] = R0[:,j] + eta ||R0[:,j]|| * unit vector in ker([J_x; J_phi])
                                                       -> eta sized in raw space
```

`main` could not excite a direction invisible to `J_phi`; `observable_null` isolated exactly
those. The two `eta` were not comparable and were never pooled.

## Why it was replaced

1. **The simple form gives the same guarantee.** `Delta_j = eta ||M_j|| n / ||J_phi n||`
   with `n` uniform on the unit sphere of `ker(J_x)` realises `||J_phi Delta_j|| =
   eta ||M_j||` exactly — which is all the `N pinv(J_phi N) E` construction bought.
2. **The two schemes agree where the question is live.** On responses with
   `rank(J_phi N) > 0`: raw amplification median 0.59 vs 0.67, and the movement of
   `range(R)` (what `theta_leak` tracks) median 0.1995 vs 0.1988.
3. **One absolute guard replaces the arm split.** The simple form's only failure is division
   by a numerically-zero gain, and the observed gains are bimodal — about `1e-15` when
   `ker(J_x) ⊆ ker(J_phi)` (rigid motion + Line parameterisation gauge + construction
   geometry) and at least `1e-2` otherwise. Discarding draws with `||J_phi n|| < 1e-9`, the
   absolute floor of the `eps_eff` cutoff used throughout, sits six orders from both sides.

A response whose whole null space is discarded is not a sampling failure: `J_phi R` is then
identical for every admissible response, so `theta_full`, `theta_rel` and `S_place` are
*exactly* invariant. That is now reported as a lemma with a coverage statistic instead of
being split off into the auxiliary arm. `theta_leak` stays response-sensitive and now also
reflects gauge components, which the current write-up states directly.
