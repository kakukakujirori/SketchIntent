# A7 Experiment: Robustness to Shared-Reference Evaluation

## Objective

Reviewer yj6z noted that the current multi-reference evaluation independently chooses the most favorable human reference for each metric. Therefore, different metrics for the same prediction may be evaluated against different complete human annotations.

We want to measure:

1. how often this actually happens;
2. whether the disagreement is numerically meaningful;
3. whether forcing all four metrics to use one shared human reference changes the conclusions of Table 1.

Do not retrain any model.

Do not construct a joint score by averaging or normalizing the raw metric values.

If a shared reference is selected, use only metric ranks.

The four metrics are:

- `theta_full_motion`: lower is better
- `theta_rel_motion`: lower is better
- `theta_dim_leak`: lower is better
- `S_place`: higher is better

Use the same evaluation samples, convergence filtering, geometry-preservation filtering, and leave-one-out protocol as the existing Table 1 evaluation.

The implementation must support arbitrary `K >= 2` references. In the held-out-human leave-one-out evaluation, there are typically two references.

---

## Step 0: Inspect the existing evaluation implementation

Before changing anything, identify:

- where scores against individual references are computed;
- where `min` / `max` multi-reference aggregation is applied;
- whether per-reference metric values are already cached or stored;
- which Table 1 rows can be recomputed without rerunning model inference.

Reuse existing per-reference scores whenever possible.

---

# Experiment 1: Best-reference consistency

For every evaluated prediction/fold, obtain all four metric values against every available human reference before multi-reference aggregation.

Let `s[m,k]` denote metric `m` evaluated against reference `k`.

Because several metric values can be numerically equal, do not use a plain `argmin` / `argmax` to determine agreement.

Define the set of best references for each metric.

For lower-is-better metrics:

    B_m = {k | s[m,k] is numerically equal to min_j s[m,j]}

For `S_place`:

    B_place = {k | s[place,k] is numerically equal to max_j s[place,j]}

Use the numerical tolerance already used by the evaluation code if one exists. Otherwise use a clearly documented `np.isclose` tolerance.

Report, separately for:

- Human held-out
- Base Vitruvion
- Solver-RL

the following statistics.

### 1. Coherent-best-reference rate

A sample is coherent if at least one reference is jointly optimal for all four metrics:

    intersection(B_full, B_rel, B_leak, B_place) != empty

Report:

- coherent rate
- incoherent rate

### 2. Pairwise best-reference incompatibility

In particular report:

    intersection(B_rel, B_place) == empty

because this directly tests whether relational behavior and dimensional placement prefer incompatible human annotations.

Also report:

- full vs place
- full vs rel
- rel vs leak

### 3. Unique-best-reference agreement

For the subset of samples where every metric has a unique best reference, report:

- all four metrics choose the same reference
- exactly three metrics agree on one reference
- selections are split across multiple references

Keep the tie-aware statistics above as the primary results.

---

# Experiment 2: Rank-based shared-reference selection

For each prediction/fold, rank the available references independently under each metric.

Ranking direction:

- `theta_full_motion`: ascending
- `theta_rel_motion`: ascending
- `theta_dim_leak`: ascending
- `S_place`: descending

Rank 1 is best.

Use average ranks for tied metric values.

For each reference `k`, compute:

    R_k =
        rank_full(k)
      + rank_rel(k)
      + rank_leak(k)
      + rank_place(k)

The rank sum is used ONLY for choosing a shared reference.

Do not report `R_k` as a new CAD metric.

## Important: rank-sum ties

If exactly one reference has the minimum rank sum, call it:

    k_joint

If multiple references have the same minimum rank sum, do NOT resolve the tie using reference index, raw metric averaging, or any other arbitrary numerical tie-break.

Mark the sample as:

    ambiguous_joint_selection = True

Report the frequency of these ambiguous cases.

For samples with a unique `k_joint`, evaluate all four original metrics against that SAME reference.

If ambiguous cases are non-negligible, additionally report a sensitivity range obtained by evaluating the original metrics against each tied rank-optimal reference. Do not independently choose a different tied reference for each metric.

---

# Experiment 3: Joint-reference regret

For samples with a unique `k_joint`, measure how much each metric worsens relative to the current independent best-reference aggregation.

For lower-is-better metrics:

    regret_m =
        score_m(k_joint)
        - min_k score_m(k)

For `S_place`:

    regret_place =
        max_k S_place(k)
        - S_place(k_joint)

All regrets should be non-negative up to numerical precision.

For each metric and each Table 1 source/model, report:

- fraction with numerically zero regret
- median
- mean
- 75th percentile
- 90th percentile
- 95th percentile

Do not average regrets across metrics.

This distinguishes genuine incompatibility from cases where two references have nearly identical metric values.

---

# Experiment 4: Independent vs shared-reference Table 1

Recompute the Human held-out, Base Vitruvion, and Solver-RL rows under the rank-based shared-reference protocol.

For the original protocol, keep the existing independent min/max aggregation.

For the shared-reference protocol:

1. rank the references using all four metrics;
2. select the unique rank-sum winner `k_joint`;
3. evaluate all four metrics against that same reference.

Use the same aggregate statistic as Table 1.

Produce:

| Variant | full independent | full shared | rel independent | rel shared | leak independent | leak shared | place independent | place shared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Also report:

- number of original samples
- number with a unique joint reference
- number / percentage with ambiguous rank-sum ties

Do not silently drop ambiguous cases.

If ambiguous cases are frequent enough that excluding them materially changes the evaluation subset, provide a sensitivity analysis over the tied rank-optimal references rather than presenting the unique-only table as directly comparable to the original Table 1.

---

# Optional secondary analysis: model-comparison flips

Only if matched predictions for the same sketches are readily available, check whether switching from independent aggregation to the shared-reference protocol changes which model has the better score on individual sketches.

This is secondary and should not require new model inference or substantial new implementation.

---

# Required saved outputs

Save:

1. a per-sample file containing:
   - sketch ID
   - source/model
   - held-out annotation/fold ID
   - reference IDs
   - all per-reference values of the four metrics
   - best-reference set for each metric
   - coherent/incoherent flag
   - rel-vs-place incompatibility flag
   - per-reference ranks
   - rank sums
   - unique `k_joint` or ambiguous flag
   - per-metric regret

2. a summary table for Experiment 1;

3. per-metric regret statistics;

4. the independent-vs-shared Table 1 comparison;

5. the exact sample counts and exclusion counts at every stage.

After running the experiment, report the numerical results without trying to write the rebuttal. Briefly note any implementation details or unexpected behavior that could affect interpretation.