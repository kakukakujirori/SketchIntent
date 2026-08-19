# Additional Experiment: Direct Pairwise Human Agreement

## Goal

Reviewer feedback asks for:
1. full held-out-human score distributions, and
2. pairwise annotator results.

Appendix Fig. 8 already provides the full held-out-human distributions under the current leave-one-out multi-reference evaluation.

However, that evaluation aggregates over multiple human references by selecting the most favorable reference for each evaluated annotation. Therefore, it does not directly characterize all human-human pairwise disagreement.

Implement an additional analysis that evaluates every valid within-sketch human annotation pair directly, without multi-reference min/max aggregation.

---

## Existing Results to Preserve

Appendix Fig. 8 reports the following held-out-human statistics under the existing leave-one-out multi-reference protocol:

- theta_full_motion:
  - mean ≈ 0.167
  - median ≈ 2.4e-8
  - IQR ≈ [1.2e-8, 0.408]

- theta_rel_motion:
  - mean ≈ 0.178
  - median ≈ 2.6e-8
  - IQR ≈ [1.2e-8, 0.408]

- theta_dim_leak:
  - mean ≈ 0.09
  - median ≈ 7.5e-15

- S_place:
  - mean ≈ 0.823
  - median ≈ 0.831

Do not modify the existing evaluation or these results.

---

## Critical Implementation Requirement

Reuse the existing evaluation implementation exactly.

Do not redefine any metric.

Reuse the same:
- sketch normalization
- observable / measurement-space representation
- Jacobian computation
- SVD threshold
- construction-geometry handling
- FreeCAD validity / convergence checks
- numerical tolerances

as the existing Table 1 / Fig. 8 evaluation.

The only conceptual difference is:

> evaluate individual human-human pairs directly instead of aggregating over multiple references.

Before implementing pair construction, inspect the existing evaluation code and determine exactly how valid annotations/sketches are filtered for Table 1 / Fig. 8.

Reuse that filtering logic rather than inferring the evaluation subset from the reported sample count.

Report the resulting:
- number of valid sketches
- number of valid annotations
- number of valid unordered human pairs

and briefly state the filtering rule used.

---

## Pair Construction

Each sketch has up to three independently authored human annotations:

    H1, H2, H3

For every pair that is valid under the existing evaluation filtering logic, evaluate:

    (H1, H2)
    (H1, H3)
    (H2, H3)

Do not perform:
- minimum over references for distance metrics
- maximum over references for S_place

Do not include self-comparisons.

---

## Metrics

### Symmetric pairwise metrics

Compute once for every unordered pair:

    theta_full_motion(Hi, Hj)
    theta_rel_motion(Hi, Hj)
    S_place(Hi, Hj)

As a sanity check, numerically verify that swapping Hi and Hj produces the same value up to numerical tolerance.

### Asymmetric metric: theta_dim_leak

theta_dim_leak(P; G) is directional.

For every unordered pair, compute both:

    theta_dim_leak(Hi; Hj)
    theta_dim_leak(Hj; Hi)

Do not replace this with a newly defined symmetrized metric.

---

## Required Outputs

### 1. Per-pair CSV

Save:

    human_pairwise_metrics.csv

Suggested columns:

    sketch_id
    annotator_i
    annotator_j
    annotation_i_id
    annotation_j_id
    theta_full_motion
    theta_rel_motion
    s_place
    theta_dim_leak_i_to_j
    theta_dim_leak_j_to_i

Preserve full floating-point precision in the saved file.

---

### 2. Aggregate summary

For each metric, report:

- N
- mean
- median
- Q1
- Q3
- IQR
- P90

For theta_dim_leak, aggregate the directional observations unless there is a strong implementation reason not to; clearly report how N is counted.

Save:

    human_pairwise_summary.csv

Also print the summary as a Markdown table to stdout.

---

### 3. Distribution figure

Produce a direct pairwise human-human distribution figure for the four metrics.

Prefer the visual format already used in Appendix Fig. 8 so that the new result can be compared easily with the existing held-out-human distribution.

For theta_dim_leak, include both directions in the distribution.

Save the figure as both PDF and PNG.

---

## Primary Questions to Inspect

After computing the results, report observations only from the data.

In particular, characterize:

1. How much larger the direct all-pair human variation is than the existing best-reference held-out-human variation.

2. Whether theta_full_motion and theta_rel_motion still have substantial mass near numerical zero.

3. Whether there is a non-negligible upper tail, indicating genuinely different redesign behaviors among human annotations.

4. Whether dimensional-placement disagreement, reflected by S_place, is larger or smaller than relation-level disagreement.

Do not force any particular conclusion.

---

## Sanity Checks

Verify and print:

1. Number of valid sketches.
2. Number of valid human annotations.
3. Number of valid unordered human pairs.
4. Number of directional theta_dim_leak evaluations.
5. No self-comparisons.
6. No multi-reference min/max aggregation.
7. Symmetry of:
   - theta_full_motion
   - theta_rel_motion
   - S_place
8. Both directions retained for theta_dim_leak.
9. Same normalization, measurement-space representation, SVD threshold, and filtering logic as the existing evaluation.
10. Near-zero floating-point values are not rounded to literal zero before statistics are computed.

---

## Nice-to-Have: Per-Sketch Diversity

Only after the required analysis is complete, optionally compute for each sketch:

    max_pairwise_theta_full
    mean_pairwise_theta_full

    max_pairwise_theta_rel
    mean_pairwise_theta_rel

    min_pairwise_S_place
    mean_pairwise_S_place

This can be used to identify sketches where all annotators agree behaviorally versus sketches with genuinely multi-modal human design intent.

Save as:

    human_pairwise_per_sketch.csv

Do not implement annotator-ID-specific 6x6 matrices unless requested separately.