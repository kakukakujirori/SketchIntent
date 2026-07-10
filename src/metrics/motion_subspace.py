"""Motion-subspace angle scalars (paper Section 4).

Single, consumer-agnostic home for the *scalar* angle formulas derived from a
pair of subspaces.  These are the paper's metric definitions; both the offline
evaluator (``src/eval``) and the training reward
(``src/models/rewards/response_metric_reward``) compose their higher-level
scores from these primitives, so the numbers reported in the paper and the
numbers the model optimizes are guaranteed to use one implementation.

The subspace *construction* (SVD range, relation Jacobians, ``J_phi``
projection, ``principal_cos2_sum``) currently lives in
``src/models/rewards/utils/geometry.py``; this module deliberately takes the
already-reduced quantities (``cos2_sum``, an overlap matrix) so it stays free of
any sketch / solver dependency.

Deliberately *not* here: the reward's symmetric two-sided motion-relational
variant and its normalization (``_motion_relational_angle`` /
``_RESPONSE_ANGLE_NORM``).  That is a training-signal design choice, not a paper
metric, so it stays in the reward layer.  The symmetric variant coincides with
``padded_subspace_angle`` exactly when the two subspace dimensions are equal and
differs only for dimension-mismatched pairs.
"""
from __future__ import annotations

import math

import torch


def padded_subspace_angle(cos2_sum: float, dim_a: int, dim_b: int) -> float:
    r"""Normalized principal-angle distance :math:`d_\angle` (paper Eq. 4).

    .. math::

        d_\angle(A, B) = \sqrt{1 - \frac{\lVert U_A^\top U_B\rVert_F^2}
                                        {\max(a, b)}} \in [0, 1]

    where ``cos2_sum`` :math:`= \lVert U_A^\top U_B\rVert_F^2 = \sum_i
    \cos^2\theta_i` (i.e. the output of ``principal_cos2_sum``) and ``dim_a`` /
    ``dim_b`` are the subspace dimensions ``a`` / ``b``.

    The ``max(a, b)`` normalization treats missing dimensions as orthogonal
    unmatched directions, so a dimension-count mismatch is penalized as a
    right-angle.  Returns 0 when both subspaces are empty (they trivially agree).

    This is the shared implementation behind the paper's full motion distance
    ``theta_motion^full`` (Eq. 5) and relational motion distance
    ``theta_motion^rel`` (Eq. 6), and behind the reward's ``_motion_full_angle``.
    """
    d_norm = max(dim_a, dim_b)
    if d_norm == 0:
        return 0.0
    angle_sq = (d_norm - cos2_sum) / d_norm
    return math.sqrt(min(max(angle_sq, 0.0), 1.0))


def leak_angle_from_overlap(overlap: torch.Tensor, dim_pred: int) -> float:
    r"""Dimensional-leak angle :math:`\theta_{leak}` (paper Section 4).

    RMS cosine between the Pred dimensional-motion directions and the GT
    relation-forbidden directions:

    .. math::

        \theta_{leak}^2 = \frac{\lVert \mathrm{orth}(\mathrm{row}(J_R^G))^\top
                                U_P \rVert_F^2}{\dim(U_P)}

    ``overlap`` is the already-formed matrix ``N_G^T @ U_P`` (GT relation row
    space basis against the Pred motion basis) and ``dim_pred`` is
    ``dim(U_P)``.  Returns 0 for an empty Pred motion subspace.

    Shared implementation behind the reward's ``_leak_relational_angle`` and the
    evaluator's ``leak_dimensional_angle``.
    """
    if dim_pred <= 0:
        return 0.0
    leak_sq = float((overlap ** 2).sum()) / float(dim_pred)
    return math.sqrt(min(max(leak_sq, 0.0), 1.0))
