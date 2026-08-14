"""Motion-subspace primitives and angle scalars (paper Section 4).

Single, consumer-agnostic home for the motion-subspace math: the SVD / Jacobian
/ null-space construction of the subspaces, the principal-cosine building block,
and the scalar angle formulas derived from a pair of subspaces.  Both the
offline evaluator (``src/eval``) and the training reward
(``src/models/rewards/response_metric_reward``) compose their higher-level
scores from these primitives, so the numbers reported in the paper and the
numbers the model optimizes are guaranteed to use one implementation.

This module depends only on the constraint differential primitives
(``src.metrics.edit_response_distance``) and the sketch context / solver; it has
no dependency on the reward or model layers.

The reward uses ``padded_subspace_angle`` directly for its default
``theta_motion^rel`` signal.  Alternative symmetric and asymmetric
motion-relational variants remain in the reward layer because they are
training-signal design choices rather than paper metrics.
"""
from __future__ import annotations

import math

import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    _constraint_refs_resolvable,
    _is_relational,
    _jacfwd_stacked_residuals,
    _probe_residual_width,
    pairwise_distance_observable,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_solver import SketchState, compute_null_space

_TINY = 1e-12


# ---------------------------------------------------------------------------
# Subspace construction primitives
# ---------------------------------------------------------------------------
def orthonormal_range(R: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
    """SVD-based orthonormal basis of range(R), keeping singular values >= eps.

    Returns (U, k) where U is (R.shape[0], k) and the columns span range(R).
    """
    if R.numel() == 0 or R.shape[1] == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    U, S, _Vh = torch.linalg.svd(R, full_matrices=False)
    keep = int((S >= eps).sum().item())
    if keep == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    return U[:, :keep].contiguous(), keep


def stack_relation_jacobians_at(
    rels,
    state: SketchState,
    context: SketchContext,
) -> tuple[torch.Tensor | None, int]:
    """Concatenate relation-constraint Jacobians evaluated at the given state.

    Pre-filters constraints with the same checks as
    :func:`src.metrics.edit_response_distance.constraint_jacobian` (handler
    lookup, ref resolvability, residual probe) and then issues a single
    ``jacfwd`` over the stacked residual vector instead of one call per relation.
    """
    entries: list[tuple[object, object, int]] = []
    for rel in rels:
        handler = context.get_constraint_handler(rel.constraint_type.name)
        if handler is None:
            continue
        if not _constraint_refs_resolvable(rel, state):
            continue
        width = _probe_residual_width(rel, handler, state)
        if width is None:
            continue
        entries.append((rel, handler, width))

    if not entries:
        return None, 0
    Jr = _jacfwd_stacked_residuals(entries, state)
    if Jr.shape[0] == 0:
        return None, 0
    return Jr, int(Jr.shape[0])


def motion_full_subspace(
    sketch: datalib.Sketch,
    native: NativeResponse,
    eps: float,
    *,
    J_phi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Orthonormal basis of the *full* motion subspace in observable space.

    "Full" here means the motion subspace driven by the full first-order
    response map ``R_C`` (dimensional + relational null-space columns
    together), projected through the pairwise-distance Jacobian ``J_phi`` so
    that Pred and GT can be compared in a common, sketch-topology-independent
    observable coordinate system:

        W_C := range(J_phi @ R_C)

    Contrast this with ``motion_relational_subspace``, which uses only
    ``null(J_R)`` and ignores the placement of dimensional drivers.

    ``J_phi`` may be passed in by the caller so that one pre-computed
    pairwise-distance Jacobian is shared between this helper and
    :func:`motion_relational_subspace` (and, in RLOO, between the K rollouts
    that share the same ``gt_sketch``).  When ``None``, J_phi is recomputed
    from ``sketch`` — the original standalone behavior used by tests.

    Returns the orthonormal basis ``U`` of ``W_C`` and its dimension ``k``.
    """
    if J_phi is None:
        J_phi = pairwise_distance_observable(
            sketch,
            dtype=native.R.dtype,
        ).to(dtype=native.R.dtype)
    if J_phi.shape[0] == 0 or native.R.shape[1] == 0:
        return torch.zeros(J_phi.shape[0], 0, dtype=native.R.dtype), 0
    return orthonormal_range(J_phi @ native.R, eps)


def motion_relational_subspace(
    sketch: datalib.Sketch,
    state: SketchState,
    context: SketchContext,
    eps: float,
    *,
    J_phi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Orthonormal basis of the *relational* motion subspace in observable space.

    "Relational" means: only the primitive motions allowed by the sketch's
    relational constraints, with dimensional drivers deliberately dropped:

        N_C := null(J_R)               # motions allowed by relations alone
        W_C := range(J_phi @ N_C)      # same motions, in pairwise-distance
                                       # observable space

    Jacobians are evaluated at ``state``.  When comparing two sketches at the
    same geometric configuration (e.g. Pred and GT, where Pred's primitive
    geometry has been overwritten with the original), the caller should pass
    the same ``state`` for both sketches so that the two subspaces live in
    coordinates that can be compared with a principal-angle SVD.

    Contrast with ``motion_full_subspace``, which uses the full first-order
    response map ``R_C`` and therefore mixes dimensional-driver placement into
    the comparison.

    ``J_phi`` may be passed in by the caller for the same caching reason
    described in :func:`motion_full_subspace`.
    """
    n_x = state.num_dof()
    dtype = state.q.dtype

    rels = [c for c in sketch.constraints.values() if _is_relational(c)]
    J_R, _ = stack_relation_jacobians_at(rels, state, context)
    if J_R is None:
        J_R = torch.zeros(0, n_x, dtype=dtype)
    N_C = compute_null_space(J_R, eps=eps)

    if J_phi is None:
        J_phi = pairwise_distance_observable(sketch).to(dtype=dtype)
    if J_phi.shape[0] == 0:
        return torch.zeros(0, 0, dtype=dtype), 0
    return orthonormal_range(J_phi @ N_C, eps)


def principal_cos2_sum(U_P: torch.Tensor, U_G: torch.Tensor) -> float:
    """Σ cos²θ_i between two orthonormal subspaces.

    Computes the sum of squared principal cosines between ``range(U_P)`` and
    ``range(U_G)`` via the SVD of ``U_G^T @ U_P``.  Returns 0 if either
    subspace is empty.

    This is the shared building block for the angle scalars used by Hθ:

    - padded principal-angle distance: ``(d_norm - cos2_sum) / d_norm``
    - symmetric projection distance:   ``(dim_P + dim_G - 2 cos2_sum) / d_norm``
    - asymmetric subset distances:     ``(dim_? - cos2_sum) / dim_?``

    Callers handle normalization by the relevant dimensions; this helper
    intentionally returns the raw quantity.
    """
    if U_P.shape[1] == 0 or U_G.shape[1] == 0:
        return 0.0
    align = U_G.T @ U_P
    sigmas = torch.linalg.svdvals(align).clamp(0.0, 1.0)
    return float((sigmas ** 2).sum())


# ---------------------------------------------------------------------------
# Angle scalars (derived from cos2_sum / an overlap matrix)
# ---------------------------------------------------------------------------
def padded_subspace_angle(cos2_sum: float, dim_a: int, dim_b: int) -> float:
    r"""Normalized principal-angle distance :math:`d_\angle` (paper Eq. 4).

    .. math::

        d_\angle(A, B) = \sqrt{1 - \frac{\lVert U_A^\top U_B\rVert_F^2}
                                        {\max(a, b)}} \in [0, 1]

    where ``cos2_sum`` :math:`= \lVert U_A^\top U_B\rVert_F^2 = \sum_i
    \cos^2\theta_i` (i.e. the output of :func:`principal_cos2_sum`) and
    ``dim_a`` / ``dim_b`` are the subspace dimensions ``a`` / ``b``.

    The ``max(a, b)`` normalization treats missing dimensions as orthogonal
    unmatched directions, so a dimension-count mismatch is penalized as a
    right-angle.  Returns 0 when both subspaces are empty (they trivially agree).

    This is the shared implementation behind the paper's full motion distance
    ``theta_motion^full`` (Eq. 5) and relational motion distance
    ``theta_motion^rel`` (Eq. 6), and behind both corresponding padded reward
    modes.
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


# ---------------------------------------------------------------------------
# Dimensional placement similarity (S_place, paper Eq. 7)
# ---------------------------------------------------------------------------
def _hungarian_matched(C: torch.Tensor) -> torch.Tensor:
    from scipy.optimize import linear_sum_assignment

    ri, ci = linear_sum_assignment((1.0 - C).numpy())
    return C[ri, ci]


def placement_sims(A: torch.Tensor, B: torch.Tensor) -> dict:
    r"""Dimensional placement similarity ``S_place`` (paper Eq. 7) + variants.

    ``A`` and ``B`` are the L2-normalized dimensional-response directions
    ``u_j = R_C[:, j] / ‖R_C[:, j]‖`` of Pred and GT (one column per driving
    dimension).  ``placement_sim`` is the paper's ``S_place``: the Hungarian
    max-weight one-to-one matching of ``|⟨u_i^P, u_j^G⟩|`` normalized by
    ``max(m_P, m_G)``.  ``asgn_min`` / ``set_sim`` / ``omp_sim`` are
    instrument-tier variants used only in the ablation diagnostics.
    """
    ka, kb = int(A.shape[1]), int(B.shape[1])
    if ka == 0 and kb == 0:
        return dict(placement_sim=1.0, asgn_min=None, set_sim=1.0, omp_sim=1.0)
    if ka == 0 or kb == 0:
        return dict(placement_sim=0.0, asgn_min=None, set_sim=0.0, omp_sim=0.0)
    C = (A.T @ B).abs()
    matched = _hungarian_matched(C)
    asgn = float(matched.sum())
    set_sim = float((C.max(dim=0).values.mean() + C.max(dim=1).values.mean()) / 2)
    return dict(
        placement_sim=asgn / max(ka, kb),
        asgn_min=asgn / min(ka, kb),
        set_sim=set_sim,
        omp_sim=_omp_sim(A, B),
    )


def _omp_sim(Ua: torch.Tensor, Ub: torch.Tensor, tol: float = 1e-9) -> float:
    A, B = Ua.clone(), Ub.clone()
    avail_a = list(range(A.shape[1]))
    avail_b = list(range(B.shape[1]))
    num = ea = eb = 0.0
    while avail_a and avail_b:
        C = (A[:, avail_a].T @ B[:, avail_b]).abs()
        flat = int(torch.argmax(C))
        i, j = flat // C.shape[1], flat % C.shape[1]
        best = float(C[i, j])
        if best < tol:
            break
        ia, jb = avail_a.pop(i), avail_b.pop(j)
        num += best
        ea += float(A[:, ia] @ A[:, ia])
        eb += float(B[:, jb] @ B[:, jb])
        da = A[:, ia] / max(float(A[:, ia].norm()), _TINY)
        db = B[:, jb] / max(float(B[:, jb].norm()), _TINY)
        if avail_a:
            Ra = A[:, avail_a]
            A[:, avail_a] = Ra - torch.outer(da, da @ Ra)
        if avail_b:
            Rb = B[:, avail_b]
            B[:, avail_b] = Rb - torch.outer(db, db @ Rb)
    ea_left = float(sum(float(A[:, i] @ A[:, i]) for i in avail_a))
    eb_left = float(sum(float(B[:, i] @ B[:, i]) for i in avail_b))
    den = math.sqrt(max((ea + ea_left) * (eb + eb_left), _TINY))
    return num / den
