from __future__ import annotations

import itertools

from _bootstrap import bootstrap

bootstrap()

import torch

from fusion360_annotation.eda_code.motion_metrics import placement_sims

torch.set_default_dtype(torch.float64)


def pairwise_dist(corners):
    return torch.stack([torch.linalg.norm(corners[i] - corners[j]) for i, j in itertools.combinations(range(4), 2)])


def jphi(corners):
    c = corners.reshape(-1).clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(lambda v: pairwise_dist(v.reshape(4, 2)), c)


def unit_cols(matrix):
    cols = [matrix[:, j] / torch.linalg.norm(matrix[:, j]) for j in range(matrix.shape[1]) if torch.linalg.norm(matrix[:, j]) > 1e-9]
    return torch.stack(cols, dim=1)


def motion_angle(A, B):
    Qa = torch.linalg.qr(A).Q
    Qb = torch.linalg.qr(B).Q
    s = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1, 1)
    k = max(A.shape[1], B.shape[1])
    cos2 = float((s**2).sum())
    return float(torch.sqrt(torch.tensor(max(0.0, (k - cos2) / k))))


def toy_square():
    corners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    Jp = jphi(corners)
    dc_ds = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]).reshape(-1, 1)
    A = unit_cols(Jp @ dc_ds)
    B = unit_cols(Jp @ (dc_ds / (2**0.5)))
    sims = placement_sims(A, B)
    print(f"square  side-vs-diagonal      asgn_sim={sims['asgn_sim']:.3f} set_sim={sims['set_sim']:.3f} theta={motion_angle(A, B):.3f}")


def toy_rectangle(w0, h0):
    corners = torch.tensor([[0.0, 0.0], [w0, 0.0], [w0, h0], [0.0, h0]])
    Jp = jphi(corners)
    Jc = torch.tensor([
        [0.0, 0.0], [0.0, 0.0],
        [1.0, 0.0], [0.0, 0.0],
        [1.0, 0.0], [0.0, 1.0],
        [0.0, 0.0], [0.0, 1.0],
    ])
    d = (w0**2 + h0**2) ** 0.5
    Jg_A = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    Jg_B = torch.tensor([[0.0, 1.0], [w0 / d, h0 / d]])
    A = unit_cols(Jp @ (Jc @ torch.linalg.inv(Jg_A)))
    B = unit_cols(Jp @ (Jc @ torch.linalg.inv(Jg_B)))
    sims = placement_sims(A, B)
    print(f"rect {w0:g}:{h0:g}  sides-vs-diagonal  asgn_sim={sims['asgn_sim']:.3f} set_sim={sims['set_sim']:.3f} theta={motion_angle(A, B):.3f}")


if __name__ == "__main__":
    toy_square()
    for w, h in ((2.0, 1.0), (3.0, 1.0), (1.5, 1.0)):
        toy_rectangle(w, h)
