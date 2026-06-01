"""Unit tests for token-space flip augmentation (research plan §16.9).

Self-contained (no data files): exercises the symmetric-grid token reflection,
the per-axis masking, the view schedule, and the K-expanded per-sample assembly.
"""
import torch

from src.sketch_handlers.constants import PRIMITIVE_VAL_OFFSET, PrimitiveAttrToken
from src.sketch_handlers.flip_augment import (
    flip_view_schedule,
    reflect_primitive_value_tokens,
    reflected_expanded_pval,
)

NB = 64
OFF = PRIMITIVE_VAL_OFFSET


def _bin(q: int) -> int:
    return OFF + q


def test_flip_view_schedule_even_and_remainder():
    assert flip_view_schedule(9, ["x", "y"]) == ["identity"] * 3 + ["x"] * 3 + ["y"] * 3
    # remainder goes to the earlier views (identity, then x)
    assert flip_view_schedule(8, ["x", "y"]) == ["identity"] * 3 + ["x"] * 3 + ["y"] * 2
    assert flip_view_schedule(8, ["x"]) == ["identity"] * 4 + ["x"] * 4
    assert flip_view_schedule(5, []) == ["identity"] * 5


def test_reflect_only_touches_the_requested_axis():
    # one token per relevant attr: StartX, StartY, Radius
    p_attr = torch.tensor([[int(PrimitiveAttrToken.StartX),
                            int(PrimitiveAttrToken.StartY),
                            int(PrimitiveAttrToken.Radius)]])
    p_val = torch.tensor([[_bin(10), _bin(20), _bin(30)]])

    out_x = reflect_primitive_value_tokens(p_val, p_attr, "x", NB)
    # X flips q -> NB-1-q; Y and Radius untouched
    assert out_x.tolist() == [[_bin(NB - 1 - 10), _bin(20), _bin(30)]]

    out_y = reflect_primitive_value_tokens(p_val, p_attr, "y", NB)
    assert out_y.tolist() == [[_bin(10), _bin(NB - 1 - 20), _bin(30)]]


def test_reflect_is_involution_and_bin_symmetric():
    p_attr = torch.full((4, 8), int(PrimitiveAttrToken.CenterX))
    p_val = torch.randint(OFF, OFF + NB, (4, 8))
    once = reflect_primitive_value_tokens(p_val, p_attr, "x", NB)
    twice = reflect_primitive_value_tokens(once, p_attr, "x", NB)
    assert torch.equal(twice, p_val)                      # involution
    # bin center negation: dequant(flip(q)) == -dequant(q) <=> flip(q) == NB-1-q
    assert torch.equal(once - OFF, (NB - 1) - (p_val - OFF))


def test_reflect_rejects_bad_axis():
    p = torch.zeros((1, 1), dtype=torch.long)
    try:
        reflect_primitive_value_tokens(p, p, "z", NB)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_reflected_expanded_pval_routes_per_schedule():
    B, K, Lp = 2, 9, 6
    axes = ["x", "y"]
    # alternate StartX / StartY so each axis view touches half the tokens
    p_attr = torch.tensor(
        [int(PrimitiveAttrToken.StartX), int(PrimitiveAttrToken.StartY)] * (Lp // 2)
    ).repeat(B * K, 1)
    p_val = torch.randint(OFF, OFF + NB, (B * K, Lp))

    out = reflected_expanded_pval(p_val, p_attr, K, axes, NB)
    sv = flip_view_schedule(K, axes)

    expected = p_val.clone()
    for b in range(B):
        for k in range(K):
            r = b * K + k
            if sv[k] != "identity":
                expected[r] = reflect_primitive_value_tokens(
                    p_val[r:r + 1], p_attr[r:r + 1], sv[k], NB
                )[0]
    assert torch.equal(out, expected)
    # identity slots are untouched
    for b in range(B):
        for k in range(K):
            if sv[k] == "identity":
                assert torch.equal(out[b * K + k], p_val[b * K + k])
    # double application returns the original (each axis reflection is an involution)
    assert torch.equal(reflected_expanded_pval(out, p_attr, K, axes, NB), p_val)
