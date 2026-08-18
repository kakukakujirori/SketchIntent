"""The Sec. 5 row structure: how the evaluated variants are built, named and classified.

Both runners construct the same variant set (held-out human, dimensional and relational
dropout controls, each model run) and both analysers order and classify it the same way, so
the vocabulary lives here rather than in whichever experiment happened to define it first.
Variant sketches are seeded exactly like ``ablation/run_motion.py``: same ``stable_seed``
inputs, so every experiment sees byte-identical dropout variants and comparisons stay paired.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ablation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ablation"))  # ablation/aggregate.py imports _bootstrap

from ablation.aggregate import VARIANT_ORDER  # noqa: E402
from ablation.controls import relation_dropout_variant  # noqa: E402
from src.eval.file_bundle import dropout_variant, stable_seed  # noqa: E402

CONTROL_PREFIXES = ("dim_drop", "rel_drop")
NON_METHOD_VARIANTS = ("source_self",)
SETTINGS = {
    "source_self": "self_consistency",
    "heldout": "human_human",
    "base": "human_model",
    "solver_rl": "human_model",
    "ours": "human_model",
}
SETTING_ORDER = ["self_consistency", "human_human", "human_model", "dim_dropout", "rel_dropout"]
"""The Sec. 5 evaluation settings the pooled table mixes together. Pooling weights the
synthetic dropout controls (six of eleven variants) as heavily as the rows the paper's
conclusions rest on, so the per-setting split is reported next to it."""


def build_variants(bundle, item, refs, heldout, model_labels, dim_rates, rel_rates):
    """The Sec. 5 variant set, seeded exactly like ``ablation/run_motion.py``."""
    rid = f"{item.batch}:{item.base_fcstd}:{'+'.join(refs)}__q={heldout}"
    variants = [("heldout", heldout, bundle.sketches[heldout])]
    for rate in dim_rates:
        seed = stable_seed(item.batch, item.base_fcstd, rid, "dim", rate)
        pred, *_ = dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=seed)
        variants.append((f"dim_drop{rate:g}", f"{rid}:d{rate:g}", pred))
    for rate in rel_rates:
        seed = stable_seed(item.batch, item.base_fcstd, rid, "rel", rate)
        pred, *_ = relation_dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=seed)
        variants.append((f"rel_drop{rate:g}", f"{rid}:r{rate:g}", pred))
    for label in model_labels:
        variants.append((label, label, bundle.sketches[label]))
    return rid, variants


def ordered_variants(pairs: pd.DataFrame) -> list[str]:
    """Sec. 5 order, then any variant the ablation list predates (e.g. the proposed model)."""
    present = set(pairs["variant"].unique())
    known = [v for v in VARIANT_ORDER if v in present]
    return known + sorted(present - set(VARIANT_ORDER))


def is_method_variant(variant: str) -> bool:
    """Held-out human and the model runs: the rows the paper's conclusions rest on."""
    return not variant.startswith(CONTROL_PREFIXES) and variant not in NON_METHOD_VARIANTS


def setting_of(variant: str) -> str:
    """Which Sec. 5 evaluation setting a variant row belongs to."""
    if variant in SETTINGS:
        return SETTINGS[variant]
    return f"{variant.split('_drop')[0]}_dropout" if variant.startswith(CONTROL_PREFIXES) else variant


__all__ = [
    "CONTROL_PREFIXES",
    "NON_METHOD_VARIANTS",
    "SETTINGS",
    "SETTING_ORDER",
    "VARIANT_ORDER",
    "build_variants",
    "is_method_variant",
    "ordered_variants",
    "setting_of",
]
