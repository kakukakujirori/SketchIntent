from .accuracy import TokenAccuracy
from .constraint_iou import ConstraintIoU
from .constraint_bleu import ConstraintBLEU

__all__ = [
    "TokenAccuracy",
    "ConstraintIoU",
    "ConstraintBLEU",
    "compute_null_space_distance",
    "compute_edit_response_distance",
    "compute_edit_invariant_relation_distance",
    "compute_native_finite_feasibility",
    "compute_finite_optimized_feasibility",
    "collect_probes",
    "pairwise_distance_observable",
    "nearest_gt",
    "human_human_variation",
    "normalized_distance",
    "pairwise_distances",
    "discover_per_annotator_dirs",
    "discover_combined_gt_dir",
    "compute_histcad_2d",
    "compute_histcad_2d_pair",
    "select_preserved_relations",
    "HistCADConfig",
    "HistCADResult",
    "HistCADProbeResult",
]


_LAZY = {
    "compute_null_space_distance":
        ("src.metrics.null_space_distance", "compute_null_space_distance"),
    "compute_edit_response_distance":
        ("src.metrics.edit_response_distance", "compute_edit_response_distance"),
    "compute_edit_invariant_relation_distance":
        ("src.metrics.edit_invariant_relation", "compute_edit_invariant_relation_distance"),
    "compute_native_finite_feasibility":
        ("src.metrics.edit_feasibility_finite", "compute_native_finite_feasibility"),
    "compute_finite_optimized_feasibility":
        ("src.metrics.edit_feasibility_optimized", "compute_finite_optimized_feasibility"),
    "collect_probes":
        ("src.metrics.edit_response_distance", "collect_probes"),
    "pairwise_distance_observable":
        ("src.metrics.edit_response_distance", "pairwise_distance_observable"),
    "nearest_gt":
        ("src.metrics.multi_gt_aggregation", "nearest_gt"),
    "human_human_variation":
        ("src.metrics.multi_gt_aggregation", "human_human_variation"),
    "normalized_distance":
        ("src.metrics.multi_gt_aggregation", "normalized_distance"),
    "pairwise_distances":
        ("src.metrics.multi_gt_aggregation", "pairwise_distances"),
    "discover_per_annotator_dirs":
        ("src.metrics.multi_gt_aggregation", "discover_per_annotator_dirs"),
    "discover_combined_gt_dir":
        ("src.metrics.multi_gt_aggregation", "discover_combined_gt_dir"),
    "compute_histcad_2d":
        ("src.metrics.histcad_2d", "compute_histcad_2d"),
    "compute_histcad_2d_pair":
        ("src.metrics.histcad_2d", "compute_histcad_2d_pair"),
    "select_preserved_relations":
        ("src.metrics.histcad_2d", "select_preserved_relations"),
    "HistCADConfig":
        ("src.metrics.histcad_2d", "HistCADConfig"),
    "HistCADResult":
        ("src.metrics.histcad_2d", "HistCADResult"),
    "HistCADProbeResult":
        ("src.metrics.histcad_2d", "HistCADProbeResult"),
}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = spec
    import importlib
    return getattr(importlib.import_module(module_path), attr)
