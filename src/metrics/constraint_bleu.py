import math
from itertools import combinations

import numpy as np
import torch
import torchmetrics

import sketchgraphs.data as datalib
from src.metrics.union_find import BCubedF1, UnionFind, UnionFindWeighted
from src.sketch_handlers.sketch_utils import cleanse_sketch_constraints


def is_same_entity(e1, e2):
    if type(e1) != type(e2):
        return False
    # Handle namedtuples or objects with __dict__
    try:
        d1 = e1._asdict()
        d2 = e2._asdict()
    except AttributeError:
        d1 = getattr(e1, "__dict__", {})
        d2 = getattr(e2, "__dict__", {})

    for k in d1:
        if k == "entityId":
            continue
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, float):
            if not math.isclose(v1, v2, rel_tol=1e-5, abs_tol=1e-8):
                return False
        elif isinstance(v1, np.ndarray):
            if not np.allclose(v1, v2, atol=1e-8):
                return False
        elif v1 != v2:
            return False
    return True


def align_constuction_indices(
    target_sketch: datalib.Sketch,
    ref_sketch: datalib.Sketch,
    base_prim_dict: dict[str, datalib.Entity],
) -> datalib.Sketch:

    # Extract added constructions
    # Validation: ensure base primitives match
    for key, base_ent in base_prim_dict.items():
        ref_ent = ref_sketch.entities.get(key)
        target_ent = target_sketch.entities.get(key)
        if ref_ent is None or not is_same_entity(base_ent, ref_ent):
            raise ValueError(f"ref_sketch does not fully include base_prim_dict at key {key}")
        if target_ent is None or not is_same_entity(base_ent, target_ent):
            raise ValueError(f"target_sketch does not fully include base_prim_dict at key {key}")

    ref_added_constructions = {key: val for key, val in ref_sketch.entities.items() if key not in base_prim_dict}
    target_added_constructions = {key: val for key, val in target_sketch.entities.items() if key not in base_prim_dict}

    # Align added constructions
    keys_modified = {}
    remaining_ref_ids = set(ref_added_constructions.keys())

    for target_id, target_ent in target_added_constructions.items():
        found_ref_id = None
        for ref_id in remaining_ref_ids:
            if is_same_entity(target_ent, ref_added_constructions[ref_id]):
                found_ref_id = ref_id
                break

        if found_ref_id:
            keys_modified[target_id] = found_ref_id
            remaining_ref_ids.remove(found_ref_id)
        else:
            # Shift ID to avoid collision with prediction IDs
            keys_modified[target_id] = f"TargetOnly_{target_id}"

    # Update entities in target_sketch
    new_entities = {}
    for k, v in target_sketch.entities.items():
        if k in base_prim_dict:
            new_entities[k] = v
        elif k in keys_modified:
            new_id = keys_modified[k]
            if hasattr(v, "_replace") and "entityId" in v._fields:
                v = v._replace(entityId=new_id)
            new_entities[new_id] = v
    target_sketch.entities.clear()
    target_sketch.entities.update(new_entities)

    # Renumber references in constraints
    for constr_id, constr in target_sketch.constraints.items():
        new_params = []
        for p in constr.parameters:
            if isinstance(p, datalib.LocalReferenceParameter):
                val = p.value
                parts = val.split(".")
                ent_id = parts[0]
                if ent_id in keys_modified:
                    parts[0] = keys_modified[ent_id]
                    val = ".".join(parts)
                new_params.append(datalib.LocalReferenceParameter(parameterId=p.parameterId, value=val))
            else:
                new_params.append(p)
        target_sketch.constraints[constr_id] = target_sketch.constraints[constr_id]._replace(parameters=new_params)

    return target_sketch


def aggregate_constraints(sketch: datalib.Sketch) -> tuple[UnionFind, UnionFind, list[tuple[str, list[tuple[int, ...]]]]]:
    """
    Convert a Sketch object into a canonical constraint set.
    - Point-Point coincidents are summarized into one: [Coincident(A,B), Coincident(B,C)] => Coincident{A,B,C}
    - Coincident is extracted from [Tangent/Perpendicular](Curve1, Point1, Curve2, Point2)
    - Horizontal/Vertical/Parallel/Perpendicular are summarized into a angle-based equivalence classes
    """

    # Modified from src/sketch_handlers/sketch_utils.py
    def _get_refs_and_value(constr: datalib.Constraint) -> tuple[list[tuple[int, int]], float | None]:
        refs = [p.value.split(".") for p in constr.parameters if isinstance(p, datalib.LocalReferenceParameter)]  # e.g., 1.start
        refs = [(int(x[0]), x[1] if (len(x) == 2 and len(x[1]) > 0) else None) for x in refs]
        assert all(y in (None, 'start', 'end', 'center') for _, y in refs)
        refs = [(x[0], {'start': 1, 'end': 2, 'center': 3}.get(x[1], 1 if isinstance(sketch.entities[str(x[0])], datalib.Point) else 0)) for x in refs]

        value = None
        for p in constr.parameters:
            if isinstance(p, datalib.QuantityParameter):
                value = p.value
                break

        return sorted(refs), value

    def _canonicalize_point_coincidents(sketch: datalib.Sketch) -> tuple[UnionFind, set[str]]:
        uf = UnionFind()
        absorbed_constraints = set()

        # register all points
        for ent_id, ent in sketch.entities.items():
            if isinstance(ent, datalib.Point):
                uf.find((ent_id, 1))  # aligned with `_get_refs_and_value``
            elif isinstance(ent, datalib.Line):
                uf.find((ent_id, 1))
                uf.find((ent_id, 2))
            elif isinstance(ent, datalib.Arc):
                uf.find((ent_id, 1))
                uf.find((ent_id, 2))
                uf.find((ent_id, 3))
            elif isinstance(ent, datalib.Circle):
                uf.find((ent_id, 3))

        # register coincidents
        for constr_id, constr in sketch.constraints.items():
            constr_type = constr.constraint_type.name
            refs, _ = _get_refs_and_value(constr)
            if constr_type == "Coincident":
                assert len(refs) == 2
                assert refs[0][1] > 0 and refs[1][1] > 0
                uf.union(refs[0], refs[1])
                absorbed_constraints.add(constr_id)
            if constr_type == "Tangent":
                assert len(refs) == 2
                if refs[0][1] > 0 and refs[1][1] > 0:
                    uf.union(refs[0], refs[1])
            if constr_type == "Perpendicular":
                assert len(refs) == 2
                if refs[0][1] > 0 and refs[1][1] > 0:
                    uf.union(refs[0], refs[1])
        return uf, absorbed_constraints

    def _canonicalize_line_orientations(sketch: datalib.Sketch) -> tuple[UnionFind, set[str]]:

        # Stage 1: Parallel equivalence classes
        uf_parallel = UnionFind()
        absorbed_constraints = set()

        # register x/y axes
        X, Y = -1, -2
        uf_parallel.find(X)
        uf_parallel.find(Y)

        # register all lines
        for ent_id, ent in sketch.entities.items():
            if isinstance(ent, datalib.Line):
                uf_parallel.find((ent_id, 0))

        # register parallel constraints
        for constr_id, constr in sketch.constraints.items():
            constr_type = constr.constraint_type.name
            refs, _ = _get_refs_and_value(constr)

            if constr_type == "Horizontal":
                assert len(refs) == 1 and refs[0][1] == 0
                uf_parallel.union(refs[0][0], X)
                absorbed_constraints.add(constr_id)
            if constr_type == "Vertical":
                assert len(refs) == 1 and refs[0][1] == 0
                uf_parallel.union(refs[0][0], Y)
                absorbed_constraints.add(constr_id)
            if constr_type == "Parallel":
                assert len(refs) == 2 and refs[0][1] == refs[1][1] == 0
                uf_parallel.union(refs[0][0], refs[1][0])
                absorbed_constraints.add(constr_id)

        # Stage 2: Angle labeling
        uf_angle = UnionFindWeighted(mod=math.pi, tol=1e-4)
        uf_angle.set_attr(uf_parallel.find(X), 0)
        uf_angle.set_attr(uf_parallel.find(Y), math.pi / 2)

        for constr_id, constr in sketch.constraints.items():
            constr_type = constr.constraint_type.name
            refs, value = _get_refs_and_value(constr)

            # # NOTE: Angle is too specific, so shouldn't be integrated into an equivalence?
            # if constr_type == "Angle":
            #     assert len(refs) == 2
            #     assert value is not None
            #     (e1, s1), (e2, s2) = refs
            #     root1, root2 = uf_parallel.find(e1), uf_parallel.find(e2)
            #     uf_angle.union(root1, root2, value % math.pi)
            if constr_type == "Perpendicular":
                assert len(refs) == 2
                (e1, s1), (e2, s2) = refs
                root1, root2 = uf_parallel.find(e1), uf_parallel.find(e2)
                if isinstance(sketch.entities[str(e1)], datalib.Line) and isinstance(sketch.entities[str(e2)], datalib.Line):
                    uf_angle.union(root1, root2, math.pi / 2)
                    absorbed_constraints.add(constr_id)

        # Stage 3: merge angle equivalence classes
        parallel_representatives = set(uf_angle.find(x)[0] for x in uf_angle.parent.keys())
        for root1, root2 in combinations(parallel_representatives, 2):
            angle1 = uf_angle.get_attr(root1)
            angle2 = uf_angle.get_attr(root2)
            if angle1 is not None and angle2 is not None and abs(angle1 - angle2) < uf_angle.tol:
                uf_parallel.union(root1, root2)

        return uf_parallel, absorbed_constraints


    # Aggregate
    uf_points, absorbed_points = None, set()
    uf_lines, absorbed_lines = None, set()
    try:
        uf_points, absorbed_points = _canonicalize_point_coincidents(sketch)
        uf_lines, absorbed_lines = _canonicalize_line_orientations(sketch)
        assert len(absorbed_points & absorbed_lines) == 0
    except ValueError as e:
        print(f"Error: {e}", flush=True)
        pass

    constraints_rest = [
        (constr.constraint_type.name, _get_refs_and_value(constr)[0])
        for constr_id, constr in sketch.constraints.items()
        if constr_id not in absorbed_points | absorbed_lines
    ]
    return uf_points, uf_lines, constraints_rest


class ConstraintBLEU(torchmetrics.Metric):
    """
    BLEU-like evaluation metric for CAD sketch constraints.

    This metric calculates similarity between a predicted constraint set (datalib.Sketch)
    and one or more reference ground truth sets. It aggregates scores across the corpus
    using a BLEU-like sum-then-divide approach.

    Attributes:
        total_hits: Accumulated numerator (intersections).
        total_count: Accumulated denominator (unions or lengths).
        num_samples: Total number of samples processed.
        num_invalid_preds: Number of invalid predictions.
    """
    def __init__(self):
        super().__init__()
        # Category-specific accumulated scores (Macro-average approach)
        # We track the sum of scores (0.0-1.0) and the number of samples that had that category.
        for cat in ["points", "lines", "others"]:
            self.add_state(f"{cat}_precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{cat}_recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{cat}_f1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{cat}_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_invalid_preds", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        base_prims: dict[str, datalib.Entity] | list[dict[str, datalib.Entity]],
        preds: datalib.Sketch | list[datalib.Sketch],
        targets: list[datalib.Sketch] | list[list[datalib.Sketch]],
    ):
        """
        Update the metric state with predictions and multiple references.
        NOTE: The set of primitives in preds and targets can be different.
              The difference should be the addition of construction lines and circles.

        Args:
            base_prims: primitives in the model input sketch or list[dict[str, Entity]]
            preds:      single Sketch or list[Sketch]
            targets:    list[Sketch] or list[list[Sketch]]
        """
        # batchfy if not
        if isinstance(base_prims, dict):
            base_prims = [base_prims]
        if isinstance(preds, datalib.Sketch):
            preds = [preds]
        if isinstance(targets[0], datalib.Sketch):
            targets = [targets]

        # Cleanse
        preds = [cleanse_sketch_constraints(sketch) for sketch in preds]
        targets = [[cleanse_sketch_constraints(sketch) for sketch in sketch_pool] for sketch_pool in targets]

        # Iterate over batch dimension
        for base_prim_dict, pred_sketch, target_pool in zip(base_prims, preds, targets):

            # Align indices of added constructions
            target_pool = [
                align_constuction_indices(target_sketch, pred_sketch, base_prim_dict)
                for target_sketch in target_pool]

            # Extract categorized constraints for pred
            pred_uf_points, pred_uf_lines, pred_constraints_rest = aggregate_constraints(pred_sketch)
            if pred_uf_points is None or pred_uf_lines is None:
                # Invalid pred sketch
                self.num_invalid_preds += 1
                continue

            pred_point_items = list(pred_uf_points.parent.keys())
            pred_line_items = list(pred_uf_lines.parent.keys())
            set_pred = set(pred_constraints_rest)

            best_sample_scores = None
            best_sample_avg_f1 = -1.0

            for target_sketch in target_pool:
                # Extract categorized constraints for target
                target_uf_points, target_uf_lines, target_constraints_rest = aggregate_constraints(target_sketch)
                target_point_items = list(target_uf_points.parent.keys())
                target_line_items = list(target_uf_lines.parent.keys())
                set_target = set(target_constraints_rest)

                # Compute point/line precision and recall using the generalized BCubedF1
                points_precision, points_recall, points_f1 = BCubedF1(target_uf_points, pred_uf_points, gt_items=target_point_items, pred_items=pred_point_items)
                lines_precision, lines_recall, lines_f1 = BCubedF1(target_uf_lines, pred_uf_lines, gt_items=target_line_items, pred_items=pred_line_items)

                # 3. Others category (Exact set match)
                others_hits = len(set_pred & set_target)
                others_precision = others_hits / len(set_pred) if len(set_pred) > 0 else 1.0
                others_recall = others_hits / len(set_target) if len(set_target) > 0 else 1.0
                others_f1 = 2 * others_precision * others_recall / (others_precision + others_recall) if (others_precision + others_recall) > 0 else 0.0

                # Determine which categories are valid for this sample
                valid_scores = {}
                if len(pred_point_items) > 0 or len(target_point_items) > 0:
                    valid_scores["points"] = (points_precision, points_recall, points_f1)
                if len(pred_line_items) > 0 or len(target_line_items) > 0:
                    valid_scores["lines"] = (lines_precision, lines_recall, lines_f1)
                if len(set_pred) > 0 or len(set_target) > 0:
                    valid_scores["others"] = (others_precision, others_recall, others_f1)

                s_avg_f1 = sum(v[2] for v in valid_scores.values()) / len(valid_scores) if valid_scores else 0.0

                if s_avg_f1 > best_sample_avg_f1:
                    best_sample_avg_f1 = s_avg_f1
                    best_sample_scores = valid_scores

            if best_sample_scores:
                for cat, (p, r, f1) in best_sample_scores.items():
                    getattr(self, f"{cat}_precision_sum").add_(p)
                    getattr(self, f"{cat}_recall_sum").add_(r)
                    getattr(self, f"{cat}_f1_sum").add_(f1)
                    getattr(self, f"{cat}_samples").add_(1)
                self.num_samples += 1

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Computes the final aggregated score (Macro-averaged across Samples and Categories).
        """
        precision_list = []
        recall_list = []
        f1_list = []

        cat_results = {}
        for cat in ["points", "lines", "others"]:
            samples = getattr(self, f"{cat}_samples")
            if samples > 0:
                p = getattr(self, f"{cat}_precision_sum") / samples
                r = getattr(self, f"{cat}_recall_sum") / samples
                f1 = getattr(self, f"{cat}_f1_sum") / samples

                precision_list.append(p)
                recall_list.append(r)
                f1_list.append(f1)
                cat_results[cat] = {"precision": p, "recall": r, "f1": f1}

        return {
            "precision": sum(precision_list) / len(precision_list) if precision_list else torch.tensor(0.0),
            "recall": sum(recall_list) / len(recall_list) if recall_list else torch.tensor(0.0),
            "f1": sum(f1_list) / len(f1_list) if f1_list else torch.tensor(0.0),
            "categories": cat_results,
            "num_samples": self.num_samples,
            "num_invalid_preds": self.num_invalid_preds
        }
