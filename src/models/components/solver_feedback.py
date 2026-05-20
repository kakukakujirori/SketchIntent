"""Solver feedback utilities for constraint evaluation.

This module provides functionality to:
1. Convert model output tokens to FreeCAD sketch
2. Run the geometry solver
3. Return solver status (DoF, conflicts, redundancies, ...)
"""
from __future__ import annotations
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import TYPE_CHECKING

import numpy as np

import sketchgraphs.data as datalib
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import complete_constraint_values

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


@dataclass
class SolverStatus:
    """Solver status returned by FreeCAD + alpha.

    NOTE: XXXConstrains can accommodate str or int.
        - If int, they are 1-indexed FreeCAD constraints.
        - If str, they are 0-indexed SketchGraphs constraints.

    NOTE: DistortingConstraints informs either
        - There is a distorting constraints (bool), or
        - Which constraints are causing distortion (list[str]).

    NOTE: RigidDimensionalConstraints lists dimensional constraints whose
        value perturbation (x0.5 or x2.0) causes the solver to no longer
        converge.  Such constraints are considered "rigid" — they encode a
        specific numeric fact rather than a topological/geometric intent —
        and are penalised during RL training.
    """
    DoF: int = -1
    Status: str = "Unknown"
    NumRelationalConstraints: int = 0
    NumDimensionalConstraints: int = 0
    MalformedConstraints: list[str] = field(default_factory=list)
    ConflictingConstraints: list[str] = field(default_factory=list)
    RedundantConstraints: list[str] = field(default_factory=list)
    DistortingConstraints: list[str] | bool = False
    RigidDimensionalConstraints: list[str] = field(default_factory=list)


class SolverFeedback:
    def __init__(self, context: SketchContext, distortion_threshold: float | None = None):
        self.sketch_tokenizer = SketchTokenizer(context)
        self.sketch_freecad_converter = SketchFreecadConverter(context, raise_on_error=False)
        self.distortion_threshold = distortion_threshold or 1 / context.num_quantization_bins
        assert self.distortion_threshold > 0, f"{self.distortion_threshold=}"

    @classmethod
    def count_relational_and_dimensional_constraints(cls, fc_sketch: "Sketcher.SketchObject",) -> tuple[int, int]:
        DIMENSIONAL_CONSTRAINT_TYPES = {"Distance", "DistanceX", "DistanceY", "Angle", "Radius", "Diameter"}
        relational = 0
        dimensional = 0
        for fc_constr in fc_sketch.Constraints:
            if fc_constr.Type in DIMENSIONAL_CONSTRAINT_TYPES:
                dimensional += 1
            else:
                relational += 1
        return (relational, dimensional)

    @classmethod
    def _extract_geometry_coords(cls, fc_sketch: "Sketcher.SketchObject") -> dict[int, np.ndarray]:
        """Extract coordinates from all geometries in a FreeCAD sketch.

        Returns dict mapping geo_id -> list of (x, y) points.
        """
        coords = {}
        for geo_id, geo in enumerate(fc_sketch.Geometry):
            geo_type = type(geo).__name__
            points = []

            if geo_type == 'Point':
                points.append((geo.X, geo.Y))
            elif geo_type == 'LineSegment':
                sp = geo.StartPoint
                ep = geo.EndPoint
                points.append((sp.x, sp.y))
                points.append((ep.x, ep.y))
            elif geo_type == 'Circle':
                c = geo.Center
                r = geo.Radius
                points.append((c.x, c.y))
                points.append((c.x + r, c.y))
            elif geo_type == 'ArcOfCircle':
                sp = geo.StartPoint
                ep = geo.EndPoint
                # Get midpoint on arc curve using center and angles
                # This handles wrap-around cases correctly
                cx, cy = geo.Center.x, geo.Center.y
                r = geo.Radius
                # Compute angles from center to start/end points
                start_angle = math.atan2(sp.y - cy, sp.x - cx)
                end_angle = math.atan2(ep.y - cy, ep.x - cx)
                # Normalize and handle wrap-around
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                mid_angle = (start_angle + end_angle) / 2.0
                mx = cx + r * math.cos(mid_angle)
                my = cy + r * math.sin(mid_angle)
                points.append((sp.x, sp.y))
                points.append((mx, my))
                points.append((ep.x, ep.y))
            else:
                raise NotImplementedError(f"Unhandled geometry type: {geo_type}")

            coords[geo_id] = np.array(points)

        return coords

    _ANISOTROPY_RATIO_TOL = 1.10

    @classmethod
    def _extract_anchor_points(cls, fc_sketch: "Sketcher.SketchObject") -> dict[int, np.ndarray]:
        """Per-entity anchor used by the alignment residual check.

        Mirrors check_submission's per-attribute comparison: line midpoint
        (pntX, pntY when round-tripped through sketchgraphs), circle/arc center,
        and point position. Rotations of a line around its midpoint therefore
        do not register as distortion, matching the annotator submission check.
        """
        anchors: dict[int, np.ndarray] = {}
        for geo_id, geo in enumerate(fc_sketch.Geometry):
            geo_type = type(geo).__name__
            if geo_type == 'Point':
                anchors[geo_id] = np.array([[geo.X, geo.Y]])
            elif geo_type == 'LineSegment':
                sp, ep = geo.StartPoint, geo.EndPoint
                anchors[geo_id] = np.array([[(sp.x + ep.x) / 2.0, (sp.y + ep.y) / 2.0]])
            elif geo_type in ('Circle', 'ArcOfCircle'):
                c = geo.Center
                anchors[geo_id] = np.array([[c.x, c.y]])
            else:
                raise NotImplementedError(f"Unhandled geometry type: {geo_type}")
        return anchors

    @classmethod
    def check_geometry_distortion(
        cls,
        fc_sketch_pre: "Sketcher.SketchObject",
        fc_sketch_post: "Sketcher.SketchObject",
        distortion_threshold: float,
    ) -> bool:
        """Detect distortion after bbox-based anisotropic affine alignment of post to pre.

        Relational constraints are scale/translation invariant, so the solver may
        legitimately settle at a different absolute scale/position. We absorb that
        freedom by aligning post to pre via per-axis bbox scaling + translation,
        then comparing anchor residuals against ``distortion_threshold * max(bw, bh)``.
        Mirrors ``fusion360_annotation/check_submission.py:align_sketch``.
        """
        coords_pre = SolverFeedback._extract_geometry_coords(fc_sketch_pre)
        coords_post = SolverFeedback._extract_geometry_coords(fc_sketch_post)
        geo_id_list = list(coords_pre.keys())
        if set(coords_pre.keys()) != set(coords_post.keys()):
            raise ValueError(f"fc_sketch_pre and fc_sketch_post must have the same set of geometry ids: {coords_pre.keys()=}, {coords_post.keys()=}")
        if not all(coords_pre[gid].shape == coords_post[gid].shape for gid in geo_id_list):
            raise ValueError(f"fc_sketch_pre and fc_sketch_post must have the same number of points per primitive: {coords_pre=}, {coords_post=}")

        # bbox uses full geometric extent (start/end of lines, ±radius of circles)
        all_pre = np.concatenate([coords_pre[gid] for gid in geo_id_list], axis=0)
        all_post = np.concatenate([coords_post[gid] for gid in geo_id_list], axis=0)

        pre_min, pre_max = all_pre.min(axis=0), all_pre.max(axis=0)
        post_min, post_max = all_post.min(axis=0), all_post.max(axis=0)
        bw, bh = float(pre_max[0] - pre_min[0]), float(pre_max[1] - pre_min[1])
        tw, th = float(post_max[0] - post_min[0]), float(post_max[1] - post_min[1])

        fx = bw / tw if tw > 1e-9 else 1.0
        fy = bh / th if th > 1e-9 else 1.0
        scale_ratio = fx / fy if fy > 1e-9 else float('inf')
        if not (1.0 / cls._ANISOTROPY_RATIO_TOL <= scale_ratio <= cls._ANISOTROPY_RATIO_TOL):
            return True

        tcx, tcy = (post_min[0] + post_max[0]) / 2.0, (post_min[1] + post_max[1]) / 2.0
        bcx, bcy = (pre_min[0] + pre_max[0]) / 2.0, (pre_min[1] + pre_max[1]) / 2.0

        # residual check uses anchor points only (line midpoint, circle/arc center, point)
        anchors_pre = cls._extract_anchor_points(fc_sketch_pre)
        anchors_post = cls._extract_anchor_points(fc_sketch_post)
        anchors_pre_arr = np.concatenate([anchors_pre[gid] for gid in geo_id_list], axis=0)
        anchors_post_arr = np.concatenate([anchors_post[gid] for gid in geo_id_list], axis=0)

        aligned_post = np.empty_like(anchors_post_arr)
        aligned_post[:, 0] = (anchors_post_arr[:, 0] - tcx) * fx + bcx
        aligned_post[:, 1] = (anchors_post_arr[:, 1] - tcy) * fy + bcy

        bbox_size = max(bw, bh) if max(bw, bh) > 1e-9 else 1.0
        pos_tol_abs = distortion_threshold * bbox_size
        residual = np.linalg.norm(anchors_pre_arr - aligned_post, axis=1)
        return bool(np.any(residual > pos_tol_abs))

    def get_solver_status(self, sketch: datalib.Sketch) -> SolverStatus:
        """Get solver status from FreeCAD sketch."""
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        doc = None
        doc_primitives_only = None
        invalid_constraints = []
        try:
            # SketchGraphs → FreeCAD
            doc, fc_sketch, primitive_freecad_mapping, constraint_freecad_mapping, invalid_constraints = \
                self.sketch_freecad_converter.to_freecad(sketch, doc_name="SolverEval", skip_invalid=True)
            doc_primitives_only, fc_sketch_primitives_only, _ = \
                self.sketch_freecad_converter.to_freecad_primitives(sketch, doc_name="SolverEvalPrimitivesOnly")

            # CAD Solver
            fc_sketch.solve()
            fc_status = fc_sketch.getSolverStatus()

            # Other traits
            relational, dimensional = SolverFeedback.count_relational_and_dimensional_constraints(fc_sketch)
            is_distorted = SolverFeedback.check_geometry_distortion(fc_sketch_primitives_only, fc_sketch, self.distortion_threshold)

            def _map_indices(fc_indices: list[int]) -> list[str]:
                sketchgraphs_indices = []
                for fc_idx in fc_indices:
                    idx_str = constraint_freecad_mapping.get(fc_idx, None)  # idx_str = 'c_xxx'
                    if idx_str is not None:
                        sketchgraphs_indices.append(idx_str)
                return sorted(list(set(sketchgraphs_indices)))

            status = SolverStatus(
                DoF = fc_status['DoF'],
                Status = fc_status['Status'],
                NumRelationalConstraints = relational,
                NumDimensionalConstraints = dimensional,
                MalformedConstraints = sorted(list(set(_map_indices(fc_status['MalformedConstraints']) + invalid_constraints))),
                ConflictingConstraints = _map_indices(fc_status['ConflictingConstraints']),
                RedundantConstraints = _map_indices(fc_status['RedundantConstraints']),
                DistortingConstraints = is_distorted,
            )
        except Exception:
            status = SolverStatus(
                Status=f"Error: {traceback.format_exc()}",
                MalformedConstraints = invalid_constraints,
            )
        finally:
            # Close document
            FreeCAD.closeDocument(doc.Name) if doc else None
            FreeCAD.closeDocument(doc_primitives_only.Name) if doc_primitives_only else None
        return status

    def evaluate_constraints(
        self,
        prim_tokens: np.ndarray,
        cons_tokens: np.ndarray,
        sketch_original: datalib.Sketch | None = None,
        check_one_by_one: bool = False,
    ) -> tuple[SolverStatus, datalib.Sketch, dict[str, int], dict[int, tuple[int, ...]]]:
        """Evaluate constraint status for a single sample.

        Parameters
        ----------
        prim_tokens : np.ndarray
            Primitive value tokens [L_prim]. Must be a numpy array.
        cons_tokens : np.ndarray
            Constraint value tokens [L_cons]. Must be a numpy array.
        sketch_original : datalib.Sketch | None
            Original sketch, used for precise primitive value assignment.
            If None, primitive values are assigned from the quantized prim_tokens.
        check_one_by_one: bool
            If True, constraints are incrementally added and verified at each time.

        Returns
        -------
        SolverStatus: SolverStatus
            Solver status
        sketchgraphs_sketch : datalib.Sketch
            Reconstructed sketch from tokens
        constraint_to_token_start : dict[str, int]
            Mapping from constraint id to token start index
        invalid_constraint_block_start : dict[int, tuple[int, ...]]
            Mapping from token start index to invalid constraint token block
        """
        if not isinstance(prim_tokens, np.ndarray):
            raise TypeError(f"prim_tokens must be np.ndarray, got {type(prim_tokens)}. Call .cpu().numpy() before passing.")
        if not isinstance(cons_tokens, np.ndarray):
            raise TypeError(f"cons_tokens must be np.ndarray, got {type(cons_tokens)}. Call .cpu().numpy() before passing.")
        assert len(prim_tokens.shape) == 1, f"{prim_tokens.shape=}"
        assert len(cons_tokens.shape) == 1, f"{cons_tokens.shape=}"

        # Tokens → SketchGraphs
        sketchgraphs_sketch, constraint_to_token_start, invalid_constraint_block_start = self.sketch_tokenizer.detokenize(
            {"p_val": prim_tokens, "c_val": cons_tokens},
            allow_invalid_constraints=True,
        )
        if sketch_original:
            # Sanity check (NOTE: Due to max_token_length, the number of primitives may be less than the original number of primitives)
            if len(sketchgraphs_sketch.entities) > len(sketch_original.entities):
                error_msg = f"Number of entities mismatch:\n"
                error_msg += "[sketch_original]\n"
                for e_id, e in sketch_original.entities.items():
                    error_msg += f"    {e_id}: {e}\n"
                error_msg += "[sketchgraphs_sketch]\n"
                for e_id, e in sketchgraphs_sketch.entities.items():
                    error_msg += f"    {e_id}: {e}\n"

                # import os, pickle
                # os.makedirs("dump", exist_ok=True)
                # pickle.dump(sketchgraphs_sketch, open("dump/sketchgraphs_sketch.pkl", "wb"))
                # pickle.dump(prim_tokens, open("dump/prim_tokens.pkl", "wb"))
                # pickle.dump(cons_tokens, open("dump/cons_tokens.pkl", "wb"))
                # pickle.dump(sketch_original, open("dump/sketch_original.pkl", "wb"))
                # error_msg += "==> Artifacts dumped to `dump/`"

                raise ValueError(error_msg)
            for ent_gen, ent_ori in zip(sketchgraphs_sketch.entities.values(), sketch_original.entities.values()):
                if type(ent_gen) != type(ent_ori):
                    raise ValueError(f"Entity type mismatch: {type(ent_gen)=} vs {type(ent_ori)=}")
            # overwrite
            sketch_original_entities_reversed = list(sketch_original.entities.values())[::-1]
            for ent_id, ent in sketchgraphs_sketch.entities.items():
                ent_ori = sketch_original_entities_reversed.pop()
                ent_ori.entityId = ent.entityId
                sketchgraphs_sketch.entities[ent_id] = ent_ori

        # Assign dimensional values (since the model doesn't infer dimensional contraint values)
        sketchgraphs_sketch, invalid_constraints = complete_constraint_values(sketchgraphs_sketch, skip_invalid_constraints=True)

        if check_one_by_one:
            # incrementally add constraints
            solver_report = SolverStatus(MalformedConstraints=invalid_constraints, DistortingConstraints=[])
            incremental_constraints: dict[str, datalib.Constraint] = {}
            for cons_id_str, cons in sketchgraphs_sketch.constraints.items():

                # invalid constraints are already registered in solver_report
                if cons_id_str in invalid_constraints:
                    continue

                sketchgraphs_sketch_incremental = datalib.Sketch(
                    entities=sketchgraphs_sketch.entities,
                    constraints=incremental_constraints | {cons_id_str: cons},
                )
                status = self.get_solver_status(sketchgraphs_sketch_incremental)

                if status.Status == "Converged":
                    incremental_constraints[cons_id_str] = cons
                    if status.DistortingConstraints:
                        assert isinstance(solver_report.DistortingConstraints, list)
                        solver_report.DistortingConstraints.append(cons_id_str)
                elif status.Status == "Redundant":
                    solver_report.RedundantConstraints.append(cons_id_str)
                elif status.Status == "Conflicting":
                    solver_report.ConflictingConstraints.append(cons_id_str)
                elif status.Status == "Malformed" or status.Status == "Failed" or status.Status.startswith("Error"):
                    solver_report.MalformedConstraints.append(cons_id_str)
                elif status.Status == "OverConstrained":
                    if status.MalformedConstraints:
                        solver_report.MalformedConstraints.append(cons_id_str)
                    elif status.ConflictingConstraints:
                        solver_report.ConflictingConstraints.append(cons_id_str)
                    elif status.RedundantConstraints:
                        solver_report.RedundantConstraints.append(cons_id_str)
                else:
                    raise NotImplementedError(f"Unhandled SolverStatus: {status.Status}")

            if solver_report.MalformedConstraints:
                solver_report.Status = "Malformed"
            elif solver_report.ConflictingConstraints:
                solver_report.Status = "Conflicting"
            elif solver_report.RedundantConstraints:
                solver_report.Status = "Redundant"
            elif solver_report.DistortingConstraints:
                solver_report.Status = "Converged"  # NOTE: We don't include 'Deviating' in the status
            else:
                solver_report.Status = "Converged"

        # Final verdict
        status = self.get_solver_status(sketchgraphs_sketch)
        status.MalformedConstraints = sorted(list(set(status.MalformedConstraints) | set(invalid_constraints)))
        if status.MalformedConstraints:
            status.Status = "Malformed"

        # ── Dimensional-constraint robustness check ───────────────────────── #
        # Only run when the full sequence is Converged; otherwise we cannot
        # distinguish "rigid" constraints from already-broken ones.
        if status.Status == "Converged":
            status.RigidDimensionalConstraints = self._check_rigid_dimensional_constraints(
                sketchgraphs_sketch
            )

        if check_one_by_one:
            solver_report.DoF = status.DoF if solver_report.Status == "Converged" else -1
            solver_report.NumRelationalConstraints = status.NumRelationalConstraints
            solver_report.NumDimensionalConstraints = status.NumDimensionalConstraints
            return solver_report, sketchgraphs_sketch, constraint_to_token_start, invalid_constraint_block_start
        else:
            return status, sketchgraphs_sketch, constraint_to_token_start, invalid_constraint_block_start

    # ------------------------------------------------------------------ #
    #  Dimensional-constraint robustness check                           #
    # ------------------------------------------------------------------ #

    _DIMENSIONAL_CONSTRAINT_TYPES = frozenset({
        "Diameter", "Distance", "DistanceX", "DistanceY", "Angle", "Radius", "Length",
    })
    _RIGIDITY_SCALE_FACTORS = (0.5, 2.0)

    def _check_rigid_dimensional_constraints(
        self,
        sketch: "datalib.Sketch",
    ) -> list[str]:
        """Return constraint IDs that break the solver when their value is scaled.

        For each dimensional constraint in *sketch* (assumed to be fully-valued
        after ``complete_constraint_values``), a copy of the sketch is built
        with that single constraint's QuantityParameter multiplied by each
        factor in ``_RIGIDITY_SCALE_FACTORS``.  If *any* scale factor causes
        the solver to return a non-Converged status for the perturbed copy, the
        constraint is labelled "rigid" and included in the returned list.

        The check is intentionally permissive: only a single constraint is
        perturbed at a time, and the original sketch already converged.
        """
        import copy as _copy
        from sketchgraphs.data import QuantityParameter as _QP

        rigid: list[str] = []

        for cons_id, cons in sketch.constraints.items():
            # Only consider dimensional constraints that carry a numeric quantity
            cons_type_name = cons.constraint_type.name
            if cons_type_name not in self._DIMENSIONAL_CONSTRAINT_TYPES:
                continue

            qty_params = [
                (idx, p) for idx, p in enumerate(cons.parameters)
                if isinstance(p, _QP)
            ]
            if not qty_params:
                continue

            is_rigid = False
            for scale in self._RIGIDITY_SCALE_FACTORS:
                # Build a perturbed copy of the sketch
                perturbed_sketch = _copy.deepcopy(sketch)
                perturbed_cons = perturbed_sketch.constraints[cons_id]
                new_params = list(perturbed_cons.parameters)
                for idx, p in qty_params:
                    try:
                        orig_value = float(p.value)
                    except (TypeError, ValueError):
                        # Cannot scale a non-numeric value; skip this constraint
                        break
                    scaled_value = orig_value * scale
                    new_params[idx] = _QP(
                        parameterId=p.parameterId,
                        value=scaled_value,
                        expression=f"RIGIDITY_CHECK:{scaled_value}",
                    )
                else:
                    # All qty_params were successfully scaled
                    perturbed_sketch.constraints[cons_id] = perturbed_cons._replace(
                        parameters=new_params
                    )
                    try:
                        perturbed_status = self.get_solver_status(perturbed_sketch)
                    except Exception:
                        # Treat solver crashes as non-convergence
                        perturbed_status = SolverStatus(Status="Error")
                    if perturbed_status.Status != "Converged":
                        is_rigid = True
                        break  # one failing scale is enough

            if is_rigid:
                rigid.append(cons_id)

        return sorted(rigid)

    def evaluate_constraints_batch(
        self,
        prim_tokens_batch: np.ndarray,
        cons_tokens_batch: np.ndarray,
        sketch_original_list: list[datalib.Sketch] | None = None,
        fully_constrained_dof: int = 2,
    ) -> dict[str, float]:
        """Evaluate constraint status for a batch of samples.

        Parameters
        ----------
        prim_tokens_batch : np.ndarray
            Primitive tokens [B, L_prim]. Must be a numpy array.
        cons_tokens_batch : np.ndarray
            Constraint tokens [B, L_cons]. Must be a numpy array.
        sketch_original_list : list[datalib.Sketch] | None
            Original sketches, used for precise primitive value assignment.
            If None, primitive values are assigned from the quantized prim_tokens.
        fully_constrained_dof : int
            Number of degrees of freedom to be regarded as fully constrained

        Returns
        -------
        dict[str, float]
            Dictionary with rates: fc_rate, under_rate, over_rate, redundant_rate
        """
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        if not isinstance(prim_tokens_batch, np.ndarray):
            raise TypeError(f"prim_tokens_batch must be np.ndarray, got {type(prim_tokens_batch)}. Call .cpu().numpy() before passing.")
        if not isinstance(cons_tokens_batch, np.ndarray):
            raise TypeError(f"cons_tokens_batch must be np.ndarray, got {type(cons_tokens_batch)}. Call .cpu().numpy() before passing.")
        assert prim_tokens_batch.shape[0] == cons_tokens_batch.shape[0], f"Batch size mismatch: {prim_tokens_batch.shape=} != {cons_tokens_batch.shape=}"
        B = prim_tokens_batch.shape[0]
        if sketch_original_list is None:
            sketch_original_list = [None] * B
        assert len(sketch_original_list) == B, f"Sketch original list size mismatch: {len(sketch_original_list)=} != {B=}"


        counts = {
            'fully_constrained': 0,
            'under_constrained': 0,
            'distorted': 0,
            'conflicting': 0,
            'redundant': 0,
            'malformed': 0,
            'failed': 0,
        }

        dofs: list[int] = []
        for prim_tokens, cons_tokens, sketch_original in tqdm(zip(prim_tokens_batch, cons_tokens_batch, sketch_original_list), desc="Solver Testing"):

            status, sketch, constraint_to_token_start, invalid_constraint_block_start = self.evaluate_constraints(
                prim_tokens, cons_tokens, sketch_original, check_one_by_one=False)

            if status.Status == "Error":
                counts['failed'] += 1
            elif status.MalformedConstraints:
                counts['malformed'] += 1
            elif status.ConflictingConstraints:
                counts['conflicting'] += 1
            elif status.RedundantConstraints:
                counts['redundant'] += 1
            elif bool(status.DistortingConstraints):
                counts['distorted'] += 1
            else:
                if status.DoF <= fully_constrained_dof:
                    counts['fully_constrained'] += 1
                else:
                    counts['under_constrained'] += 1
                if status.DoF >= 0:
                    dofs.append(status.DoF)

        # Compute rates
        total_valid = B - counts['failed']
        if total_valid == 0:
            return {}

        return {
            'fc_rate': counts['fully_constrained'] / total_valid,
            'under_rate': counts['under_constrained'] / total_valid,
            'distorted_rate': counts['distorted'] / total_valid,
            'conflicting_rate': counts['conflicting'] / total_valid,
            'redundant_rate': counts['redundant'] / total_valid,
            'malformed_rate': counts['malformed'] / total_valid,
            'failed_rate': counts['failed'] / B,
            'valid_dof_mean': sum(dofs) / len(dofs) if dofs else 0.0,
        }
