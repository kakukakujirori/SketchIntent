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
import torch

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
    """
    DoF: int = -1
    Status: str = "Unknown"
    NumRelationalConstraints: int = 0
    NumDimensionalConstraints: int = 0
    MalformedConstraints: list[str] = field(default_factory=list)
    ConflictingConstraints: list[str] = field(default_factory=list)
    RedundantConstraints: list[str] = field(default_factory=list)
    DistortingConstraints: list[str] | bool = False


class SolverFeedback:
    def __init__(self, context: SketchContext, distortion_threshold: float | None = None):
        self.sketch_tokenizer = SketchTokenizer(context)
        self.sketch_freecad_converter = SketchFreecadConverter(context)
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

    @classmethod
    def check_geometry_distortion(
        cls,
        fc_sketch_pre: "Sketcher.SketchObject",
        fc_sketch_post: "Sketcher.SketchObject",
        distortion_threshold: float,
    ) -> bool:
        coords_pre = SolverFeedback._extract_geometry_coords(fc_sketch_pre)
        coords_post = SolverFeedback._extract_geometry_coords(fc_sketch_post)
        geo_id_list = coords_pre.keys()
        if len(coords_pre) != len(coords_post):
            raise ValueError(f"fc_sketch_pre and fc_sketch_post must have the same number and order of primitives: {len(coords_pre)=}, {len(coords_post)=}")
        if set(coords_pre.keys()) != set(coords_post.keys()):
            raise ValueError(f"fc_sketch_pre and fc_sketch_post must have the same set of geometry ids: {coords_pre.keys()=}, {coords_post.keys()=}")
        if not all(coords_pre[gid].shape == coords_post[gid].shape for gid in geo_id_list):
            raise ValueError(f"fc_sketch_pre and fc_sketch_post must have the same number of points per primitive: {coords_pre=}, {coords_post=}")

        is_distorted = False
        for gid in geo_id_list:
            p_pre = coords_pre[gid]
            p_post = coords_post[gid]
            dist = np.linalg.norm(p_pre - p_post, axis=1)
            if np.any(dist > distortion_threshold):
                is_distorted = True
                break

        return is_distorted

    def get_solver_status(self, sketch: datalib.Sketch) -> SolverStatus:
        """Get solver status from FreeCAD sketch."""
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        doc = None
        doc_primitives_only = None
        try:
            # SketchGraphs → FreeCAD
            doc, fc_sketch, primitive_freecad_mapping, constraint_freecad_mapping = \
                self.sketch_freecad_converter.to_freecad(sketch, doc_name="SolverEval")
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
                MalformedConstraints = _map_indices(fc_status['MalformedConstraints']),
                ConflictingConstraints = _map_indices(fc_status['ConflictingConstraints']),
                RedundantConstraints = _map_indices(fc_status['RedundantConstraints']),
                DistortingConstraints = is_distorted,
            )
        except Exception:
            status = SolverStatus(Status=f"Error: {traceback.format_exc()}")
        finally:
            # Close document
            FreeCAD.closeDocument(doc.Name) if doc else None
            FreeCAD.closeDocument(doc_primitives_only.Name) if doc_primitives_only else None
        return status

    def evaluate_constraints(
        self,
        prim_tokens: np.ndarray | torch.Tensor,
        cons_tokens: np.ndarray | torch.Tensor,
        sketch_original: datalib.Sketch | None = None,
        check_one_by_one: bool = False,
    ) -> tuple[SolverStatus, datalib.Sketch, list[tuple[int, ...]]]:
        """Evaluate constraint status for a single sample.

        Parameters
        ----------
        prim_tokens : array-like
            Primitive value tokens [L_prim]
        cons_tokens : array-like
            Constraint value tokens [L_cons]
        sketch_original : datalib.Sketch | None
            Original sketch, used for precise primitive value assignment.
            If None, primitive values are assigned from the quantized prim_tokens.
        check_one_by_one: bool
            If True, constraints are incrementally added and verified at each time.

        Returns
        -------
        SolverStatus
            Solver status
        """
        assert len(prim_tokens.shape) == 1, f"{prim_tokens.shape=}"
        assert len(cons_tokens.shape) == 1, f"{cons_tokens.shape=}"

        # Convert to numpy if needed
        if isinstance(prim_tokens, torch.Tensor):
            prim_tokens = prim_tokens.cpu().numpy()
        if isinstance(cons_tokens, torch.Tensor):
            cons_tokens = cons_tokens.cpu().numpy()

        # Tokens → SketchGraphs
        sketchgraphs_sketch, invalid_cons_token_blocks = self.sketch_tokenizer.detokenize(
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
        sketchgraphs_sketch = complete_constraint_values(sketchgraphs_sketch)

        if check_one_by_one:
            # incrementally add constraints
            solver_report = SolverStatus(DistortingConstraints=[])
            incremental_constraints: dict[str, datalib.Constraint] = {}
            for cons_id_str, cons in sketchgraphs_sketch.constraints.items():

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

        if check_one_by_one:
            solver_report.DoF = status.DoF if solver_report.Status == "Converged" else -1
            solver_report.NumRelationalConstraints = status.NumRelationalConstraints
            solver_report.NumDimensionalConstraints = status.NumDimensionalConstraints
            return solver_report, sketchgraphs_sketch, invalid_cons_token_blocks
        else:
            return status, sketchgraphs_sketch, invalid_cons_token_blocks

    def evaluate_constraints_batch(
        self,
        prim_tokens_batch: torch.Tensor | np.ndarray,
        cons_tokens_batch: torch.Tensor | np.ndarray,
        sketch_original_list: list[datalib.Sketch] | None = None,
        fully_constrained_dof: int = 2,
    ) -> dict[str, float]:
        """Evaluate constraint status for a batch of samples.

        Parameters
        ----------
        prim_tokens_batch : torch.Tensor
            Primitive tokens [B, L_prim]
        cons_tokens_batch : torch.Tensor
            Constraint tokens [B, L_cons]
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

        assert prim_tokens_batch.shape[0] == cons_tokens_batch.shape[0], f"Batch size mismatch: {prim_tokens_batch.shape=} != {cons_tokens_batch.shape=}"
        B = prim_tokens_batch.shape[0]
        if sketch_original_list is None:
            sketch_original_list = [None] * B
        assert len(sketch_original_list) == B, f"Sketch original list size mismatch: {len(sketch_original_list)=} != {B=}"

        # Convert to numpy if needed
        if isinstance(prim_tokens_batch, torch.Tensor):
            prim_tokens_batch = prim_tokens_batch.cpu().numpy()
        if isinstance(cons_tokens_batch, torch.Tensor):
            cons_tokens_batch = cons_tokens_batch.cpu().numpy()

        counts = {
            'fully_constrained': 0,
            'under_constrained': 0,
            'over_constrained': 0,
            'redundant': 0,
            'malformed': 0,
            'conversion_failed': 0,
        }

        for prim_tokens, cons_tokens, sketch_original in tqdm(zip(prim_tokens_batch, cons_tokens_batch, sketch_original_list), desc="Solver Testing"):

            status, sketch, invalid_cons_token_blocks = self.evaluate_constraints(
                prim_tokens, cons_tokens, sketch_original, check_one_by_one=False)

            if status.Status == "Error":
                counts['conversion_failed'] += 1
            elif status.MalformedConstraints:
                counts['malformed'] += 1
            elif status.ConflictingConstraints:
                counts['over_constrained'] += 1
            elif status.RedundantConstraints:
                counts['redundant'] += 1
            elif status.DoF <= fully_constrained_dof:
                counts['fully_constrained'] += 1
            else:
                counts['under_constrained'] += 1

        # Compute rates
        total_valid = B - counts['conversion_failed']
        if total_valid == 0:
            return {}

        return {
            'fc_rate': counts['fully_constrained'] / total_valid,
            'under_rate': counts['under_constrained'] / total_valid,
            'over_rate': counts['over_constrained'] / total_valid,
            'redundant_rate': counts['redundant'] / total_valid,
            'malformed_rate': counts['malformed'] / total_valid,
            'conversion_fail_rate': counts['conversion_failed'] / B,
        }
