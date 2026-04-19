import traceback
from typing import TYPE_CHECKING

import sketchgraphs.data as datalib
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import dump_freecad_commands, ensure_freecad_imported, load_freecad_commands
from src.sketch_handlers.sketch_utils import complete_constraint_values

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


# Subnode mapping: entity type name -> list of (suffix, pointID) pairs
_SUBNODE_MAP = {
    'Point': [],  # Point entity itself maps to (gid, 1)
    'Line': [('.start', 1), ('.end', 2)],
    'Circle': [('.center', 3)],
    'Arc_clockwise': [('.start', 2), ('.end', 1), ('.center', 3)],
    'Arc_counter_clockwise': [('.start', 1), ('.end', 2), ('.center', 3)],
}

# Default pointID for entity-level (edge) reference
_ENTITY_POINT_ID = {
    'Point': 1,  # Point entity itself is vertex 1
    'Line': 0,
    'Circle': 0,
    'Arc': 0,
}


class SketchFreecadConverter:
    """
    Data converter between FreeCAD and SketchGraphs.
    """
    def __init__(self, context: SketchContext, verbose: bool = True, raise_on_error: bool = True):
        # Lazy import
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        self.context = context
        self.verbose = verbose
        self.raise_on_error = raise_on_error
        self.dprint = lambda msg: print(msg) if self.verbose else None
        self.freecad_to_sketch_type_mapping = {
            Part.LineSegment: 'Line',
            Part.Circle: 'Circle',
            Part.ArcOfCircle: 'Arc',
            Part.Point: 'Point'
        }

    def to_freecad_primitives(
        self,
        sketch: datalib.Sketch,
        doc_name: str = "Sketch",
        skip_invalid_primitives: bool = False,
    ) -> tuple['FreeCAD.Document', 'Sketcher.SketchObject', dict[str, tuple[int, int]]]:
        """
        Converts primitives in datalib.Sketch to a FreeCAD SketchObject.
        Returns (doc, sketch_obj, primitive_freecad_mapping).

        primitive_freecad_mapping maps reference strings to (GeoID, PointID):
            "entityId" -> (geoID, default_pointID)
            "entityId.start" -> (geoID, 1)
            "entityId.end" -> (geoID, 2)
            "entityId.center" -> (geoID, 3)
        """
        # Lazy import
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        doc = None
        primitive_freecad_mapping: dict[str, tuple[int, int]] = {}

        try:
            doc = FreeCAD.newDocument(doc_name)
            fc_sketch = doc.addObject("Sketcher::SketchObject", "Sketch")

            for entity in sketch.entities.values():
                entity_name = entity.__class__.__name__
                handler = self.context.get_primitive_handler(entity_name)

                if handler is None:
                    self.dprint(f"Warning: No handler for entity {entity}")
                    continue

                # Register to FreeCAD
                try:
                    geo, is_construction = handler.to_freecad(entity)
                    gid = fc_sketch.addGeometry(geo, is_construction)
                except Exception as e:
                    if skip_invalid_primitives:
                        self.dprint(f"Warning: Failed to add primitive {entity}: {traceback.format_exc()}")
                        continue
                    else:
                        raise e

                eid = entity.entityId

                # Entity-level reference
                default_pid = _ENTITY_POINT_ID.get(entity_name, 0)
                primitive_freecad_mapping[eid] = (gid, default_pid)

                # Subnode references (NOTE: For simplicity, configure StartPoint -> EndPoint counter-clockwise in FreeCAD)
                if entity_name == 'Arc':
                    assert isinstance(entity, datalib.Arc)
                    entity_name = f"Arc_{'clockwise' if entity.clockwise else 'counter_clockwise'}"
                subnodes = _SUBNODE_MAP.get(entity_name, [])
                for suffix, pid in subnodes:
                    primitive_freecad_mapping[f"{eid}{suffix}"] = (gid, pid)

            # FreeCAD internal axes: GeoID -1 = X-axis (HAxis), -2 = Y-axis (VAxis)
            # These are used by constraints that reference the Origin or axes
            primitive_freecad_mapping['-1'] = (-1, 0)       # X-axis edge
            primitive_freecad_mapping['-1.start'] = (-1, 1) # Origin (start of X-axis)
            primitive_freecad_mapping['-1.end'] = (-1, 2)   # end of X-axis
            primitive_freecad_mapping['-2'] = (-2, 0)       # Y-axis edge
            primitive_freecad_mapping['-2.start'] = (-2, 1) # Origin (start of Y-axis)
            primitive_freecad_mapping['-2.end'] = (-2, 2)   # end of Y-axis

            doc.recompute()

        except Exception as e:
            self.dprint(f"Error adding FreeCAD primitive {entity}: {e}")
            FreeCAD.closeDocument(doc.Name) if doc else None
            raise e

        return doc, fc_sketch, primitive_freecad_mapping

    def to_freecad_constraints(
        self,
        sketch: datalib.Sketch,
        fc_sketch: 'Sketcher.SketchObject',
        primitive_freecad_mapping: dict[str, tuple[int, int]],
        skip_invalid_constraints: bool = False,
    ) -> tuple[dict[int, str], list[str]]:
        """
        Adds constraints to the FreeCAD SketchObject.
        Returns constraint_freecad_mapping.

        constraint_freecad_mapping maps FreeCAD's internal 0-based constraint indices to original reference strings:
            0 -> "c_0"
            1 -> "c_1"
        """
        FreeCAD, Part, Sketcher = ensure_freecad_imported()
        constraint_freecad_mapping: dict[int, str] = {}
        invalid_constraints = []

        for c_id, constr in sketch.constraints.items():
            handler_name = constr.constraint_type.name
            handler = self.context.get_constraint_handler(handler_name)
            if not handler:
                self.dprint(f"Warning: No handler for constraint {handler_name}")
                continue

            # Handler returns list of (type_name, args_list, datum_or_none)
            # NOTE: Dimension constraints define values in datum, while
            #       Geometric constraints can define them in args_list (Tangent, Perpendicular)
            try:
                fc_constraints = handler.to_freecad(constr, sketch.entities, primitive_freecad_mapping)
                for fc_type, fc_args, fc_datum in fc_constraints:
                    idx = fc_sketch.addConstraint(Sketcher.Constraint(fc_type, *fc_args))
                    constraint_freecad_mapping[idx] = c_id
                    if fc_datum is not None:
                        fc_sketch.setDatum(idx, fc_datum)
            except Exception as e:
                if skip_invalid_constraints:
                    self.dprint(f"Warning: Failed to add constraint {handler_name}: {traceback.format_exc()}")
                    invalid_constraints.append(c_id)
                    continue
                else:
                    raise e

        return constraint_freecad_mapping, invalid_constraints

    def to_freecad(
        self,
        sketch: datalib.Sketch,
        doc_name: str = "Sketch",
        skip_invalid: bool = False,
    ) -> tuple['FreeCAD.Document', 'Sketcher.SketchObject', dict[str, tuple[int, int]], dict[int, str], list[str]]:
        """
        Full conversion: Primitives + Constraints.
        """
        sketch, _ = complete_constraint_values(sketch)
        doc, fc_sketch, primitive_freecad_mapping = self.to_freecad_primitives(sketch, doc_name, skip_invalid)
        constraint_freecad_mapping, invalid_constraints = self.to_freecad_constraints(sketch, fc_sketch, primitive_freecad_mapping, skip_invalid)
        doc.recompute()
        return doc, fc_sketch, primitive_freecad_mapping, constraint_freecad_mapping, invalid_constraints

    def to_freecad_commands(self, sketch: datalib.Sketch, skip_invalid: bool = False) -> list[str]:
        """
        Wrapper of to_freecad to output FreeCAD commands.
        """
        doc = None
        try:
            doc, fc_sketch, _, _, _ = self.to_freecad(sketch, skip_invalid=skip_invalid)
            commands = dump_freecad_commands(fc_sketch)
            return commands
        finally:
            FreeCAD, _, _ = ensure_freecad_imported()
            FreeCAD.closeDocument(doc.Name) if doc else None

    def from_freecad_primitives(self, sketch_obj: 'Sketcher.SketchObject') -> tuple[dict[str, datalib.Entity], dict[tuple[int, int], str]]:
        """
        Converts FreeCAD SketchObject geometry to datalib Entities.
        Returns (entities_dict, geoID_to_primitive).

        geoID_to_primitive maps (GeoID, PointID) to reference strings:
            (geoID, 0) -> "entityId"       (edge reference)
            (geoID, 1) -> "entityId.start"  or "entityId" for Point
            (geoID, 2) -> "entityId.end"
            (geoID, 3) -> "entityId.center"
        """
        entities = {}
        geoID_to_primitive: dict[tuple[int, int], str] = {}

        for i, geo in enumerate(sketch_obj.Geometry):
            freecad_geo_type = type(geo)
            p_name = self.freecad_to_sketch_type_mapping.get(freecad_geo_type)
            if not p_name:
                self.dprint(f"Warning: No handler for geometry type {freecad_geo_type}")
                continue

            handler = self.context.get_primitive_handler(p_name)
            if handler:
                try:
                    entity = handler.from_freecad(output_id=str(i), fc_geo=geo)
                    if entity:
                        entity.isConstruction = sketch_obj.getConstruction(i)
                        entities[entity.entityId] = entity
                        eid = entity.entityId

                        # Entity-level reference
                        default_pid = _ENTITY_POINT_ID.get(p_name, 0)
                        geoID_to_primitive[(i, default_pid)] = eid

                        # Subnode references
                        if p_name == 'Arc':
                            mapped_name = f"Arc_{'clockwise' if getattr(entity, 'clockwise', False) else 'counter_clockwise'}"
                        else:
                            mapped_name = p_name
                        subnodes = _SUBNODE_MAP.get(mapped_name, [])
                        for suffix, pid in subnodes:
                            geoID_to_primitive[(i, pid)] = f"{eid}{suffix}"

                except Exception as e:
                    self.dprint(f"Error converting geometry {i} ({freecad_geo_type}): {e}")
                    if self.raise_on_error:
                        raise e

        # FreeCAD internal axes: GeoID -1 = X-axis (HAxis), -2 = Y-axis (VAxis)
        geoID_to_primitive[(-1, 0)] = '-1'        # X-axis edge
        geoID_to_primitive[(-1, 1)] = '-1.start'  # Origin
        geoID_to_primitive[(-1, 2)] = '-1.end'
        geoID_to_primitive[(-2, 0)] = '-2'        # Y-axis edge
        geoID_to_primitive[(-2, 1)] = '-2.start'   # Origin
        geoID_to_primitive[(-2, 2)] = '-2.end'

        return entities, geoID_to_primitive

    def from_freecad_constraints(self, fc_sketch: 'Sketcher.SketchObject', geoID_to_primitive: dict[tuple[int, int], str]) -> dict[str, datalib.Constraint]:
        """
        Extracts constraints from FreeCAD SketchObject.

        Uses multi-handler dispatch for ambiguous FC types:
        - 'Symmetric'  -> try Midpoint (same-line start/end), then Mirror
        - 'PointOnObject' -> Coincident handler
        """
        # Build handler candidate lists for FC types unexisting in SketchGraphs
        _HANDLER_ORDER: dict[str, list[str]] = {
            'Symmetric': ['Midpoint', 'Mirror'],
            'PointOnObject': ['Coincident'],
        }

        constraints = {}
        for i, fc_constr in enumerate(fc_sketch.Constraints):
            c_type = fc_constr.Type
            sg_constr = None

            # Determine handler candidates
            if c_type in _HANDLER_ORDER:
                candidates = _HANDLER_ORDER[c_type]
            else:
                candidates = [c_type]

            # Try each candidate handler in order
            for handler_idx, handler_name in enumerate(candidates):
                handler = self.context.get_constraint_handler(handler_name)
                if handler:
                    try:
                        sg_constr = handler.from_freecad(fc_constr, fc_sketch.Geometry, geoID_to_primitive)
                        if sg_constr:
                            break
                    except Exception as e:
                        # If there are multiple candidates, intermediate failures are false positives
                        if handler_idx == len(candidates) - 1:
                            self.dprint(f"Error in {handler_name}.from_freecad for constraint {i} ({c_type}): {traceback.format_exc()}")
                            if self.raise_on_error:
                                raise e

            if sg_constr:
                c_id = f"c_{i}"
                # Make identifier a real UID
                sg_constr = sg_constr._replace(identifier=c_id)
                constraints[c_id] = sg_constr

        return constraints

    def from_freecad(self, sketch_obj: 'Sketcher.SketchObject') -> datalib.Sketch:
        """
        Extracts full datalib.Sketch from FreeCAD SketchObject.
        """
        entities, geoID_to_primitive = self.from_freecad_primitives(sketch_obj)
        constraints = self.from_freecad_constraints(sketch_obj, geoID_to_primitive)
        return datalib.Sketch(entities=entities, constraints=constraints)

    def from_freecad_commands(self, commands_or_file: list[str] | str) -> datalib.Sketch:
        """
        Wrapper of from_freecad to create a FreeCAD document from python commands and extract the sketch.
        """
        doc = None
        try:
            doc, fc_sketch = load_freecad_commands(commands_or_file)
            sg_sketch = self.from_freecad(fc_sketch)
            return sg_sketch
        finally:
            FreeCAD, _, _ = ensure_freecad_imported()
            FreeCAD.closeDocument(doc.Name) if doc else None