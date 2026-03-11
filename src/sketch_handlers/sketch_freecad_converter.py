from typing import Any, TYPE_CHECKING

import sketchgraphs.data as datalib
from sketchgraphs.data._constraint import DirectionValue
from .context import SketchContext
from .freecad_utils import ensure_freecad_imported

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


# Subnode mapping: entity type name -> list of (suffix, pointID) pairs
_SUBNODE_MAP = {
    'Point': [],  # Point entity itself maps to (gid, 1)
    'Line': [('.start', 1), ('.end', 2)],
    'Circle': [('.center', 3)],
    'Arc': [('.start', 1), ('.end', 2), ('.center', 3)],
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
    def _get_handler_name(self, constr: datalib.Constraint) -> str:
        """
        Helper method to correctly identify the required handler.
        Certain constraint types (like Distance) use parameters to differentiate
        between variants (DistanceX vs DistanceY).
        """
        c_name = constr.constraint_type.name
        if c_name == 'Distance':
            for p in constr.parameters:
                if getattr(p, 'parameterId', None) == 'direction':
                    val = getattr(p, 'value', None)
                    if val == DirectionValue.HORIZONTAL or val == 'HORIZONTAL':
                        return 'DistanceX'
                    elif val == DirectionValue.VERTICAL or val == 'VERTICAL':
                        return 'DistanceY'
        return c_name

    def __init__(self, context: SketchContext):
        # Lazy import
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        self.context = context
        self.freecad_to_sketch_type_mapping = {
            Part.LineSegment: 'Line',
            Part.Circle: 'Circle',
            Part.ArcOfCircle: 'Arc',
            Part.Point: 'Point'
        }

    def to_freecad_primitives(self, sketch: datalib.Sketch, doc_name: str = "Sketch") -> tuple['FreeCAD.Document', 'Sketcher.SketchObject', dict[str, tuple[int, int]]]:
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

        doc = FreeCAD.newDocument(doc_name)
        sketch_obj = doc.addObject("Sketcher::SketchObject", "Sketch")

        primitive_freecad_mapping: dict[str, tuple[int, int]] = {}

        for entity in sketch.entities.values():
            entity_name = entity.__class__.__name__
            handler = self.context.get_primitive_handler(entity_name)

            if handler is None:
                print(f"Warning: No handler for entity {entity}")
                continue

            # Register to FreeCAD
            geo, is_construction = handler.to_freecad(entity)
            gid = sketch_obj.addGeometry(geo, is_construction)

            eid = entity.entityId

            # Entity-level reference
            default_pid = _ENTITY_POINT_ID.get(entity_name, 0)
            primitive_freecad_mapping[eid] = (gid, default_pid)

            # Subnode references
            subnodes = _SUBNODE_MAP.get(entity_name, [])
            for suffix, pid in subnodes:
                primitive_freecad_mapping[f"{eid}{suffix}"] = (gid, pid)

        # FreeCAD internal axes: GeoID -1 = X-axis (HAxis), -2 = Y-axis (VAxis)
        # These are used by constraints that reference the Origin or axes
        primitive_freecad_mapping['xaxis'] = (-1, 0)       # X-axis edge
        primitive_freecad_mapping['xaxis.start'] = (-1, 1) # Origin (start of X-axis)
        primitive_freecad_mapping['xaxis.end'] = (-1, 2)   # end of X-axis
        primitive_freecad_mapping['yaxis'] = (-2, 0)       # Y-axis edge
        primitive_freecad_mapping['yaxis.start'] = (-2, 1) # Origin (start of Y-axis)
        primitive_freecad_mapping['yaxis.end'] = (-2, 2)   # end of Y-axis
        primitive_freecad_mapping['origin'] = (-1, 1)      # Origin point (same as xaxis.start)

        doc.recompute()
        return doc, sketch_obj, primitive_freecad_mapping

    def to_freecad_constraints(self, sketch: datalib.Sketch, sketch_obj: 'Sketcher.SketchObject', primitive_freecad_mapping: dict[str, tuple[int, int]]) -> dict[int, str]:
        """
        Adds constraints to the FreeCAD SketchObject.
        Returns constraint_freecad_mapping.

        constraint_freecad_mapping maps FreeCAD's internal 0-based constraint indices to original reference strings:
            0 -> "c_0"
            1 -> "c_1"
        """
        FreeCAD, Part, Sketcher = ensure_freecad_imported()
        constraint_freecad_mapping: dict[int, str] = {}

        for c_id, constr in sketch.constraints.items():
            try:
                handler_name = self._get_handler_name(constr)
                handler = self.context.get_constraint_handler(handler_name)
                if not handler:
                    print(f"Warning: No handler for constraint {handler_name}")
                    continue

                # Handler returns list of (type_name, args_list, datum_or_none)
                # NOTE: Dimension constraints define values in datum, while
                #       Geometric constraints can define them in args_list (Tangent, Perpendicular)
                fc_constraints = handler.to_freecad(constr, primitive_freecad_mapping)
                for fc_type, fc_args, fc_datum in fc_constraints:
                    try:
                        idx = sketch_obj.addConstraint(Sketcher.Constraint(fc_type, *fc_args))
                        constraint_freecad_mapping[idx] = c_id
                        if fc_datum is not None:
                            sketch_obj.setDatum(idx, fc_datum)
                    except Exception as e:
                        print(f"Error adding FreeCAD constraint {fc_type}({fc_args}): {e}")

            except Exception as e:
                print(f"Error creating FreeCAD constraint {c_id}: {e}")

        return constraint_freecad_mapping

    def to_freecad(self, sketch: datalib.Sketch, doc_name: str = "Sketch") -> tuple['FreeCAD.Document', 'Sketcher.SketchObject', dict[str, tuple[int, int]], dict[int, str]]:
        """
        Full conversion: Primitives + Constraints.
        """
        doc, sketch_obj, primitive_freecad_mapping = self.to_freecad_primitives(sketch, doc_name)
        constraint_freecad_mapping = self.to_freecad_constraints(sketch, sketch_obj, primitive_freecad_mapping)
        doc.recompute()
        return doc, sketch_obj, primitive_freecad_mapping, constraint_freecad_mapping

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
                print(f"Warning: No handler for geometry type {freecad_geo_type}")
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
                        subnodes = _SUBNODE_MAP.get(p_name, [])
                        for suffix, pid in subnodes:
                            geoID_to_primitive[(i, pid)] = f"{eid}{suffix}"

                except Exception as e:
                    print(f"Error converting geometry {i} ({freecad_geo_type}): {e}")

        # FreeCAD internal axes: GeoID -1 = X-axis (HAxis), -2 = Y-axis (VAxis)
        geoID_to_primitive[(-1, 0)] = 'xaxis'        # X-axis edge
        geoID_to_primitive[(-1, 1)] = 'xaxis.start'  # Origin
        geoID_to_primitive[(-1, 2)] = 'xaxis.end'
        geoID_to_primitive[(-2, 0)] = 'yaxis'        # Y-axis edge
        geoID_to_primitive[(-2, 1)] = 'yaxis.start'   # Origin
        geoID_to_primitive[(-2, 2)] = 'yaxis.end'

        return entities, geoID_to_primitive

    def from_freecad_constraints(self, sketch_obj: 'Sketcher.SketchObject', geoID_to_primitive: dict[tuple[int, int], str]) -> dict[str, datalib.Constraint]:
        """
        Extracts constraints from FreeCAD SketchObject.

        Uses multi-handler dispatch for ambiguous FC types:
        - 'Coincident' -> try Concentric (center-center), then Coincident
        - 'Symmetric'  -> try Midpoint (same-line start/end), then Mirror
        - 'PointOnObject' -> Coincident handler
        """
        # Build handler candidate lists for ambiguous FC types
        _HANDLER_ORDER: dict[str, list[str]] = {
            'Coincident': ['Concentric', 'Coincident'],
            'Symmetric': ['Midpoint', 'Mirror'],
            'PointOnObject': ['Coincident'],
        }

        constraints = {}
        for i, fc_constr in enumerate(sketch_obj.Constraints):
            c_type = fc_constr.Type
            sg_constr = None

            # Determine handler candidates
            if c_type in _HANDLER_ORDER:
                candidates = _HANDLER_ORDER[c_type]
            else:
                candidates = [c_type]

            # Try each candidate handler in order
            for handler_name in candidates:
                handler = self.context.get_constraint_handler(handler_name)
                if handler:
                    try:
                        sg_constr = handler.from_freecad(fc_constr, geoID_to_primitive)
                        if sg_constr:
                            break
                    except Exception as e:
                        print(f"Error in {handler_name}.from_freecad for constraint {i} ({c_type}): {e}")

            # Fallback: type name mapping
            if not sg_constr:
                fallback_name = None
                if c_type in ['DistanceX', 'DistanceY']:
                    fallback_name = c_type
                elif c_type == 'Radius':
                    fallback_name = 'Radius'
                elif c_type == 'Diameter':
                    fallback_name = 'Diameter'

                if fallback_name:
                    handler = self.context.get_constraint_handler(fallback_name)
                    if handler:
                        try:
                            sg_constr = handler.from_freecad(fc_constr, geoID_to_primitive)
                        except Exception as e:
                            print(f"Error converting constraint {i} ({c_type}): {e}")

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