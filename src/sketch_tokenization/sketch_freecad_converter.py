from typing import Any, TYPE_CHECKING

import sketchgraphs.data as datalib
from .context import SketchContext
from .freecad_utils import ensure_freecad_imported

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


class SketchFreecadConverter:
    """
    Tokenizer/Detokenizer for SketchGraphs data.
    """
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

    def to_freecad_primitives(self, sketch: datalib.Sketch, doc_name: str = "Sketch") -> tuple['FreeCAD.Document', 'Sketcher.SketchObject', dict[str, int]]:
        """
        Converts primitives in datalib.Sketch to a FreeCAD SketchObject.
        Returns (doc, sketch_obj, entity_id_to_geo_id_map).
        """
        # Lazy import
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        doc = FreeCAD.newDocument(doc_name)
        sketch_obj = doc.addObject("Sketcher::SketchObject", "Sketch")

        # Map EntityID -> GeoID (index in FreeCAD)
        primitive_geoID: dict[str, int] = {}

        for entity in sketch.entities.values():
            entity_name = entity.__class__.__name__
            handler = self.context.get_primitive_handler(entity_name)

            if handler is None:
                print(f"Warning: No handler for entity {entity}")
                continue

            # Register to FreeCAD
            geo, is_construction = handler.to_freecad(entity)
            idx = sketch_obj.addGeometry(geo, is_construction)
            primitive_geoID[entity.entityId] = idx

        doc.recompute()
        return doc, sketch_obj, primitive_geoID

    def to_freecad_constraints(self, sketch: datalib.Sketch, sketch_obj: 'Sketcher.SketchObject', entity_map: dict[str, int]) -> None:
        """
        Adds constraints to the FreeCAD SketchObject.
        """
        sorted_constraints = sorted(sketch.constraints.items(), key=lambda x: x[0])

        for c_id, constr in sorted_constraints:
            try:
                handler = self.context.get_constraint_handler(constr.constraint_type)
                if not handler:
                    # Fallback for integer type or name mismatch
                    if isinstance(constr.constraint_type, int):
                        try:
                            c_name = datalib.ConstraintType(constr.constraint_type).name
                            handler = self.context.get_constraint_handler(c_name)
                        except (ValueError, AttributeError):
                            pass

                if not handler:
                    print(f"Warning: No handler for constraint {constr.constraint_type}")
                    continue

                handler.to_freecad(constr, sketch_obj, entity_map)

            except Exception as e:
                print(f"Error creating FreeCAD constraint {c_id}: {e}")

    def to_freecad(self, sketch: datalib.Sketch, doc_name: str = "Sketch") -> tuple['FreeCAD.Document', 'Sketcher.SketchObject']:
        """
        Full conversion: Primitives + Constraints.
        """
        doc, sketch_obj, entity_map = self.to_freecad_primitives(sketch, doc_name)
        self.to_freecad_constraints(sketch, sketch_obj, entity_map)
        doc.recompute()
        return doc, sketch_obj

    def from_freecad_primitives(self, sketch_obj: 'Sketcher.SketchObject') -> tuple[dict[str, datalib.Entity], dict[int, str]]:
        """
        Converts FreeCAD SketchObject geometry to datalib Entities.
        Returns (entities_dict, geoID_to_primitive).
        """
        entities = {}
        geoID_to_primitive = {}

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
                        geoID_to_primitive[i] = entity.entityId
                except Exception as e:
                    print(f"Error converting geometry {i} ({freecad_geo_type}): {e}")

        return entities, geoID_to_primitive

    def from_freecad_constraints(self, sketch_obj: 'Sketcher.SketchObject', geoID_to_primitive: dict[int, str]) -> dict[str, datalib.Constraint]:
        """
        Extracts constraints from FreeCAD SketchObject.
        """
        constraints = {}
        for i, fc_constr in enumerate(sketch_obj.Constraints):
            c_type = fc_constr.Type
            handler = self.context.get_constraint_handler(c_type)

            if not handler:
                # Type name mapping fallbacks
                if c_type in ['DistanceX', 'DistanceY']:
                    handler = self.context.get_constraint_handler('Distance')
                elif c_type == 'Radius':
                    handler = self.context.get_constraint_handler('Radius')
                elif c_type == 'Diameter':
                    handler = self.context.get_constraint_handler('Diameter')

            if handler:
                try:
                    # handler.from_freecad expects entity_map (GeoID -> EntityID) but actually uses it
                    # to map GeoID refs to EntityID refs.
                    # Wait, our verify script passed `entity_map` where keys are GeoIDs (ints).
                    # `from_freecad` usually takes (fc_constraint, entity_map).
                    sg_constr = handler.from_freecad(fc_constr, geoID_to_primitive)
                    if sg_constr:
                        # Assign unique ID
                        c_id = f"c_{i}"
                        constraints[c_id] = sg_constr
                except Exception as e:
                    print(f"Error converting constraint {i} ({c_type}): {e}")
            else:
                # Warning: No handler
                pass

        return constraints

    def from_freecad(self, sketch_obj: 'Sketcher.SketchObject') -> datalib.Sketch:
        """
        Extracts full datalib.Sketch from FreeCAD SketchObject.
        """
        entities, geoID_to_primitive = self.from_freecad_primitives(sketch_obj)
        constraints = self.from_freecad_constraints(sketch_obj, geoID_to_primitive)
        return datalib.Sketch(entities=entities, constraints=constraints)