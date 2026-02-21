from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np
import sketchgraphs.data as datalib
from .base import PrimitiveHandler
from src.sketch_handlers.constants import PrimitiveToken, PrimitiveAttrToken
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.registry import PRIMITIVE_HANDLERS

if TYPE_CHECKING:
    FreeCAD, Part, _ = ensure_freecad_imported()


@PRIMITIVE_HANDLERS.register_module()
class PointHandler(PrimitiveHandler):
    @property
    def token_id(self) -> int:
        return PrimitiveToken.Point

    @property
    def name(self) -> str:
        return "Point"

    @property
    def layout(self) -> tuple[int, ...]:
        return (PrimitiveAttrToken.StartX, PrimitiveAttrToken.StartY)

    def sketch_type_check(self, entity: datalib.Entity):
        if not isinstance(entity, datalib.Point):
            raise TypeError(f"{entity=} must be of type datalib.Point")
        return entity

    def freecad_type_check(self, fc_geo: Part.Entity) -> Part.Point:
        # Lazy import
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        if not isinstance(fc_geo, Part.Point):
            raise TypeError(f"{fc_geo=} must be of type Part.Point")
        return fc_geo

    def encode(self, entity: datalib.Entity) -> tuple[np.ndarray, tuple[int, ...]]:
        entity = self.sketch_type_check(entity)
        values = np.array([entity.x, entity.y])
        return values, self.layout

    def decode(self, params: np.ndarray, entity_id: str) -> datalib.Point:
        return datalib.Point(entity_id, x=params[0], y=params[1])

    def to_freecad(self, entity: datalib.Entity) -> tuple[Part.Point, bool]:
        # Lazy import
        FreeCAD, Part, _ = ensure_freecad_imported()

        entity = self.sketch_type_check(entity)

        geo = Part.Point(FreeCAD.Vector(entity.x, entity.y, 0))
        return geo, entity.isConstruction

    def from_freecad(self, fc_geo: Any, output_id: str) -> datalib.Point:
        # Expecting Part.Point
        return datalib.Point(
            entityId=output_id,
            x=fc_geo.X,
            y=fc_geo.Y,
            isConstruction=False,  # NOTE: Construction flag is set by SketchFreeCADConverter
        )

    def get_bbox(self, entity: datalib.Entity) -> np.ndarray:
        entity = self.sketch_type_check(entity)
        return np.array([[entity.x, entity.y], [entity.x, entity.y]])
