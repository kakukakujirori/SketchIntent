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
class CircleHandler(PrimitiveHandler):
    @property
    def token_id(self) -> int:
        return PrimitiveToken.Circle

    @property
    def name(self) -> str:
        return "Circle"

    @property
    def layout(self) -> tuple[int, ...]:
        return (
            PrimitiveAttrToken.CenterX, PrimitiveAttrToken.CenterY,
            PrimitiveAttrToken.Radius,
        )

    def sketch_type_check(self, entity: datalib.Entity) -> datalib.Circle:
        if not isinstance(entity, datalib.Circle):
            raise TypeError(f"{entity=} must be of type datalib.Circle")
        return entity

    def freecad_type_check(self, fc_geo: Part.Geometry) -> Part.Circle:
        # Lazy import
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        if not isinstance(fc_geo, Part.Circle):
            raise TypeError(f"{fc_geo=} must be of type Part.Circle")
        return fc_geo

    def encode(self, entity: datalib.Entity) -> tuple[np.ndarray, tuple[int, ...]]:
        entity = self.sketch_type_check(entity)
        # Circle: CenterX, CenterY, Radius
        values = np.append(entity.center_point, entity.radius)
        return values, self.layout

    def decode(self, params: np.ndarray, entity_id: str) -> datalib.Circle:
        # params: [cx, cy, r]
        radius = abs(params[2])
        if radius < 1e-9: radius = 1e-9
        return datalib.Circle(entity_id, xCenter=params[0], yCenter=params[1], radius=radius)

    def to_freecad(self, entity: datalib.Entity) -> tuple[Part.Circle, bool]:
        # Lazy import
        FreeCAD, Part, _ = ensure_freecad_imported()

        entity = self.sketch_type_check(entity)

        center = FreeCAD.Vector(entity.xCenter, entity.yCenter, 0)
        radius = entity.radius if entity.radius > 1e-7 else 1e-3

        geo = Part.Circle(center, FreeCAD.Vector(0,0,1), radius)
        return geo, entity.isConstruction

    def from_freecad(self, fc_geo: Any, output_id: str) -> datalib.Circle:
        fc_geo = self.freecad_type_check(fc_geo)
        center = fc_geo.Center
        return datalib.Circle(
            entityId=output_id,
            isConstruction=False,  # NOTE: Construction flag is set by SketchFreeCADConverter
            xCenter=center.x,
            yCenter=center.y,
            radius=fc_geo.Radius,
            clockwise=False,  # This field is meaningless?
        )

    def get_bbox(self, entity: datalib.Entity) -> np.ndarray:
        entity = self.sketch_type_check(entity)
        return np.array([
            [entity.xCenter-entity.radius, entity.yCenter-entity.radius],
            [entity.xCenter+entity.radius, entity.yCenter+entity.radius]
        ])
