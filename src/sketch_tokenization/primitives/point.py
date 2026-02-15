from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import PrimitiveHandler
from ..constants import PrimitiveToken, PrimitiveAttrToken
from ..freecad_utils import ensure_freecad_imported
from ..registry import PRIMITIVE_HANDLERS

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

    def encode(self, entity: datalib.Point) -> tuple[np.ndarray, tuple[int, ...]]:
        values = np.array([entity.x, entity.y])
        return values, self.layout

    def decode(self, params: np.ndarray, entity_id: str) -> datalib.Point:
        return datalib.Point(entity_id, x=params[0], y=params[1])

    def to_freecad(self, entity: datalib.Point, sketch: Any, **kwargs) -> Any:
        FreeCAD, Part, _ = ensure_freecad_imported()

        geo = Part.Point(FreeCAD.Vector(entity.x, entity.y, 0))
        idx = sketch.addGeometry(geo, entity.isConstruction)
        return idx

    def from_freecad(self, fc_geo: Any, **kwargs) -> datalib.Point:
        # Expecting Part.Point
        return datalib.Point(
            entityId=kwargs.get('id', '0'),
            x=fc_geo.X,
            y=fc_geo.Y,
            isConstruction=kwargs.get('isConstruction', False)
        )

    def get_bbox(self, entity: datalib.Point) -> np.ndarray:
        return np.array([[entity.x, entity.y], [entity.x, entity.y]])
