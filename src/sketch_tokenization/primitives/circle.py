from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import PrimitiveHandler
from ..constants import PrimitiveToken, PrimitiveAttrToken
from ..freecad_utils import ensure_freecad_imported
from ..registry import PRIMITIVE_HANDLERS


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

    def encode(self, entity: datalib.Circle) -> tuple[np.ndarray, tuple[int, ...]]:
        # Circle: CenterX, CenterY, Radius
        values = np.append(entity.center_point, entity.radius)
        return values, self.layout

    def decode(self, params: np.ndarray, entity_id: str) -> datalib.Circle:
        # params: [cx, cy, r]
        radius = abs(params[2])
        if radius < 1e-9: radius = 1e-9
        return datalib.Circle(entity_id, xCenter=params[0], yCenter=params[1], radius=radius)

    def to_freecad(self, entity: datalib.Circle, sketch: Any, **kwargs) -> Any:
        FreeCAD, Part, _ = ensure_freecad_imported()
        center = FreeCAD.Vector(entity.xCenter, entity.yCenter, 0)
        radius = entity.radius if entity.radius > 1e-7 else 1e-3

        geo = Part.Circle(center, FreeCAD.Vector(0,0,1), radius)
        idx = sketch.addGeometry(geo, entity.isConstruction)
        return idx

    def from_freecad(self, fc_geo: Any, **kwargs) -> datalib.Circle:
        # Expecting Part.Circle
        center = fc_geo.Center
        return datalib.Circle(
            entityId=kwargs.get('id', '0'),
            xCenter=center.x,
            yCenter=center.y,
            radius=fc_geo.Radius,
            isConstruction=kwargs.get('isConstruction', False)
        )

    def get_bbox(self, entity: datalib.Circle) -> np.ndarray:
        return np.array([[entity.xCenter-entity.radius, entity.yCenter-entity.radius],
                         [entity.xCenter+entity.radius, entity.yCenter+entity.radius]])
