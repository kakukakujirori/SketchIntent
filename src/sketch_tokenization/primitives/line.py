from typing import Any
import numpy as np
import math
import sketchgraphs.data as datalib
from .base import PrimitiveHandler
from ..constants import PrimitiveToken, PrimitiveAttrToken
from ..freecad_utils import ensure_freecad_imported
from ..registry import PRIMITIVE_HANDLERS

@PRIMITIVE_HANDLERS.register_module()
class LineHandler(PrimitiveHandler):
    @property
    def token_id(self) -> int:
        return PrimitiveToken.Line

    @property
    def name(self) -> str:
        return "Line"

    @property
    def layout(self) -> tuple[int, ...]:
        return (
            PrimitiveAttrToken.StartX, PrimitiveAttrToken.StartY,
            PrimitiveAttrToken.EndX, PrimitiveAttrToken.EndY,
        )

    def encode(self, entity: datalib.Line) -> tuple[np.ndarray, tuple[int, ...]]:
        # Line: StartX, StartY, EndX, EndY
        values = np.concatenate([entity.start_point, entity.end_point])
        return values, self.layout

    def decode(self, params: np.ndarray, entity_id: str) -> datalib.Line:
        # params: [sx, sy, ex, ey]
        info = {
            'id': entity_id,
            'startPoint': params[:2],
            'endPoint': params[2:]
        }
        return datalib.Line.from_info(info)

    def to_freecad(self, entity: datalib.Line, sketch: Any, **kwargs) -> Any:
        FreeCAD, Part, _ = ensure_freecad_imported()

        start = FreeCAD.Vector(entity.start_point[0], entity.start_point[1], 0)
        end = FreeCAD.Vector(entity.end_point[0], entity.end_point[1], 0)

        # Validation for min length
        if (start - end).Length < 1e-7:
            end = start + FreeCAD.Vector(1e-3, 0, 0)

        geo = Part.LineSegment(start, end)
        idx = sketch.addGeometry(geo, entity.isConstruction)
        return idx

    def from_freecad(self, fc_geo: Any, **kwargs) -> datalib.Line:
        # Expecting Part.LineSegment
        start = fc_geo.StartPoint
        end = fc_geo.EndPoint

        return datalib.Line.from_info({
            'id': kwargs.get('id', '0'),
            'startPoint': np.array([start.x, start.y]),
            'endPoint': np.array([end.x, end.y]),
            'isConstruction': kwargs.get('isConstruction', False)
        })

    def get_bbox(self, entity: datalib.Line) -> np.ndarray:
        start_x, start_y = entity.start_point
        end_x, end_y = entity.end_point
        return np.array([[min(start_x, end_x), min(start_y, end_y)],
                         [max(start_x, end_x), max(start_y, end_y)]])
