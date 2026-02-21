from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np
import math
import sketchgraphs.data as datalib
from .base import PrimitiveHandler
from src.sketch_handlers.constants import PrimitiveToken, PrimitiveAttrToken
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.registry import PRIMITIVE_HANDLERS

if TYPE_CHECKING:
    FreeCAD, Part, _ = ensure_freecad_imported()


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

    def sketch_type_check(self, entity: datalib.Entity) -> datalib.Line:
        if not isinstance(entity, datalib.Line):
            raise TypeError(f"{entity=} must be of type datalib.Line")
        return entity

    def freecad_type_check(self, fc_geo: Part.Geometry) -> Part.LineSegment:
        # Lazy import
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        if not isinstance(fc_geo, Part.LineSegment):
            raise TypeError(f"{fc_geo=} must be of type Part.LineSegment")
        return fc_geo

    def encode(self, entity: datalib.Entity) -> tuple[np.ndarray, tuple[int, ...]]:
        entity = self.sketch_type_check(entity)
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

    def to_freecad(self, entity: datalib.Entity) -> tuple[Part.LineSegment, bool]:
        # Lazy import
        FreeCAD, Part, _ = ensure_freecad_imported()

        entity = self.sketch_type_check(entity)

        start = FreeCAD.Vector(entity.start_point[0], entity.start_point[1], 0)
        end = FreeCAD.Vector(entity.end_point[0], entity.end_point[1], 0)

        # Validation for min length
        if (start - end).Length < 1e-7:
            end = start + FreeCAD.Vector(1e-3, 0, 0)

        geo = Part.LineSegment(start, end)
        return geo, entity.isConstruction

    def from_freecad(self, fc_geo: Any, output_id: str) -> datalib.Line:
        fc_geo = self.freecad_type_check(fc_geo)
        start = fc_geo.StartPoint
        end = fc_geo.EndPoint

        return datalib.Line.from_info({
            'id': output_id,
            'startPoint': np.array([start.x, start.y]),
            'endPoint': np.array([end.x, end.y]),
            'isConstruction': False,  # NOTE: Construction flag is set by SketchFreeCADConverter
        })

    def get_bbox(self, entity: datalib.Entity) -> np.ndarray:
        entity = self.sketch_type_check(entity)
        start_x, start_y = entity.start_point
        end_x, end_y = entity.end_point
        return np.array([[min(start_x, end_x), min(start_y, end_y)],
                         [max(start_x, end_x), max(start_y, end_y)]])
