from typing import Any
import numpy as np
import math
import sketchgraphs.data as datalib
from .base import PrimitiveHandler
from ..constants import PrimitiveToken, PrimitiveAttrToken
from ..freecad_utils import ensure_freecad_imported
from ..registry import PRIMITIVE_HANDLERS


def arc_from_three_points(entity_id, p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if abs(D) < 1e-9:
            # Fallback (Line-like or flat)
        chord_dx = x3 - x1
        chord_dy = y3 - y1
        chord_len_sq = chord_dx**2 + chord_dy**2

        if chord_len_sq < 1e-9:
            cx, cy = x1, y1
            radius = 1e-6
            start_angle = 0.0
            end_angle = 0.0
            is_clockwise = False
        else:
            mx, my = (x1 + x3) / 2, (y1 + y3) / 2
            chord_len = np.sqrt(chord_len_sq)
            nx, ny = -chord_dy / chord_len, chord_dx / chord_len
            radius = 10000.0
            cx = mx + nx * radius
            cy = my + ny * radius
            start_angle = math.atan2(y1 - cy, x1 - cx)
            end_angle = math.atan2(y3 - cy, x3 - cx)
            is_clockwise = False
    else:
        cx = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
        cy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
        radius = np.sqrt((cx - x1)**2 + (cy - y1)**2)

        start_angle = math.atan2(y1 - cy, x1 - cx)
        mid_angle = math.atan2(y2 - cy, x2 - cx)
        end_angle = math.atan2(y3 - cy, x3 - cx)

        def norm_angle(a, ref):
            diff = a - ref
            while diff < 0: diff += 2*math.pi
            while diff >= 2*math.pi: diff -= 2*math.pi
            return diff

        mid_rel_ccw = norm_angle(mid_angle, start_angle)
        end_rel_ccw = norm_angle(end_angle, start_angle)
        is_clockwise = not (mid_rel_ccw < end_rel_ccw)

    if is_clockwise:
        start_param = -start_angle
        end_param = -end_angle
        if start_param > end_param:
            start_param -= 2 * math.pi
    else:
        start_param = start_angle
        end_param = end_angle
        if end_param <= start_param:
            end_param += 2 * math.pi

    return datalib.Arc(
        entityId=entity_id,
        xCenter=cx,
        yCenter=cy,
        radius=radius,
        clockwise=is_clockwise,
        xDir=1.0,
        yDir=0.0,
        startParam=start_param,
        endParam=end_param
    )


@PRIMITIVE_HANDLERS.register_module()
class ArcHandler(PrimitiveHandler):
    @property
    def token_id(self) -> int:
        return PrimitiveToken.Arc

    @property
    def name(self) -> str:
        return "Arc"

    @property
    def layout(self) -> tuple[int, ...]:
        return (
            PrimitiveAttrToken.StartX, PrimitiveAttrToken.StartY,
            PrimitiveAttrToken.MidX, PrimitiveAttrToken.MidY,
            PrimitiveAttrToken.EndX, PrimitiveAttrToken.EndY,
            PrimitiveAttrToken.CenterX, PrimitiveAttrToken.CenterY,
        )

    def encode(self, entity: datalib.Arc) -> tuple[np.ndarray, tuple[int, ...]]:
        values = np.concatenate([entity.start_point, entity.mid_point, entity.end_point, entity.center_point])
        return values, self.layout

    def decode(self, params: np.ndarray, entity_id: str) -> datalib.Arc:
        p1 = params[0:2] # Start
        p2 = params[2:4] # Mid
        p3 = params[4:6] # End
        return arc_from_three_points(entity_id, p1, p2, p3)

    def to_freecad(self, entity: datalib.Arc, sketch: Any, **kwargs) -> Any:
        FreeCAD, Part, _ = ensure_freecad_imported()

        p1 = FreeCAD.Vector(entity.start_point[0], entity.start_point[1], 0)
        p2 = FreeCAD.Vector(entity.mid_point[0], entity.mid_point[1], 0)
        p3 = FreeCAD.Vector(entity.end_point[0], entity.end_point[1], 0)

        try:
            geo = Part.ArcOfCircle(p1, p2, p3)
            idx = sketch.addGeometry(geo, entity.isConstruction)
            return idx
        except Exception:
            return -1

    def from_freecad(self, fc_geo: Any, **kwargs) -> datalib.Arc:
        # Expecting Part.ArcOfCircle
        p_start = np.array([fc_geo.StartPoint.x, fc_geo.StartPoint.y])
        p_end = np.array([fc_geo.EndPoint.x, fc_geo.EndPoint.y])

        # Midpoint estimation
        mid_param = (fc_geo.FirstParameter + fc_geo.LastParameter) / 2.0
        val_mid = fc_geo.value(mid_param)
        p_mid = np.array([val_mid.x, val_mid.y])

        return arc_from_three_points(
            kwargs.get('id', '0'),
            p_start, p_mid, p_end
        )

    def get_bbox(self, entity: datalib.Arc) -> np.ndarray:
        pts = np.vstack([entity.start_point, entity.mid_point, entity.end_point])
        return np.array([pts.min(axis=0), pts.max(axis=0)])
