"""Small synthetic sketches shared by the rebuttal tests. No FreeCAD required."""
from __future__ import annotations

from collections import OrderedDict

import sketchgraphs.data as datalib

from src.metrics.histcad_2d import _set_quantity
from src.sketch_handlers.context import SketchContext


def line(eid: str, x0: float, y0: float, x1: float, y1: float) -> datalib.Line:
    return datalib.Line.from_info(dict(id=eid, startPoint=(x0, y0), endPoint=(x1, y1)))


def circle(eid: str, cx: float, cy: float, r: float) -> datalib.Circle:
    return datalib.Circle(
        entityId=eid, isConstruction=False, xCenter=cx, yCenter=cy, radius=r, clockwise=False
    )


def arc(eid: str, cx: float, cy: float, r: float) -> datalib.Arc:
    return datalib.Arc(
        entityId=eid, isConstruction=False, xCenter=cx, yCenter=cy,
        xDir=1.0, yDir=0.0, radius=r, clockwise=False, startParam=0.0, endParam=1.2,
    )


def point(eid: str, x: float, y: float) -> datalib.Point:
    return datalib.Point(entityId=eid, isConstruction=False, x=x, y=y)


def constraint(context: SketchContext, name: str, refs: tuple[str, ...], value=None, **kwargs):
    """Build a constraint whose value carries the watermark handlers require to read it."""
    handler = context.get_constraint_handler(name)
    assert handler is not None, f"no handler for {name}"
    built = handler.create_constraint(name, refs, value=value, **kwargs)
    return built if value is None else _set_quantity(built, value)


def mixed_primitive_sketch() -> datalib.Sketch:
    """One of each supported primitive, in a deliberately non-alphabetical entity order."""
    entities = OrderedDict([
        ("e_line", line("e_line", 0.0, 0.0, 2.0, 0.0)),
        ("e_circle", circle("e_circle", 4.0, 1.0, 0.75)),
        ("e_arc", arc("e_arc", -2.0, 3.0, 1.5)),
        ("e_point", point("e_point", 1.0, -2.0)),
    ])
    return datalib.Sketch(entities=entities, constraints=OrderedDict())


def reordered(sketch: datalib.Sketch, order: tuple[str, ...]) -> datalib.Sketch:
    entities = OrderedDict((eid, sketch.entities[eid]) for eid in order)
    return datalib.Sketch(entities=entities, constraints=sketch.constraints)
