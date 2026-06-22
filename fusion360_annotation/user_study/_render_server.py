"""Persistent FreeCAD-GUI render/edit server for ``compare.py``.

Boots the real FreeCAD GUI once (headless, under ``xvfb-run``) and then services
requests over a localhost TCP socket.  Unlike a plain screenshot server it keeps
each sketch **loaded as a live session**, so the client can drag primitive
vertices (``movePoint`` -> the solver re-solves -> the under-constrained sketch
deforms) and drag dimension labels, re-rendering after every edit.

Each render enters the Sketcher *edit* mode so the native constraint
glyphs/dimensions are drawn, frames the geometry, and captures only the GL
viewport (no toolbar / task-panel chrome).

Run with the FreeCAD-build Python and a display::

    xvfb-run -a <freecad-python> _render_server.py --port 8765

Protocol: newline-delimited JSON.  Request ``{"op": ...}``:

* ``view``  {src, kind, w, h, [cx, cy, height], [refit]}  -> ensure loaded; render
* ``move``  {src, kind, geo, pos, x, y, w, h, cx, cy, height}  -> drag a vertex
* ``move_label`` {src, kind, constr, x, y, ...frame}  -> drag a dimension label
* ``reset`` {src, kind, w, h}  -> reload from disk (discard edits), refit

Response::

    {"ok": true, "png": "<base64>", "handles": {...}, "frame": {...}, "dof": N}
    {"ok": false, "error": "<msg>"}

``frame`` = {cx, cy, height, w, h} is the orthographic camera framing the client
needs to map world<->pixel.  ``handles`` = draggable points in *world* coords::

    {"v": [[geo, pos, x, y, construction], ...],   # geometry vertices
     "l": [[constr, x, y, type], ...]}             # dimension labels
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import socketserver
import sys
import tempfile
import traceback
from collections import OrderedDict

# FreeCAD's GUI hijacks sys.stdout/stderr once it starts; keep the real fd to log.
_REAL_OUT = os.dup(1)


def log(msg: str) -> None:
    try:
        os.write(_REAL_OUT, (msg + "\n").encode())
    except Exception:
        pass


# Fallback for a source-built FreeCAD (the author's headless dev box). Only used
# when FreeCAD is NOT already loaded -- i.e. the xvfb-subprocess path. When this
# module is imported in-process from inside a running FreeCAD (the annotator's
# official app / AppImage), FreeCAD is already in sys.modules, so we must NOT add
# this path: doing so could pull mismatched submodules (Part/Sketcher) from a
# DIFFERENT FreeCAD build into the running one and hard-crash it (ABI mismatch).
_FC_LIB = os.path.expanduser("~/github/FreeCAD/build/release/lib")
if "FreeCAD" not in sys.modules and os.path.isdir(_FC_LIB):
    sys.path.insert(0, _FC_LIB)

import FreeCAD  # noqa: E402
import FreeCADGui as Gui  # noqa: E402
import Part  # noqa: E402
import Sketcher  # noqa: E402

_DIM_TYPES = {"Distance", "DistanceX", "DistanceY", "Radius", "Diameter", "Angle"}
# "Remove Dimension" also drops Block, which fully pins a primitive's DoF.
_STRIP_TYPES = _DIM_TYPES | {"Block"}
MARGIN = 2.5           # camera-height / geometry-extent (room to deform + zoom-out)
LABEL_FRAC = 0.07      # datum-label leader length as a fraction of extent
MAX_SESSIONS = 8
TMP_PNG = os.path.join(tempfile.gettempdir(), f"_compare_render_{os.getpid()}.png")
_BG_COLOR = "#1a4060"    # flat bg for offscreen render (midpoint of the gradient)

# SoDatumLabel::Type enum order (FreeCAD src/Gui/SoDatumLabel.h), used to read
# the exact, FreeCAD-computed label position out of the live edit scene graph.
(_DL_ANGLE, _DL_DISTANCE, _DL_DISTANCEX, _DL_DISTANCEY,
 _DL_RADIUS, _DL_DIAMETER, _DL_SYMMETRIC, _DL_ARCLENGTH) = range(8)
_FLT_EPS = 1.1920929e-7


# ===========================================================================
# Session = one loaded, editable sketch document
# ===========================================================================
class Session:
    def __init__(self, doc, sketch, new_docs):
        self.doc = doc
        self.sketch = sketch
        self.new_docs = new_docs        # doc names to close on eviction
        self.frame = None               # (cx, cy, height) base framing
        self.gdoc = None                # Gui document
        self.view = None                # 3D inventor view
        self.dims_stripped = False      # dimensional constraints removed?


_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
_CURRENT_EDIT: "Session | None" = None  # the session currently in Sketcher edit


def _find_sketch(doc):
    for o in doc.Objects:
        if getattr(o, "TypeId", "") == "Sketcher::SketchObject":
            return o
    raise ValueError("no Sketcher::SketchObject in document")


def _build_from_py(src: str):
    before = set(FreeCAD.listDocuments())
    g = {"FreeCAD": FreeCAD, "App": FreeCAD, "Part": Part,
         "Sketcher": Sketcher, "Vector": FreeCAD.Vector}
    with open(src) as f:
        exec(compile(f.read(), src, "exec"), g)
    # Only trust documents THIS script actually created.  Invalid predictions
    # (e.g. the SolverError ones whose body is fully commented out) create no
    # document, so we must NOT fall back to FreeCAD.ActiveDocument: that is
    # whatever was rendered last, and returning it silently shows a foreign
    # sketch (a previous one, the other panel's, or a GT) in this panel.
    new_docs = [n for n in FreeCAD.listDocuments() if n not in before]
    if not new_docs:
        raise ValueError("invalid prediction: no document was created")
    active = FreeCAD.ActiveDocument
    name = active.Name if (active is not None and active.Name in new_docs) else new_docs[-1]
    doc = FreeCAD.getDocument(name)
    return doc, _find_sketch(doc), new_docs


def _open_fcstd(src: str):
    before = set(FreeCAD.listDocuments())
    doc = FreeCAD.openDocument(src)
    new_docs = [n for n in FreeCAD.listDocuments() if n not in before]
    return doc, _find_sketch(doc), new_docs


def _close_session(sess: Session) -> None:
    global _CURRENT_EDIT
    try:
        if sess.gdoc is not None:
            sess.gdoc.resetEdit()
    except Exception:
        pass
    if _CURRENT_EDIT is sess:
        _CURRENT_EDIT = None
    for n in sess.new_docs:
        try:
            FreeCAD.closeDocument(n)
        except Exception:
            pass


def _setup_session(sess: Session) -> None:
    """One-time per session: recompute, hide clutter, pull in datum labels, and
    prime the orthographic top view.  Keeps the per-render path cheap."""
    sketch, doc = sess.sketch, sess.doc
    sketch.recompute()
    try:
        doc.recompute()
    except Exception:
        pass
    # Pull absurdly-far default datum labels in close to the geometry, once.
    bb = sketch.Shape.BoundBox
    extent = max(bb.XLength, bb.YLength, 1e-6) if bb.isValid() else 1.0
    for i, c in enumerate(sketch.Constraints):
        if c.Type in _DIM_TYPES:
            try:
                if abs(sketch.getLabelDistance(i)) >= extent:
                    sketch.setLabelDistance(i, extent * LABEL_FRAC)
            except Exception:
                pass
    try:
        doc.recompute()
    except Exception:
        pass
    sess.gdoc = Gui.getDocument(doc.Name)
    for o in doc.Objects:
        if o.Name != sketch.Name:
            try:
                sess.gdoc.getObject(o.Name).Visibility = False
            except Exception:
                pass
    sess.view = sess.gdoc.activeView()
    sess.view.setCameraType("Orthographic")
    sess.view.viewTop()


def _session_alive(sess: "Session") -> bool:
    """True if the session's document still exists. In the live GUI the user can
    close or delete a document; touching a stale (deleted) session crashes
    FreeCAD, so callers must drop it instead of using it."""
    try:
        return sess.doc is not None and sess.doc.Name in FreeCAD.listDocuments()
    except Exception:
        return False


def get_session(src: str, kind: str, reload: bool = False) -> Session:
    global _CURRENT_EDIT
    if reload and src in _SESSIONS:
        _close_session(_SESSIONS.pop(src))
    if src in _SESSIONS:
        sess = _SESSIONS[src]
        if _session_alive(sess):
            _SESSIONS.move_to_end(src)
            return sess
        # Document was closed/deleted in the GUI -> drop the stale session.
        # Do NOT _close_session it: that would call into the dangling GUI doc.
        if _CURRENT_EDIT is sess:
            _CURRENT_EDIT = None
        _SESSIONS.pop(src, None)
    if kind == "fcstd" or src.lower().endswith(".fcstd"):
        doc, sketch, new_docs = _open_fcstd(src)
    else:
        doc, sketch, new_docs = _build_from_py(src)
    sess = Session(doc, sketch, new_docs)
    _setup_session(sess)
    _SESSIONS[src] = sess
    while len(_SESSIONS) > MAX_SESSIONS:
        _, old = _SESSIONS.popitem(last=False)
        _close_session(old)
    return sess


# ===========================================================================
# Geometry helpers
# ===========================================================================
def _vertices(sketch):
    """Draggable vertices as [geo, pos, x, y, construction] in world coords."""
    P = sketch.Placement
    out = []
    for i, geo in enumerate(sketch.Geometry):
        try:
            cons = bool(sketch.getConstruction(i))
        except Exception:
            cons = False
        pts = []
        if isinstance(geo, Part.LineSegment):
            pts = [(1, geo.StartPoint), (2, geo.EndPoint)]
        elif isinstance(geo, Part.Circle):
            pts = [(3, geo.Center)]
        elif isinstance(geo, Part.ArcOfCircle):
            pts = [(1, geo.StartPoint), (2, geo.EndPoint), (3, geo.Center)]
        elif isinstance(geo, Part.Point):
            pts = [(1, FreeCAD.Vector(geo.X, geo.Y, 0))]
        for pos, v in pts:
            g = P.multVec(v)
            out.append([i, pos, round(g.x, 6), round(g.y, 6), cons])
    return out


def _datum_text_center(lab):
    """Port of ``SoDatumLabel::getLabelTextCenter()`` -> (x, y) in sketch-local
    coords (or None).  Reads the live node fields the Sketcher set when it drew
    the label, so this is *exactly* where FreeCAD renders the dimension text –
    no heuristic guessing.  (The C++ method itself isn't exposed to pivy.)"""
    if lab.pnts.getNum() < 2:
        return None
    pts = lab.pnts.getValues()
    p1, p2 = pts[0].getValue(), pts[1].getValue()
    dt = lab.datumtype.getValue()
    if dt in (_DL_DISTANCE, _DL_DISTANCEX, _DL_DISTANCEY):
        length, length2 = lab.param1.getValue(), lab.param2.getValue()
        if dt == _DL_DISTANCE:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        elif dt == _DL_DISTANCEX:
            dx, dy = (1.0 if p2[0] - p1[0] >= _FLT_EPS else -1.0), 0.0
        else:
            dx, dy = 0.0, (1.0 if p2[1] - p1[1] >= _FLT_EPS else -1.0)
        L = math.hypot(dx, dy) or 1.0
        dx, dy = dx / L, dy / L
        nx, ny = -dy, dx                                   # label normal
        normproj = (p2[0] - p1[0]) * nx + (p2[1] - p1[1]) * ny
        midx = ((p1[0] + normproj * nx) + p2[0]) / 2.0
        midy = ((p1[1] + normproj * ny) + p2[1]) / 2.0
        return (midx + nx * length + dx * length2, midy + ny * length + dy * length2)
    if dt in (_DL_RADIUS, _DL_DIAMETER):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1.0
        length = lab.param1.getValue()
        return (p2[0] + length * dx / L, p2[1] + length * dy / L)
    if dt == _DL_ANGLE:
        # Angle needs param3 (the swept range), which pivy does not expose on the
        # node -> fall back to the heuristic anchor for angle labels.
        try:
            rng = lab.param3.getValue()
        except Exception:
            return None
        ang = lab.param2.getValue() + rng / 2.0
        length = lab.param1.getValue()
        return (p1[0] + math.cos(ang) * 2.0 * length, p1[1] + math.sin(ang) * 2.0 * length)
    return (p1[0], p1[1])


def _find_datum_label(node):
    """First SoDatumLabel at or under ``node`` (depth-first), else None."""
    if node.getTypeId().getName() == "SoDatumLabel":
        return node
    if hasattr(node, "getNumChildren"):
        for k in range(node.getNumChildren()):
            r = _find_datum_label(node.getChild(k))
            if r is not None:
                return r
    return None


def _find_constr_group(sess: Session):
    """The Sketcher edit 'constraint group' node, whose i-th child is the coin
    separator for constraint i.  Identified as the node with exactly one child
    per constraint whose first dimensional child carries a SoDatumLabel.  None if
    not found (callers then fall back to the heuristic)."""
    sketch = sess.sketch
    ncon = len(sketch.Constraints)
    dim0 = next((i for i, c in enumerate(sketch.Constraints) if c.Type in _DIM_TYPES), None)
    if dim0 is None or sess.view is None:
        return None
    stack = [sess.view.getSceneGraph()]
    while stack:
        n = stack.pop()
        if not hasattr(n, "getNumChildren"):
            continue
        nc = n.getNumChildren()
        if nc == ncon and _find_datum_label(n.getChild(dim0)) is not None:
            return n
        for k in range(nc):
            stack.append(n.getChild(k))
    return None


def _datum_label_node(sess: Session, ci: int):
    """The live SoDatumLabel node for constraint ``ci`` (or None)."""
    grp = _find_constr_group(sess)
    if grp is None or not (0 <= ci < grp.getNumChildren()):
        return None
    return _find_datum_label(grp.getChild(ci))


def _scene_label_centers(sess: Session) -> dict:
    """{constraint index -> (local_x, local_y)} read from the live Sketcher edit
    scene graph.  Returns {} if the constraint group can't be located, so callers
    fall back to the heuristic anchor."""
    grp = _find_constr_group(sess)
    if grp is None:
        return {}
    out = {}
    for i, c in enumerate(sess.sketch.Constraints):
        if c.Type not in _DIM_TYPES:
            continue
        # Per-label guard: one awkward label (e.g. an Angle) must never discard
        # the exact positions of all the others.
        try:
            lab = _find_datum_label(grp.getChild(i))
            ctr = _datum_text_center(lab) if lab is not None else None
        except Exception:
            ctr = None
        if ctr is not None:
            out[i] = ctr
    return out


def _labels(sketch, centers=None):
    """Draggable dimension-label anchors as [constr, x, y, type] (world coords).
    Uses the exact label centre from the edit scene graph (``centers``) when
    available, falling back to the heuristic anchor only when it isn't."""
    P = sketch.Placement
    out = []
    geom = sketch.Geometry
    bb = sketch.Shape.BoundBox
    extent = max(bb.XLength, bb.YLength, 1e-6) if bb.isValid() else 1.0
    centers = centers or {}
    for i, c in enumerate(sketch.Constraints):
        if c.Type not in _DIM_TYPES:
            continue
        anchor = centers.get(i)
        if anchor is None:
            try:
                anchor = _label_anchor(sketch, i, c, geom, extent)
            except Exception:
                anchor = None
        if anchor is not None:
            g = P.multVec(FreeCAD.Vector(anchor[0], anchor[1], 0))
            out.append([i, round(g.x, 6), round(g.y, 6), c.Type])
    return out


def _point_at(g, pos):
    if pos == 1 and hasattr(g, "StartPoint"):
        return g.StartPoint
    if pos == 2 and hasattr(g, "EndPoint"):
        return g.EndPoint
    if pos == 3 and hasattr(g, "Center"):
        return g.Center
    if isinstance(g, Part.Point):
        return FreeCAD.Vector(g.X, g.Y, 0)
    if hasattr(g, "StartPoint"):
        return g.StartPoint
    return None


def _ref_points(c, geom):
    """Reference point(s) + measured direction for a dimensional constraint.
    Handles point refs (PosId 1/2/3) AND edge refs (PosId 0: a line -> its
    midpoint+direction, a circle -> its centre).  Returns (points, direction)."""
    pts, direction = [], None
    for gid, pos in ((c.First, c.FirstPos), (c.Second, c.SecondPos)):
        if gid is None or gid < 0 or gid >= len(geom):
            continue
        g = geom[gid]
        if pos == 0:  # whole-edge reference (e.g. a line-length Distance)
            if isinstance(g, Part.LineSegment):
                a, b = g.StartPoint, g.EndPoint
                pts.append(FreeCAD.Vector((a.x + b.x) / 2, (a.y + b.y) / 2, 0))
                direction = (b.x - a.x, b.y - a.y)
            elif hasattr(g, "Center"):
                pts.append(g.Center)
        else:
            p = _point_at(g, pos)
            if p is not None:
                pts.append(p)
    if direction is None and len(pts) == 2:
        direction = (pts[1].x - pts[0].x, pts[1].y - pts[0].y)
    return pts, direction


def _label_anchor(sketch, ci, c, geom, extent=1.0):
    """Best-effort world anchor of a dimension label (see _render notes).

    *extent* is the max(XLength, YLength) of the sketch bounding box and is
    used for scale-proportional fallback positioning.  The final anchor is
    clamped so it never drifts more than ``MARGIN * extent`` from the
    reference-geometry midpoint – this prevents handles from landing outside
    the rendered camera frustum.
    """
    try:
        d = sketch.getLabelDistance(ci)
    except Exception:
        d = 0.0
    # Scale-proportional fallback when FreeCAD returns 0 (label not yet placed).
    default_off = extent * LABEL_FRAC
    # Maximum distance the label anchor may be from the ref-point midpoint.
    max_off = extent * (MARGIN - 0.1)

    if c.Type in ("Radius", "Diameter"):
        g = geom[c.First]
        if hasattr(g, "Center") and hasattr(g, "Radius"):
            ang = sketch.getLabelPosition(ci)
            # LabelPosition is the leader angle (rad); the default sentinel (10.0,
            # i.e. outside [-2pi, 2pi]) renders along +X, so clamp it to 0.
            if abs(ang) > math.tau:
                ang = 0.0
            r = g.Radius + (d if d > 0 else g.Radius * 0.3)
            r = min(r, max_off)  # clamp so the handle stays in view
            return (g.Center.x + r * math.cos(ang), g.Center.y + r * math.sin(ang))
        return None
    # Distance family / Angle.
    pts, direction = _ref_points(c, geom)
    if not pts:
        return None
    mx = sum(p.x for p in pts) / len(pts)
    my = sum(p.y for p in pts) / len(pts)
    off = d if d != 0 else default_off
    # Clamp to prevent the label handle from flying off-canvas.
    off = max(-max_off, min(max_off, off))
    # DistanceX label sits above the ref points (+y); DistanceY to the left (-x).
    if c.Type == "DistanceX":
        return (mx, my + off)
    if c.Type == "DistanceY":
        return (mx - off, my)
    # Aligned/edge Distance (and Angle): offset perpendicular to the segment.
    if direction is not None:
        dx, dy = direction
        L = math.hypot(dx, dy)
        if L > 1e-9:
            return (mx - off * dy / L, my + off * dx / L)
        # Degenerate direction (e.g. concentric circles): fall back to
        # radial placement similar to Radius/Diameter labels.
        try:
            ang = sketch.getLabelPosition(ci)
            if abs(ang) > math.tau:
                ang = 0.0
        except Exception:
            ang = 0.0
        r = abs(off)
        for gid in (c.First, c.Second):
            if 0 <= gid < len(geom) and hasattr(geom[gid], 'Radius'):
                r = max(r, geom[gid].Radius)
        r = min(r, max_off)
        return (mx + r * math.cos(ang), my + r * math.sin(ang))
    return (mx, my + off)


# ===========================================================================
# Rendering
# ===========================================================================
def _frame_session(sess: Session):
    bb = sess.sketch.Shape.BoundBox
    if bb.isValid() and max(bb.XLength, bb.YLength) > 1e-9:
        extent = max(bb.XLength, bb.YLength)
        return ((bb.XMin + bb.XMax) / 2.0, (bb.YMin + bb.YMax) / 2.0, extent * MARGIN)
    return (0.0, 0.0, 10.0)


def _geom_extent(sess: Session) -> float:
    """Raw geometry extent (without MARGIN), for the client to fit to geometry."""
    bb = sess.sketch.Shape.BoundBox
    if bb.isValid() and max(bb.XLength, bb.YLength) > 1e-9:
        return max(bb.XLength, bb.YLength)
    return 1.0


def _ensure_edit(sess: Session) -> None:
    """Enter Sketcher edit mode for this session (so glyphs/dimensions draw),
    switching away from any other session.  Persistent: a no-op when already
    editing this one, which is what makes repeated drag renders fast."""
    global _CURRENT_EDIT
    if _CURRENT_EDIT is sess:
        return False
    # Only reset edit on the previous session if its document is still alive;
    # calling into a closed/deleted GUI doc can hard-crash FreeCAD.
    if (_CURRENT_EDIT is not None and _CURRENT_EDIT.gdoc is not None
            and _session_alive(_CURRENT_EDIT)):
        try:
            _CURRENT_EDIT.gdoc.resetEdit()
        except Exception:
            pass
    sess.gdoc.setEdit(sess.sketch, 0)
    _CURRENT_EDIT = sess
    for _ in range(3):
        Gui.updateGui()
    return True  # the edited sketch was switched (caller may need an extra render)


def _render(sess: Session, w: int, h: int, frame, settle: int = 2) -> str:
    switched = _ensure_edit(sess)
    view = sess.view
    cam = view.getCameraNode()
    cx, cy, height = frame
    cam.position.setValue(cx, cy, 10.0)
    cam.height.setValue(height)
    try:
        cam.aspectRatio.setValue(1.0)
    except Exception:
        pass
    for _ in range(settle):
        Gui.updateGui()
    # Use an explicit background colour (not "Current") to force FreeCAD to
    # do a fresh offscreen render from the scene graph rather than grabbing
    # the viewport framebuffer, which can be stale after a session switch.
    view.saveImage(TMP_PNG, int(w), int(h), _BG_COLOR)
    if switched:
        # Right after switching the edited sketch, that first grab can still
        # capture a stale/empty framebuffer -- the symptom is that the handles
        # (blue/orange markers) appear but the image is blank until you drag
        # (which forces a re-render). Settle a bit more and grab again.
        for _ in range(2):
            Gui.updateGui()
        view.saveImage(TMP_PNG, int(w), int(h), _BG_COLOR)
    with open(TMP_PNG, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _response(sess: Session, w: int, h: int, frame) -> dict:
    png = _render(sess, w, h, frame)
    try:
        sess.sketch.solve()
        dof = int(sess.sketch.DoF)
    except Exception:
        dof = -1
    try:
        centers = _scene_label_centers(sess)
    except Exception:
        centers = {}
    return {
        "ok": True, "png": png,
        "handles": {"v": _vertices(sess.sketch), "l": _labels(sess.sketch, centers)},
        "frame": {"cx": frame[0], "cy": frame[1], "height": frame[2], "w": w, "h": h,
                  "geom_height": _geom_extent(sess)},
        "dof": dof,
    }


def _strip_dimensions(sess: Session) -> None:
    """Remove every dimensional (and Block) constraint from the session's sketch
    and recompute.  Geometry stays put -- the sketch just gains DoF and loses its
    dimension labels.  Marks the session so we don't strip twice."""
    sketch = sess.sketch
    dim_ids = [i for i, c in enumerate(sketch.Constraints) if c.Type in _STRIP_TYPES]
    for i in reversed(dim_ids):          # delete high->low so indices stay valid
        try:
            sketch.delConstraint(i)
        except Exception:
            pass
    sketch.recompute()
    sess.dims_stripped = True


def _apply_strip_mode(src: str, kind: str, sess: Session, want_strip: bool) -> Session:
    """Make the session match the 'remove dimensions' toggle.  Stripping is
    destructive, so restoring the dimensions means reloading from disk (which
    also discards edits); returns the (possibly reloaded) session."""
    if want_strip and not sess.dims_stripped:
        _strip_dimensions(sess)
    elif not want_strip and sess.dims_stripped:
        sess = get_session(src, kind, reload=True)
    return sess


# ===========================================================================
# Ops
# ===========================================================================
def handle_request(req: dict) -> dict:
    op = req.get("op", "view")
    src = req["src"]
    kind = req.get("kind") or ("fcstd" if src.lower().endswith(".fcstd") else "py")
    w, h = int(req.get("w", 1600)), int(req.get("h", 1600))
    want_strip = bool(req.get("strip_dims"))

    try:
        if op == "reset":
            sess = get_session(src, kind, reload=True)
            if want_strip:
                _strip_dimensions(sess)
            sess.frame = _frame_session(sess)
            return _response(sess, w, h, sess.frame)

        sess = get_session(src, kind)
        sess = _apply_strip_mode(src, kind, sess, want_strip)
        if sess.frame is None or req.get("refit"):
            sess.frame = _frame_session(sess)
        # Client may override framing (pan/zoom that wants a fresh crisp render).
        if req.get("cx") is not None and not req.get("refit"):
            frame = (float(req["cx"]), float(req["cy"]), float(req["height"]))
        else:
            frame = sess.frame

        if op == "move":
            P = sess.sketch.Placement
            local = P.inverse().multVec(FreeCAD.Vector(float(req["x"]), float(req["y"]), 0))
            # FreeCAD 1.0 exposes ``movePoint``; 1.1 renamed it to ``moveGeometry``
            # (identical signature). Support both so dragging works on either.
            mover = (getattr(sess.sketch, "movePoint", None)
                     or getattr(sess.sketch, "moveGeometry", None))
            if mover is None:
                raise AttributeError("SketchObject has no movePoint/moveGeometry")
            mover(int(req["geo"]), int(req["pos"]), local, 0)
            sess.sketch.recompute()
        elif op == "move_label":
            _apply_label_move(sess, int(req["constr"]),
                              float(req["x"]), float(req["y"]))
        return _response(sess, w, h, frame)
    except Exception as e:
        log(f"[{op} error] {src}\n{traceback.format_exc()}")
        return {"ok": False, "error": str(e) or type(e).__name__}


def _apply_label_move(sess: Session, ci: int, wx: float, wy: float) -> None:
    """Place dimension *ci*'s label so its text centre follows the cursor.

    For the Distance family this inverts ``SoDatumLabel::getLabelTextCenter`` on
    the live node: the label has two DoF -- ``LabelDistance`` (param1, normal to
    the measured line) and ``LabelPosition`` (param2, along it) -- so we set both
    from the cursor instead of only the perpendicular one.  That makes the handle
    track the cursor exactly, with no sign flips or off-axis drift."""
    sketch = sess.sketch
    P = sketch.Placement
    loc = P.inverse().multVec(FreeCAD.Vector(wx, wy, 0))
    c = sketch.Constraints[ci]
    geom = sketch.Geometry
    lab = _datum_label_node(sess, ci)

    if c.Type in ("Radius", "Diameter"):
        g = geom[c.First]
        if hasattr(g, "Center"):
            dx, dy = loc.x - g.Center.x, loc.y - g.Center.y
            sketch.setLabelPosition(ci, math.atan2(dy, dx))
            sketch.setLabelDistance(ci, max(math.hypot(dx, dy) - getattr(g, "Radius", 0.0), 0.01))
    elif c.Type in ("Distance", "DistanceX", "DistanceY") and lab is not None and lab.pnts.getNum() >= 2:
        pts = lab.pnts.getValues()
        p1, p2 = pts[0].getValue(), pts[1].getValue()
        if c.Type == "Distance":
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        elif c.Type == "DistanceX":
            dx, dy = (1.0 if p2[0] - p1[0] >= _FLT_EPS else -1.0), 0.0
        else:
            dx, dy = 0.0, (1.0 if p2[1] - p1[1] >= _FLT_EPS else -1.0)
        L = math.hypot(dx, dy) or 1.0
        dx, dy = dx / L, dy / L
        nx, ny = -dy, dx
        normproj = (p2[0] - p1[0]) * nx + (p2[1] - p1[1]) * ny
        midx = ((p1[0] + normproj * nx) + p2[0]) / 2.0
        midy = ((p1[1] + normproj * ny) + p2[1]) / 2.0
        tx, ty = loc.x - midx, loc.y - midy
        sketch.setLabelDistance(ci, tx * nx + ty * ny)   # param1: perpendicular
        sketch.setLabelPosition(ci, tx * dx + ty * dy)   # param2: along the line
    else:
        # Fallback (node not found, or Angle): heuristic perpendicular offset.
        pts, direction = _ref_points(c, geom)
        if pts:
            mx = sum(p.x for p in pts) / len(pts)
            my = sum(p.y for p in pts) / len(pts)
            if c.Type == "DistanceX":
                sketch.setLabelDistance(ci, loc.y - my)
            elif c.Type == "DistanceY":
                sketch.setLabelDistance(ci, mx - loc.x)
            elif direction is not None:
                dx, dy = direction
                L = math.hypot(dx, dy)
                if L > 1e-9:
                    sketch.setLabelDistance(ci, -(loc.x - mx) * dy / L + (loc.y - my) * dx / L)
                else:
                    sketch.setLabelPosition(ci, math.atan2(loc.y - my, loc.x - mx))
                    sketch.setLabelDistance(ci, max(math.hypot(loc.x - mx, loc.y - my), 0.01))
            else:
                sketch.setLabelDistance(ci, loc.y - my)

    # Presentation changes only show up in the datum-label scene after a recompute
    # (Gui.updateGui alone is not enough); the vertex-move path does the same.
    sketch.recompute()


# ===========================================================================
# Socket server
# ===========================================================================
class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                resp = handle_request(json.loads(line))
            except Exception as e:
                resp = {"ok": False, "error": f"bad request: {e}"}
            self.wfile.write((json.dumps(resp) + "\n").encode())
            self.wfile.flush()


def _configure_background() -> None:
    pv = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/View")
    pv.SetBool("Gradient", True)
    pv.SetBool("UseBackgroundColorMid", False)
    pv.SetUnsigned("BackgroundColor", 0x336699FF)
    pv.SetUnsigned("BackgroundColor2", 0x002040FF)
    # Construction geometry can otherwise be invisible: a centreline on the sketch
    # Y-axis hides under the (brighter) axis line, and thin construction merges
    # with nearby edges.  A distinct cyan + a wider line keeps it readable.
    pv.SetUnsigned("ConstructionColor", 0x00FFFFFF)  # cyan, 0xRRGGBBAA
    FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Sketcher/View").SetInt(
        "ConstructionWidth", 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    _configure_background()
    Gui.showMainWindow()
    log("FreeCAD " + ".".join(FreeCAD.Version()[0:3]))

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), Handler) as srv:
        log(f"RENDER_SERVER_READY {args.host}:{args.port}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
