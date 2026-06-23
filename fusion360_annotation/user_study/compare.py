#!/usr/bin/env python
"""User-study viewer: pick the better of **two** CAD sketches (left vs right).

Shows the real FreeCAD Sketcher rendering of two model folders side by side, with
an optional **ground-truth** reference (the 3 human annotations from ``data_eval``,
viewable via the GT toggle). For every sketch the annotator clicks one of three
buttons -- *Prefer left sketch* / *Tie* / *Prefer right sketch* -- and the choice
is appended to a results file. Each panel is interactive (drag a vertex to deform
an under-constrained sketch, drag a dimension label, wheel-zoom, pan).

OS-AGNOSTIC BY DESIGN
---------------------
The only dependencies are FreeCAD/Qt (cross-platform) and the Python standard
library (no ``fastapi``/``uvicorn`` to install). There is **no OS-specific code**:

* **Annotators (any OS: Windows/macOS/Linux)** -- open the official **FreeCAD GUI**
  and, in its *Python console*, run::

      import sys; sys.path.insert(0, r"<path-to>/user_study")
      import compare; compare.main()

  FreeCAD is already running, so rendering happens **in-process** (no ``xvfb``,
  no subprocess). The 3D view is exported to PNG via FreeCAD's own offscreen
  ``View3DInventor.saveImage`` -- not an OS screen-grab -- which works everywhere.
  A browser tab opens with the viewer. Run ``compare.stop()`` to shut down.

* **Dev / headless Linux (the author)** -- run as a plain script::

      python compare.py <folderA> <folderB> --gt-dir <data_eval>

  With no FreeCAD GUI present this falls back to spawning ``_render_server.py``
  under ``xvfb-run`` (the original architecture), so the existing workflow is
  unchanged.

The mode is auto-detected from ``FreeCAD.GuiUp``.

DATA LAYOUT (for the no-argument annotator run)
-----------------------------------------------
The data folders live either directly under ``user_study/`` or under
``user_study/data/`` (both are searched; override with ``data_dir=...``)::

    user_study/            (or user_study/data/)
      <modelA>/<batch>/<stem>.py          # left  (A)
      <modelB>/<batch>/<stem>.py          # right (B)
      data_eval/<batch>/<stem>_<V>.FCStd  # GT reference variants (D/J/R/S ...)

The two model folders are auto-discovered (preferring names starting with
``Vitruvion``); ``data_eval`` is the GT root. Model names are hidden in the UI
(blind A/B) so annotators can't tell which model is which.

RESULTS
-------
Votes are written to ``data/user_study_results.json`` after every click (atomic,
de-duplicated by sketch, resumable across sessions). When all sketches are voted,
a SHA-256 over the (stem-sorted, canonical) votes is appended as ``integrity`` so
the shared file can be checked for accidental corruption.

Options (script mode): ``--gt-dir DIR  --data-dir DIR  --port N  --res N  --no-browser``
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import socket
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Study configuration (tweak here)
# ===========================================================================
BLIND = True                       # hide real model-folder names in the UI (A/B)
SHUFFLE_SIDES = True               # randomize which model is shown left/right
SHUFFLE_SEED = "user-study-v1"     # fixed -> a given sketch always gets the same sides
RESULTS_NAME = "user_study_results.json"
DEFAULT_DATA_DIR = HERE / "data"
GT_DIRNAME = "data_eval"           # ground-truth root under the data dir

# In the in-process (annotator GUI) path, FreeCAD's own window pops to the front
# every time we switch the edited sketch (Sketcher ``setEdit`` activates its MDI
# window), stealing the screen from the browser on a single-monitor setup. Keep
# it out of the way after each render. OS-agnostic (plain Qt show*/lower calls):
#   "minimize" -> iconify it; the browser keeps focus (restored on stop()).
#   "lower"    -> only push it behind the browser (use if minimizing ever blanks
#                 renders on your OS -- the window stays mapped).
#   "off"      -> leave it alone (old behavior).
FREECAD_WINDOW = "minimize"

_RUNNING: dict | None = None       # holds the in-process server/refs (keeps alive)

_LAYOUT_HELP = (
    "Need two model folders under {data_dir} (found: {found}).\n"
    "Expected layout:\n"
    "  {data_dir}/<modelA>/<batch>/*.py\n"
    "  {data_dir}/<modelB>/<batch>/*.py\n"
    "  {data_dir}/" + GT_DIRNAME + "/<batch>/*_<V>.FCStd   (ground-truth reference)\n"
    "Or pass explicit folders: compare.main('<modelA>', '<modelB>', gt_dir='<data_eval>')"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ===========================================================================
# Discovery: shared sketches (batch-aware), captions, ground-truth variants
# ===========================================================================
def index_folder(folder: Path) -> dict[str, tuple[str, Path]]:
    """Map ``<stem>`` -> (batch, predicted .py path) over the 0..N batch subdirs."""
    out: dict[str, tuple[str, Path]] = {}
    for p in sorted(folder.glob("*/*.py")):
        out[p.stem] = (p.parent.name, p)
    return out


def load_captions(folder: Path) -> dict[str, dict]:
    caps: dict[str, dict] = {}
    for rj in sorted(folder.glob("*/results.json")):
        try:
            rows = json.loads(rj.read_text())
        except Exception:
            continue
        for r in rows:
            sf = r.get("source_file") or ""
            stem = Path(sf).stem if sf else r.get("sketch_id", "")
            if stem:
                caps[stem] = {"status": r.get("status"), "dof": r.get("final_dof"),
                              "final": r.get("final_status")}
    return caps


def gt_variants(gt_dir: Path, batch: str, stem: str) -> dict[str, str]:
    """Map every available variant letter -> abs FCStd path (variants vary per
    sketch: D/E/J/R/S/T ...). Look in the sketch's batch, then any batch."""
    found: dict[str, str] = {}
    pats = [gt_dir / batch] if (gt_dir / batch).is_dir() else []
    pats.append(None)  # fallback: search every batch
    for base in pats:
        globber = (base.glob if base else gt_dir.glob)
        pattern = f"{stem}_*.FCStd" if base else f"*/{stem}_*.FCStd"
        for f in globber(pattern):
            v = f.name[len(stem) + 1:-len(".FCStd")]
            if v and v not in found:
                found[v] = str(f.resolve())
        if found:
            break
    return dict(sorted(found.items()))


def discover(data_dir: Path, gt_dir: Path | None) -> tuple[list[Path], Path | None]:
    """Find the two model folders and the GT root under ``data_dir``.

    GT = ``data_eval`` (or the first subdir that holds ``*.FCStd``). Model folders
    = the other subdirs that hold ``*/*.py`` (preferring ``Vitruvion*`` names)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return [], gt_dir
    subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    if gt_dir is None:
        cand = data_dir / GT_DIRNAME
        if cand.is_dir():
            gt_dir = cand
        else:
            for d in subdirs:
                if any(d.glob("*/*.FCStd")) or any(d.glob("*.FCStd")):
                    gt_dir = d
                    break

    gt_res = Path(gt_dir).resolve() if gt_dir else None
    models = []
    for d in subdirs:
        if gt_res and d.resolve() == gt_res:
            continue
        if any(d.glob("*/*.py")):
            models.append(d)
    vitr = [d for d in models if d.name.lower().startswith("vitruvion")]
    if len(vitr) >= 2:
        models = vitr
    return models, gt_dir


# ===========================================================================
# Render backend A: subprocess + xvfb (headless dev fallback)
# ===========================================================================
class RenderClient:
    """One persistent, serialized socket connection to ``_render_server.py``."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._fp = None

    def _ensure(self) -> None:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=180)
            self._fp = self._sock.makefile("rwb")

    def request(self, req: dict) -> dict:
        with self._lock:
            for attempt in (1, 2):
                try:
                    self._ensure()
                    self._fp.write((json.dumps(req) + "\n").encode())
                    self._fp.flush()
                    line = self._fp.readline()
                    if not line:
                        raise ConnectionError("render server closed connection")
                    return json.loads(line)
                except (OSError, ConnectionError) as e:
                    self._close()
                    if attempt == 2:
                        return {"ok": False, "error": f"render-server IO: {e}"}
            return {"ok": False, "error": "unreachable"}

    def _close(self) -> None:
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock, self._fp = None, None


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def start_render_server(port: int):
    import subprocess
    cmd = ["xvfb-run", "-a", sys.executable,
           str(HERE / "_render_server.py"), "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    def pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(f"[render] {line}")
            sys.stdout.flush()

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"render server exited early (code {proc.returncode})")
        if _port_open(port):
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("render server did not become ready within 120s")


class SubprocessRenderer:
    """xvfb subprocess + socket client. Used on headless Linux (no FreeCAD GUI)."""

    def __init__(self, res: int):
        self._res = res
        self._proc = None
        self._client: RenderClient | None = None

    def start(self) -> None:
        port = free_port()
        print(f"Starting FreeCAD render server on port {port} (xvfb) ...")
        self._proc = start_render_server(port)
        self._client = RenderClient("127.0.0.1", port)

    def request(self, req: dict) -> dict:
        return self._client.request(req)  # type: ignore[union-attr]

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass


# ===========================================================================
# Render backend B: in-process (runs inside the official FreeCAD GUI, any OS)
# ===========================================================================
def _import_qtcore():
    """Grab QtCore from the SAME binding FreeCAD's GUI uses, so our QTimer lives in
    the app's event loop. FreeCAD installs a ``PySide`` shim that re-exports the
    active binding, so try that first; fall back to PySide2/6 directly."""
    for mod in ("PySide.QtCore", "PySide2.QtCore", "PySide6.QtCore"):
        try:
            return importlib.import_module(mod)
        except Exception:
            continue
    raise RuntimeError("Could not import Qt (PySide) for the main-thread executor")


class MainThreadExecutor:
    """Run callables on FreeCAD's main GUI thread.

    Qt/Coin are not thread-safe, so the HTTP worker threads must never touch the
    GUI. They submit a job here; a ``QTimer`` living on the main thread drains the
    queue inside the FreeCAD event loop and runs each job there, while the worker
    blocks on a result. ``Gui.updateGui()`` pumps a nested event loop, so the
    drain is guarded against re-entrancy."""

    def __init__(self):
        QtCore = _import_qtcore()
        self._main_thread = threading.current_thread()
        self._jobs: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._draining = False
        self._timer = QtCore.QTimer()
        self._timer.setInterval(8)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    def submit(self, fn):
        if threading.current_thread() is self._main_thread:
            return fn()  # already on the GUI thread (e.g. one-time setup)
        done = threading.Event()
        box: dict = {}
        with self._lock:
            self._jobs.append((fn, box, done))
        if not done.wait(timeout=180):
            raise TimeoutError("render timed out (FreeCAD GUI thread busy)")
        if "err" in box:
            raise box["err"]
        return box["result"]

    def _drain(self):
        if self._draining:
            return
        self._draining = True
        try:
            while True:
                with self._lock:
                    if not self._jobs:
                        return
                    fn, box, done = self._jobs.popleft()
                try:
                    box["result"] = fn()
                except Exception as e:  # noqa: BLE001
                    box["err"] = e
                finally:
                    done.set()
        finally:
            self._draining = False

    def stop(self):
        try:
            self._timer.stop()
        except Exception:
            pass


# Seconds after a render during which a window raise is treated as ours (setEdit
# raising FreeCAD) and re-tucked. We don't just wait for the raise *event* -- on a
# real WM/compositor it can arrive hundreds of ms late -- so for this whole window
# we actively re-tuck on a poll. Outside it (the annotator stopped navigating) a
# raise is a deliberate click on FreeCAD and is left alone. Bump this if a slow
# machine still lets FreeCAD pop up after a switch.
_TUCK_GRACE_S = 1.5
_TUCK_POLL_MS = 100


class InProcessRenderer:
    """Render using the already-running FreeCAD GUI -- no xvfb, no subprocess."""

    def __init__(self):
        self._exec = MainThreadExecutor()
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        self._rs = importlib.import_module("_render_server")
        try:  # match the author's render look (bg gradient, construction colour)
            self._exec.submit(self._rs._configure_background)
        except Exception:
            pass
        # Stop FreeCAD's own window from grabbing the screen on every sketch
        # switch (see FREECAD_WINDOW). Grab the main window, install a filter that
        # re-tucks it on any later raise, and tuck it now -- all on the GUI thread.
        self._mw = None
        self._filter = None
        self._poll = None
        self._tucking = False
        self._active_until = 0.0     # re-tuck raises only inside a render window
        if FREECAD_WINDOW in ("minimize", "lower"):
            try:
                import FreeCADGui as Gui
                self._mw = self._exec.submit(Gui.getMainWindow)
            except Exception:
                self._mw = None
            if self._mw is not None:
                self._exec.submit(self._install_filter)
                self._exec.submit(self._tuck_window)

    def _tuck_window(self) -> None:
        """Keep FreeCAD's window from covering the browser. Runs on the GUI
        thread; called after every render and from the event filter on any raise.
        Cheap no-op once already tucked; re-entrancy-guarded so the state changes
        it triggers can't recurse through the filter."""
        mw = self._mw
        if mw is None or self._tucking:
            return
        self._tucking = True
        try:
            if FREECAD_WINDOW == "minimize":
                if not mw.isMinimized():
                    mw.showMinimized()
            elif FREECAD_WINDOW == "lower":
                mw.lower()
        except Exception:
            pass
        finally:
            self._tucking = False

    def _restore_window(self) -> None:
        mw = self._mw
        if mw is None:
            return
        try:
            if mw.isMinimized():
                mw.showNormal()
            mw.raise_()
        except Exception:
            pass

    def _poll_tuck(self) -> None:
        """Active re-tuck while we're inside a render's grace window. Catches the
        raise even if its event arrives late or never (WM/compositor lag), which a
        purely event-driven filter would miss. Idle -> cheap no-op."""
        if time.monotonic() < self._active_until:
            self._tuck_window()

    def _install_filter(self) -> None:
        """Keep FreeCAD's window tucked whenever a render brings it forward. Two
        layers, both gated to the render grace window so an *idle* (deliberate)
        click on FreeCAD is left alone: (1) an event filter that re-tucks promptly
        on a raise event, and (2) a poll timer that re-tucks even when the event is
        late/absent. Runs on the GUI thread (QObject/QTimer thread affinity)."""
        if self._mw is None:
            return
        try:
            QtCore = _import_qtcore()
        except Exception:
            return
        E = QtCore.QEvent
        ET = getattr(E, "Type", E)        # PySide6 scopes enums under QEvent.Type
        wanted = set()
        for n in ("WindowActivate", "WindowStateChange", "Show"):
            v = getattr(ET, n, None)
            if v is not None:
                try:
                    wanted.add(int(v))
                except Exception:
                    pass
        renderer = self

        class _TuckFilter(QtCore.QObject):
            def eventFilter(self, obj, ev):
                try:
                    if (int(ev.type()) in wanted
                            and time.monotonic() < renderer._active_until):
                        renderer._tuck_window()
                except Exception:
                    pass
                return False          # observe only; never consume the event

        self._filter = _TuckFilter()
        self._mw.installEventFilter(self._filter)

        self._poll = QtCore.QTimer()
        self._poll.setInterval(_TUCK_POLL_MS)
        self._poll.timeout.connect(self._poll_tuck)
        self._poll.start()

    def request(self, req: dict) -> dict:
        def job():
            # Open the "this raise is ours" window around the render so the filter
            # re-tucks setEdit/openDocument raises -- including deferred ones that
            # land just after we return -- but not a later, idle user click.
            self._active_until = time.monotonic() + _TUCK_GRACE_S
            try:
                resp = self._rs.handle_request(req)
                self._tuck_window()
                return resp
            finally:
                self._active_until = time.monotonic() + _TUCK_GRACE_S
        return self._exec.submit(job)

    def _teardown_window(self) -> None:
        if self._mw is None:
            return
        try:
            if self._poll is not None:
                self._poll.stop()
        except Exception:
            pass
        self._poll = None
        try:
            if self._filter is not None:
                self._mw.removeEventFilter(self._filter)
        except Exception:
            pass
        self._filter = None
        self._restore_window()

    def close(self) -> None:
        if self._mw is not None:          # give the annotator their window back
            try:
                self._exec.submit(self._teardown_window)
            except Exception:
                pass
        self._exec.stop()


def _gui_up() -> bool:
    try:
        import FreeCAD  # core import is headless-safe
        return bool(getattr(FreeCAD, "GuiUp", False))
    except Exception:
        return False


def make_renderer(res: int, gui: bool):
    if gui:
        print("FreeCAD GUI detected -> in-process rendering (OS-agnostic).")
        return InProcessRenderer()
    print("No FreeCAD GUI -> launching xvfb render-server subprocess (headless Linux).")
    r = SubprocessRenderer(res)
    r.start()
    return r


# ===========================================================================
# Request helpers (backend-agnostic)
# ===========================================================================
_OP_KEYS = ("op", "geo", "pos", "x", "y", "constr", "cx", "cy", "height",
            "refit", "strip_dims")


def _swapped(stem: str) -> bool:
    """Deterministic per-sketch left/right flip (stable across reloads/sessions)."""
    if not SHUFFLE_SIDES:
        return False
    return bool(hashlib.sha256((SHUFFLE_SEED + "|" + stem).encode()).digest()[0] & 1)


def _sides(cfg: dict, stem: str):
    """(left_name, left_path, right_name, right_path) for this sketch, honoring the
    per-sketch shuffle. The two model folders are cfg name1/idx1 and name2/idx2."""
    if _swapped(stem):
        return (cfg["name2"], cfg["idx2"][stem][1], cfg["name1"], cfg["idx1"][stem][1])
    return (cfg["name1"], cfg["idx1"][stem][1], cfg["name2"], cfg["idx2"][stem][1])


def resolve_src(cfg: dict, panel: str, stem: str, variant: str | None) -> tuple[str, str]:
    if panel == "left":
        idx = cfg["idx2"] if _swapped(stem) else cfg["idx1"]
        return str(idx[stem][1]), "py"
    if panel == "right":
        idx = cfg["idx1"] if _swapped(stem) else cfg["idx2"]
        return str(idx[stem][1]), "py"
    if panel == "gt":
        gv = cfg["gt"].get(stem, {})
        if variant not in gv:
            raise ValueError("no such GT variant")
        return gv[variant], "fcstd"
    raise ValueError("bad panel")


def run_op(cfg: dict, renderer, body: dict) -> dict:
    stem = body.get("stem")
    if stem not in cfg["stem_set"]:
        raise KeyError(stem or "?")
    src, kind = resolve_src(cfg, body.get("panel"), stem, body.get("variant"))
    req = {k: body[k] for k in _OP_KEYS if body.get(k) is not None}
    req.update({"src": src, "kind": kind, "w": cfg["res"], "h": cfg["res"]})
    return renderer.request(req)


def build_meta(cfg: dict) -> dict:
    sketches = []
    for stem in cfg["stems"]:
        swap = _swapped(stem)
        sketches.append({
            "stem": stem,
            "batch": cfg["idx1"][stem][0],
            "left": cfg["cap2"].get(stem) if swap else cfg["cap1"].get(stem),
            "right": cfg["cap1"].get(stem) if swap else cfg["cap2"].get(stem),
            "gt": list(cfg["gt"].get(stem, {}).keys()),
        })
    f1 = "Sketch A" if BLIND else cfg["name1"]
    f2 = "Sketch B" if BLIND else cfg["name2"]
    return {"folder1": f1, "folder2": f2, "res": cfg["res"], "blind": BLIND,
            "results_name": RESULTS_NAME, "sketches": sketches}


# ===========================================================================
# Vote storage (one JSON file, atomic, resumable, integrity-hashed when done)
# ===========================================================================
def _relpath(p, base: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(p)


class VotesManager:
    CHOICES = ("A", "B", "Tie")

    def __init__(self, cfg: dict, path: Path):
        self.cfg = cfg
        self.path = Path(path)
        self.lock = threading.Lock()
        self.votes: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            for e in data.get("votes", []):
                if e.get("stem") in self.cfg["stem_set"]:
                    self.votes[e["stem"]] = e
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not read existing results ({e}); starting fresh")

    def _snapshot(self) -> dict:
        total = len(self.cfg["stems"])
        return {"votes": {s: e["choice"] for s, e in self.votes.items()},
                "total": total, "done": len(self.votes),
                "all_done": len(self.votes) >= total}

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def record(self, stem, choice) -> dict:
        if stem not in self.cfg["stem_set"]:
            raise ValueError("unknown sketch")
        choice = (choice or "").strip()
        if choice not in self.CHOICES:
            raise ValueError("choice must be one of A / B / Tie")
        cfg = self.cfg
        lname, lpath, rname, rpath = _sides(cfg, stem)
        with self.lock:
            self.votes[stem] = {
                "stem": stem,
                "batch": cfg["idx1"][stem][0],
                "left": lname, "right": rname,
                "left_path": _relpath(lpath, self.path.parent),
                "right_path": _relpath(rpath, self.path.parent),
                "choice": choice,
                "preferred": lname if choice == "A" else
                             rname if choice == "B" else "Tie",
                "ts": _now_iso(),
            }
            self._write()
            return self._snapshot()

    def _write(self) -> None:
        cfg = self.cfg
        ordered = [self.votes[s] for s in cfg["stems"] if s in self.votes]
        doc = {
            "meta": {
                "model1": cfg["name1"], "model2": cfg["name2"],
                "shuffle_sides": SHUFFLE_SIDES,
                "shuffle_seed": SHUFFLE_SEED if SHUFFLE_SIDES else None,
                "gt_dir": cfg.get("gt_dir_str"), "total": len(cfg["stems"]),
                "blind": BLIND, "created_at": cfg["created_at"],
                "hash_recipe": ("sha256 of json.dumps(votes sorted by stem, "
                                "sort_keys=True, separators=(',',':'), "
                                "ensure_ascii=False)"),
            },
            "votes": ordered,
        }
        if len(self.votes) >= len(cfg["stems"]):
            canon = json.dumps(sorted(self.votes.values(), key=lambda e: e["stem"]),
                               sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False)
            doc["completed_at"] = _now_iso()
            doc["integrity"] = {"algo": "sha256", "scope": "votes",
                                "value": hashlib.sha256(canon.encode("utf-8")).hexdigest()}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        tmp.replace(self.path)


# ===========================================================================
# HTTP server (stdlib only)
# ===========================================================================
def make_handler(cfg: dict, renderer, votes: VotesManager):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, payload, ctype="application/json"):
            if ctype == "application/json":
                data = json.dumps(payload).encode("utf-8")
            elif isinstance(payload, bytes):
                data = payload
            else:
                data = payload.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except Exception:
                pass

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw or b"{}")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, INDEX_HTML, "text/html")
            elif path == "/api/meta":
                self._send(200, build_meta(cfg))
            elif path == "/api/votes":
                self._send(200, votes.snapshot())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._body()
            except Exception:
                self._send(400, {"error": "bad json"})
                return
            if path == "/api/op":
                try:
                    resp = run_op(cfg, renderer, body)
                except KeyError as e:
                    self._send(404, {"error": f"unknown sketch: {e}"})
                    return
                except ValueError as e:
                    self._send(400, {"error": str(e)})
                    return
                except Exception as e:  # noqa: BLE001
                    self._send(422, {"error": str(e)})
                    return
                if not resp.get("ok"):
                    self._send(422, {"error": resp.get("error", "render failed")})
                    return
                self._send(200, resp)
            elif path == "/api/vote":
                try:
                    out = votes.record(body.get("stem"), body.get("choice"))
                except ValueError as e:
                    self._send(400, {"error": str(e)})
                    return
                self._send(200, out)
            else:
                self._send(404, {"error": "not found"})

    return Handler


# ===========================================================================
# CLI / entry point
# ===========================================================================
def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="User-study side-by-side sketch preference viewer.")
    ap.add_argument("folder1", nargs="?", type=Path, help="left model folder (A)")
    ap.add_argument("folder2", nargs="?", type=Path, help="right model folder (B)")
    ap.add_argument("--gt-dir", type=Path, help=f"GT root (default: <data-dir>/{GT_DIRNAME})")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="base data dir for auto-discovery + results file")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--res", type=int, default=1500)
    ap.add_argument("--no-browser", action="store_true")
    return ap.parse_args(argv)


def main(folder1=None, folder2=None, gt_dir=None, data_dir=None,
         port=8000, res=1500, open_browser=True, argv=None):
    """Start the viewer. Call with no args inside the FreeCAD GUI console, or pass
    ``argv`` (a list) / explicit folders from a script."""
    global _RUNNING
    if argv is not None:
        a = _parse_args(argv)
        folder1, folder2, gt_dir = a.folder1, a.folder2, a.gt_dir
        data_dir, port, res = a.data_dir, a.port, a.res
        open_browser = not a.no_browser

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    if folder1 and folder2:
        f1, f2 = Path(folder1).resolve(), Path(folder2).resolve()
        results_dir = data_dir if data_dir.exists() else f1.parent
    else:
        # Look in <data-dir> first, then alongside compare.py (the model folders
        # may sit directly under user_study/ or under user_study/data/).
        used_dir, models, gt_found, seen = None, [], None, []
        for cand in (Path(data_dir), HERE):
            if cand in seen:
                continue
            seen.append(cand)
            m, g = discover(cand, gt_dir)
            if len(m) >= 2:
                used_dir, models, gt_found = cand, m, g
                break
        if used_dir is None:
            sys.exit(_LAYOUT_HELP.format(data_dir=data_dir, found=[]))
        f1, f2 = models[0].resolve(), models[1].resolve()
        if gt_dir is None:
            gt_dir = gt_found
        results_dir = used_dir

    idx1, idx2 = index_folder(f1), index_folder(f2)

    def _sk(s):
        b = idx1[s][0]
        return (int(b) if b.isdigit() else float("inf"), b, s)

    stems = sorted(set(idx1) & set(idx2), key=_sk)
    if not stems:
        sys.exit(f"No shared sketches between {f1} and {f2}")

    gt: dict = {}
    gt_res = Path(gt_dir).resolve() if gt_dir else None
    if gt_res and gt_res.exists():
        gt = {s: gt_variants(gt_res, idx1[s][0], s) for s in stems}

    cfg = {
        "idx1": idx1, "idx2": idx2, "stems": stems, "stem_set": set(stems),
        "cap1": load_captions(f1), "cap2": load_captions(f2),
        "gt": gt, "name1": f1.name, "name2": f2.name, "res": res,
        "gt_dir_str": str(gt_res) if gt_res else None, "created_at": _now_iso(),
    }
    n_gt = sum(1 for s in stems if gt.get(s))
    print(f"Shared sketches: {len(stems)}  |  with GT: {n_gt}")
    print(f"Left  (A): {f1.name}\nRight (B): {f2.name}"
          + ("    [names BLIND in UI]" if BLIND else ""))

    results_path = Path(results_dir) / RESULTS_NAME
    votes = VotesManager(cfg, results_path)
    print(f"Results: {results_path}  ({len(votes.votes)}/{len(stems)} already recorded)")

    if _RUNNING is not None:          # re-run inside the same FreeCAD session
        print("Stopping previous viewer before restarting ...")
        stop()

    gui = _gui_up()
    renderer = make_renderer(res, gui)
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                    make_handler(cfg, renderer, votes))
    except OSError as e:
        close = getattr(renderer, "close", None)
        if close:
            close()
        sys.exit(f"Could not bind 127.0.0.1:{port} ({e}). A previous viewer may "
                 f"still be running -- run compare.stop(), or use port=<other>.")
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Viewer:  {url}\n")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if gui:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        _RUNNING = {"httpd": httpd, "renderer": renderer, "votes": votes, "thread": t}
        print("Serving inside FreeCAD (in-process). Leave FreeCAD open and use the "
              "browser tab.\nRun  compare.stop()  in the console to shut down.")
        return _RUNNING

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        close = getattr(renderer, "close", None)
        if close:
            close()
    return None


def stop():
    """Shut down the in-process server started by ``main()`` (annotator console)."""
    global _RUNNING
    if not _RUNNING:
        print("Not running.")
        return
    r, _RUNNING = _RUNNING, None
    try:
        r["httpd"].shutdown()
        r["httpd"].server_close()
    except Exception:
        pass
    close = getattr(r.get("renderer"), "close", None)
    if close:
        try:
            close()
        except Exception:
            pass
    print("Stopped.")


# ===========================================================================
# Front-end (single page; vanilla JS; preference voting over FreeCAD renders)
# ===========================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sketch preference study</title>
<style>
  :root{--bg:#1b2230;--bar:#11161f;--line:#2c3647;--fg:#dfe6f0;--muted:#8a97ab;--accent:#4a90d9;--good:#7fe0a0;}
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
            font:13px/1.4 system-ui,Segoe UI,Roboto,sans-serif;overflow:hidden;}
  #bar{height:46px;display:flex;align-items:center;gap:8px;padding:0 10px;background:var(--bar);
       border-bottom:1px solid var(--line);position:relative;z-index:5;}
  .nav{background:#222c3a;color:var(--fg);border:1px solid var(--line);border-radius:6px;
       width:34px;height:30px;font-size:17px;cursor:pointer;}
  .nav:hover{background:#2c3a4d;}
  #title{flex:0 1 auto;min-width:240px;max-width:42vw;text-align:center;cursor:pointer;
         background:#222c3a;border:1px solid var(--line);border-radius:6px;height:30px;display:flex;
         align-items:center;justify-content:center;gap:8px;padding:0 12px;font-weight:600;
         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  #title:hover{background:#2c3a4d;}
  #title .batch{color:var(--accent);}
  #counter{color:var(--muted);font-weight:400;font-size:12px;}
  #vcount{color:var(--muted);font-size:12px;white-space:nowrap;margin-left:2px;}
  #spacer{flex:1;}
  .ctrl{display:flex;align-items:center;gap:6px;color:var(--muted);}
  /* preference bar */
  #votebar{height:56px;display:flex;align-items:stretch;gap:8px;padding:8px 10px;
           background:#0f141d;border-bottom:1px solid var(--line);}
  .vbtn{flex:1;border:1px solid var(--line);border-radius:8px;background:#222c3a;color:var(--fg);
        font-size:14px;font-weight:600;cursor:pointer;display:flex;align-items:center;
        justify-content:center;gap:8px;transition:background .12s,box-shadow .12s;}
  .vbtn:hover{background:#2c3a4d;}
  .vbtn.tie{flex:0 0 180px;}
  .vbtn.selected{background:var(--accent);border-color:#7fb2ea;color:#fff;
                 box-shadow:0 0 0 2px rgba(127,178,234,.4) inset;}
  .vbtn.flash{background:#8ad4ff !important;color:#06283d !important;}
  .vbtn:disabled{opacity:.55;cursor:default;}
  #panels{display:flex;height:calc(100% - 102px);}
  .panel{flex:1;min-width:0;display:flex;flex-direction:column;border-left:1px solid var(--line);}
  .panel:first-child{border-left:0;}
  .phead{display:flex;align-items:center;gap:8px;padding:5px 10px;background:#161d28;
         border-bottom:1px solid var(--line);}
  .pname{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;}
  .pcap{color:var(--muted);font-size:12px;white-space:nowrap;}
  .pcap .badge{padding:0 6px;border-radius:8px;background:#243042;margin-left:4px;}
  .pcap .dof{padding:0 6px;border-radius:8px;background:#2a3b53;color:#bcd;margin-left:4px;}
  .pbtn{background:#222c3a;color:var(--fg);border:1px solid var(--line);border-radius:5px;
        height:24px;padding:0 8px;font-size:12px;cursor:pointer;}
  .pbtn:hover{background:#2c3a4d;}
  .view{position:relative;flex:1;overflow:hidden;cursor:grab;background:#1a4060;}
  .view.grab{cursor:grabbing;}
  .view img{position:absolute;left:0;top:0;transform-origin:0 0;user-select:none;
            -webkit-user-drag:none;user-drag:none;-webkit-user-select:none;}
  .hd{position:absolute;inset:0;pointer-events:none;}
  .h{position:absolute;width:11px;height:11px;margin:-6px 0 0 -6px;border-radius:50%;
     border:1.5px solid #19e0e0;background:rgba(20,224,224,.25);}
  .h.con{border-color:#9aa;background:rgba(160,170,180,.2);}
  .h.lab{border-radius:2px;border-color:#ffae42;background:rgba(255,174,66,.25);
         width:13px;height:13px;margin:-7px 0 0 -7px;}
  .h.hot{box-shadow:0 0 0 3px rgba(255,255,255,.25);}
  .msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
       color:var(--muted);pointer-events:none;}
  .msg.err{color:#e0746a;}
  #picker{position:absolute;top:48px;left:50%;transform:translateX(-50%);width:min(700px,93vw);
          max-height:72vh;background:#0f141d;border:1px solid var(--line);border-radius:8px;
          box-shadow:0 12px 40px #000a;display:none;flex-direction:column;z-index:10;}
  #picker.open{display:flex;}
  #search{margin:8px;padding:8px 10px;background:#1a212d;border:1px solid var(--line);
          border-radius:6px;color:var(--fg);font-size:13px;}
  #list{overflow:auto;padding:0 6px 8px;}
  .item{display:flex;gap:10px;align-items:baseline;padding:7px 10px;border-radius:6px;cursor:pointer;}
  .item:hover{background:#1c2533;} .item.cur{background:#23344a;}
  .item .chk{width:14px;flex:none;color:var(--good);font-weight:700;}
  .item .bt{color:var(--accent);width:34px;flex:none;font-variant-numeric:tabular-nums;}
  .item .st{color:var(--muted);font-size:11px;margin-left:auto;white-space:nowrap;}
  .hint{position:absolute;left:8px;bottom:6px;color:#6b7a90;font-size:11px;pointer-events:none;}
  /* all-done overlay */
  #done{position:absolute;inset:0;background:rgba(10,14,20,.94);display:none;flex-direction:column;
        align-items:center;justify-content:center;z-index:60;gap:14px;text-align:center;padding:24px;}
  #done.show{display:flex;}
  #done h1{margin:0;font-size:32px;color:var(--good);}
  #done p{margin:0;color:var(--muted);max-width:560px;}
  #done button{margin-top:6px;background:#222c3a;color:var(--fg);border:1px solid var(--line);
               border-radius:8px;height:36px;padding:0 18px;cursor:pointer;font-size:13px;}
  #done button:hover{background:#2c3a4d;}
</style></head><body>
  <div id="bar">
    <button class="nav" id="prev" title="Previous (←)">‹</button>
    <div id="title"><span id="ttext">loading…</span><span id="counter"></span></div>
    <button class="nav" id="next" title="Next (→)">›</button>
    <span id="vcount"></span>
    <div id="spacer"></div>
    <label class="ctrl"><input type="checkbox" id="dimToggle"/> Remove Dimension</label>
  </div>
  <div id="votebar">
    <button class="vbtn left" id="vL">Prefer left sketch</button>
    <button class="vbtn tie" id="vT">Tie</button>
    <button class="vbtn right" id="vR">Prefer right sketch</button>
  </div>
  <div id="picker"><input id="search" placeholder="Filter by batch / sketch…" autocomplete="off"/>
    <div id="list"></div></div>
  <div id="panels"></div>
  <div id="done">
    <h1>All Done! You can close the screen.</h1>
    <p id="doneSub">Thanks! Your responses are saved. Please send the results file back to the organizer.</p>
    <button id="review">Review / change answers</button>
  </div>
<script>
const $=s=>document.querySelector(s);
let META=null,I=0,RES=1500,STRIP_DIMS=false;
let VOTES={},voting=false;
const viewers={},idxByStem={};

async function boot(){
  META=await (await fetch('/api/meta')).json(); RES=META.res;
  META.sketches.forEach((sk,i)=>idxByStem[sk.stem]=i);
  try{ const v=await (await fetch('/api/votes')).json(); VOTES=v.votes||{}; }catch(e){ VOTES={}; }
  buildPanels(); buildList(); wireVotes(); updateChecks();
  const start=firstUnvoted();
  if(start<0){ show(0); showDone(); } else { show(start); }
}
function folderLabels(){
  return [{k:'left',n:META.folder1},{k:'right',n:META.folder2}];
}
function buildPanels(){
  const root=$('#panels'); root.innerHTML='';
  for(const k in viewers) delete viewers[k];   // drop stale viewers (e.g. gt once GT is turned off, so it stops loading in the background)
  for(const f of folderLabels()){
    const p=document.createElement('div'); p.className='panel';
    p.innerHTML=`<div class="phead"><span class="pname" title="${f.n}">${f.n}</span>
      <span class="pcap" id="cap-${f.k}"></span>
      <button class="pbtn" id="rst-${f.k}" title="Discard edits & refit">↺ reset</button></div>
      <div class="view" id="view-${f.k}"><img id="img-${f.k}" draggable="false"/><div class="hd" id="hd-${f.k}"></div>
      <div class="msg" id="msg-${f.k}"></div>
      <div class="hint">drag vertex = deform · drag □ = move label · wheel = zoom · drag bg = pan</div></div>`;
    root.appendChild(p);
    viewers[f.k]=makeViewer(f.k);
    $('#rst-'+f.k).onclick=()=>viewers[f.k].reset();
  }
}

function makeViewer(key){
  const view=$('#view-'+key), img=$('#img-'+key), hd=$('#hd-'+key), msg=$('#msg-'+key);
  let cam=null, tf={s:1,tx:0,ty:0}, H={v:[],l:[]}, ctx={stem:null,variant:null};
  let token=0, dots=[], drag=null, inflight=false, queued=null;

  const w2i=(x,y)=>[(x-cam.cx)/cam.height*cam.w+cam.w/2, cam.h/2-(y-cam.cy)/cam.height*cam.h];
  const i2w=(ix,iy)=>[cam.cx+(ix-cam.w/2)*cam.height/cam.w, cam.cy+(cam.h/2-iy)*cam.height/cam.h];
  const screen=(x,y)=>{const[ix,iy]=w2i(x,y);return[ix*tf.s+tf.tx, iy*tf.s+tf.ty];};
  const scr2w=(sx,sy)=>i2w((sx-tf.tx)/tf.s,(sy-tf.ty)/tf.s);
  const applyImg=()=>{img.style.transform=`translate(${tf.tx}px,${tf.ty}px) scale(${tf.s})`;};

  function fit(){ if(!cam) return; const vw=view.clientWidth,vh=view.clientHeight;
    const zoomIn=cam.geom_height>0?cam.height/cam.geom_height:1;
    tf.s=Math.min(vw/cam.w, vh/cam.h)*0.98*zoomIn; tf.tx=(vw-cam.w*tf.s)/2; tf.ty=(vh-cam.h*tf.s)/2;
    applyImg(); drawHandles(); }

  function drawHandles(){
    const items=[]; for(const v of H.v) items.push({t:'v',a:v});
    for(const l of H.l) items.push({t:'l',a:l});
    while(dots.length<items.length){const d=document.createElement('div');d.className='h';hd.appendChild(d);dots.push(d);}
    while(dots.length>items.length){hd.removeChild(dots.pop());}
    items.forEach((it,k)=>{
      const d=dots[k]; const a=it.a;
      const x=a[it.t==='v'?2:1], y=a[it.t==='v'?3:2];
      const[sx,sy]=screen(x,y);
      d.style.left=sx+'px'; d.style.top=sy+'px';
      d.className='h'+(it.t==='l'?' lab':'')+(it.t==='v'&&a[4]?' con':'')+(drag&&drag.k===k?' hot':'');
      d._it=it; d._k=k;
    });
  }
  function hitTest(sx,sy){
    let best=null,bd=14*14;
    dots.forEach((d,k)=>{const dx=parseFloat(d.style.left)-sx,dy=parseFloat(d.style.top)-sy;
      const q=dx*dx+dy*dy; if(q<bd){bd=q;best={k,it:d._it};}});
    return best;
  }

  function showMsg(t,err){msg.textContent=t;msg.className='msg'+(err?' err':'');}
  function setData(r){
    cam={cx:r.frame.cx,cy:r.frame.cy,height:r.frame.height,w:r.frame.w,h:r.frame.h,geom_height:r.frame.geom_height};
    H=r.handles||{v:[],l:[]};
    setCap(r.dof,r.status); syncListDof(r.dof);
    img.onload=()=>{ if(!cam._fitted){cam._fitted=true; fit();} else { applyImg(); drawHandles(); } showMsg('',false); };
    img.src='data:image/png;base64,'+r.png;
  }
  function setCap(dof,status){
    // status is the LIVE solver result from this render (not the cached
    // results.json); DoF is also live. Keeps the caption truthful about
    // conflicting/redundant sketches that the cached status mislabels.
    let s = key==='gt' ? `variant ${ctx.variant||'-'}` : (status||'');
    $('#cap-'+key).innerHTML = s + (dof!=null&&dof>=0?`<span class="dof">DoF ${dof}</span>`:'');
  }
  function syncListDof(dof){
    // Self-heal the sidebar DoF to the live value as each sketch is viewed, so it
    // never disagrees with the panel. Skip GT and strip-dims so the list keeps the
    // canonical (full-constraint) DoF rather than the stripped one.
    if(key==='gt'||STRIP_DIMS||dof==null||dof<0) return;
    const sk=META.sketches[I]; if(!sk) return;
    if(!sk[key]) sk[key]={}; sk[key].dof=dof;
    const row=document.querySelector('.item[data-idx="'+I+'"]');
    if(row){ const st=row.querySelector('.st'); if(st) st.textContent=listDofText(sk); }
  }

  async function post(body){
    body.panel=key; body.stem=ctx.stem; if(key==='gt') body.variant=ctx.variant;
    body.strip_dims=STRIP_DIMS;
    const r=await fetch('/api/op',{method:'POST',headers:{'Content-Type':'application/json'},
                                   body:JSON.stringify(body)});
    if(!r.ok){ let d='error'; try{d=(await r.json()).error||d;}catch(e){} throw new Error(d); }
    return r.json();
  }
  async function load(stem,variant){
    ctx={stem,variant}; const my=++token;
    if(!stem||(key==='gt'&&!variant)){ showMsg('—'); H={v:[],l:[]}; drawHandles(); img.removeAttribute('src'); return; }
    showMsg('rendering…'); img.removeAttribute('src');
    try{ const r=await post({op:'view'}); if(my!==token)return; setData(r); }
    catch(e){ if(my===token){ showMsg('⚠ '+e.message,true); H={v:[],l:[]}; drawHandles(); img.removeAttribute('src'); } }
  }
  async function reset(){ const my=++token; showMsg('resetting…');
    try{ const r=await post({op:'reset'}); if(my!==token)return; cam=null; setData(r); }
    catch(e){ showMsg('⚠ '+e.message,true); } }

  function sendMove(){
    if(inflight){return;} const job=queued; queued=null; if(!job) return; inflight=true;
    post(job).then(r=>{ inflight=false;
      cam={cx:r.frame.cx,cy:r.frame.cy,height:r.frame.height,w:r.frame.w,h:r.frame.h,geom_height:r.frame.geom_height};
      H=r.handles; setCap(r.dof,r.status); syncListDof(r.dof);
      img.onload=()=>{applyImg();drawHandles();}; img.src='data:image/png;base64,'+r.png;
      if(queued) sendMove();
    }).catch(e=>{ inflight=false; showMsg('⚠ '+e.message,true); });
  }
  view.addEventListener('wheel',e=>{ e.preventDefault(); if(!cam)return;
    const r=view.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const f=e.deltaY<0?1.12:1/1.12, ns=Math.max(.05,Math.min(80,tf.s*f));
    tf.tx=mx-(mx-tf.tx)*(ns/tf.s); tf.ty=my-(my-tf.ty)*(ns/tf.s); tf.s=ns; applyImg(); drawHandles();
  },{passive:false});
  view.addEventListener('mousedown',e=>{ if(!cam)return;
    const r=view.getBoundingClientRect(),sx=e.clientX-r.left,sy=e.clientY-r.top;
    const hit=hitTest(sx,sy);
    if(hit){ drag={mode:hit.it.t,k:hit.k,it:hit.it}; drawHandles(); }
    else { drag={mode:'pan',lx:e.clientX,ly:e.clientY}; }
    view.classList.add('grab');
  });
  window.addEventListener('mousemove',e=>{ if(!drag)return;
    if(drag.mode==='pan'){ tf.tx+=e.clientX-drag.lx; tf.ty+=e.clientY-drag.ly;
      drag.lx=e.clientX; drag.ly=e.clientY; applyImg(); drawHandles(); return; }
    const r=view.getBoundingClientRect(); const[wx,wy]=scr2w(e.clientX-r.left,e.clientY-r.top);
    const d=dots[drag.k]; if(d){const[ssx,ssy]=screen(wx,wy);d.style.left=ssx+'px';d.style.top=ssy+'px';}
    const a=drag.it.a;
    if(drag.mode==='v') queued={op:'move',geo:a[0],pos:a[1],x:wx,y:wy,cx:cam.cx,cy:cam.cy,height:cam.height};
    else queued={op:'move_label',constr:a[0],x:wx,y:wy,cx:cam.cx,cy:cam.cy,height:cam.height};
    sendMove();
  });
  window.addEventListener('mouseup',()=>{ if(drag){ if(queued)sendMove(); drag=null; view.classList.remove('grab'); drawHandles(); }});
  view.addEventListener('dblclick',fit);

  return {load,reset,fit};
}

function setTitle(){ const sk=META.sketches[I];
  $('#ttext').innerHTML=`<span class="batch">${sk.batch}/</span>${sk.stem}`;
  $('#counter').textContent=`(${I+1}/${META.sketches.length})`;
}
function show(i){ if(!META.sketches.length)return;
  const ni=(i+META.sketches.length)%META.sketches.length;
  if(ni!==I && STRIP_DIMS){ STRIP_DIMS=false; $('#dimToggle').checked=false; }
  I=ni; const sk=META.sketches[I];
  setTitle();
  viewers.left.load(sk.stem,null); viewers.right.load(sk.stem,null);
  hideDone(); setVoteEnabled(true); updateVoteButtons(VOTES[sk.stem]||null,false);
  syncList();
}

// ---- preference voting ----
function wireVotes(){
  $('#vL').onclick=()=>vote('A');
  $('#vT').onclick=()=>vote('Tie');
  $('#vR').onclick=()=>vote('B');
  $('#review').onclick=()=>{ hideDone(); openPicker(); };
}
function firstUnvoted(){ for(let i=0;i<META.sketches.length;i++){ if(!(META.sketches[i].stem in VOTES)) return i; } return -1; }
function nextUnvotedFrom(i){ const n=META.sketches.length;
  for(let k=1;k<=n;k++){ const j=(i+k)%n; if(!(META.sketches[j].stem in VOTES)) return j; } return -1; }
function setVoteEnabled(on){ ['#vL','#vT','#vR'].forEach(s=>{$(s).disabled=!on;}); }
function updateVoteButtons(choice,flash){
  const map={A:$('#vL'),B:$('#vR'),Tie:$('#vT')};
  Object.values(map).forEach(b=>b.classList.remove('selected','flash'));
  if(choice&&map[choice]){ const b=map[choice]; b.classList.add('selected');
    if(flash){ b.classList.add('flash'); setTimeout(()=>b.classList.remove('flash'),450); } }
}
function vote(choice){
  if(voting||!META.sketches.length) return;
  const sk=META.sketches[I];
  VOTES[sk.stem]=choice;
  updateVoteButtons(choice,true); updateChecks();
  voting=true; setVoteEnabled(false);
  postVote(sk.stem,choice);
  setTimeout(()=>{ voting=false;
    const n=nextUnvotedFrom(I);
    if(n<0){ setVoteEnabled(true); showDone(); } else { show(n); }
  },500);
}
async function postVote(stem,choice){
  try{
    const r=await fetch('/api/vote',{method:'POST',headers:{'Content-Type':'application/json'},
                                     body:JSON.stringify({stem,choice})});
    if(!r.ok){ console.warn('vote save failed',(await r.json()).error); }
  }catch(e){ console.warn('vote save failed',e); }
}
function updateChecks(){
  document.querySelectorAll('#list .item').forEach(el=>{
    const sk=META.sketches[+el.dataset.idx];
    const voted=sk&&(sk.stem in VOTES);
    el.classList.toggle('voted',!!voted);
    const c=el.querySelector('.chk'); if(c) c.textContent=voted?'✓':'';
  });
  const done=Object.keys(VOTES).length;
  $('#vcount').textContent=`voted ${done}/${META.sketches.length}`;
}
function showDone(){ $('#done').classList.add('show'); }
function hideDone(){ $('#done').classList.remove('show'); }

// ---- filename / batch jump list ----
function listDofText(sk){  // shared by buildList and live syncListDof
  const ld=sk&&sk.left&&sk.left.dof!=null?`L DoF ${sk.left.dof}`:'';
  const rd=sk&&sk.right&&sk.right.dof!=null?` · R DoF ${sk.right.dof}`:'';
  return ld+rd;
}
function buildList(){ const list=$('#list'); list.innerHTML='';
  META.sketches.forEach((sk,idx)=>{ const d=document.createElement('div');
    d.className='item'; d.dataset.idx=idx;
    d.dataset.key=(sk.batch+'/'+sk.stem).toLowerCase();
    d.innerHTML=`<span class="chk"></span><span class="bt">${sk.batch}</span><span class="stem">${sk.stem}</span>
                 <span class="st">${listDofText(sk)}</span>`;
    d.onclick=()=>{show(idx);closePicker();}; list.appendChild(d); });
}
function syncList(){ document.querySelectorAll('.item').forEach(el=>el.classList.toggle('cur',+el.dataset.idx===I));
  const cur=document.querySelector('.item.cur'); if(cur&&$('#picker').classList.contains('open'))cur.scrollIntoView({block:'nearest'}); }
function openPicker(){ $('#picker').classList.add('open'); const s=$('#search'); s.value='';filter('');s.focus();syncList(); }
function closePicker(){ $('#picker').classList.remove('open'); }
function filter(q){ q=q.toLowerCase(); document.querySelectorAll('.item').forEach(el=>
  el.style.display=el.dataset.key.includes(q)?'':'none'); }

$('#prev').onclick=()=>show(I-1);
$('#next').onclick=()=>show(I+1);
$('#title').onclick=()=>$('#picker').classList.contains('open')?closePicker():openPicker();
$('#search').addEventListener('input',e=>filter(e.target.value));
$('#dimToggle').addEventListener('change',e=>{ STRIP_DIMS=e.target.checked; show(I); });
document.addEventListener('keydown',e=>{ if(e.target.id==='search'){if(e.key==='Escape')closePicker();return;}
  if(e.key==='ArrowLeft')show(I-1); else if(e.key==='ArrowRight')show(I+1); else if(e.key==='Escape')closePicker(); });
document.addEventListener('click',e=>{ if(!$('#picker').contains(e.target)&&e.target.id!=='title'&&!$('#title').contains(e.target))closePicker(); });
window.addEventListener('resize',()=>Object.values(viewers).forEach(v=>v.fit()));
document.addEventListener('dragstart',e=>e.preventDefault());
boot();
</script></body></html>
"""


if __name__ == "__main__":
    main(argv=sys.argv[1:])
