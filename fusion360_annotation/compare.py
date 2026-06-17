#!/usr/bin/env python
"""Side-by-side **editable** FreeCAD viewer for two ``model_outputs`` folders.

    python compare.py <folder1> <folder2> [options]

For every sketch the two runs share, shows the **real FreeCAD Sketcher rendering**
of folder1's and folder2's predicted constraint sets side by side (and, optionally,
the human ground-truth from ``data_eval``).  Each panel is interactive:

* **drag a primitive vertex** -> ``movePoint`` -> the solver re-solves -> an
  under-constrained sketch *deforms* so you can see its remaining DoF;
* **drag a dimension label** to reposition it;
* mouse wheel = zoom, drag empty space = pan; per-panel reset discards edits.

How it works: a persistent FreeCAD-GUI **render server** (``_render_server.py``,
under ``xvfb-run``) keeps each sketch loaded as a live session and returns, for
every edit, a fresh PNG of the viewport (no toolbar chrome) plus the draggable
handle positions and the camera framing.  This process is a thin FastAPI app that
proxies those ops and serves a single-page UI.

Run with the FreeCAD-build Python (the one that can ``import FreeCAD``).

Options:
    --gt-dir DIR   ground-truth root (default: <repo>/data_eval)
    --port N       web port (default 8000)
    --res N        render resolution px (default 1500)
    --no-browser   do not auto-open a browser
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


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


# ===========================================================================
# Render-server client (one persistent, serialized connection)
# ===========================================================================
class RenderClient:
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


# ===========================================================================
# Web app
# ===========================================================================
def build_app(cfg: dict, client: RenderClient) -> FastAPI:
    app = FastAPI()

    def resolve_src(panel: str, stem: str, variant: str | None) -> tuple[str, str]:
        if panel == "left":
            return str(cfg["idx1"][stem][1]), "py"
        if panel == "right":
            return str(cfg["idx2"][stem][1]), "py"
        if panel == "gt":
            gv = cfg["gt"].get(stem, {})
            if variant not in gv:
                raise HTTPException(404, "no such GT variant")
            return gv[variant], "fcstd"
        raise HTTPException(400, "bad panel")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        sketches = []
        for stem in cfg["stems"]:
            sketches.append({
                "stem": stem,
                "batch": cfg["idx1"][stem][0],
                "left": cfg["cap1"].get(stem),
                "right": cfg["cap2"].get(stem),
                "gt": list(cfg["gt"].get(stem, {}).keys()),
            })
        return JSONResponse({"folder1": cfg["name1"], "folder2": cfg["name2"],
                             "res": cfg["res"], "sketches": sketches})

    @app.post("/api/op")
    def op(body: dict = Body(...)):
        panel = body.get("panel")
        stem = body.get("stem")
        if stem not in cfg["stem_set"]:
            raise HTTPException(404, "unknown sketch")
        src, kind = resolve_src(panel, stem, body.get("variant"))
        req = {k: body[k] for k in
               ("op", "geo", "pos", "x", "y", "constr", "cx", "cy", "height", "refit")
               if k in body and body[k] is not None}
        req.update({"src": src, "kind": kind,
                    "w": cfg["res"], "h": cfg["res"]})
        resp = client.request(req)
        if not resp.get("ok"):
            raise HTTPException(422, resp.get("error", "render failed"))
        return JSONResponse(resp)

    return app


# ===========================================================================
# Render-server process management
# ===========================================================================
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


def start_render_server(port: int) -> subprocess.Popen:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder1", type=Path)
    ap.add_argument("folder2", type=Path)
    ap.add_argument("--gt-dir", type=Path, default=REPO_ROOT / "data_eval")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--res", type=int, default=1500)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    f1, f2 = args.folder1.resolve(), args.folder2.resolve()
    idx1, idx2 = index_folder(f1), index_folder(f2)
    def _sort_key(s):
        b = idx1[s][0]
        return (int(b) if b.isdigit() else float('inf'), b, s)
    stems = sorted(set(idx1) & set(idx2), key=_sort_key)
    if not stems:
        sys.exit(f"No shared sketches between {f1} and {f2}")

    gt_dir = args.gt_dir.resolve()
    gt = {}
    if gt_dir.exists():
        gt = {s: gt_variants(gt_dir, idx1[s][0], s) for s in stems}
    cfg = {
        "idx1": idx1, "idx2": idx2, "stems": stems, "stem_set": set(stems),
        "cap1": load_captions(f1), "cap2": load_captions(f2),
        "gt": gt, "name1": f1.name, "name2": f2.name, "res": args.res,
    }
    n_gt = sum(1 for s in stems if gt.get(s))
    print(f"Shared sketches: {len(stems)}  |  with GT: {n_gt}")

    rport = free_port()
    print(f"Starting FreeCAD render server on port {rport} (xvfb) ...")
    proc = start_render_server(rport)
    client = RenderClient("127.0.0.1", rport)
    app = build_app(cfg, client)

    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  Viewer:  {url}\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        proc.terminate()


# ===========================================================================
# Front-end (single page; vanilla JS; draggable handles over FreeCAD renders)
# ===========================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FreeCAD compare</title>
<style>
  :root{--bg:#1b2230;--bar:#11161f;--line:#2c3647;--fg:#dfe6f0;--muted:#8a97ab;--accent:#4a90d9;}
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
            font:13px/1.4 system-ui,Segoe UI,Roboto,sans-serif;overflow:hidden;}
  #bar{height:46px;display:flex;align-items:center;gap:8px;padding:0 10px;background:var(--bar);
       border-bottom:1px solid var(--line);position:relative;z-index:5;}
  .nav{background:#222c3a;color:var(--fg);border:1px solid var(--line);border-radius:6px;
       width:34px;height:30px;font-size:17px;cursor:pointer;}
  .nav:hover{background:#2c3a4d;}
  #title{flex:0 1 auto;min-width:260px;max-width:48vw;text-align:center;cursor:pointer;
         background:#222c3a;border:1px solid var(--line);border-radius:6px;height:30px;display:flex;
         align-items:center;justify-content:center;gap:8px;padding:0 12px;font-weight:600;
         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  #title:hover{background:#2c3a4d;}
  #title .batch{color:var(--accent);}
  #counter{color:var(--muted);font-weight:400;font-size:12px;}
  #spacer{flex:1;}
  .ctrl{display:flex;align-items:center;gap:6px;color:var(--muted);}
  #gtSeg{display:none;border:1px solid var(--line);border-radius:6px;overflow:hidden;}
  #gtSeg button{background:#222c3a;color:var(--fg);border:0;border-left:1px solid var(--line);
                min-width:30px;height:28px;padding:0 8px;cursor:pointer;}
  #gtSeg button:first-child{border-left:0;}
  #gtSeg button.on{background:var(--accent);color:#fff;}
  #panels{display:flex;height:calc(100% - 46px);}
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
  .item .bt{color:var(--accent);width:34px;flex:none;font-variant-numeric:tabular-nums;}
  .item .st{color:var(--muted);font-size:11px;margin-left:auto;white-space:nowrap;}
  .item.has-gt .stem::after{content:"GT";font-size:9px;color:#6fae5e;margin-left:6px;
          border:1px solid #3c5a31;border-radius:6px;padding:0 4px;}
  .hint{position:absolute;left:8px;bottom:6px;color:#6b7a90;font-size:11px;pointer-events:none;}
</style></head><body>
  <div id="bar">
    <button class="nav" id="prev" title="Previous (←)">‹</button>
    <div id="title"><span id="ttext">loading…</span><span id="counter"></span></div>
    <button class="nav" id="next" title="Next (→)">›</button>
    <div id="spacer"></div>
    <label class="ctrl"><input type="checkbox" id="gtToggle"/> GT</label>
    <div id="gtSeg"></div>
    <button class="nav" id="fit" title="Fit all views">⤢</button>
  </div>
  <div id="picker"><input id="search" placeholder="Filter by batch / sketch…" autocomplete="off"/>
    <div id="list"></div></div>
  <div id="panels"></div>
<script>
const $=s=>document.querySelector(s);
let META=null,I=0,GT_ON=false,GT_V=null,RES=1500;
const viewers={};

async function boot(){
  META=await (await fetch('/api/meta')).json(); RES=META.res;
  buildPanels(); buildList(); show(0);
}
function folderLabels(){
  const a=[{k:'left',n:META.folder1},{k:'right',n:META.folder2}];
  if(GT_ON) a.push({k:'gt',n:'Ground truth'});
  return a;
}
function buildPanels(){
  const root=$('#panels'); root.innerHTML='';
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
    // reconcile dot pool
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
    setCap(r.dof);
    img.onload=()=>{ if(!cam._fitted){cam._fitted=true; fit();} else { applyImg(); drawHandles(); } showMsg('',false); };
    img.src='data:image/png;base64,'+r.png;
  }
  function setCap(dof){
    const sk=META.sketches[I]; const c=key==='gt'?null:sk[key];
    let s=''; if(key==='gt') s=`variant ${ctx.variant||'-'}`;
    else if(c) s=`${c.status||'?'}`;
    $('#cap-'+key).innerHTML = s + (dof!=null&&dof>=0?`<span class="dof">DoF ${dof}</span>`:'');
  }

  async function post(body){
    body.panel=key; body.stem=ctx.stem; if(key==='gt') body.variant=ctx.variant;
    const r=await fetch('/api/op',{method:'POST',headers:{'Content-Type':'application/json'},
                                   body:JSON.stringify(body)});
    if(!r.ok){ let d='error'; try{d=(await r.json()).detail||d;}catch(e){} throw new Error(d); }
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

  // ---- dragging: vertex deform / label move (throttled, coalesced) ----
  function sendMove(){
    if(inflight){return;} const job=queued; queued=null; if(!job) return; inflight=true;
    post(job).then(r=>{ inflight=false;
      cam={cx:r.frame.cx,cy:r.frame.cy,height:r.frame.height,w:r.frame.w,h:r.frame.h,geom_height:r.frame.geom_height};
      H=r.handles; setCap(r.dof);
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
    // move the grabbed dot immediately for responsiveness
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
function buildGtSeg(){
  const sk=META.sketches[I], seg=$('#gtSeg'); seg.innerHTML='';
  sk.gt.forEach(v=>{ const b=document.createElement('button'); b.textContent=v;
    b.classList.toggle('on',GT_ON&&GT_V===v);
    b.onclick=()=>{ GT_V=v; buildGtSeg(); if(viewers.gt) viewers.gt.load(sk.stem,GT_V); };
    seg.appendChild(b); });
  if(!sk.gt.length){ const s=document.createElement('span'); s.textContent='no GT';
    s.style.cssText='color:#8a97ab;padding:0 8px;line-height:28px'; seg.appendChild(s); }
}
function show(i){ if(!META.sketches.length)return;
  I=(i+META.sketches.length)%META.sketches.length; const sk=META.sketches[I];
  if(GT_ON && (!GT_V||!sk.gt.includes(GT_V))) GT_V=sk.gt[0]||null;
  setTitle(); if(GT_ON) buildGtSeg();
  viewers.left.load(sk.stem,null); viewers.right.load(sk.stem,null);
  if(viewers.gt) viewers.gt.load(sk.stem,GT_V);
  syncList();
}

// ---- filename / batch jump list ----
function buildList(){ const list=$('#list'); list.innerHTML='';
  META.sketches.forEach((sk,idx)=>{ const d=document.createElement('div');
    d.className='item'+(sk.gt.length?' has-gt':''); d.dataset.idx=idx;
    d.dataset.key=(sk.batch+'/'+sk.stem).toLowerCase();
    const ld=sk.left&&sk.left.dof!=null?`L DoF ${sk.left.dof}`:'';
    const rd=sk.right&&sk.right.dof!=null?` · R DoF ${sk.right.dof}`:'';
    d.innerHTML=`<span class="bt">${sk.batch}</span><span class="stem">${sk.stem}</span>
                 <span class="st">${ld}${rd}</span>`;
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
$('#fit').onclick=()=>Object.values(viewers).forEach(v=>v.fit());
$('#title').onclick=()=>$('#picker').classList.contains('open')?closePicker():openPicker();
$('#search').addEventListener('input',e=>filter(e.target.value));
$('#gtToggle').addEventListener('change',e=>{ GT_ON=e.target.checked;
  $('#gtSeg').style.display=GT_ON?'flex':'none';
  if(GT_ON){const sk=META.sketches[I]; GT_V=GT_V&&sk.gt.includes(GT_V)?GT_V:(sk.gt[0]||null);}
  buildPanels(); if(GT_ON)buildGtSeg(); show(I); });
document.addEventListener('keydown',e=>{ if(e.target.id==='search'){if(e.key==='Escape')closePicker();return;}
  if(e.key==='ArrowLeft')show(I-1); else if(e.key==='ArrowRight')show(I+1); else if(e.key==='Escape')closePicker(); });
document.addEventListener('click',e=>{ if(!$('#picker').contains(e.target)&&e.target.id!=='title'&&!$('#title').contains(e.target))closePicker(); });
window.addEventListener('resize',()=>Object.values(viewers).forEach(v=>v.fit()));
// Block the browser's native image drag-and-drop (which would otherwise hijack a
// vertex/label drag, dragging a ghost image that snaps back on release).
document.addEventListener('dragstart',e=>e.preventDefault());
boot();
</script></body></html>
"""


if __name__ == "__main__":
    main()
