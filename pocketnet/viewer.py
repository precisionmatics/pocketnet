"""Interactive visualization: py3Dmol Jupyter widget and a self-contained browser viewer."""
from __future__ import annotations
import json, os, re, tempfile, webbrowser
from pathlib import Path
from typing import Optional

COLORS = [
    "#E8743B", "#4F81BD", "#6AA84F", "#8E44AD", "#D4AC0D",
    "#C0392B", "#17A589", "#2E86C1", "#BA4A00", "#1A5276",
]


def _pdb_minimal(st) -> str:
    """Minimal ATOM PDB text from a Structure (no PKT pseudo-atoms)."""
    lines = []
    for i in range(len(st)):
        nm = st.name[i]; nm = f" {nm:<3}" if len(nm) < 4 else nm
        lines.append(
            f"ATOM  {i + 1:>5} {nm} {st.res[i]:>3} {st.chain[i][:1]:1}"
            f"{int(st.rid[i]):>4}    "
            f"{st.xyz[i, 0]:8.3f}{st.xyz[i, 1]:8.3f}{st.xyz[i, 2]:8.3f}"
            f"  1.00  0.00"
        )
    return "\n".join(lines + ["END", ""])


def _res_to_parts(res: str):
    """Split 'A:45' or 'A:45A' into (chain, numeric_resno_str)."""
    chain, resid = res.split(":", 1)
    m = re.match(r"-?\d+", resid)
    return chain, (m.group(0) if m else resid)


# ─────────────────────────────────────────────────────── py3Dmol (Jupyter) ──

def show(result, width: int = 880, height: int = 540):
    """Render protein and pockets as an interactive 3-D widget inside a Jupyter notebook.

    Requires ``pip install py3Dmol``.  In JupyterLab you may also need
    ``pip install jupyterlab-3dmol`` and a notebook reload.

    Each pocket's lining residues are coloured distinctly; translucent spheres
    mark the pocket centres.  Numbered labels float above each sphere.

    Parameters
    ----------
    result  : :class:`pocketnet.PredictionResult`
    width, height : pixel dimensions of the viewer panel.
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError("pip install py3Dmol") from None

    pdb = _pdb_minimal(result.structure)
    v = py3Dmol.view(width=width, height=height)
    v.addModel(pdb, "pdb")

    # base: light grey cartoon
    v.setStyle({}, {"cartoon": {"color": "#AAAAAA", "opacity": 0.65}})

    # per-pocket: colour residues + sphere + label
    for i, p in enumerate(result.pockets):
        col = COLORS[i % len(COLORS)]
        for res in p.residues:
            chain, rnum = _res_to_parts(res)
            v.addStyle(
                {"chain": chain, "resi": rnum},
                {"cartoon": {"color": col, "opacity": 1.0},
                 "stick":   {"color": col, "radius": 0.18, "opacity": 0.85}},
            )
        x, y, z = p.center
        v.addSphere({"center": {"x": x, "y": y, "z": z},
                     "radius": 2.2, "color": col, "opacity": 0.50})
        v.addLabel(
            f"  {p.rank} | p={p.probability:.2f}  ",
            {"position":          {"x": x, "y": y + 3.2, "z": z},
             "fontSize":          11,
             "fontColor":         "white",
             "backgroundColor":   col,
             "backgroundOpacity": 0.88,
             "borderRadius":      3,
             "padding":           2,
             "inFront":           True},
        )

    v.zoomTo()
    v.spin(False)
    return v.show()


# ──────────────────────────────────────────────────── browser popup viewer ──

def view(result, open_browser: bool = True) -> str:
    """Open a full interactive browser viewer (NGL.js) with the protein and pockets.

    Writes a self-contained HTML file to a temporary location and opens it in
    your default browser.  No server is started — everything is embedded.

    Features: cartoon + pocket colouring, translucent spheres, surface toggle,
    pocket-list sidebar with probability bars, click-to-focus, spin, screenshot.

    Returns the path to the temporary HTML file.
    """
    pdb_text = _pdb_minimal(result.structure)
    vols = result.volumes
    pocket_data = [
        {"rank": p.rank, "score": round(p.score, 3),
         "probability": round(p.probability, 3),
         "is_druggable": p.is_druggable,
         "center": p.center, "n_points": p.n_points,
         "volume": round(v, 1), "n_residues": len(p.residues),
         "residues": p.residues}
        for p, v in zip(result.pockets, vols)
    ]
    html = _build_html(pdb_text, pocket_data,
                       result.source or "structure",
                       len(result.structure), len(result.points))
    tmp = Path(tempfile.mktemp(suffix="_pocketnet.html"))
    tmp.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(tmp.as_uri())
    return str(tmp)


def _build_html(pdb_text: str, pockets: list, source: str, n_atoms: int, n_points: int) -> str:
    pdb_js     = json.dumps(pdb_text)
    pockets_js = json.dumps(pockets)
    colors_js  = json.dumps(COLORS)
    src_safe   = source.replace("'", "\\'")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PocketNet — {source}</title>
<script src="https://cdn.jsdelivr.net/npm/ngl@2.0.0-dev.37/dist/ngl.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{display:flex;flex-direction:column;height:100vh;background:#ffffff;color:#1a1a2e;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}}
/* ── header ── */
.hdr{{display:flex;align-items:center;justify-content:space-between;padding:7px 14px;background:#f5f6fa;border-bottom:1px solid #dde1ea;flex-shrink:0}}
.logo{{font-size:17px;font-weight:700;color:#2563eb;letter-spacing:-.3px}}
.src {{font-size:13px;color:#666;margin-left:8px}}
.stats{{font-size:11px;color:#999}}
.controls{{display:flex;gap:6px}}
.btn{{padding:4px 11px;border:1px solid #d0d5df;border-radius:5px;background:#ffffff;color:#444;font-size:11px;cursor:pointer;transition:all .15s}}
.btn:hover{{background:#f0f2f8;color:#1a1a2e;border-color:#aab0c0}}
.btn.on{{background:#eff6ff;color:#2563eb;border-color:#93c5fd}}
/* ── main ── */
.main{{display:flex;flex:1;overflow:hidden}}
/* ── sidebar ── */
.sb{{width:235px;background:#f9fafc;border-right:1px solid #dde1ea;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}}
.sb-title{{padding:9px 11px;font-size:10px;font-weight:600;color:#7a8aaa;text-transform:uppercase;letter-spacing:.7px;border-bottom:1px solid #dde1ea;flex-shrink:0}}
.pc-list{{flex:1;overflow-y:auto;padding:5px}}
.pc-list::-webkit-scrollbar{{width:3px}}
.pc-list::-webkit-scrollbar-thumb{{background:#d0d5df;border-radius:2px}}
.card{{border-radius:7px;padding:8px 10px;margin-bottom:4px;cursor:pointer;border:1.5px solid transparent;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,.06);transition:all .13s}}
.card:hover{{background:#f5f7fc;border-color:#c8d0e0}}
.card.sel{{border-color:var(--c);background:#f0f5ff;box-shadow:0 0 0 3px color-mix(in srgb,var(--c) 15%,transparent)}}
.c-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}}
.c-rank{{font-size:12px;font-weight:700;color:#1a1a2e;display:flex;align-items:center;gap:5px}}
.dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.pill{{font-size:9px;padding:1px 5px;border-radius:9px;font-weight:600}}
.yes{{background:#dcfce7;color:#16a34a;border:1px solid #86efac}}
.no {{background:#f3f4f6;color:#9ca3af;border:1px solid #e5e7eb}}
.bar-bg{{background:#e8eaf0;border-radius:3px;height:3.5px;margin-bottom:5px;overflow:hidden}}
.bar{{height:3.5px;border-radius:3px}}
.meta{{display:flex;gap:7px;font-size:10px;color:#6b7a9a}}
.meta span{{white-space:nowrap}}
/* ── viewport ── */
#vp{{flex:1;position:relative}}
#stage{{width:100%;height:100%}}
.hint{{position:absolute;bottom:9px;right:11px;font-size:10px;color:#bbb;pointer-events:none}}
</style>
</head>
<body>
<div class="hdr">
  <div style="display:flex;align-items:center">
    <span class="logo">PocketNet</span>
    <span class="src">{source}</span>
  </div>
  <span class="stats">{n_atoms:,} atoms &nbsp;·&nbsp; {n_points:,} surface pts &nbsp;·&nbsp; {len(pockets)} pockets</span>
  <div class="controls">
    <button class="btn" id="b-surf"    onclick="togSurf()">Surface</button>
    <button class="btn on" id="b-lbl" onclick="togLabels()">Labels</button>
    <button class="btn" id="b-spin"    onclick="togSpin()">Spin</button>
    <button class="btn" onclick="reset()">Reset</button>
    <button class="btn" onclick="screenshot()">&#128247; Save</button>
  </div>
</div>
<div class="main">
  <div class="sb">
    <div class="sb-title">Predicted pockets</div>
    <div class="pc-list" id="pc-list"></div>
  </div>
  <div id="vp">
    <div id="stage"></div>
    <div class="hint">Scroll = zoom &nbsp;·&nbsp; Drag = rotate &nbsp;·&nbsp; Right-drag = pan</div>
  </div>
</div>
<script>
const PDB  = {pdb_js};
const PKT  = {pockets_js};
const COLS = {colors_js};
const DIM  = '#d0d4de';   // colour for non-selected pockets

var stg, pComp, sphComps={{}}, surfRep, showLabels=true, spinning=false, active=null;
var pocketReps={{}};  // rank -> {{cartoon, licorice}}

function h2f(h){{var x=parseInt(h.slice(1),16);return[(x>>16&255)/255,(x>>8&255)/255,(x&255)/255]}}
function makeSele(residues){{
  return residues.map(function(r){{
    var pt=r.split(':'); return pt[1].replace(/[^\\d\\-]/g,'')+':'+pt[0];
  }}).join(' or ');
}}

// ── build sidebar ──
(function(){{
  var ul=document.getElementById('pc-list');
  PKT.forEach(function(p,i){{
    var c=COLS[i%COLS.length], pct=Math.round(p.probability*100);
    var d=document.createElement('div');
    d.className='card'; d.id='card'+p.rank; d.style.setProperty('--c',c);
    d.innerHTML='<div class="c-top">'
      +'<div class="c-rank"><div class="dot" style="background:'+c+'"></div>Pocket '+p.rank+'</div>'
      +'<span class="pill '+(p.is_druggable?'yes':'no')+'">'+(p.is_druggable?'✓ druggable':'low conf')+'</span>'
      +'</div>'
      +'<div class="bar-bg"><div class="bar" style="width:'+pct+'%;background:'+c+'"></div></div>'
      +'<div class="meta">'
      +'<span>p = '+p.probability.toFixed(2)+'</span>'
      +'<span>'+p.n_residues+' res</span>'
      +'<span>'+(p.volume>0?p.volume.toFixed(0)+' Å³':'—')+'</span>'
      +'<span>score '+p.score.toFixed(1)+'</span>'
      +'</div>';
    d.onclick=function(){{focus(p.rank);}};
    ul.appendChild(d);
  }});
}})();

// ── rebuild pocket representations with given highlight ──
function applyColors(selRank){{
  // remove old pocket reps
  var toRm=[];
  pComp.reprList.forEach(function(r){{
    if(r.name&&(r.name.slice(0,2)==='pc'||r.name.slice(0,2)==='lc'))toRm.push(r);
  }});
  toRm.forEach(function(r){{pComp.removeRepresentation(r);}});
  pocketReps={{}};

  PKT.forEach(function(p,i){{
    if(!p.residues.length)return;
    var isSelected=(selRank===null||p.rank===selRank);
    var c=isSelected?COLS[(p.rank-1)%COLS.length]:DIM;
    var opC=isSelected?0.98:0.30;
    var opL=isSelected?0.85:0.18;
    var sele=makeSele(p.residues);
    var cr=pComp.addRepresentation('cartoon', {{sele:sele,color:c,opacity:opC,name:'pc'+p.rank}});
    var lr=pComp.addRepresentation('licorice',{{sele:sele,color:c,radius:0.20,opacity:opL,name:'lc'+p.rank}});
    pocketReps[p.rank]={{cartoon:cr,licorice:lr}};
    // update sphere opacity too
    if(sphComps[p.rank])sphComps[p.rank].setVisibility(isSelected);
  }});
}}

// ── init NGL ──
document.addEventListener('DOMContentLoaded',function(){{
  stg=new NGL.Stage('stage',{{backgroundColor:'#ffffff',quality:'medium',antialias:true}});
  window.addEventListener('resize',function(){{stg.handleResize();}});

  var blob=new Blob([PDB],{{type:'text/plain'}});
  stg.loadFile(blob,{{ext:'pdb',defaultRepresentation:false}}).then(function(comp){{
    pComp=comp;

    comp.addRepresentation('cartoon',{{color:'#c8ccd8',opacity:0.72,name:'base'}});
    surfRep=comp.addRepresentation('surface',{{color:'#d0d4e0',opacity:0.14,surfaceType:'sas',visible:false,name:'surf'}});

    // initial pocket reps + spheres
    PKT.forEach(function(p,i){{
      var c=COLS[i%COLS.length];
      if(p.residues.length){{
        var sele=makeSele(p.residues);
        var cr=comp.addRepresentation('cartoon', {{sele:sele,color:c,opacity:0.98,name:'pc'+p.rank}});
        var lr=comp.addRepresentation('licorice',{{sele:sele,color:c,radius:0.20,opacity:0.85,name:'lc'+p.rank}});
        pocketReps[p.rank]={{cartoon:cr,licorice:lr}};
      }}
      var sh=new NGL.Shape('sh'+p.rank);
      sh.addSphere(p.center,h2f(c),2.0,'Pocket '+p.rank+' p='+p.probability.toFixed(2));
      var sc=stg.addComponentFromObject(sh); sc.addRepresentation('buffer',{{opacity:0.72}});
      sphComps[p.rank]=sc;
    }});

    // double-rAF: ensures canvas has correct pixel dimensions before centring
    requestAnimationFrame(function(){{
      requestAnimationFrame(function(){{
        stg.handleResize();
        comp.autoView(600);
      }});
    }});
  }});
}});

// ── interactions ──
function focus(rank){{
  if(active!==null){{
    var old=document.getElementById('card'+active);
    if(old)old.classList.remove('sel');
  }}
  active=rank;
  var card=document.getElementById('card'+rank);
  if(card){{card.classList.add('sel');card.scrollIntoView({{block:'nearest'}});}}

  applyColors(rank);

  var p=PKT[rank-1];
  if(pComp&&p.residues.length){{
    pComp.autoView(makeSele(p.residues),700);
  }}
}}

function togSurf(){{
  if(!surfRep)return;
  surfRep.setVisibility(!surfRep.visible);
  document.getElementById('b-surf').classList.toggle('on',surfRep.visible);
}}
function togLabels(){{
  showLabels=!showLabels;
  Object.values(sphComps).forEach(function(c){{c.setVisibility(showLabels);}});
  document.getElementById('b-lbl').classList.toggle('on',showLabels);
}}
function togSpin(){{
  spinning=!spinning;
  stg.setSpin(spinning?[0,1,0]:false);
  document.getElementById('b-spin').classList.toggle('on',spinning);
}}
function reset(){{
  if(active!==null){{
    var old=document.getElementById('card'+active);
    if(old)old.classList.remove('sel');
    active=null;
  }}
  applyColors(null);
  if(pComp)pComp.autoView(600);
}}
function screenshot(){{
  stg.makeImage({{factor:2,antialias:true,trim:false}}).then(function(blob){{
    var url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download='pocketnet_{src_safe}.png';a.click();URL.revokeObjectURL(url);
  }});
}}
</script>
</body>
</html>"""
