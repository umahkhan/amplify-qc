"""
Amplify QC -- qPCR plate review console.

Two independent models score every well, answering two *different* questions:

  1. Clinical call (RandomForest on curve-shape features)
     -> "Did this well amplify?"  AMP / NO AMP.  Benchmarked
     against the naive "did the trace ever cross 0.15 RFU" rule.

  2. Quality flag (3-channel conv autoencoder: raw + 1st + 2nd derivative,
     replicate-padded, scored per cycle and per window against per-class
     clean-data thresholds)
     -> "Does this curve carry an artifact, and where?"  A review flag, NOT a
     call -- a genuine positive can still carry an artifact worth a human's eyes.

Both models are pretrained (train_model.py -> qpcr_model.pkl); the app only runs
inference. Ground-truth columns in an uploaded plate are optional and only unlock
the validation section.
"""
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs
import streamlit as st
import torch

from qpcr_pipeline import benchmark, CYCLE_COLS
import qpcr_v2  # noqa: F401  (must be importable for unpickling the model)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "qpcr_model.pkl"
DEMO_PATH = HERE / "demo_plate.csv"

CLINICAL_THRESHOLD = 0.15
ROWS = list("ABCDEFGH")
REQUIRED_COLS = ["well", "row", "col"] + CYCLE_COLS

# ---- palette
NAVY = "#0A2540"
BLUE = "#2563EB"
INK = "#0F1B2D"
MUTED = "#64748B"
AMP_COLOR = "#2563EB"
NOAMP_COLOR = "#CBD5E1"
TIER_COLOR = {"HIGH": "#C62828", "REVIEW": "#F59E0B", "BORDERLINE": "#FDE68A"}
TIER_TEXT = {"HIGH": "High priority", "REVIEW": "Review", "BORDERLINE": "Borderline call", "OK": "OK"}
HOT_COLOR = "#C62828"

st.set_page_config(page_title="Amplify QC", page_icon="🧬", layout="wide")

CSS = f"""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{display:none !important;}}
[data-testid="stHeader"] {{background:transparent;height:0;}}
.stApp {{background:#F4F7FB;}}
.block-container {{padding:0.5rem 1.4rem 0.6rem 1.4rem;max-width:1800px;}}
[data-testid="stMain"] [data-testid="stVerticalBlock"] {{gap:0.6rem;}}
html, body, [class*="css"] {{font-family:Inter,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:{INK};}}

/* summary panel */
.hero {{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:20px;background:#fff;
  border:1px solid #E2E8F0;border-top:3px solid {BLUE};border-radius:12px;padding:9px 22px 10px 22px;
  box-shadow:0 1px 3px rgba(15,27,45,.06),0 6px 18px rgba(15,27,45,.05);margin-bottom:10px;}}
.hero-t {{font-size:17px;font-weight:800;letter-spacing:-.2px;color:{INK};line-height:1.15;}}
.hero-s {{font-size:12px;color:{MUTED};margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.hero-k {{display:flex;gap:52px;align-items:center;justify-content:center;}}
.hk {{border-left:4px solid {BLUE};padding-left:10px;display:flex;flex-direction:column;line-height:1.1;}}
.hk-v {{font-size:26px;font-weight:800;color:{INK};}}
.hk-l {{font-size:10.5px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{MUTED};}}

/* KPI cards */
.kpi {{background:#fff;border-radius:10px;padding:7px 14px 8px 14px;border-left:5px solid {BLUE};
  box-shadow:0 2px 10px rgba(15,27,45,.07);}}
.kpi-l {{font-size:11px;font-weight:700;letter-spacing:1.1px;text-transform:uppercase;color:{MUTED};}}
.kpi-v {{display:inline-block;font-size:26px;font-weight:800;line-height:1.1;color:{INK};}}
.kpi-s {{display:inline-block;margin-left:8px;font-size:11.5px;color:{MUTED};}}
.stTabs [data-baseweb="tab-list"] {{gap:6px;}}
.stTabs [data-baseweb="tab"] {{font-weight:700;font-size:15px;padding:6px 18px;}}

/* cards */
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {{background:#fff;border:0 !important;
  border-radius:14px;box-shadow:0 2px 12px rgba(15,27,45,.08);}}
.card-t {{font-size:12.5px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;color:{NAVY};
  margin:0 0 4px;display:flex;align-items:center;gap:8px;}}
.card-t:before {{content:"";width:4px;height:16px;background:{BLUE};border-radius:2px;display:inline-block;}}

/* pills + well detail */
.pill {{display:inline-block;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.4px;}}
.wd-head {{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}}
.wd-id {{font-size:24px;font-weight:800;letter-spacing:-.5px;}}
.wd-grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:2px 0 6px;}}
.wd-cell {{background:#F5F8FC;border-radius:8px;padding:4px 10px;}}
.wd-cell .l {{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{MUTED};}}
.wd-cell .v {{font-size:17px;font-weight:800;line-height:1.2;}}
.wd-note {{border-radius:8px;padding:7px 12px;font-size:13px;line-height:1.4;}}

/* legend */
.legend {{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:{MUTED};margin-top:2px;}}
.legend i {{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px;border:1px solid rgba(0,0,0,.15);}}

/* empty state */
.empty {{text-align:center;padding:70px 20px;background:#fff;border-radius:14px;box-shadow:0 2px 12px rgba(15,27,45,.08);}}
.empty h2 {{margin:10px 0 6px;color:{NAVY};}}
.empty p {{color:{MUTED};max-width:560px;margin:0 auto;}}

/* sidebar */
[data-testid="stSidebar"] {{background:#fff;border-right:1px solid #E2E8F0;width:250px !important;min-width:250px !important;}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {{display:none;}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{padding:8px 10px;min-height:0;background:#F1F5F9;border:1px dashed #93C5FD;}}
[data-testid="stSidebar"] button {{background:{BLUE};color:#fff;border:0;font-weight:700;border-radius:8px;}}
[data-testid="stSidebar"] button:hover {{background:#1D4ED8;}}
[data-testid="stSidebar"] button * {{color:#fff !important;}}
[data-testid="stSidebar"] small, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{color:{MUTED} !important;}}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{background:transparent;box-shadow:none;}}
.brand {{display:flex;align-items:center;gap:11px;padding:2px 0 14px 0;border-bottom:1px solid #E2E8F0;margin-bottom:4px;}}
.brand-t {{font-size:17px;font-weight:800;letter-spacing:-.2px;color:{INK};line-height:1.15;}}
.brand-s {{font-size:11px;color:{MUTED};letter-spacing:.3px;}}
[data-testid="stSidebarHeader"] {{display:none;}}
[data-testid="stSidebarContent"] {{padding-top:18px;}}
.side-h {{font-size:11px;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;color:{MUTED};margin:14px 0 6px;}}
.side-card {{background:#F1F5F9;border:1px solid #E2E8F0;border-radius:10px;padding:10px 12px;font-size:12.5px;line-height:1.6;word-break:break-all;}}
.side-card b {{color:{INK};}}

/* primary buttons in main area */
[data-testid="stDownloadButton"] button {{background:{NAVY};color:#fff;border:0;font-weight:700;border-radius:8px;}}
[data-testid="stDownloadButton"] button:hover {{background:{BLUE};color:#fff;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- model / data

def validate(df):
    """Return a list of human-readable problems with an uploaded frame."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    problems = []
    if missing:
        cyc = [c for c in missing if c in CYCLE_COLS]
        meta = [c for c in missing if c not in CYCLE_COLS]
        if meta:
            problems.append(f"Missing well-position columns: {', '.join(meta)}.")
        if cyc:
            problems.append(
                f"Missing {len(cyc)} cycle column(s) (need c1..c40). "
                f"First few missing: {', '.join(cyc[:6])}" + (" ..." if len(cyc) > 6 else ""))
    return problems


@st.cache_resource(show_spinner=False)
def load_model(path: str, mtime: float):
    """Load the pretrained pipeline. mtime is part of the cache key so retraining
    (train_model.py) is picked up without restarting the app."""
    return torch.load(path, weights_only=False)


@st.cache_data(show_spinner=False)
def score_plate(test_bytes: bytes, model_key: str, _pipe):
    """Inference only. model_key (not _pipe) is what invalidates the cache."""
    test_df = pd.read_csv(BytesIO(test_bytes))
    results, reconstructions, extras = _pipe.predict(test_df)
    return test_df, results, np.asarray(reconstructions), extras


# ---------------------------------------------------------------- small html helpers

def kpi(label, value, color, sub=""):
    return (f'<div class="kpi" style="border-left-color:{color}"><div class="kpi-l">{label}</div>'
            f'<div><span class="kpi-v">{value}</span><span class="kpi-s">{sub}</span></div></div>')


def pill(text, bg, fg="#fff"):
    return f'<span class="pill" style="background:{bg};color:{fg}">{text}</span>'


def card_title(text):
    st.markdown(f'<div class="card-t">{text}</div>', unsafe_allow_html=True)


def hero(stats, plate_name=None):
    """Summary panel: title top-left, high-level plate statistics centred."""
    kp = "".join(f'<div class="hk" style="border-left-color:{c}"><span class="hk-v">{v}</span>'
                 f'<span class="hk-l">{l}</span></div>' for v, l, c in stats)
    sub = f'<div class="hero-s">{plate_name}</div>' if plate_name else ""
    st.markdown(f'<div class="hero"><div><div class="hero-t">Summary</div>{sub}</div>'
                f'<div class="hero-k">{kp}</div><div></div></div>', unsafe_allow_html=True)


def brand():
    """Product name, top-left of the sidebar."""
    logo = ('<svg width="36" height="36" viewBox="0 0 40 40"><rect width="40" height="40" rx="10" '
            'fill="#2563EB"/><path d="M6 30 C14 30 16 28 20 20 C24 12 26 10 34 10" '
            'stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round"/></svg>')
    st.markdown(f'<div class="brand">{logo}<div><div class="brand-t">Amplify QC</div>'
                f'<div class="brand-s">qPCR curve quality control</div></div></div>', unsafe_allow_html=True)


REVIEW_TEMPLATE = r'''<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:transparent;font-family:Inter,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:#0F1B2D}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:14px;height:calc(100vh - 6px)}
.col{display:flex;flex-direction:column;gap:12px;min-height:0}
.card{background:#fff;border-radius:14px;box-shadow:0 2px 12px rgba(15,27,45,.08);padding:12px 14px;display:flex;flex-direction:column;min-height:0}
.ct{font-size:12.5px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;color:#0A2540;margin:0 0 6px;display:flex;align-items:center;gap:8px}
.ct:before{content:"";width:4px;height:16px;background:#2563EB;border-radius:2px;display:inline-block}
.bar{display:flex;align-items:center;gap:14px;margin-bottom:2px}
.seg{display:inline-flex;border:1px solid #CBD5E1;border-radius:8px;overflow:hidden}
.seg button{border:0;background:#F8FAFC;padding:6px 13px;font:600 13px inherit;font-family:inherit;color:#334155;cursor:pointer;border-right:1px solid #CBD5E1}
.seg button:last-child{border-right:0}
.seg button.on{background:#DBEAFE;color:#1D4ED8}
.hint{font-size:12px;color:#64748B}
#plot{flex:1;min-height:0}
.plate{display:grid;grid-template-columns:20px repeat(12,1fr);gap:6px;align-items:center;justify-items:center;padding:2px 2px 4px}
.ph{font-size:12.5px;color:#64748B;font-weight:600}
.w{width:100%;aspect-ratio:1;max-height:44px;max-width:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:19px;cursor:pointer;border:1px solid rgba(15,27,45,.25);user-select:none;transition:transform .07s}
.w:hover{transform:scale(1.1)}
.w.flag{border-width:2.5px}
.w.sel{box-shadow:0 0 0 2.5px #fff,0 0 0 5.5px #0A2540}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#64748B;margin-top:4px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px;vertical-align:-1px;border:1px solid rgba(0,0,0,.15)}
.wd-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.wd-id{font-size:24px;font-weight:800;letter-spacing:-.5px}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.4px;margin-left:6px}
.grid{display:grid;grid-template-columns:repeat(1,1fr);gap:8px;margin:2px 0 8px}
.cell{background:#F5F8FC;border-radius:8px;padding:4px 10px}
.cell .l{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748B}
.cell .v{font-size:17px;font-weight:800;line-height:1.2}
.note{border-radius:8px;padding:7px 12px;font-size:13px;line-height:1.4}
#tip{position:fixed;pointer-events:none;background:#0F1B2D;color:#fff;font-size:12px;line-height:1.45;padding:7px 10px;border-radius:8px;display:none;z-index:9;box-shadow:0 4px 14px rgba(0,0,0,.3);white-space:nowrap}
</style></head><body>
<div class="wrap">
  <div class="col"><div class="card" style="flex:1">
    <div class="ct">Amplification curve</div>
    <div class="bar"><span class="hint">Drag to zoom · double-click to reset · shaded band = flagged region, red circles = unusual cycles</span></div>
    <div id="plot"></div>
  </div></div>
  <div class="col">
    <div class="card"><div class="ct">Plate map — click a well <span class="hint" style="font-weight:500;letter-spacing:0;text-transform:none">(arrow keys also work)</span></div>
      <div class="plate" id="plate"></div>
      <div class="legend"><span><i style="background:#2563EB"></i>Amplified</span><span><i style="background:#CBD5E1"></i>Not amplified</span><span><i style="background:#F59E0B"></i>Review</span><span><i style="background:#C62828"></i>High priority</span><span><i style="background:#FDE68A"></i>Borderline</span></div>
    </div>
    <div class="card" id="detail"></div>
  </div>
</div>
<div id="tip"></div>
<script>/*PLOTLY*/</script>
<script>
const D = /*DATA*/null, W = D.wells, ROWS = "ABCDEFGH".split(""), BLUE = "#2563EB", RED = "#C62828";
const TIER = {HIGH:{bg:RED,fg:"#fff",text:"High priority",edge:"#7F1D1D"}, REVIEW:{bg:"#F59E0B",fg:"#0F1B2D",text:"Review",edge:"#B45309"},
  BORDERLINE:{bg:"#FDE68A",fg:"#0F1B2D",text:"Borderline call",edge:"rgba(15,27,45,.25)"}, OK:{bg:null,fg:null,text:"OK",edge:"rgba(15,27,45,.25)"}};
const CYC = Array.from({length:40},(_,i)=>i+1);
const idx = {}; W.forEach((w,i)=>idx[w.id]=i);
let cur = (D.init in idx) ? idx[D.init] : 0;
const wellEls = [], tip = document.getElementById("tip");
const flagged = w => w.tier==="HIGH" || w.tier==="REVIEW";
// Unusual cycles = cycles over their own limit near the worst spot; the flagged region spans
// from the first to the last of them (and always includes the worst cycle/window itself).
W.forEach(w=>{ w.hot=[]; w.r0=w.s0; w.r1=w.s1;
  if (flagged(w)){ CYC.forEach((c,k)=>{ if (w.ratio[k]>1 && c>=w.s0-3 && c<=w.s1+3) w.hot.push(k); });
    w.hot.forEach(k=>{ w.r0=Math.min(w.r0,CYC[k]); w.r1=Math.max(w.r1,CYC[k]); }); }
  w.reg = w.r0===w.r1 ? ""+w.r0 : w.r0+"–"+w.r1; });

// ---- plate (built once; selection only toggles a class)
const plate = document.getElementById("plate");
plate.appendChild(document.createElement("div"));
for (let c=1;c<=12;c++){ const h=document.createElement("div"); h.className="ph"; h.textContent=c; plate.appendChild(h); }
const pos = {}; W.forEach((w,i)=>pos[w.row+"|"+w.col]=i);
ROWS.forEach(r=>{
  const l=document.createElement("div"); l.className="ph"; l.textContent=r; plate.appendChild(l);
  for (let c=1;c<=12;c++){
    const i=pos[r+"|"+c], el=document.createElement("div");
    if (i===undefined){ plate.appendChild(el); continue; }
    const w=W[i], t=TIER[w.tier], amp=w.call==="AMP";
    el.className="w"+(flagged(w)?" flag":"");
    el.style.background = t.bg || (amp?BLUE:"#CBD5E1");
    el.style.borderColor = flagged(w) ? t.edge : "rgba(15,27,45,.25)";
    el.style.color = w.tier==="OK" ? (amp?"#fff":"#475569") : t.fg;
    el.textContent = amp ? "+" : "–";
    el.onclick = ()=>select(i);
    el.onmousemove = e=>{
      tip.innerHTML = "<b>"+w.id+"</b><br>Call: "+w.call+"<br>Review: "+t.text+(flagged(w)?"<br>Unusual signal: cycles "+w.reg:"");
      tip.style.display="block"; tip.style.left=Math.min(e.clientX+14, innerWidth-230)+"px"; tip.style.top=(e.clientY+14)+"px"; };
    el.onmouseleave = ()=>tip.style.display="none";
    wellEls[i]=el; plate.appendChild(el);
  }
});

// ---- curve (Plotly.react: swaps data in place, no re-layout of the page)
function drawCurve(){
  const w=W[cur], anom=flagged(w), hot=w.hot;
  const traces=[{x:CYC,y:w.raw,mode:"lines+markers",name:"Fluorescence",line:{color:BLUE,width:3},marker:{size:6},
                 hovertemplate:"cycle %{x}<br>%{y:.3f} RFU<extra></extra>"}];
  if (hot.length) traces.push({x:hot.map(k=>CYC[k]),y:hot.map(k=>w.raw[k]),mode:"markers",name:"Unusual cycle",
    marker:{color:"rgba(0,0,0,0)",size:15,line:{color:RED,width:2.5}},customdata:hot.map(k=>w.ratio[k]),
    hovertemplate:"cycle %{x}: %{customdata:.1f}× its normal limit<extra></extra>"});
  const shapes=[];
  const ann=[];
  if (anom){ shapes.push({type:"rect",xref:"x",yref:"paper",x0:w.r0-0.5,x1:w.r1+0.5,y0:0,y1:1,fillcolor:"rgba(198,40,40,.14)",line:{width:0},layer:"below"});
    const right=w.r0>22; ann.push({xref:"x",x:right?w.r1+0.5:w.r0-0.5,yref:"paper",y:1,text:"unusual region · cycle"+(w.r0===w.r1?" ":"s ")+w.reg,showarrow:false,xanchor:right?"right":"left",yanchor:"top",font:{size:11,color:RED}}); }
  const t=TIER[w.tier], title="Well "+w.id+" · "+w.call+(anom?" · "+t.text:"");
  Plotly.react("plot", traces, {title:{text:title,font:{size:16,color:"#0A2540"},x:0,y:0.98}, margin:{l:50,r:10,t:62,b:44},
    plot_bgcolor:"#fff", paper_bgcolor:"rgba(0,0,0,0)", legend:{orientation:"h",yanchor:"bottom",y:1.0,xanchor:"left",x:0},
    dragmode:"zoom", hovermode:"closest", uirevision:w.id, shapes:shapes, annotations:ann,
    xaxis:{title:"Cycle",range:[0.5,40.5],gridcolor:"#EDF1F7",zeroline:false}, yaxis:{title:"RFU",autorange:true,gridcolor:"#EDF1F7",zeroline:false}},
    {displaylogo:false,responsive:true,scrollZoom:false,modeBarButtons:[["zoom2d","pan2d","zoomIn2d","zoomOut2d","resetScale2d"]]});
}

// ---- detail card
function drawDetail(){
  const w=W[cur], t=TIER[w.tier], amp=w.call==="AMP", fl=flagged(w);
  const callPill='<span class="pill" style="background:'+(amp?BLUE:"#64748B")+';color:#fff">'+w.call+'</span>';
  const rbg = w.tier==="OK" ? "#16A34A" : t.bg, rfg = (w.tier==="REVIEW"||w.tier==="BORDERLINE") ? "#0F1B2D" : "#fff";
  const revPill='<span class="pill" style="background:'+rbg+';color:'+rfg+'">'+t.text.toUpperCase()+'</span>';
  let nbg,nfg,note;
  if (fl){ nbg = w.tier==="HIGH"?"#FDECEC":"#FEF3C7"; nfg = w.tier==="HIGH"?"#7F1D1D":"#78350F";
    note="<b>Unusual signal at cycle"+(w.r0===w.r1?"":"s")+" "+w.reg+"</b> — "+(w.tier==="HIGH"?"well outside normal. Check before reporting.":"just over the review limit. Worth a quick look."); }
  else if (w.tier==="BORDERLINE"){ nbg="#FEF9C3"; nfg="#713F12"; note="Borderline amplification call. Consider a repeat."; }
  else { nbg="#DCFCE7"; nfg="#14532D"; note="No unusual signal. The curve looks like a normal qPCR trace."; }
  const cells=[["Unusual cycles",fl?w.reg:"—"]];
  document.getElementById("detail").innerHTML='<div class="wd-head"><div class="wd-id">'+w.id+'</div><div>'+callPill+revPill+'</div></div>'+
    '<div class="grid">'+cells.map(c=>'<div class="cell"><div class="l">'+c[0]+'</div><div class="v">'+c[1]+'</div></div>').join("")+'</div>'+
    '<div class="note" style="background:'+nbg+';color:'+nfg+'">'+note+'</div>';
}

function select(i){
  if (wellEls[cur]) wellEls[cur].classList.remove("sel");
  cur=i; wellEls[cur].classList.add("sel"); drawDetail(); drawCurve();
}
document.addEventListener("keydown", e=>{
  const w=W[cur], r=ROWS.indexOf(w.row), c=w.col; let nr=r,nc=c;
  if (e.key==="ArrowRight") nc=Math.min(12,c+1); else if (e.key==="ArrowLeft") nc=Math.max(1,c-1);
  else if (e.key==="ArrowDown") nr=Math.min(7,r+1); else if (e.key==="ArrowUp") nr=Math.max(0,r-1); else return;
  const j=pos[ROWS[nr]+"|"+nc]; if (j!==undefined){ e.preventDefault(); select(j); }
});
select(cur);
</script></body></html>'''

REVIEW_HEIGHT = 650


@st.cache_resource(show_spinner=False)
def plotly_js():
    return get_plotlyjs()


def review_html(test_df, results, extras, init_well):
    """Plate map + curve + detail card as ONE client-side component. All wells'
    data is embedded, so selecting a well swaps the curve in the browser with no
    Streamlit rerun (the previous server-side version needed two reruns per click)."""
    ratio = extras["cycle_ratio"]
    wells = []
    for i in range(len(results)):
        r = results.iloc[i]
        wells.append(dict(
            id=str(r["well"]), row=str(r["row"]), col=int(r["col"]), call=str(r["call"]),
            conf=round(float(r["confidence"]), 4), tier=str(r["review_flag"]), region=str(r["worst_region"]),
            s0=int(r["span_start"]), s1=int(r["span_end"]), sev=round(float(r["severity"]), 3),
            raw=[round(float(v), 4) for v in test_df.iloc[i][CYCLE_COLS]],
            ratio=[round(float(v), 3) for v in ratio[i]]))
    return (REVIEW_TEMPLATE.replace("/*PLOTLY*/", plotly_js())
            .replace("null, W = D.wells", json.dumps(dict(wells=wells, init=init_well)) + ", W = D.wells"))


# ---------------------------------------------------------------- app

def main():
    if not MODEL_PATH.exists():
        st.error("No pretrained model found. Run `python train_model.py` once to create `qpcr_model.pkl`.")
        st.stop()
    pipe = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    meta = pipe.meta
    model_key = f"{meta['trained_at']}|{MODEL_PATH.stat().st_mtime}"

    with st.sidebar:
        brand()
        st.markdown('<div class="side-h">Plate data</div>', unsafe_allow_html=True)
        test_file = st.file_uploader("Upload plate CSV (one row per well)", type="csv",
                                     label_visibility="collapsed")
        if DEMO_PATH.exists() and st.button("Load demo plate", use_container_width=True):
            st.session_state.use_demo = True
        loaded_slot = st.empty()
        st.caption("Required columns: well, row, col, c1…c40")

    if st.query_params.get("demo") == "1":  # shareable link that opens straight onto the demo plate
        st.session_state.use_demo = True
    test_bytes, plate_name = None, None
    if test_file is not None:
        test_bytes, plate_name = test_file.getvalue(), test_file.name
    elif st.session_state.get("use_demo") and DEMO_PATH.exists():
        test_bytes, plate_name = DEMO_PATH.read_bytes(), DEMO_PATH.name

    if plate_name:
        loaded_slot.markdown(f'<div class="side-card">📄 <b>{plate_name}</b></div>', unsafe_allow_html=True)

    if test_bytes is None:
        st.markdown(
            '<div class="empty"><div style="font-size:46px">🧪</div><h2>Upload a plate to begin</h2>'
            '<p>Drop a plate CSV in the sidebar (one row per well, columns <code>well</code>, <code>row</code>, '
            '<code>col</code>, <code>c1</code>…<code>c40</code>). Every well is called AMP / NO AMP and checked '
            'for unusual signal, with the suspect cycles highlighted for review.</p></div>', unsafe_allow_html=True)
        st.stop()

    try:
        raw_test_df = pd.read_csv(BytesIO(test_bytes))
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    problems = validate(raw_test_df)
    if problems:
        st.error("This file can't be scored:\n\n" + "\n".join(f"- {p}" for p in problems))
        st.stop()

    test_df, results, reconstructions, extras = score_plate(test_bytes, model_key, pipe)
    test_df = test_df.reset_index(drop=True)
    results = results.reset_index(drop=True)
    cut = float(extras["flag_cut"])
    results["severity"] = results["anomaly_score"] / cut
    well_to_pos = {w: i for i, w in enumerate(test_df["well"])}
    all_wells = list(test_df["well"])
    has_truth = "true_label" in test_df.columns


    if "selected_well" not in st.session_state or st.session_state.selected_well not in well_to_pos:
        st.session_state.selected_well = all_wells[0]
    st.session_state.setdefault("_plate_tok", None)
    st.session_state.setdefault("_table_tok", None)

    # ---- header with inline KPIs
    n_total = len(results)
    n_amp = int((results["call"] == "AMP").sum())
    n_high = int((results["review_flag"] == "HIGH").sum())
    n_review = n_high + int((results["review_flag"] == "REVIEW").sum())
    hero([(n_total, "Wells", "#94A3B8"), (n_amp, "Amplified", AMP_COLOR),
          (n_total - n_amp, "Not amplified", "#CBD5E1"),
          (n_review, "Needs review", TIER_COLOR["REVIEW"]), (n_high, "High priority", TIER_COLOR["HIGH"])],
         plate_name)

    tab_review, tab_data = st.tabs(["Plate review", "Plate data"])

    # ---- PLATE DATA tab (built first in code so its row selection is read this run)
    with tab_data:
        with st.container(border=True):
            card_title("Well results")
            f1, f2, f3 = st.columns([3, 2, 5])
            show = f1.segmented_control("Show", ["All", "Review", "High"], default="All",
                                        label_visibility="collapsed") or "All"
            call_filter = f2.selectbox("Call", ["All calls", "AMP", "NO AMP"], label_visibility="collapsed")
            f3.caption("Wells in plate order · select a row to open it on the Plate review tab.")

            view = results.copy()
            if show == "Review":
                view = view[view["review_flag"].isin(["HIGH", "REVIEW"])]
            elif show == "High":
                view = view[view["review_flag"] == "HIGH"]
            if call_filter != "All calls":
                view = view[view["call"] == call_filter]
            # plate order: A1, A2 ... A12, B1 ... H12 (column headers can still re-sort)
            view = view.assign(_r=view["row"].map(ROWS.index), _c=view["col"].astype(int)).sort_values(["_r", "_c"])
            view = view[["well", "call", "review_flag", "worst_region"]].reset_index(drop=True)
            view.loc[~view["review_flag"].isin(["HIGH", "REVIEW"]), "worst_region"] = "—"  # only flagged wells have cycles to check
            view["review_flag"] = view["review_flag"].map({"HIGH": "High", "REVIEW": "Review", "BORDERLINE": "Borderline", "OK": "OK"})

            table_event = st.dataframe(
                view, hide_index=True, use_container_width=True, height=470,
                on_select="rerun", selection_mode="single-row", key="results_table",
                column_config={
                    "well": "Well", "call": "Call",
                    "review_flag": "Review", "worst_region": "Worst cycles",
                })
            st.download_button("⬇ Download results CSV", data=results.to_csv(index=False).encode(),
                               file_name="qpcr_results.csv", mime="text/csv")

        if has_truth:
            with st.expander("Validation against ground truth"):
                bm = benchmark(results, test_df, verbose=False)
                if bm is not None:
                    st.markdown("**Clinical call: ML classifier vs. naive threshold caller**")
                    st.table(pd.DataFrame({
                        "Sensitivity": [bm["sens_ml"], bm["sens_naive"]],
                        "Specificity": [bm["spec_ml"], bm["spec_naive"]],
                    }, index=["ML classifier", "Naive threshold"]).round(3))
                    d1, d2 = st.columns(2)
                    d1.metric("Δ Sensitivity", f"{bm['d_sens']:+.1f} pts")
                    d2.metric("Δ Specificity", f"{bm['d_spec']:+.1f} pts")
                    if "anomaly_precision" in bm:
                        st.markdown("**Artifact detection**")
                        a1, a2 = st.columns(2)
                        a1.metric("Precision", f"{bm['anomaly_precision']:.3f}")
                        a2.metric("Recall", f"{bm['anomaly_recall']:.3f}")
                        st.caption(f"TP={bm['anomaly_tp']}  FP={bm['anomaly_fp']}  FN={bm['anomaly_fn']}")

    # ---- a row picked in the Plate data tab becomes the starting well of the review view
    rows = table_event.selection.rows if table_event and table_event.selection else []
    if rows and len(view) and 0 <= rows[0] < len(view):
        table_well = view.iloc[rows[0]]["well"]
        if table_well != st.session_state._table_tok:
            st.session_state._table_tok = table_well
            st.session_state.selected_well = table_well

    with tab_review:
        components.html(review_html(test_df, results, extras, st.session_state.selected_well),
                        height=REVIEW_HEIGHT, scrolling=False)


if __name__ == "__main__" or st.runtime.exists():
    main()
