"""CyberBoard-AI: Streamlit dashboard for the AI advisory board agent."""

import streamlit as st
import streamlit.components.v1 as components
import json
import time
import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.pipeline.vector_store import confidence_from_result

_FAVICON = Path(__file__).parent / "assets" / "cyberboard_favicon.png"
st.set_page_config(
    page_title="CyberBoard-AI",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "🏛️",
    layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, [data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
}

/* App background: dark with emerald radial glow (liquid glass base) */
.stApp, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(1000px 620px at 88% -8%, rgba(16,185,129,0.30), transparent 60%),
      radial-gradient(760px 520px at 2% 12%, rgba(5,150,105,0.16), transparent 55%),
      linear-gradient(160deg, #051510 0%, #030C08 55%, #010604 100%);
    background-attachment: fixed;
    color: #E9F6EF;
}
[data-testid="stHeader"] { background: transparent; }

/* Hero banner - liquid glass */
.cb-hero {
    position: relative;
    background: linear-gradient(180deg, rgba(255,255,255,0.08), transparent 42%), linear-gradient(135deg, rgba(16,185,129,0.20) 0%, rgba(6,78,59,0.30) 55%, rgba(2,20,14,0.55) 100%);
    backdrop-filter: blur(26px) saturate(145%);
    -webkit-backdrop-filter: blur(26px) saturate(145%);
    border: 1px solid rgba(52,211,153,0.26);
    border-radius: 26px;
    padding: 30px 34px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.20), 0 22px 60px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.22);
    overflow: hidden;
}
.cb-hero::before {
    content:""; position:absolute; inset:0;
    background: radial-gradient(620px 130px at 82% -34%, rgba(52,211,153,0.38), transparent 70%);
    pointer-events:none;
}
.cb-hero h1 {
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    padding: 0;
    position: relative;
}
.cb-hero h1 .gold {
    background: linear-gradient(90deg, #6EE7B7, #22C55E);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.cb-hero p {
    color: #A7CFBE;
    margin: 8px 0 0 0;
    font-size: 1rem;
    letter-spacing: 0.2px;
    position: relative;
}

/* Result cards - frosted glass */
.cb-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), transparent 46%), rgba(15,32,25,0.50);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border: 1px solid rgba(52,211,153,0.14);
    border-left: 3px solid #22C55E;
    border-radius: 20px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.20), 0 14px 38px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.12);
    transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
}
.cb-card:hover { border-left-color: #6EE7B7; transform: translateY(-2px); box-shadow: 0 12px 34px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.12); }
.cb-card .cb-meta { margin-bottom: 8px; }
.cb-chip {
    display: inline-block;
    background: rgba(52,211,153,0.10);
    border: 1px solid rgba(52,211,153,0.30);
    color: #7FE7B4;
    border-radius: 999px;
    padding: 3px 13px;
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 8px;
    backdrop-filter: blur(6px);
}
.cb-chip.ticker { background: linear-gradient(135deg, #6EE7B7, #22C55E); color: #05231A; font-weight: 800; border: none; }
.cb-card .cb-text {
    color: #CDE7DA;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-top: 8px;
}

/* Confidence pills - glass */
.cb-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 13px;
    font-size: 0.76rem;
    font-weight: 700;
    float: right;
    backdrop-filter: blur(6px);
}
.cb-pill.high { background: rgba(34,197,94,0.16); color: #4ADE80; border: 1px solid rgba(34,197,94,0.45); }
.cb-pill.mid  { background: rgba(250,204,21,0.12); color: #FACC15; border: 1px solid rgba(250,204,21,0.40); }
.cb-pill.low  { background: rgba(248,113,113,0.12); color: #F87171; border: 1px solid rgba(248,113,113,0.40); }

/* Trace stage cards - glass */
.cb-stage {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), transparent 46%), rgba(11,26,20,0.52);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(52,211,153,0.12);
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.16), 0 8px 26px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.09);
    animation: cb-stage-in .5s ease both;
    transition: border-color .2s ease, box-shadow .2s ease;
}
.cb-stage:hover { border-color: rgba(52,211,153,0.32); box-shadow: 0 1px 2px rgba(0,0,0,0.16), 0 12px 32px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.12); }
@keyframes cb-stage-in { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform: translateY(0); } }
/* trace rerank table: hover-highlight rows to inspect a passage's score */
[data-testid="stExpander"] table tbody tr { transition: background .15s ease; }
[data-testid="stExpander"] table tbody tr:hover { background: rgba(52,211,153,0.10); }
.cb-stage .cb-stage-title {
    color: #34D399;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.cb-stage .cb-stage-body { color: #CDE7DA; font-size: 0.85rem; }

/* Sidebar - glass */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(6,22,15,0.78) 0%, rgba(2,10,7,0.88) 100%);
    backdrop-filter: blur(26px) saturate(130%);
    -webkit-backdrop-filter: blur(26px) saturate(130%);
    border-right: 1px solid rgba(52,211,153,0.16);
}

/* Inputs - glass pill */
.stTextInput input, .stTextArea textarea {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent 40%), rgba(11,26,20,0.55) !important;
    border: 1px solid rgba(52,211,153,0.20) !important;
    border-radius: 16px !important;
    color: #E9F6EF !important;
    backdrop-filter: blur(14px) saturate(140%);
    box-shadow: 0 8px 24px rgba(0,0,0,0.20), inset 0 1px 0 rgba(255,255,255,0.06);
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #34D399 !important;
    box-shadow: 0 0 0 3px rgba(52,211,153,0.18) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #6E8F80; }

/* Primary button - emerald gradient */
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.32), transparent 46%), linear-gradient(135deg, #6EE7B7 0%, #22C55E 60%, #059669 100%);
    color: #04120C;
    font-weight: 700;
    border: none;
    border-radius: 999px;
    padding: 10px 34px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.18), 0 8px 24px rgba(34,197,94,0.38), inset 0 1px 0 rgba(255,255,255,0.55);
    transition: transform .15s ease, box-shadow .2s ease;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 12px 30px rgba(34,197,94,0.48); color: #04120C; }
.stButton > button[kind="primary"]:disabled { background: rgba(52,211,153,0.20); color: #6E8F80; box-shadow: none; }

/* Secondary button - glass suggestion card */
.stButton > button[kind="secondary"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), transparent 46%), rgba(15,32,25,0.48);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(52,211,153,0.14);
    color: #DCEFE6;
    border-radius: 20px;
    padding: 18px 20px;
    min-height: 92px;
    font-weight: 500;
    font-size: 0.94rem;
    line-height: 1.4;
    text-align: left;
    box-shadow: 0 1px 2px rgba(0,0,0,0.18), 0 10px 30px rgba(0,0,0,0.24), inset 0 1px 0 rgba(255,255,255,0.10);
    transition: transform .2s ease, border-color .2s ease, background .2s ease, box-shadow .2s ease;
}
.stButton > button[kind="secondary"] p { text-align: left; width: 100%; }
.stButton > button[kind="secondary"]:hover {
    border-color: #34D399;
    background: rgba(18,44,33,0.62);
    transform: translateY(-3px);
    color: #FFFFFF;
    box-shadow: 0 12px 30px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.10);
}

/* Selectbox - glass */
[data-baseweb="select"] > div {
    background: rgba(10,24,18,0.60) !important;
    border: 1px solid rgba(52,211,153,0.22) !important;
    border-radius: 12px !important;
}

/* ===== Greeting ===== */
.cb-greet { color:#FFFFFF; font-size:2.1rem; font-weight:800; letter-spacing:-0.5px; text-align:center; margin:6px 0 0; }
.cb-greet span { background:linear-gradient(90deg,#6EE7B7,#22C55E); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.cb-greet-sub { color:#A7CFBE; text-align:center; font-size:1rem; margin:6px 0 22px; }
.cb-hint { color:#8FB3A3; font-size:0.82rem; margin:16px 0 8px; text-transform:uppercase; letter-spacing:1.4px; }

/* Metric - glass tiles */
[data-testid="stMetricValue"] { color: #34D399; }
[data-testid="stMetric"] {
    background: rgba(10,24,18,0.45);
    border: 1px solid rgba(52,211,153,0.14);
    border-radius: 14px;
    padding: 12px 16px;
    backdrop-filter: blur(10px);
}

/* Slider accent */
[data-testid="stSlider"] [role="slider"] { background-color: #34D399 !important; }

/* Checkbox accent - emerald when checked */
[data-testid="stCheckbox"] label:has(input:checked) > span:first-child {
    background-color: #34D399 !important;
    border-color: #34D399 !important;
}

/* Slider value label + filled track - emerald */
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * { color: #34D399 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="255, 75, 75"] { background: #34D399 !important; }

/* Tabs - emerald active state */
.stTabs button[data-baseweb="tab"][aria-selected="true"],
.stTabs button[data-baseweb="tab"][aria-selected="true"] * { color: #34D399 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #34D399 !important; }
.stTabs button[data-baseweb="tab"]:hover { color: #6EE7B7 !important; }

/* Links + spinner + progress - emerald */
.stMarkdown a, .stMarkdown a:visited { color: #34D399 !important; }
.stSpinner > div { border-top-color: #34D399 !important; }
[data-testid="stProgress"] div[role="progressbar"] > div { background-color: #34D399 !important; }

/* General text tone */
.stMarkdown p, .stMarkdown li, label, .stCaption, [data-testid="stCaptionContainer"] { color: #C6E3D6; }

/* ===== Minimalist sidebar ===== */
[data-testid="stSidebar"] > div:first-child { padding-top: 26px; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #6E9484 !important;
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: 3px; margin-top: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    padding: 9px 12px;
    border-radius: 12px;
    margin: 0;
    width: 100%;
    transition: background .18s ease, box-shadow .18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: rgba(52,211,153,0.07); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: rgba(52,211,153,0.13);
    box-shadow: inset 2px 0 0 #34D399;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p { color: #FFFFFF !important; font-weight: 600; }
[data-testid="stSidebar"] hr { border-color: rgba(52,211,153,0.14); margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

TICKERS = list(config.COMPANIES.keys())


def hero(title_white: str, title_gold: str, subtitle: str):
    st.markdown(
        f'<div class="cb-hero"><h1>{title_white} <span class="gold">{title_gold}</span></h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True)


_MASCOT_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;padding:0;background:transparent;overflow:hidden;}
  .wrap{display:flex;justify-content:center;align-items:center;height:196px;}
  .mascot{width:150px;height:150px;position:relative;cursor:pointer;animation:float 4.6s ease-in-out infinite;-webkit-tap-highlight-color:transparent;}
  @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
  .beam{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:300px;height:3px;border-radius:50%;
    background:linear-gradient(90deg,transparent,rgba(110,231,183,.15),rgba(52,211,153,.85),rgba(110,231,183,.15),transparent);
    filter:blur(1px);box-shadow:0 0 28px 7px rgba(52,211,153,.35);animation:beam 4.2s ease-in-out infinite;}
  @keyframes beam{0%,100%{opacity:.55;width:240px}50%{opacity:1;width:320px}}
  .halo{position:absolute;inset:-24%;border-radius:50%;
    background:radial-gradient(circle,rgba(52,211,153,.42),rgba(52,211,153,.10) 45%,transparent 70%);
    filter:blur(7px);opacity:.7;animation:breathe 3.6s ease-in-out infinite;transition:opacity .3s;}
  @keyframes breathe{0%,100%{transform:scale(.9);opacity:.5}50%{transform:scale(1.1);opacity:.85}}
  .face{position:relative;width:150px;height:150px;transform-origin:center;transition:transform .25s ease;}
  .mascot.hovering .face{animation:wobble 1.1s ease-in-out infinite;}
  .mascot.hovering .halo{opacity:1;}
  @keyframes wobble{0%,100%{transform:rotate(-5deg)}50%{transform:rotate(5deg)}}
  .mascot.smiling .face{animation:bounce .55s ease;}
  .mascot.smiling .halo{opacity:1;}
  @keyframes bounce{0%{transform:scale(1)}30%{transform:scale(1.14)}55%{transform:scale(.95)}100%{transform:scale(1)}}
  .eyes{transition:transform .15s ease;}
  .eye{transform-box:fill-box;transform-origin:center;animation:blink 5s infinite;transition:transform .2s ease;}
  @keyframes blink{0%,92%,100%{transform:scaleY(1)}96%{transform:scaleY(.1)}}
  .mascot.smiling .eye{animation:none;transform:scaleY(.35) translateY(-7px);}
  .mouth{opacity:0;fill:none;stroke:#05231A;stroke-width:5;stroke-linecap:round;transition:opacity .2s ease;}
  .mascot.smiling .mouth{opacity:1;}
  /* thinking state (while the board reviews the filings) */
  .mascot.thinking .face{animation:think-rock 1.5s ease-in-out infinite;}
  @keyframes think-rock{0%,100%{transform:rotate(-4deg)}50%{transform:rotate(4deg)}}
  .mascot.thinking .eyes{animation:scan 1.1s ease-in-out infinite;}
  @keyframes scan{0%,100%{transform:translateX(-4px)}50%{transform:translateX(4px)}}
  .mascot.thinking .eye{animation:none;}
  .mascot.thinking .halo{opacity:1;animation:think-pulse .9s ease-in-out infinite;}
  @keyframes think-pulse{0%,100%{opacity:.55;transform:scale(.95)}50%{opacity:1;transform:scale(1.14)}}
  @media (prefers-reduced-motion:reduce){.mascot,.halo,.beam,.eye,.face,.eyes{animation:none!important;}}
</style></head><body>
<div class="wrap"><div class="mascot" id="m" title="Say hi to your AI board member">
  <div class="beam"></div><div class="halo"></div>
  <div class="face"><svg viewBox="0 0 148 148" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
    <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#A7F3D0"/><stop offset=".5" stop-color="#34D399"/><stop offset="1" stop-color="#059669"/>
    </linearGradient></defs>
    <g transform="rotate(45 74 74)">
      <rect x="36" y="36" width="76" height="76" rx="20" fill="url(#g)"/>
      <rect x="36" y="36" width="76" height="76" rx="20" fill="none" stroke="rgba(255,255,255,.4)" stroke-width="1.5"/>
    </g>
    <g class="eyes" id="eyes">
      <rect class="eye" x="60" y="64" width="8" height="20" rx="4" fill="#05231A"/>
      <rect class="eye" x="80" y="64" width="8" height="20" rx="4" fill="#05231A"/>
    </g>
    <path class="mouth" d="M61 90 Q74 103 87 90"/>
  </svg></div>
</div></div>
<script>
  const m=document.getElementById('m'), eyes=document.getElementById('eyes');
  m.addEventListener('mouseenter',()=>{ if(!m.classList.contains('thinking')) m.classList.add('hovering'); });
  m.addEventListener('mouseleave',()=>{m.classList.remove('hovering');eyes.style.transform='';});
  m.addEventListener('mousemove',e=>{
    if(m.classList.contains('thinking'))return;
    const r=m.getBoundingClientRect();
    const dx=(e.clientX-(r.left+r.width/2))/(r.width/2);
    const dy=(e.clientY-(r.top+r.height/2))/(r.height/2);
    eyes.style.transform='translate('+(dx*4).toFixed(1)+'px,'+(dy*3).toFixed(1)+'px)';
  });
  m.addEventListener('click',()=>{
    if(m.classList.contains('thinking'))return;
    m.classList.remove('smiling'); void m.offsetWidth; m.classList.add('smiling');
    clearTimeout(m._t); m._t=setTimeout(()=>m.classList.remove('smiling'),2000);
  });
  // react to the board working: watch the parent page for Streamlit's spinner
  let wasThinking=false;
  setInterval(()=>{
    let running=false;
    try{ running=!!parent.document.querySelector('[data-testid="stSpinner"]'); }catch(e){}
    if(running){
      if(!m.classList.contains('thinking')){ m.classList.add('thinking'); m.classList.remove('hovering','smiling'); eyes.style.transform=''; }
      wasThinking=true;
    }else{
      if(m.classList.contains('thinking')) m.classList.remove('thinking');
      if(wasThinking){ wasThinking=false; m.classList.add('smiling'); clearTimeout(m._t); m._t=setTimeout(()=>m.classList.remove('smiling'),1600); }
    }
  },180);
</script>
</body></html>"""


def mascot():
    """Interactive AI board member: floats, eyes follow the cursor, wobbles on hover, smiles on click."""
    components.html(_MASCOT_HTML, height=200)


def callout(text: str):
    """Emerald glass info callout (replaces st.info so it matches the theme)."""
    st.markdown(
        '<div style="background:linear-gradient(180deg, rgba(52,211,153,0.10), rgba(52,211,153,0.04));'
        'border:1px solid rgba(52,211,153,0.22); border-left:3px solid #34D399; border-radius:14px;'
        'padding:14px 18px; color:#C6E3D6; font-size:0.9rem; line-height:1.6; backdrop-filter:blur(12px);'
        'box-shadow:inset 0 1px 0 rgba(255,255,255,0.06); margin-bottom:12px;">' + text + '</div>',
        unsafe_allow_html=True)


def emerald_bar(data: list, cat: str, val: str, title: str, tooltip=None):
    """Interactive horizontal emerald bar chart (hover for tooltips)."""
    import altair as alt
    import pandas as pd
    chart = (
        alt.Chart(pd.DataFrame(data))
        .mark_bar(color="#34D399", cornerRadius=3)
        .encode(
            x=alt.X(f"{val}:Q", title="Hit rate (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y(f"{cat}:N", sort="-x", title=None),
            tooltip=tooltip or [cat, val],
        )
        .properties(title=title, height=max(150, len(data) * 30))
    )
    st.altair_chart(chart, use_container_width=True)


def confidence_pill(confidence: float) -> str:
    cls = "high" if confidence >= 0.7 else ("mid" if confidence >= 0.4 else "low")
    return f'<span class="cb-pill {cls}">{confidence:.0%} confidence</span>'


def result_card(meta: dict, confidence: float, text: str):
    import html
    snippet = html.escape(text[:400] + ("..." if len(text) > 400 else ""))
    section = html.escape(meta["section"].replace("_", " "))
    st.markdown(
        f'<div class="cb-card">'
        f'<div class="cb-meta">'
        f'<span class="cb-chip ticker">{meta["ticker"]}</span>'
        f'<span class="cb-chip">{meta["filing_type"]} · {meta["filing_date"]}</span>'
        f'<span class="cb-chip">{section}</span>'
        f'{confidence_pill(confidence)}'
        f'</div>'
        f'<div class="cb-text">{snippet}</div>'
        f'</div>',
        unsafe_allow_html=True)


@st.cache_resource
def load_store():
    from src.pipeline.vector_store import HybridStore
    store = HybridStore()
    if not store.bm25:
        store.rebuild_bm25_from_collection()
    return store


@st.cache_resource
def load_ajyad_store():
    """Separate demo collection (Ajyad Capital annual reports) — not part of the thesis's formal evaluation corpus."""
    from src.pipeline.vector_store import HybridStore
    store = HybridStore(collection_name=config.AJYAD_DEMO_COLLECTION_NAME)
    if not store.bm25:
        store.rebuild_bm25_from_collection()
    return store


def retrieve(query: str, ticker: str, k: int = 5, with_trace: bool = False):
    store = load_store()
    query_emb = None
    if config.gemini_available():
        from src.pipeline.embedder import embed_batch, get_client
        try:
            query_emb = embed_batch([query], get_client())[0]
        except Exception:
            # Fall back to sparse-only search if the embedding call fails
            # (e.g. transient provider outage); no need to hard-fail the query.
            pass
    trace = [] if with_trace else None
    results = store.search(query, query_embedding=query_emb, k=k, ticker=ticker, trace=trace)
    if with_trace:
        return results, trace
    return results


def render_trace(trace: list):
    """Reasoning trace visualization: how the pipeline produced these results."""
    for i, step in enumerate(trace):
        stage = step.get("stage", "?")
        delay = f'style="animation-delay:{i*0.15:.2f}s"'
        if step.get("skipped"):
            st.markdown(f'<div class="cb-stage" {delay}><div class="cb-stage-title">{stage}</div>'
                        f'<div class="cb-stage-body">Skipped</div></div>', unsafe_allow_html=True)
            continue
        with st.container():
            st.markdown(f'<div class="cb-stage" {delay}><div class="cb-stage-title">{stage}</div></div>',
                        unsafe_allow_html=True)
            if "sparse_candidates" in step:
                cols = st.columns(3)
                cols[0].metric("Sparse candidates (BM25)", step["sparse_candidates"])
                cols[1].metric("Dense candidates", step["dense_candidates"],
                               help="0 when Gemini embeddings are not configured")
                cols[2].metric("Company filter", step["ticker_filter"])
                if step["top_sparse"]:
                    st.caption("Top BM25 matches: " + ", ".join(
                        f"{t['id']} ({t['bm25']})" for t in step["top_sparse"][:3]))
            elif "alpha" in step:
                st.caption(f"Fusion: {step['formula']}")
                cols = st.columns(2)
                cols[0].metric("Candidate pool", step["pool_size"])
                cols[1].metric("Kept for reranking", step["kept_for_rerank"])
            elif "rerank_scores" in step:
                st.caption(f"Model: {step['model']}")
                changed = step["order_before"] != step["order_after"]
                st.caption("Reranker changed the order: " + ("yes" if changed else "no"))
                st.table([
                    {"Rank": i + 1, "Chunk": r["id"], "Logit": r["logit"], "Confidence": f"{r['confidence']:.0%}"}
                    for i, r in enumerate(step["rerank_scores"])
                ])


def run_agent(question: str):
    from src.agent.react_agent import ask
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(ask(question))
    finally:
        loop.close()


def run_agent_safe(question: str):
    """Run the agent and return (answer, error_message). Never raises."""
    from src.agent.react_agent import AgentUnavailableError
    try:
        return run_agent(question), None
    except AgentUnavailableError as e:
        return None, str(e)
    except Exception:
        return None, "Something went wrong while contacting the AI model. Please try again."


# Sidebar
st.sidebar.markdown(
    '<div style="display:flex; align-items:center; gap:11px; margin-bottom:4px;">'
    '<svg width="32" height="32" viewBox="0 0 30 30" style="filter:drop-shadow(0 0 6px rgba(52,211,153,0.55)); flex-shrink:0;">'
    '<defs><linearGradient id="cblogo" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#A7F3D0"/><stop offset="0.5" stop-color="#34D399"/><stop offset="1" stop-color="#059669"/>'
    '</linearGradient></defs>'
    '<g transform="rotate(45 15 15)">'
    '<rect x="5" y="5" width="20" height="20" rx="6" fill="url(#cblogo)"/>'
    '<rect x="5" y="5" width="20" height="20" rx="6" fill="none" stroke="rgba(255,255,255,0.45)" stroke-width="1"/>'
    '</g>'
    '<rect x="11" y="12" width="2.4" height="6" rx="1.2" fill="#05231A"/>'
    '<rect x="16.6" y="12" width="2.4" height="6" rx="1.2" fill="#05231A"/>'
    '</svg>'
    '<span style="font-size:1.4rem; font-weight:800; letter-spacing:-0.5px; color:#FFFFFF; line-height:1;">'
    'Cyber<span style="background:linear-gradient(90deg,#6EE7B7,#22C55E);'
    '-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;">Board</span>-AI</span>'
    '</div>'
    '<p style="color:#8FB3A3; font-size:0.72rem; letter-spacing:1.5px; text-transform:uppercase; margin:0 0 4px;">'
    'AI Advisory Board Agent</p>',
    unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["Ask Agent", "Search Filings", "Ajyad Capital (Demo)", "Evaluation Results"])
selected_ticker = st.sidebar.selectbox("Company", ["All"] + TICKERS)

st.sidebar.markdown("---")
gemini_ok = config.gemini_available()
st.sidebar.markdown(
    f'<span class="cb-chip">{load_store().collection.count():,} chunks</span>'
    f'<span class="cb-chip">{len(TICKERS)} companies</span><br><br>'
    f'<span class="cb-pill {"high" if gemini_ok else "low"}" style="float:none;">'
    f'Gemini {"connected" if gemini_ok else "not configured"}</span>',
    unsafe_allow_html=True)

# Main content
if mode == "Ask Agent":
    mascot()
    st.markdown(
        "<h1 class='cb-greet'>Ask the <span>Advisory Board</span></h1>"
        "<p class='cb-greet-sub'>Board level answers, grounded in SEC filings.</p>",
        unsafe_allow_html=True)

    if not config.gemini_available():
        st.warning("Gemini API not configured. Add GEMINI_API_KEY to .env to enable the agent.")

    # apply an example question chosen on the previous run (before the widget is created)
    if "pending_q" in st.session_state:
        st.session_state.agent_q = st.session_state.pop("pending_q")

    question = st.text_area(
        "Your question:", key="agent_q",
        placeholder="What are Apple's main risk factors?",
        height=90, label_visibility="collapsed")

    ask = st.button("Ask the board", type="primary", disabled=not config.gemini_available())

    st.markdown("<p class='cb-hint'>Or start with an example</p>", unsafe_allow_html=True)
    examples = [
        "What are the main risk factors?",
        "Summarize revenue and earnings",
        "How is board oversight handled?",
    ]
    ecols = st.columns(3)
    for col, ex in zip(ecols, examples):
        if col.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_q = ex
            st.rerun()

    if ask:
        with st.spinner("The board is reviewing the filings..."):
            answer, error = run_agent_safe(question)

        if error:
            st.error(error)
        else:
            st.markdown("### Response")
            ph = st.empty()
            shown = ""
            for tok in answer.split(" "):
                shown += tok + " "
                ph.markdown(shown)
                time.sleep(0.016)

        ticker = selected_ticker if selected_ticker != "All" else None
        if not error and ticker:
            with st.expander("View retrieved sources"):
                results = retrieve(question, ticker)
                for i, r in enumerate(results):
                    result_card(r["metadata"], confidence_from_result(r), r["text"])

elif mode == "Search Filings":
    hero("Search", "SEC Filings", "Hybrid retrieval with cross encoder reranking")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search query:", placeholder="revenue growth drivers")
    with col2:
        k = st.slider("Results", 3, 20, 5)

    show_trace = st.checkbox("Show retrieval reasoning trace", value=True,
                             help="Explainability: see how each stage of the pipeline produced these results")

    if query:
        ticker = selected_ticker if selected_ticker != "All" else None
        results, trace = retrieve(query, ticker, k=k, with_trace=True)

        st.markdown(f"**{len(results)} results**" + (f" for {ticker}" if ticker else ""))

        for r in results:
            result_card(r["metadata"], confidence_from_result(r), r["text"])

        if show_trace and trace:
            with st.expander("Retrieval reasoning trace", expanded=True):
                render_trace(trace)

elif mode == "Ajyad Capital (Demo)":
    hero("Ajyad Capital", "Demo Showcase", "Live retrieval over Ajyad Capital annual reports")

    callout(
        "This is a showcase corpus (Ajyad Capital annual reports, 2021 to 2025), included to demonstrate "
        "the pipeline on a real GCC company the author works with. It is a separate ChromaDB collection "
        "and is not part of the thesis's formal evaluation methodology, which is scoped to SEC EDGAR "
        "filings and the Bahrain Bourse GCC evaluation set."
    )

    ajyad_store = load_ajyad_store()
    st.caption(f"{ajyad_store.collection.count():,} chunks indexed")

    col1, col2 = st.columns([3, 1])
    with col1:
        ajyad_query = st.text_input("Search query:", placeholder="capital adequacy ratio",
                                     key="ajyad_query")
    with col2:
        ajyad_k = st.slider("Results", 3, 20, 5, key="ajyad_k")

    if ajyad_query:
        query_emb = None
        if config.gemini_available():
            from src.pipeline.embedder import embed_batch, get_client
            try:
                query_emb = embed_batch([ajyad_query], get_client())[0]
            except Exception:
                pass
        ajyad_results = ajyad_store.search(ajyad_query, query_embedding=query_emb, k=ajyad_k)

        st.markdown(f"**{len(ajyad_results)} results**")
        for r in ajyad_results:
            result_card(r["metadata"], confidence_from_result(r), r["text"])

elif mode == "Evaluation Results":
    hero("Evaluation", "Results", "FinanceBench retrieval accuracy and ablation study")

    eval_path = config.DATA_DIR / "financebench" / "eval_results.json"
    ablation_path = config.DATA_DIR / "financebench" / "ablation_results.json"

    tab1, tab2 = st.tabs(["FinanceBench Eval", "Ablation Study"])

    with tab1:
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)

            total = len(eval_data)
            hits = sum(1 for r in eval_data if r["retrieval_hit"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Questions", total)
            col2.metric("Retrieval Hits", hits)
            col3.metric("Hit Rate", f"{100*hits/total:.1f}%")

            # Per company
            st.subheader("By company")
            by_company = {}
            for r in eval_data:
                by_company.setdefault(r["ticker"], []).append(r)

            company_data = []
            for ticker, rs in sorted(by_company.items()):
                h = sum(1 for r in rs if r["retrieval_hit"])
                company_data.append({"Company": ticker, "Questions": len(rs), "Hits": h,
                                     "Rate": f"{100*h/len(rs):.0f}%", "RateNum": round(100 * h / len(rs), 1)})
            emerald_bar(company_data, "Company", "RateNum", "Retrieval hit rate by company",
                        tooltip=["Company", "Questions", "Hits", "Rate"])
            with st.expander("View data table"):
                st.table([{k: v for k, v in d.items() if k != "RateNum"} for d in company_data])

            # Detailed results
            with st.expander("Detailed results"):
                for r in eval_data:
                    badge = ('<span style="color:#4ADE80; font-weight:700;">HIT</span>'
                             if r["retrieval_hit"]
                             else '<span style="color:#F87171; font-weight:700;">MISS</span>')
                    st.markdown(f'{badge} &nbsp; <b>{r["ticker"]}</b> &nbsp;|&nbsp; {r["question"][:80]}...',
                                unsafe_allow_html=True)
                    st.caption(f"Gold: {r['gold_answer'][:100]}")
        else:
            callout("Run evaluation first: python -m src.evaluation.financebench_eval")

    with tab2:
        if ablation_path.exists():
            with open(ablation_path) as f:
                ablation_data = json.load(f)

            st.subheader("Configuration comparison")
            table_data = []
            for r in sorted(ablation_data, key=lambda x: x["hit_rate"], reverse=True):
                table_data.append({
                    "Config": r["config_name"],
                    "Hits": r["retrieval_hits"],
                    "Hit Rate": f"{100*r['hit_rate']:.1f}%",
                    "Avg Latency": f"{r['avg_latency']:.2f}s",
                    "RateNum": round(100 * r["hit_rate"], 1),
                })
            emerald_bar(table_data, "Config", "RateNum", "Retrieval hit rate by configuration",
                        tooltip=["Config", "Hit Rate", "Avg Latency"])
            with st.expander("View data table"):
                st.table([{k: v for k, v in d.items() if k != "RateNum"} for d in table_data])
        else:
            callout("Run ablation first: python -m src.evaluation.ablation_study")
