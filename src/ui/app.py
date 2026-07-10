"""CyberBoard-AI: Streamlit dashboard for the AI advisory board agent."""

import streamlit as st
import json
import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.pipeline.vector_store import confidence_from_result

st.set_page_config(page_title="CyberBoard-AI", page_icon="🏛️", layout="wide")

NAVY = "#121A2F"
GOLD = "#C9A84C"
MUTED = "#8fa0bd"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput {
    font-family: 'Exo', sans-serif;
}

/* Hero banner */
.cb-hero {
    background: linear-gradient(135deg, #121A2F 0%, #1b2a4a 55%, #0B1120 100%);
    border: 1px solid rgba(201, 168, 76, 0.35);
    border-radius: 14px;
    padding: 26px 32px 22px 32px;
    margin-bottom: 22px;
}
.cb-hero h1 {
    color: #FFFFFF;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin: 0 0 2px 0;
    padding: 0;
}
.cb-hero h1 .gold { color: #C9A84C; }
.cb-hero p {
    color: #8fa0bd;
    margin: 0;
    font-size: 0.95rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Result cards */
.cb-card {
    background: #121A2F;
    border: 1px solid rgba(201, 168, 76, 0.18);
    border-left: 4px solid #C9A84C;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.cb-card .cb-meta { margin-bottom: 8px; }
.cb-chip {
    display: inline-block;
    background: rgba(201, 168, 76, 0.12);
    border: 1px solid rgba(201, 168, 76, 0.35);
    color: #C9A84C;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
}
.cb-chip.ticker { background: #C9A84C; color: #121A2F; font-weight: 800; }
.cb-card .cb-text {
    color: #c7d2e4;
    font-size: 0.88rem;
    line-height: 1.55;
    margin-top: 6px;
}

/* Confidence pills */
.cb-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    float: right;
}
.cb-pill.high { background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }
.cb-pill.mid  { background: rgba(241, 196, 15, 0.12); color: #f1c40f; border: 1px solid rgba(241,196,15,0.4); }
.cb-pill.low  { background: rgba(231, 76, 60, 0.12); color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }

/* Trace stage cards */
.cb-stage {
    background: #0f1729;
    border: 1px solid rgba(143, 160, 189, 0.15);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.cb-stage .cb-stage-title {
    color: #C9A84C;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.cb-stage .cb-stage-body { color: #c7d2e4; font-size: 0.85rem; }

/* Sidebar polish */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121A2F 0%, #0B1120 100%);
    border-right: 1px solid rgba(201, 168, 76, 0.2);
}

/* Metric styling */
[data-testid="stMetricValue"] { color: #C9A84C; }
</style>
""", unsafe_allow_html=True)

TICKERS = list(config.COMPANIES.keys())


def hero(title_white: str, title_gold: str, subtitle: str):
    st.markdown(
        f'<div class="cb-hero"><h1>{title_white} <span class="gold">{title_gold}</span></h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True)


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


def retrieve(query: str, ticker: str, k: int = 5, with_trace: bool = False):
    store = load_store()
    query_emb = None
    if config.azure_available():
        from src.pipeline.embedder import embed_batch, get_client
        query_emb = embed_batch([query], get_client())[0]
    trace = [] if with_trace else None
    results = store.search(query, query_embedding=query_emb, k=k, ticker=ticker, trace=trace)
    if with_trace:
        return results, trace
    return results


def render_trace(trace: list):
    """Reasoning trace visualization: how the pipeline produced these results."""
    for step in trace:
        stage = step.get("stage", "?")
        if step.get("skipped"):
            st.markdown(f'<div class="cb-stage"><div class="cb-stage-title">{stage}</div>'
                        f'<div class="cb-stage-body">Skipped</div></div>', unsafe_allow_html=True)
            continue
        with st.container():
            st.markdown(f'<div class="cb-stage"><div class="cb-stage-title">{stage}</div></div>',
                        unsafe_allow_html=True)
            if "sparse_candidates" in step:
                cols = st.columns(3)
                cols[0].metric("Sparse candidates (BM25)", step["sparse_candidates"])
                cols[1].metric("Dense candidates", step["dense_candidates"],
                               help="0 when Azure OpenAI embeddings are not configured")
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


# Sidebar
st.sidebar.markdown(
    '<h2 style="margin-bottom:0;">🏛️ Cyber<span style="color:#C9A84C;">Board</span>-AI</h2>'
    '<p style="color:#8fa0bd; font-size:0.78rem; letter-spacing:1px; text-transform:uppercase;">'
    'AI Advisory Board Agent</p>',
    unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["Ask Agent", "Search Filings", "Evaluation Results"])
selected_ticker = st.sidebar.selectbox("Company", ["All"] + TICKERS)

st.sidebar.markdown("---")
azure_ok = config.azure_available()
st.sidebar.markdown(
    f'<span class="cb-chip">{load_store().collection.count():,} chunks</span>'
    f'<span class="cb-chip">{len(TICKERS)} companies</span><br><br>'
    f'<span class="cb-pill {"high" if azure_ok else "low"}" style="float:none;">'
    f'Azure {"connected" if azure_ok else "not configured"}</span>',
    unsafe_allow_html=True)

# Main content
if mode == "Ask Agent":
    hero("Ask the", "Advisory Board", "Board level answers grounded in SEC filings")

    if not config.azure_available():
        st.warning("Azure OpenAI not configured. Add credentials to .env to enable the agent.")

    question = st.text_area("Your question:", placeholder="What are Apple's main risk factors?", height=80)

    if st.button("Ask", type="primary", disabled=not config.azure_available()):
        with st.spinner("Agent is reasoning..."):
            answer = run_agent(question)
        st.markdown("### Response")
        st.markdown(answer)

        # Show sources
        ticker = selected_ticker if selected_ticker != "All" else None
        if ticker:
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
                company_data.append({"Company": ticker, "Questions": len(rs), "Hits": h, "Rate": f"{100*h/len(rs):.0f}%"})
            st.table(company_data)

            # Detailed results
            with st.expander("Detailed results"):
                for r in eval_data:
                    icon = "✅" if r["retrieval_hit"] else "❌"
                    st.markdown(f"{icon} **{r['ticker']}** | {r['question'][:80]}...")
                    st.caption(f"Gold: {r['gold_answer'][:100]}")
        else:
            st.info("Run evaluation first: python -m src.evaluation.financebench_eval")

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
                })
            st.table(table_data)
        else:
            st.info("Run ablation first: python -m src.evaluation.ablation_study")
