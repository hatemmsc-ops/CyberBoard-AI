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

TICKERS = list(config.COMPANIES.keys())


@st.cache_resource
def load_store():
    from src.pipeline.vector_store import HybridStore
    store = HybridStore()
    if not store.bm25:
        store.rebuild_bm25_from_collection()
    return store


def retrieve(query: str, ticker: str, k: int = 5):
    store = load_store()
    query_emb = None
    if config.azure_available():
        from src.pipeline.embedder import embed_batch, get_client
        query_emb = embed_batch([query], get_client())[0]
    return store.search(query, query_embedding=query_emb, k=k, ticker=ticker)


def run_agent(question: str):
    from src.agent.react_agent import ask
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(ask(question))
    finally:
        loop.close()


# Sidebar
st.sidebar.title("CyberBoard-AI")
st.sidebar.caption("AI Advisory Board Agent for Corporate Governance")

mode = st.sidebar.radio("Mode", ["Ask Agent", "Search Filings", "Evaluation Results"])
selected_ticker = st.sidebar.selectbox("Company", ["All"] + TICKERS)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data:** {load_store().collection.count():,} chunks")
st.sidebar.markdown(f"**Companies:** {len(TICKERS)}")
st.sidebar.markdown(f"**Azure:** {'Connected' if config.azure_available() else 'Not configured'}")

# Main content
if mode == "Ask Agent":
    st.header("Ask the Advisory Board")

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
                    meta = r["metadata"]
                    confidence = confidence_from_result(r)
                    st.markdown(f"**Source {i+1}** | {meta['section']} | {meta['filing_type']} {meta['filing_date']} | Confidence: {confidence:.0%}")
                    st.text(r["text"][:500])
                    st.divider()

elif mode == "Search Filings":
    st.header("Search SEC Filings")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search query:", placeholder="revenue growth drivers")
    with col2:
        k = st.slider("Results", 3, 20, 5)

    if query:
        ticker = selected_ticker if selected_ticker != "All" else None
        results = retrieve(query, ticker, k=k)

        st.markdown(f"**{len(results)} results**" + (f" for {ticker}" if ticker else ""))

        for i, r in enumerate(results):
            meta = r["metadata"]
            confidence = confidence_from_result(r)

            with st.container():
                cols = st.columns([1, 1, 1, 1])
                cols[0].markdown(f"**{meta['ticker']}**")
                cols[1].markdown(f"{meta['filing_type']} ({meta['filing_date']})")
                cols[2].markdown(f"Section: {meta['section']}")
                cols[3].markdown(f"Confidence: {confidence:.0%}")

                st.text(r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"])
                st.divider()

elif mode == "Evaluation Results":
    st.header("Evaluation Results")

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
