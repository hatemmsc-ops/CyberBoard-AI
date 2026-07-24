"""Tool 3: Check governance and compliance posture from SEC filings."""

from agents import function_tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@function_tool
def compliance_checker(ticker: str, topic: str = "") -> str:
    """Retrieve governance, compliance, and internal controls information from SEC filings.
    Use this tool when the user asks about corporate governance, regulatory compliance,
    internal controls, audit findings, board oversight, or ESG disclosures.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        topic: Optional focus area like 'internal controls', 'audit', 'board governance', 'ESG', 'cybersecurity governance'.
    """
    from src.pipeline.vector_store import HybridStore, confidence_from_result
    from src.pipeline.embedder import embed_batch, get_client
    import config

    store = HybridStore(collection_name=config.collection_for_ticker(ticker))
    if not store.bm25:
        store.rebuild_bm25_from_collection()

    query = f"{ticker} corporate governance compliance controls"
    if topic:
        query += f" {topic}"

    query_emb = None
    if config.gemini_available():
        try:
            query_emb = embed_batch([query], get_client())[0]
        except Exception:
            # Fall back to sparse-only retrieval if the embedding call fails
            # (e.g. transient provider outage); still return real, grounded results.
            pass

    results = store.search(query, query_embedding=query_emb, k=6, ticker=ticker.upper())

    compliance_results = [r for r in results if any(
        kw in r["metadata"].get("section", "").lower()
        for kw in ["controls", "governance", "procedures"]
    )]
    if not compliance_results:
        compliance_results = results[:5]

    if not compliance_results:
        return f"No compliance/governance information found for {ticker}."

    context_parts = []
    for i, r in enumerate(compliance_results):
        meta = r["metadata"]
        confidence = confidence_from_result(r)
        source = f"{meta['ticker']} {meta['filing_type']} ({meta['filing_date']}) - {meta['section']} (confidence: {confidence:.0%})"
        context_parts.append(f"[Source {i+1}: {source}]\n{r['text']}")

    return f"Retrieved {len(compliance_results)} governance/compliance passages for {ticker}:\n\n" + "\n\n---\n\n".join(context_parts)
