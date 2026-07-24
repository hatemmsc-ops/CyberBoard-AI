"""Tool 2: Identify and rank risk factors from SEC filings."""

from agents import function_tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@function_tool
def risk_identifier(ticker: str, risk_category: str = "") -> str:
    """Retrieve risk factors disclosed in a company's filings (US SEC reports or
    Bahrain Bourse annual reports). Use this tool when the user asks about risks,
    threats, vulnerabilities, challenges, or concerns facing a company.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT for US; NBB, BBK, ALBH for Bahrain Bourse).
        risk_category: Optional filter like 'regulatory', 'market', 'operational', 'cybersecurity', 'supply chain'.
    """
    from src.pipeline.vector_store import get_cached_store, confidence_from_result
    from src.pipeline.embedder import embed_batch, get_client
    import config

    store = get_cached_store(config.collection_for_ticker(ticker))

    query = f"{ticker} risk factors"
    if risk_category:
        query += f" {risk_category}"

    query_emb = None
    if config.gemini_available():
        try:
            query_emb = embed_batch([query], get_client())[0]
        except Exception:
            # Fall back to sparse-only retrieval if the embedding call fails
            # (e.g. transient provider outage); still return real, grounded results.
            pass

    results = store.search(query, query_embedding=query_emb, k=8, ticker=ticker.upper())

    risk_results = [r for r in results if "risk" in r["metadata"].get("section", "").lower()]
    if not risk_results:
        risk_results = results[:5]

    if not risk_results:
        return f"No risk information found for {ticker}."

    context_parts = []
    for i, r in enumerate(risk_results):
        meta = r["metadata"]
        confidence = confidence_from_result(r)
        source = f"{meta['ticker']} {meta['filing_type']} ({meta['filing_date']}) - {meta['section']} (confidence: {confidence:.0%})"
        context_parts.append(f"[Source {i+1}: {source}]\n{r['text']}")

    return f"Retrieved {len(risk_results)} risk-related passages for {ticker}:\n\n" + "\n\n---\n\n".join(context_parts)
