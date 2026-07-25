"""Tool 1: Summarize financial data from retrieved filing chunks."""

from agents import function_tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@function_tool
def financial_summarizer(query: str, ticker: str) -> str:
    """Retrieve and summarize financial information from a company's filings (US SEC
    reports or Bahrain Bourse annual reports) for a given company. Use this tool when
    the user asks about revenue, earnings, financial performance, cash flow, balance
    sheet items, or any quantitative financial data.

    Args:
        query: The financial question to answer.
        ticker: Stock ticker symbol (e.g. AAPL, MSFT for US; NBB, BBK, ALBH for Bahrain Bourse).
    """
    from src.pipeline.vector_store import get_cached_store, confidence_from_result
    from src.pipeline.embedder import embed_batch, get_client
    import config

    store = get_cached_store(config.collection_for_ticker(ticker))

    query_emb = None
    if config.gemini_available():
        try:
            query_emb = embed_batch([query], get_client())[0]
        except Exception:
            # Fall back to sparse-only retrieval if the embedding call fails
            # (e.g. transient provider outage); still return real, grounded results.
            pass

    results = store.search(query, query_embedding=query_emb, k=5, ticker=ticker.upper())

    if not results:
        return f"No financial data found for {ticker} related to: {query}"

    context_parts = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        confidence = confidence_from_result(r)
        source = f"{meta['ticker']} {meta['filing_type']} ({meta['filing_date']}) - {meta['section']} (confidence: {confidence:.0%})"
        context_parts.append(f"[Source {i+1}: {source}]\n{r['text']}")

    return f"Retrieved {len(results)} relevant passages for {ticker}:\n\n" + "\n\n---\n\n".join(context_parts)
