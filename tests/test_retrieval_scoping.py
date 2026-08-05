"""Regression tests for the retrieval scoping defect found in Week 5.

Ticker filtering used to run AFTER the global top-k selection, so a company's
own passages could be discarded before the company filter ever applied. For the
example query the top matching JPMorgan passage ranked nineteenth globally, just
beyond the threshold in use, and the filtered result came back empty even though
every JPMorgan filing was indexed.

Scoping the candidate pool to the target company BEFORE ranking raised the
FinanceBench retrieval hit rate from 54.8 to 96.8 percent, and raised the
measured contribution of cross encoder reranking from 3.2 to 12.9 percentage
points, because the reranker had previously been handed a pool that often did
not contain the correct passage at all.

These tests fail if that ordering is ever reversed. They run offline: no
network, no ChromaDB, no reranker model.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

BM25Okapi = pytest.importorskip("rank_bm25").BM25Okapi
from src.pipeline.vector_store import HybridStore  # noqa: E402


class _StubCollection:
    """Minimal stand-in for the ChromaDB collection: no disk, no network."""

    def __init__(self, docs):
        self._docs = docs

    def get(self, ids, include=None):
        present = [i for i in ids if i in self._docs]
        return {
            "ids": present,
            "documents": [self._docs[i] for i in present],
            "metadatas": [{"ticker": i.split("_")[0]} for i in present],
        }


def _store_with(docs):
    """A HybridStore with only the sparse path wired up.

    __init__ is bypassed deliberately so the test never touches ChromaDB.
    """
    store = object.__new__(HybridStore)
    store.bm25_ids = list(docs)
    store.bm25 = BM25Okapi([docs[i].lower().split() for i in store.bm25_ids])
    store.collection = _StubCollection(docs)
    store.reranker = None
    return store


def _corpus_where_target_is_outranked():
    """30 Apple chunks that match the query strongly, one JPMorgan chunk that
    matches it weakly. Ranked globally, the JPMorgan chunk falls far below any
    sensible cut-off, which is exactly the situation that exposed the defect.
    """
    docs = {
        f"AAPL_10-K_2024_risk_{n}": "capital requirements capital requirements capital"
        for n in range(30)
    }
    docs["JPM_10-K_2024_risk_0"] = "capital requirements for the group"
    return docs


def test_target_company_survives_even_when_outranked_globally():
    docs = _corpus_where_target_is_outranked()
    store = _store_with(docs)

    results = store.search(
        "capital requirements", query_embedding=None, k=5, rerank=False, ticker="JPM"
    )

    ids = [r["id"] for r in results]
    assert ids, (
        "Scoped search returned nothing. The company filter is probably running "
        "after global ranking again, which is the Week 5 defect."
    )
    assert "JPM_10-K_2024_risk_0" in ids


def test_scoped_search_never_leaks_another_company():
    docs = _corpus_where_target_is_outranked()
    store = _store_with(docs)

    results = store.search(
        "capital requirements", query_embedding=None, k=5, rerank=False, ticker="JPM"
    )

    leaked = [r["id"] for r in results if not r["id"].startswith("JPM_")]
    assert not leaked, f"Scoped search leaked other companies: {leaked}"


def test_unscoped_search_still_returns_the_global_best():
    """The fix must not break the unscoped path, which should still rank
    globally across every company."""
    docs = _corpus_where_target_is_outranked()
    store = _store_with(docs)

    results = store.search(
        "capital requirements", query_embedding=None, k=5, rerank=False, ticker=None
    )

    ids = [r["id"] for r in results]
    assert ids, "Unscoped search returned nothing."
    assert any(i.startswith("AAPL_") for i in ids), (
        "Unscoped search should surface the globally strongest matches."
    )


def test_scoping_is_not_a_substring_match():
    """A ticker filter must match the identifier segment, not merely a prefix
    of a longer ticker, or one company's chunks would contaminate another's."""
    docs = {
        "JPM_10-K_2024_risk_0": "capital requirements for the group",
        "JPMX_10-K_2024_risk_0": "capital requirements for the group",
    }
    store = _store_with(docs)

    results = store.search(
        "capital requirements", query_embedding=None, k=5, rerank=False, ticker="JPM"
    )

    ids = [r["id"] for r in results]
    assert "JPMX_10-K_2024_risk_0" not in ids, (
        "Ticker scoping matched a different company whose ticker shares a prefix."
    )
