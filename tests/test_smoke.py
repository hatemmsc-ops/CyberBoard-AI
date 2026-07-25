"""Offline smoke tests protecting the eval-critical contracts.

No network or ChromaDB required. Run with: python -m pytest tests/ -q
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# --- collection routing (the bug that sent GCC questions to the SEC corpus) ---

def test_gcc_tickers_route_to_bahrain_bourse():
    for t in ("NBB", "BBK", "ALBH", "BEYON", "GFH", "BISB", "KFH"):
        assert config.collection_for_ticker(t) == config.BAHRAIN_BOURSE_COLLECTION_NAME

def test_sec_tickers_route_to_sec_filings():
    for t in ("AAPL", "MSFT", "JPM"):
        assert config.collection_for_ticker(t) == config.COLLECTION_NAME

def test_routing_is_case_insensitive_and_defaults_to_sec():
    assert config.collection_for_ticker("bbk") == config.BAHRAIN_BOURSE_COLLECTION_NAME
    assert config.collection_for_ticker("UNKNOWN") == config.COLLECTION_NAME

def test_bahrain_ticker_set_matches_company_map():
    expected = {t for t, _ in config.BAHRAIN_BOURSE_COMPANIES.values()}
    assert config.BAHRAIN_BOURSE_TICKERS == expected


# --- GCC gold set integrity ---

def _load_gcc():
    return json.load(open(config.DATA_DIR / "gcc_eval" / "questions.json"))["questions"]

def test_gcc_set_has_35_questions_all_verified():
    qs = _load_gcc()
    assert len(qs) == 35
    assert all(q["verified"] is True for q in qs)

def test_gcc_questions_have_required_fields():
    for q in _load_gcc():
        for key in ("id", "company", "ticker", "question", "question_type", "answer", "evidence_text"):
            assert q.get(key), f"{q.get('id')} missing {key}"
        assert q["question_type"] in ("numerical", "qualitative")

def test_gcc_tickers_are_known():
    for q in _load_gcc():
        assert q["ticker"] in config.BAHRAIN_BOURSE_TICKERS


# --- judge contract (empty prediction takes the no-network fast path) ---

def test_judge_empty_prediction_is_hallucination():
    from src.evaluation.llm_judge import judge_answer
    r = judge_answer("Q?", "gold", "", "context")
    assert r["correct"] is False
    assert r["faithful"] is False
    assert r["hallucinated"] is True


# --- full_eval loader handles both schemas ---

def test_loader_handles_both_schemas(tmp_path):
    from src.evaluation.full_eval import _load_questions
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps([{"id": "x"}]))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"questions": [{"id": "y"}]}))
    assert _load_questions(flat)[0]["id"] == "x"
    assert _load_questions(wrapped)[0]["id"] == "y"
