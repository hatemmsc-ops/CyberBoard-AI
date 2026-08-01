"""LLM-as-judge scoring for the agent's answers.

Two independent judgments per answer, using Gemini at temperature 0 with
JSON-structured output:

  - correct:   does the predicted answer match the gold answer (same figure /
               same fact), allowing for rounding and unit restatement?
  - faithful:  is every factual claim in the predicted answer supported by the
               retrieved context that was actually shown to the agent?

An answer that is unfaithful is counted as a hallucination. Faithfulness is
judged only against the retrieved context (not the gold answer), so it measures
grounding rather than correctness.
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

_JUDGE_PROMPT = """You are a strict evaluator of a financial question-answering system. \
Judge only from the material given. Do not use outside knowledge.

QUESTION:
{question}

GOLD ANSWER (verified from the source report):
{gold}

PREDICTED ANSWER (from the system):
{predicted}

RETRIEVED CONTEXT (the only passages the system was shown):
{context}

Return a JSON object with exactly these keys:
- "correct": true if the PREDICTED ANSWER states the same core fact/figure as the \
GOLD ANSWER (accept rounding, unit restatement such as thousands vs millions, and \
extra correct detail); false otherwise.
- "faithful": true if EVERY factual claim in the PREDICTED ANSWER is supported by \
the RETRIEVED CONTEXT; false if any figure or claim is absent from or contradicts \
the context.
- "rationale": one short sentence citing the specific figure or phrase that decided it.

Output only the JSON object."""


def judge_answer(question: str, gold: str, predicted: str, context: str) -> dict:
    """Return {"correct": bool, "faithful": bool, "hallucinated": bool, "rationale": str}."""
    if not predicted or not predicted.strip():
        return {"correct": False, "faithful": False, "hallucinated": True,
                "rationale": "empty prediction"}

    from google.genai import types
    from src.pipeline.embedder import get_client, _executor

    # The judge must see all of the context the agent actually retrieved, or a
    # correctly grounded answer can be scored unfaithful because its supporting
    # chunk fell past the cutoff. GCC's full_document chunks push a 5-chunk tool
    # output to 20k-77k characters, so the cap is set well above that. Gemini
    # Flash has a ~1M token window, so even the largest context fits with room.
    MAX_CONTEXT_CHARS = 400000
    prompt = _JUDGE_PROMPT.format(
        question=question, gold=gold, predicted=predicted,
        context=context[:MAX_CONTEXT_CHARS] if context else "(no context retrieved)",
    )

    client = get_client()
    # google-genai's own http_options timeout is unreliable (see the embedder
    # note), so the judge call is bounded with the same thread-pool timeout. A
    # hung provider socket would otherwise freeze the whole evaluation run
    # indefinitely; on timeout this raises and the caller records the question
    # as an error and moves on.
    JUDGE_TIMEOUT_SECONDS = 90
    resp = _executor.submit(
        client.models.generate_content,
        model=config.GEMINI_CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    ).result(timeout=JUDGE_TIMEOUT_SECONDS)

    try:
        data = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        return {"correct": False, "faithful": False, "hallucinated": True,
                "rationale": f"unparseable judge output: {str(resp.text)[:120]}"}

    correct = bool(data.get("correct", False))
    faithful = bool(data.get("faithful", False))
    return {
        "correct": correct,
        "faithful": faithful,
        "hallucinated": not faithful,
        "rationale": str(data.get("rationale", ""))[:300],
    }
