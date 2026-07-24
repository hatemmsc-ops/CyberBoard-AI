"""CyberBoard-AI: ReAct agent for corporate governance advisory."""

import asyncio
import time
from agents import Agent, Runner, ModelSettings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

from src.tools.financial_summarizer import financial_summarizer
from src.tools.risk_identifier import risk_identifier
from src.tools.compliance_checker import compliance_checker

SYSTEM_PROMPT = """You are CyberBoard-AI, an AI advisory board agent for corporate governance.
You help board members make informed decisions by analyzing two corpora: US SEC filings
(10-K and 10-Q reports) from Fortune 500 companies, and annual reports from companies listed
on Bahrain Bourse (a GCC market). Both are reached through the same tools.

Your role:
1. Answer board-level financial queries with evidence from actual filings
2. Identify and prioritize risk factors relevant to governance decisions
3. Assess compliance posture and internal controls effectiveness

Guidelines:
- Always cite which document (company, type, date, section) your information comes from
- Present findings in a structured, board-ready format
- Flag any data gaps or limitations in available documents
- Compare across time periods when relevant
- Answer only what was asked. Report the figure the question calls for and stop; do not
  volunteer additional precise numbers (share counts, ratios, currency conversions, cash-flow
  figures) unless they appear verbatim in the retrieved content and the question asks for them.
- Do not convert currencies yourself. If a report is in USD, report USD; if in BHD, report BHD.
  Never compute or invent a converted figure.
- Be precise with financial figures and avoid speculation
- If a tool call fails or returns no grounded results, say so plainly and stop there.
  Never substitute your own general/background knowledge about a company for retrieved
  document content, even to be helpful — an ungrounded answer is worse than no answer, since
  the board member cannot tell it apart from a cited one.
- End every response with: "This output is advisory only and should not be construed as financial advice."

Available companies:
- US (SEC): AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, JNJ, V, WMT, PG, MA, UNH, KO,
  PFE, CVX, CSCO, INTC, GS
- Bahrain Bourse (GCC): NBB, BBK, ALBH (Alba), BEYON (Batelco), GFH, BISB, KFH

You have three tools:
- financial_summarizer: For revenue, earnings, cash flow, and financial performance questions
- risk_identifier: For risk factors, threats, and challenges
- compliance_checker: For governance, compliance, internal controls, and audit information

Use multiple tools when a question spans different areas. Think step by step."""


def create_agent() -> Agent:
    """Create the CyberBoard-AI advisory agent."""
    model_name = f"litellm/gemini/{config.GEMINI_CHAT_MODEL}" if config.gemini_available() else "gpt-4o"

    return Agent(
        name="CyberBoard-AI",
        instructions=SYSTEM_PROMPT,
        tools=[financial_summarizer, risk_identifier, compliance_checker],
        model=model_name,
        model_settings=ModelSettings(temperature=0.1),
    )


class AgentUnavailableError(Exception):
    """Raised when the model API fails after retries. Safe to show to end users."""


def _is_transient(error: Exception) -> bool:
    msg = str(error).lower()
    return any(s in msg for s in ("503", "429", "unavailable", "rate", "resource_exhausted", "overloaded"))


async def ask(question: str, max_retries: int = 3) -> str:
    """Run a single query through the agent, retrying on transient model-provider errors."""
    agent = create_agent()
    for attempt in range(max_retries):
        try:
            result = await Runner.run(agent, question)
            return result.final_output
        except Exception as e:
            if _is_transient(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Model temporarily unavailable, retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            raise AgentUnavailableError(
                "The AI model is temporarily unavailable or overloaded. Please try again in a moment."
            ) from e


def ask_sync(question: str) -> str:
    """Synchronous wrapper for ask()."""
    return asyncio.run(ask(question))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are Apple's main risk factors according to their most recent 10-K filing?"
    print(f"Question: {question}\n")
    try:
        print(ask_sync(question))
    except AgentUnavailableError as e:
        print(f"Error: {e}")
