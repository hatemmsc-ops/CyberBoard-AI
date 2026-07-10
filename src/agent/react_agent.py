"""CyberBoard-AI: ReAct agent for corporate governance advisory."""

import asyncio
from agents import Agent, Runner, ModelSettings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

from src.tools.financial_summarizer import financial_summarizer
from src.tools.risk_identifier import risk_identifier
from src.tools.compliance_checker import compliance_checker

SYSTEM_PROMPT = """You are CyberBoard-AI, an AI advisory board agent for corporate governance.
You help board members make informed decisions by analyzing SEC filings (10-K and 10-Q reports)
from Fortune 500 companies.

Your role:
1. Answer board-level financial queries with evidence from actual filings
2. Identify and prioritize risk factors relevant to governance decisions
3. Assess compliance posture and internal controls effectiveness

Guidelines:
- Always cite which filing (company, type, date, section) your information comes from
- Present findings in a structured, board-ready format
- Flag any data gaps or limitations in available filings
- Compare across time periods when relevant
- Be precise with financial figures and avoid speculation
- End every response with: "This output is advisory only and should not be construed as financial advice."

Available companies: AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, JNJ, V,
WMT, PG, MA, UNH, KO, PFE, CVX, CSCO, INTC, GS

You have three tools:
- financial_summarizer: For revenue, earnings, cash flow, and financial performance questions
- risk_identifier: For risk factors, threats, and challenges
- compliance_checker: For governance, compliance, internal controls, and audit information

Use multiple tools when a question spans different areas. Think step by step."""


def create_agent() -> Agent:
    """Create the CyberBoard-AI advisory agent."""
    model_name = f"azure/{config.AZURE_OPENAI_DEPLOYMENT}" if config.azure_available() else "gpt-4o"

    return Agent(
        name="CyberBoard-AI",
        instructions=SYSTEM_PROMPT,
        tools=[financial_summarizer, risk_identifier, compliance_checker],
        model=model_name,
        model_settings=ModelSettings(temperature=0.1),
    )


async def ask(question: str) -> str:
    """Run a single query through the agent."""
    agent = create_agent()
    result = await Runner.run(agent, question)
    return result.final_output


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
    print(ask_sync(question))
