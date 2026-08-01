# CyberBoard-AI

An AI advisory board agent for corporate governance: a RAG-based agentic system with explainable financial reasoning.

MSc Artificial Intelligence thesis project, Bahrain Polytechnic, 2026. Student: Hatem Isa (202508993). Supervisor: Dr. Joshua Samuel.

## What it does

CyberBoard-AI helps board members analyze corporate filings without reading hundreds of pages each, across two markets: United States SEC 10-K and 10-Q filings and Bahrain Bourse (GCC) annual reports. It combines hybrid retrieval (BM25 + dense embeddings + cross-encoder reranking) with a ReAct agent that calls three specialized tools:

- **Financial Summarizer**: revenue, earnings, cash flow questions
- **Risk Identifier**: risk factor extraction and categorization
- **Compliance Checker**: governance, internal controls, audit questions

Every answer includes source attribution (which filing, section, date) and a confidence score derived from the retrieval pipeline. A live reasoning trace shows exactly how each result was scored, fused, and reranked.

## Architecture

```
SEC EDGAR filings → parse → chunk (512 tok, 64 overlap) → embed → ChromaDB
                                                                       │
Query → BM25 (sparse) ─┐                                              │
                        ├─ alpha-weighted fusion → cross-encoder rerank ┘
Query → dense embed ────┘
                        │
                        ▼
              ReAct agent + 3 tools → source-attributed answer
```

## Results

**Answer quality.** Averaged over five runs on each corpus, the full agent reaches:

| Corpus | Accuracy | Faithfulness |
|---|---|---|
| Bahrain Bourse (GCC), 35 questions | 96.8% ± 0.1% | 99.4% ± 1.2% |
| SEC-recent (US), 19 questions | 98.9% ± 2.1% | 96.8% ± 4.2% |

On the GCC set this compares with 82.9% for single-pass RAG and 11.4% for a closed-book model with no retrieval. A paired McNemar test on the primary run finds the agent strictly superior (five wins, zero losses, exact p 0.0625). Full detail in [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md).

**Retrieval and a fixed defect.** An early version had a bug: the BM25 sparse retriever filtered by company ticker *after* selecting the global top-k candidates across all 20 companies, so a company's own filings could be silently dropped from its own results. Fixing this raised FinanceBench retrieval hit rate from **54.8% to 96.8%**, and the measured contribution of cross-encoder reranking rose from **+3.2 to +12.9 percentage points** (see `data/financebench/ablation_results.json`).

| Config | Hit rate | Avg latency |
|---|---|---|
| Hybrid + rerank | 96.8% | 0.66s |
| Hybrid, no rerank | 83.9% | 0.05s |
| Sparse only | 96.8% | 0.27s |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in a Gemini API key (optional; the pipeline runs on BM25 alone without it)
```

## Running the pipeline

```bash
python -m src.ingestion.sec_downloader      # download filings (needs SEC_EDGAR_USER_AGENT in .env)
python -m src.ingestion.filing_parser       # HTML → structured JSON
python -m src.pipeline.chunker              # chunk + index into ChromaDB
python -m src.evaluation.financebench_eval  # retrieval evaluation
python -m src.evaluation.ablation_study     # ablation study
streamlit run src/ui/app.py                 # dashboard
```

The evaluation and dashboard both work without a Gemini API key, running on BM25 sparse retrieval only. Dense embeddings, the full ReAct agent, and the "Ask Agent" UI mode require a Gemini API key (`GEMINI_API_KEY`) configured in `.env`.

## Data

- **US corpus**: 116 SEC EDGAR 10-K/10-Q filings from 20 large-cap companies, 18,136 chunks (see `data/sec_filings/manifest.csv`). Public regulatory filings, no license restrictions, re-downloadable via `sec_downloader.py`.
- **GCC corpus**: Bahrain Bourse annual reports from 7 companies, 845 chunks, for cross-market evaluation.
- **Evaluation sets**: [FinanceBench](https://arxiv.org/abs/2311.11944) (31 questions matched to the US corpus) for retrieval; 35 hand-verified GCC questions and 19 recent US numerical questions for answer quality.

## Repository structure

```
config.py                  Central configuration (companies, chunking, retrieval params)
src/ingestion/              SEC EDGAR download, HTML parsing, FinanceBench/Bahrain Bourse loaders
src/pipeline/                Chunking, embedding, hybrid vector store (BM25 + ChromaDB + reranker)
src/tools/                   Agent tools (financial summarizer, risk identifier, compliance checker)
src/agent/                   ReAct agent (OpenAI Agents SDK)
src/evaluation/              FinanceBench evaluation + ablation study
src/ui/                      Streamlit dashboard
docs/EVALUATION_RESULTS.md   Full evaluation results and methodology
```

## Responsible AI

All outputs are advisory only and are not financial advice. Every claim is cited to its source filing. This system does not process personal data; all data sources are public disclosure filings.

## License

MIT. See [LICENSE](LICENSE).
