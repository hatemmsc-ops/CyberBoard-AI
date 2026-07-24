# CyberBoard-AI: Evaluation Methodology and Results

This document records the evaluation of the CyberBoard-AI agentic RAG system. It is written to be adapted directly into the thesis results chapter. All figures come from runs recorded in the repository.

## 0. Abstract (draft for the thesis introduction)

CyberBoard-AI is an agentic retrieval system designed to answer governance questions at board level from corporate filings while citing the source of each answer. This study evaluates the system across two document collections of differing character: the annual reports of seven companies listed on Bahrain Bourse, and the recent 10-K and 10-Q filings of twenty large United States corporations. Each answer is assessed on two dimensions: accuracy, meaning whether the reported figure matches the source, and faithfulness, meaning whether every claim can be traced to the passages the agent retrieved. On the Bahrain Bourse collection the agent answered every completed question correctly, and the correct passage appeared among the ten highest ranked results in 97 percent of cases. Faithfulness proved more difficult to achieve than accuracy. Approximately one third of the correct answers included a detail that the retrieved text did not support, most often a ratio or a currency conversion the model had derived independently. This divergence is the principal result of the study, because it locates the weakness in how an answer is presented rather than in whether the relevant evidence can be found.

Two further findings emerged from the evaluation itself. Where a filing required to answer a question was absent from the collection, the agent reported the absence rather than fabricating a value, which is the behaviour the design was intended to produce. In three separate cases the agent contradicted a gold answer that had been verified by hand, and the agent proved correct. The manual answer had matched a figure that does appear in the report yet responds to a subtly different question, for example profit before minority interest rather than profit attributable to shareholders. Detecting this class of error required a grounded agent and an independent judge, and it could not be achieved by rereading the statements alone. Confirming that a figure is present is therefore not equivalent to confirming that it answers the question posed.

## 1. Evaluation design

The system was evaluated on two corpora that are held in separate ChromaDB collections so that results on each stay reproducible and independent.

| Corpus | Collection | Chunks | Contents |
|---|---|---|---|
| US SEC filings | `sec_filings` | 18,136 | 10-K and 10-Q filings for 20 large-cap US companies, filing dates 2024 to 2026 |
| Bahrain Bourse (GCC) | `bahrain_bourse` | 845 | Annual reports for 7 GCC-listed companies (NBB, BBK, Alba, Batelco/Beyon, GFH, BisB, KFH) |

Retrieval uses a hybrid design: sparse BM25 over the collection, dense cosine similarity over Gemini embeddings (`gemini-embedding-001`, 1536 dimensions), fused with weight alpha, then re-ranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`). Chunks are 512 tokens with 64-token overlap. The agent is a ReAct loop over three retrieval tools, routed to Gemini (`gemini-flash-latest`, temperature 0.1) through LiteLLM.

### 1.1 Question sets

Three gold question sets were prepared. Each answer was verified by hand against the source document before scoring.

1. **GCC set** (35 questions, 5 per company). Numerical and qualitative questions answerable from the ingested annual reports. Every figure was confirmed in the extractable source text with the correct fiscal-year column checked.
2. **SEC-recent set** (19 questions). Revenue and net income questions drawn from the most recent 10-K of 10 large-cap tickers (fiscal years 2024 and 2025). Each figure was confirmed present in the ingested corpus and cross-checked against the company reported value. This set was built because the standard FinanceBench set could not be used (Section 4).
3. **FinanceBench** (31 questions). The published financial-QA benchmark. Used as a robustness probe rather than an accuracy benchmark, for the reason given in Section 4.

### 1.2 Metrics

Two independent judgments are produced per answer by an LLM judge (Gemini, temperature 0, JSON output):

* **Answer accuracy.** Does the predicted answer state the same figure or fact as the gold answer, allowing for rounding and unit restatement.
* **Faithfulness.** Is every claim in the predicted answer supported by the context the agent actually retrieved. An unfaithful answer is counted as a hallucination. Faithfulness is judged only against retrieved context, so it measures grounding rather than truth, and it is bounded above by retrieval quality.

Retrieval quality is measured separately as hit@10: whether a chunk overlapping the gold evidence appears in the top 10 retrieved results.

## 2. GCC results (primary benchmark)

### 2.1 Retrieval

Hit@10 on the GCC set was **34 of 35 (97.1 percent)**. The single miss is a labelling artefact of the word-overlap rule, not a retrieval failure: the gold chunk was retrieved at rank 5, but the compressed evidence string did not reach the 30 percent overlap threshold.

### 2.2 Retrieval ablation

Seven retrieval configurations were compared on hit@10.

| Configuration | Hit rate | Latency (s) |
|---|---|---|
| hybrid, no rerank | 100.0 percent (35/35) | 0.80 |
| hybrid with rerank | 97.1 percent (34/35) | 1.35 |
| dense only | 97.1 percent (34/35) | 1.12 |
| alpha 0.5 | 97.1 percent (34/35) | 1.05 |
| alpha 0.9 | 97.1 percent (34/35) | 1.05 |
| sparse with rerank | 94.3 percent (33/35) | 0.22 |
| sparse, no rerank | 91.4 percent (32/35) | 0.00 |

Observations. Hybrid retrieval without re-ranking was strongest. The cross-encoder re-ranker slightly reduced recall@10 (from 100 to 97.1 percent) while adding about 0.5 seconds of latency. The re-ranker is trained on general web question answering rather than numeric financial tables, so on distinctive-figure lookups it occasionally demotes a correct chunk. This result is specific to recall@10: re-ranking mainly reorders within the top k, so it can still improve precision at rank 1, which the agent depends on because it reads the top 5. Dense retrieval alone carried most of the signal, sparse alone was weakest but still strong because financial figures are distinctive lexical tokens, and the fusion weight alpha had little effect across 0.5 to 0.9.

### 2.3 End-to-end agent accuracy and faithfulness

On the corrected gold set, every question the agent completed was answered correctly.

| Metric | Value |
|---|---|
| Answer accuracy | 32 of 32 completed (100 percent); 91.4 percent if the 3 errored questions are counted as failures |
| Faithfulness | 22 of 32 (68.8 percent) |
| Hallucination rate | 31.2 percent |
| Average latency | 9.4 seconds |
| Errors | 3 (2 transient provider errors, 1 max-turns loop on an image-only page) |

The gap between accuracy and faithfulness is the central finding. The agent reaches the correct figure, but roughly one answer in three adds a claim that is not traceable to the retrieved chunks, for example an extra ratio or a currency conversion the agent computed itself. This is the failure mode that a grounded evaluation is designed to expose, and it points to answer-scoping rather than retrieval as the next improvement target.

## 3. A methodological result: grounded evaluation surfaced gold errors

Three of the four answers the judge marked as "wrong but faithful" were not agent errors. They were errors in the hand-verified gold set that two prior rounds of manual figure-matching had missed. In each case the agent, grounded in the source, was correct.

| Question | Original gold | Correct value | Cause |
|---|---|---|---|
| Batelco net profit attributable to equity holders | 82,036 (profit for the year) | 72,049 | The 82,036 figure includes 9,987 of non-controlling interest; the question asks for the attributable figure |
| GFH Tier 1 capital adequacy ratio | 21.68 percent | 15.92 percent (Group) | 21.68 percent is the Bank solo ratio; the Group consolidated Tier 1 ratio is 15.92 percent |
| BisB total income | 58.1 million (summary) | 74,488 thousand | The report labels two different figures "Total income"; the primary income statement value is 74,488 thousand |

The lesson for the methodology chapter is that manual figure-matching confirms a number exists but does not confirm it answers the question. Attaching a figure to the wrong concept (attributable versus total, solo versus consolidated, one definition of a line item versus another) is a class of error that only a grounded agent plus an independent judge surfaced. This strengthens the case for the evaluation design rather than weakening the gold set.

## 4. Why FinanceBench could not be used as an accuracy benchmark

FinanceBench questions reference fiscal years 2016 to 2023. The ingested SEC corpus contains only 2024 to 2026 filings. The overlap is zero of 31 questions. When run against this corpus the agent correctly reported that the requested data was not present, so it scored 14.8 percent accuracy but 92.6 percent faithfulness. The low accuracy is therefore an artefact of a corpus-question year mismatch, not a reasoning failure.

The high faithfulness is itself a positive result and is reported as such: when the underlying filing is absent, the agent abstains rather than fabricating a plausible number. This is the intended anti-hallucination behaviour. To obtain a valid SEC accuracy figure comparable to the GCC set, the SEC-recent question set (Section 1.1) was built from filings that are actually in the corpus.

## 5. Limitations

1. **Image-only statement pages.** Some GCC filings, notably NBB, render primary financial statements as images with no text layer. Those figures enter the corpus only through the Notes, which reduces retrieval grounding for them and produced one max-turns agent failure.
2. **SEC parser section labels are unreliable.** The parser routed large amounts of filing content, including financial statement figures, into a catch-all `controls_and_procedures` section. The figures remain retrievable because retrieval filters by ticker, but the section metadata cannot be trusted for section-scoped analysis.
3. **Faithfulness ceiling.** Because faithfulness is judged against retrieved context, a correct answer whose supporting figure was not retrieved is counted as unfaithful. The KFH questions show this: correct figures, low faithfulness, because the exact supporting chunk was not surfaced.
4. **Answer embellishment.** The agent volunteers ungrounded supporting detail on lookup questions. Constraining the agent to answer only what is asked reduced but did not eliminate this behaviour.

## 6. Reproduction

```bash
# GCC retrieval and ablation (retrieval only)
python -m src.evaluation.gcc_eval
python -m src.evaluation.ablation_study --set gcc

# End-to-end agent evaluation with the LLM judge
python -m src.evaluation.full_eval --set gcc
python -m src.evaluation.full_eval --set sec_recent
```

Offline contract tests: `python -m pytest tests/ -q`.
