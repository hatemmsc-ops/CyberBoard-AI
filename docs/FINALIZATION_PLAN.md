# CyberBoard-AI Thesis Finalization Plan

Handoff plan for any Claude model (Sonnet, Opus, Fable) continuing this work. Last updated 13 July 2026 (Week 6 of 13). Read this whole file before acting.

## Scope rule

This plan and any session using it concern ONLY the thesis (CyberBoard-AI). All Semester 2 courses (IT9201, IT9203, IT9204, IT9205) are finished. Never touch them.

## Context snapshot (verified 13 July 2026)

- Student: Hatem Isa, 202508993, MSc AI, Bahrain Polytechnic. Supervisor: Dr. Joshua Samuel. Weekly Teams meetings Mon/Tue 9:00 PM India time. Weekly progress update expected (what was done, what is next).
- Code: this repository, https://github.com/hatemmsc-ops/CyberBoard-AI (README, MIT licence, evaluation JSONs included; data/chromadb and raw filings correctly gitignored).
- Thesis docs (outside this repo): /Users/hatemisa/Desktop/Thesis. Papers folder has 58 verified sources (44 downloaded PDFs plus 14 already cited in the proposal).
- Data: 116 SEC filings (20 Fortune 500 companies) parsed to JSON; 18,136 chunks in ChromaDB (data/chromadb, 351 MB, gitignored). Bahrain Bourse: 0 files (open item).
- Key results (all reproducible from data/financebench/eval_results.json and ablation_results.json):
  - FinanceBench retrieval hit rate 96.8% (30/31), was 54.8% before a BM25 ticker-filtering bug fix (filtering ran after global top-k selection; fixed in src/pipeline/vector_store.py).
  - Reranking contribution +12.9 points (83.9% to 96.8%). Sparse-only equals hybrid at 96.8% because dense retrieval is inactive without cloud LLM/embedding credentials.
- UI: Streamlit (src/ui/app.py), navy/gold theme, three modes, retrieval reasoning trace visualization working. Launch: streamlit run src/ui/app.py (or .claude/launch.json cyberboard-ui).

### Mid Review Report status

The report (outside this repo, in the Thesis folder) is complete to Bahrain Polytechnic Thesis Handbook specification and pending only final user steps: a live dashboard screenshot, the student's signature, and submission before the 19 July 2026 deadline. Content covers ~3,760 words across 9 sections, 30 IEEE-numbered references, an updated Gantt chart, a 5-item risk register, and a reflection section, all grounded in the real evaluation numbers above.

### Compute access blocker

The institution will not provide an Azure OpenAI subscription. The current plan is to apply for the AWS Cloud Credit for Research programme to fund an OpenAI-compatible endpoint or Amazon Bedrock, with a personal subscription as fallback if that is delayed. This blocks dense embeddings, the full agent reasoning loop, LLM-judge faithfulness scoring, and the two stated hypothesis tests (H1, H2), which in turn blocks full completion of Objectives O3, O4, and O5.

## Hard deadlines

| Date | Deliverable |
|---|---|
| 19 Jul 2026 | Mid Review Report (20% of grade) |
| 21 Aug 2026 | Final Thesis (Template C) |
| 23 Aug 2026 | Final demo and PPT (Template D) |
| 26 to 27 Aug 2026 | Demonstration |

## Phase 0. User prerequisites (ask, do not start work that depends on them)

1. Compute access: cloud credit application or personal subscription fallback, then credentials into `.env` (see `.env.example`).
2. Bahrain Bourse annual report PDFs (5 to 10 companies, English) from bahrainbourse.com. The site blocks automated access, so this is a manual step.
3. Mid Review Report: dashboard screenshot, signature, final read-through, submission.

## Phase 1. Compute activation (first coding work once credentials exist)

1. Verify: `python3 -c "import config; print(config.azure_available())"` prints True (or the equivalent check if the provider changed).
2. Embed all 18,136 chunks with `src/pipeline/embedder.py` (batched, has backoff). Store into ChromaDB via `HybridStore.add_chunks`. Expect real API cost; confirm with the user before running.
3. Verify: a dense query returns nonzero dense candidates in the search trace.
4. Smoke test the agent end to end; confirm the advisory disclaimer appears.
5. Re-run `python -m src.evaluation.financebench_eval` then `python -m src.evaluation.ablation_study`. Dense-only and true hybrid configs should now produce distinct numbers. H2 test: is reranking at least +10% precision over dense alone.
6. Commit and push.

## Phase 2. Bahrain Bourse corpus and GCC evaluation set

1. File PDFs into `data/bahrain_bourse/{company_name}/`. Parse with `src/ingestion/bahrain_bourse_loader.py`, chunk and index with ticker-style ids so ticker filtering works.
2. Verify: chunks appear in ChromaDB; a search filtered to a Bourse company returns its text.
3. Build a custom GCC evaluation set (30 to 50 QA pairs), written together with the user against the actual source reports; never fabricate gold answers. Store as `data/gcc_eval/questions.json` mirroring `matched_questions.json`.
4. Run retrieval evaluation on the GCC set; save results. Commit.

## Phase 3. Full evaluation and hypothesis testing (needs Phases 1 and 2)

1. Full agent evaluation (answer correctness, latency) and LLM-judge faithfulness scoring.
2. Hallucination rate: percent of claims not traceable to retrieved sources.
3. H1 test: agentic RAG vs single-pass RAG on faithfulness, paired t-test or Wilcoxon, target p < 0.05.
4. Ablation per Objective O4: RAG only vs RAG+agent vs RAG+agent+rerank, with statistics.
5. Save every number to JSON; never quote a number in the thesis that does not exist in a results file.

## Phase 4. Explainability completion (O5 remainder)

1. Agent-level reasoning trace: capture the ReAct tool-call chain and render it in the UI's Ask Agent mode, consistent with the existing retrieval trace styling.
2. Verify in browser with a live query.
3. Commit.

## Phase 5. Final thesis write-up (Template C, start by Week 10)

1. Structure from the Final Project Report Template. Reuse proposal and Mid Review content where valid; update all numbers from the final JSON results.
2. Literature review: 58 sources available, themed across RAG, agentic AI, hybrid retrieval, explainability, hallucination, GRC, GCC regulation, and financial NLP benchmarks. Cite in IEEE numeric style.
3. Figures to produce: system architecture diagram; retrieval pipeline trace example; bug-fix before/after comparison; ablation comparison chart; per-company results; latency table.
4. Writing rules: no dashes as connectors anywhere; consistent spelling with the proposal.
5. Every reconstructed or assumed figure gets flagged clearly until the user confirms it against real data.

## Phase 6. Demo and PPT (Template D, due 23 Aug; demo 26 to 27 Aug)

1. Follow the Demonstration Template's requirements. House style consistent with the UI theme.
2. Demo script: live Search Filings query with the reasoning trace open; Ask Agent end to end; Evaluation Results tab; the retrieval bug-fix story as the narrative centerpiece.
3. Rehearse timing; record a fallback screen capture in case the network fails during the live demo.

## Standing guardrails

- Verify before asserting: read actual files and rerun evaluations rather than trusting remembered numbers.
- Never fabricate data, quotes, gold answers, or citations. Verify any new source PDF actually opens before citing it.
- Commit after each working milestone; `data/` stays gitignored.
- Office file edits: check for lock files first; always do a visual QA pass (render to PDF, look at it) before declaring any document done.
- After any edit that touches multiple paragraphs of a Word document, re-verify structure: no duplicated paragraphs, no corrupted headings, all cross-section references still point to the correct section numbers, and no pagination artifacts (split tables, stray blank pages) in the rendered output.
