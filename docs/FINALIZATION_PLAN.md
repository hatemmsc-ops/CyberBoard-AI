# CyberBoard-AI Thesis Finalization Plan

Handoff plan for any Claude model (Sonnet, Opus, Fable) continuing this work. Written 10 July 2026 (Week 5 of 13). Follow phases in order; each has a verify step. Read this whole file before acting.

## Scope rule

This plan and any session using it concern ONLY the thesis (CyberBoard-AI). All Semester 2 courses (IT9201, IT9203, IT9204, IT9205) are finished. Never touch them. See memory file feedback_thesis_session_scope.md.

## Context snapshot (verified 10 July 2026)

- Student: Hatem Isa, 202508993, MSc AI, Bahrain Polytechnic. Supervisor: Dr. Joshua Samuel (Joshua.Samuel@polytechnic.bh, office 26.106). Weekly Teams meetings Mon/Tue 9:00 PM India time. Weekly progress email expected (what was done, what is next).
- Code: /Users/hatemisa/Desktop/CyberBoard-AI. Git initialized, public remote live at https://github.com/hatemmsc-ops/CyberBoard-AI (README, MIT licence, evaluation JSONs included; data/chromadb and raw filings correctly gitignored).
- Thesis docs: /Users/hatemisa/Desktop/Thesis. Papers: /Users/hatemisa/Desktop/Thesis/Papers (44 verified PDFs; 58 sources total with proposal's 14).
- Data: 116 SEC filings (20 Fortune 500 companies) parsed to JSON; 18,136 chunks in ChromaDB (data/chromadb, 351 MB, gitignored). Bahrain Bourse: 0 files (blocker).
- Key results (all reproducible from data/financebench/eval_results.json and ablation_results.json):
  - FinanceBench retrieval hit rate 96.8% (30/31), was 54.8% before the BM25 ticker filter bug fix (filtering ran after global top k selection; fixed in src/pipeline/vector_store.py).
  - Reranking contribution +12.9 points (83.9% to 96.8%). Sparse only equals hybrid at 96.8% because dense is inactive without Azure.
- UI: Streamlit (src/ui/app.py), navy/gold theme, three modes, retrieval reasoning trace visualization working. Launch: streamlit run src/ui/app.py (or .claude/launch.json cyberboard-ui).
- Mid Review Report: the FINAL, handbook-compliant version is at Thesis/202508993_MidReview.docx and .pdf (filename per handbook 6.1 convention: StudentID_MidReview). This supersedes the earlier short Mid_Review_Report_Hatem_202508993.docx (built against Template B's brief prompts, which understated the handbook's real 3,000-4,000 word / 20+ reference / Gantt+risk-register requirement; keep for reference but do not resubmit or re-edit it). The current version: ~3,300 words, 30 IEEE references, updated Gantt chart, risk register, reflection section, AI Usage Statement. Not yet submitted (due 19 Jul).
- AI Usage Statement wording: per Dr. Joshua's explicit instruction (12 Jul 2026), keep this section minimal. State only that Claude Code was used for code implementation under student supervision, and Perplexity was used to find research articles (verified against source PDFs before citing). Do NOT mention the humanizer/AI-detection-reduction tool or any AI-detection score in the submitted document. See "AI usage disclosure" guardrail below.
- BLOCKER: no Azure OpenAI credentials (.env missing). Blocks dense embeddings, agent runs, H1/H2 testing, O3/O4/O5 completion.

## Hard deadlines

| Date | Deliverable |
|---|---|
| 19 Jul 2026 | Mid Review Report (Template B), 20% of grade |
| 21 Aug 2026 | Final Thesis (Template C) |
| 23 Aug 2026 | Final demo and PPT (Template D) |
| 26 to 27 Aug 2026 | Demonstration |

## Phase 0. User prerequisites (ask, do not start work that depends on them)

1. Azure OpenAI credentials into /Users/hatemisa/Desktop/CyberBoard-AI/.env (template in .env.example: endpoint, key, gpt-4o deployment, text-embedding-3-small deployment). Fallback per risk register: personal OpenAI subscription (adjust embedder/get_client accordingly).
2. Bahrain Bourse annual report PDFs (5 to 10 companies, English) from bahrainbourse.com. Suggested: NBB, BBK, Ahli United Bank, Alba, Batelco, GFH, Bahrain Islamic Bank. User downloads manually (site blocks automation); check ~/Downloads first.
3. GitHub remote: user creates repo, then push and update the Repository URL field in the Mid Review Report before submission.
4. Mid Review Report: user reads, decides on optional Streamlit screenshot in Section 4, submits to Moodle before 19 July.

## Phase 1. Azure activation (first coding work once .env exists)

1. Verify: python3 -c "import config; print(config.azure_available())" prints True.
2. Embed all 18,136 chunks with src/pipeline/embedder.py (batched, has backoff). Store embeddings into ChromaDB via HybridStore.add_chunks or a re-embed script. Expect real API cost; confirm with user before running.
3. Verify: a dense query returns nonzero dense_candidates in the search trace.
4. Smoke test agent: python3 -m src.agent.react_agent "What are Apple's main risk factors?" returns a grounded answer ending with the advisory disclaimer.
5. Re-run: python3 -m src.evaluation.financebench_eval then python3 -m src.evaluation.ablation_study. Now dense_only and true hybrid configs produce distinct numbers. Record all in the JSON files. H2 test: is rerank at least +10% precision over dense alone.
6. Commit.

## Phase 2. Bahrain Bourse corpus and GCC evaluation set

1. File PDFs into data/bahrain_bourse/{company_name}/. Parse with src/ingestion/bahrain_bourse_loader.py (pdfplumber path exists; extend to chunk and index with ticker style ids like NBB_AR_2025_section_N so ticker filtering works).
2. Verify: chunks appear in ChromaDB; a search filtered to a Bourse company returns its text.
3. Build custom GCC evaluation set: 30 to 50 QA pairs, hand written WITH THE USER (never fabricate gold answers; data integrity rule). Store as data/gcc_eval/questions.json mirroring matched_questions.json schema.
4. Run retrieval eval on the GCC set; save results. Commit.

## Phase 3. Full evaluation and hypothesis testing (needs Phases 1 and 2)

1. Full agent evaluation: run_full_eval in src/evaluation/financebench_eval.py (answer correctness, latency). Add LLM judge faithfulness scoring per Zheng et al. rubric (paper in Thesis/Papers/LLM_as_Judge_MTBench.pdf; RAG specific: Judge_as_Judge_RAG_Consistency.pdf).
2. Hallucination rate: percent of claims not traceable to retrieved sources (proposal Section 6.5).
3. H1 test: agentic RAG vs single pass RAG on faithfulness, paired t test or Wilcoxon, p < 0.05. Baselines per proposal: (1) single pass dense RAG, (2) closed book LLM.
4. Ablation per proposal O4: RAG only vs RAG+agent vs RAG+agent+rerank, with statistics.
5. Save every number to JSON; never quote a number in the thesis that does not exist in a results file.

## Phase 4. Explainability completion (O5 remainder)

1. Agent level reasoning trace: capture ReAct tool call chain (which tool, arguments, retrieved sources per call) and render in the UI Ask Agent mode, same cb-stage styling as the retrieval trace.
2. Verify in browser (launch.json cyberboard-ui; preview tools) with a live query.
3. Commit.

## Phase 5. Final thesis write up (Template C, start by Week 10)

1. Structure from Template_C_Final_Project_Report.docx. Reuse proposal Sections and Mid Review content where valid; update all numbers from the final JSON results.
2. Literature review: 58 sources available. Papers folder has themed coverage (RAG, agents, hybrid retrieval, XAI, hallucination, GRC, GCC regulation incl. the actual Bahrain CG Code 2022 and CBB Rulebook Vol 6, benchmarks, BM25, transformers, Sentence-BERT, HNSW, Loughran McDonald). Cite in IEEE numeric style.
3. Figures to produce (matplotlib or similar, save to CyberBoard-AI/docs/figures/): system architecture diagram; retrieval pipeline trace example; bug fix before/after bar (54.8 vs 96.8); ablation comparison chart; FinanceBench per company results; GCC set results; latency table.
4. Writing rules (mandatory): NO dashes as connectors anywhere (no em dash, en dash, or connector hyphen; reference codes like 10-K and model ids are fine). British or US spelling consistent with proposal.
5. Humanization workflow, revised 12 Jul 2026: walter_batch_humanize reliably CORRUPTS citation brackets in reference-dense text (tested three ways on the Mid Review Report: dropped [1]-[30] markers, duplicated a citation onto the wrong source, even with word-token substitutes like "CITEA" instead of "[1]"). Do NOT run any section containing IEEE citation markers through the humanizer, ever, regardless of preserve/exact_phrases settings. For citation-dense sections (Literature Review, Methodology, References), write directly and disclose as such; do not chase a low AI-detection score there. For citation-free narrative sections (Progress, Reflection, Ethics, Baseline prose), humanizing is viable but ALWAYS re-verify facts and proper-noun capitalization afterward: documented failures include misspelling "Claude Code" as "Claud Code", rewriting solo work as "the team", and lowercasing proper nouns/acronyms (azure openai, chromadb, bm25). Manually fix every such error before use. Best achieved score on narrative sections so far: 17/100 (not the originally hoped 7/100); trying harder degrades accuracy faster than it lowers the score, so stop rather than keep iterating.
6. Every reconstructed or assumed figure gets an italic red note (C0392B) until user confirms. Zero formula errors if any xlsx is produced (recalc via LibreOffice per CLAUDE.md).

## Phase 6. Demo and PPT (Template D, due 23 Aug; demo 26 to 27 Aug)

1. PPTX per Template_D_Demonstration.docx requirements. House style: Calibri unless template dictates; navy 121A2F headers consistent with the UI theme.
2. Demo script: Search Filings live query with reasoning trace open; Ask Agent end to end; Evaluation Results tab; the bug fix story (JPMorgan zero results anecdote) as the narrative centerpiece.
3. Rehearse timing; record fallback screen capture in case campus network fails.

## Standing guardrails

- AI Usage Statement wording (Dr. Joshua's explicit instruction, 12 Jul 2026): keep minimal. Only state that Claude Code was used for code implementation under student supervision, and Perplexity was used to find research articles (verified against source PDFs before citing). Never mention the humanizer, any AI-detection-reduction tool, or an AI-detection score in any submitted document (Mid Review, Final Thesis, or otherwise). This applies to every future assessment's AI Usage Statement, not just the Mid Review Report.
- Verify before asserting: read actual files and rerun evals rather than trusting remembered numbers.
- Never fabricate data, quotes, gold answers, or citations. All 44 PDFs in Thesis/Papers were download verified; if adding papers, verify the PDF opens (pypdf) before citing.
- Commit after each working milestone; data/ stays gitignored.
- Weekly: refresh the progress email draft for Dr. Joshua with real numbers.
- Office file edits: check for ~$ lock files first; visual QA loop (soffice to PDF, pdftoppm, actually look) before declaring any document done.
