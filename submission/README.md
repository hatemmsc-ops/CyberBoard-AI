# Submission archive

Deliverables for IT9099 Master's Thesis, MSc Artificial Intelligence, Bahrain Polytechnic.

**Hatem Isa, 202508993** — supervised by Dr. Joshua Samuel.

| File | Assessment | Submitted |
|---|---|---|
| `202508993_Thesis.pdf` | Final Project Thesis (30%) | 25 August 2026 |
| `202508993_Paper_ECIR2027_LNCS.pdf` | Essay / Paper (20%), ECIR 2027 LNCS format | due 30 August 2026 |
| `202508993_SimilarityReport.pdf` | Similarity self-check accompanying the thesis | 25 August 2026 |
| `CyberBoard-AI_Demonstration_Pack.pdf` | Demonstration and viva pack (30%) | due 31 August 2026 |
| `CyberBoard-AI_Demonstration_Slides.pptx` | Demonstration slides | due 31 August 2026 |

The demonstration is delivered on **1 September 2026**.

## Reproducing the results

Chapter 5 of the thesis belongs to commit `9660fb81282a491d96f404a50422adfde8fb7eb9`.
Cite that commit rather than the branch, since the branch moves.

```
git checkout 9660fb81282a491d96f404a50422adfde8fb7eb9
```

From the repository root, five commands rebuild every reported number:

```
python -m src.evaluation.financebench_eval   # United States retrieval hit rate
python -m src.evaluation.ablation_study      # retrieval ablation
python -m src.evaluation.full_eval --set gcc # Gulf Cooperation Council answer quality
python -m src.evaluation.baselines --set gcc # no retrieval and single pass baselines
python -m src.evaluation.multi_run --set gcc # repeated runs for variance
```

Result files live in `data/` and are tracked in version control. Continuous integration
fails the build if any result file cited in the thesis goes missing.

See the repository root `README.md` for setup and the full architecture description.
