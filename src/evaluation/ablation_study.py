"""Ablation study: test different retrieval configurations."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.evaluation.financebench_eval import evaluate_retrieval


@dataclass
class AblationConfig:
    name: str
    use_dense: bool = True
    use_sparse: bool = True
    use_rerank: bool = True
    alpha: float = 0.7
    chunk_size: int = 512
    k: int = 10


@dataclass
class AblationResult:
    config_name: str
    total_questions: int
    retrieval_hits: int
    hit_rate: float
    avg_latency: float


DEFAULT_CONFIGS = [
    AblationConfig("hybrid_rerank", use_dense=True, use_sparse=True, use_rerank=True, alpha=0.7),
    AblationConfig("hybrid_no_rerank", use_dense=True, use_sparse=True, use_rerank=False, alpha=0.7),
    AblationConfig("dense_only", use_dense=True, use_sparse=False, use_rerank=True, alpha=1.0),
    AblationConfig("sparse_only", use_dense=False, use_sparse=True, use_rerank=True, alpha=0.0),
    AblationConfig("sparse_no_rerank", use_dense=False, use_sparse=True, use_rerank=False, alpha=0.0),
    AblationConfig("alpha_0.5", use_dense=True, use_sparse=True, use_rerank=True, alpha=0.5),
    AblationConfig("alpha_0.9", use_dense=True, use_sparse=True, use_rerank=True, alpha=0.9),
]


def run_ablation(configs: list[AblationConfig] = None, matched_path: Path = None) -> list[AblationResult]:
    """Run ablation study across different retrieval configurations."""
    if configs is None:
        configs = DEFAULT_CONFIGS
    if matched_path is None:
        matched_path = config.DATA_DIR / "financebench" / "matched_questions.json"

    with open(matched_path) as f:
        questions = json.load(f)

    from src.pipeline.vector_store import HybridStore
    store = HybridStore()
    if not store.bm25:
        store.rebuild_bm25_from_collection()

    all_results = []

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Config: {cfg.name}")
        print(f"  dense={cfg.use_dense}, sparse={cfg.use_sparse}, rerank={cfg.use_rerank}, alpha={cfg.alpha}")

        # Skip dense-only configs if Gemini not available
        if cfg.use_dense and cfg.alpha > 0 and not config.gemini_available():
            if not cfg.use_sparse:
                print("  SKIPPED (requires Gemini API key for dense retrieval)")
                continue
            print("  NOTE: Running sparse-only (no Gemini key for dense)")

        hits = 0
        total_time = 0

        for q in questions:
            ticker = q["ticker"]
            evidence = q.get("evidence", [])

            start = time.time()

            query_emb = None
            if cfg.use_dense and config.gemini_available():
                from src.pipeline.embedder import embed_batch, get_client
                query_emb = embed_batch([q["question"]], get_client())[0]

            effective_alpha = cfg.alpha
            if not cfg.use_dense or query_emb is None:
                effective_alpha = 0.0
            if not cfg.use_sparse:
                effective_alpha = 1.0

            results = store.search(
                q["question"],
                query_embedding=query_emb,
                k=cfg.k,
                alpha=effective_alpha,
                rerank=cfg.use_rerank,
                ticker=ticker,
            )

            elapsed = time.time() - start
            total_time += elapsed

            retrieved_texts = [r["text"] for r in results]
            hit = False
            if evidence:
                for ev in evidence:
                    ev_text = ev.get("evidence_text", "") if isinstance(ev, dict) else str(ev)
                    ev_words = set(ev_text.lower().split()[:20])
                    for rt in retrieved_texts:
                        if len(ev_words & set(rt.lower().split())) > len(ev_words) * 0.3:
                            hit = True
                            break
                    if hit:
                        break
            if hit:
                hits += 1

        hit_rate = hits / len(questions) if questions else 0
        avg_latency = total_time / len(questions) if questions else 0

        ar = AblationResult(
            config_name=cfg.name,
            total_questions=len(questions),
            retrieval_hits=hits,
            hit_rate=hit_rate,
            avg_latency=avg_latency,
        )
        all_results.append(ar)
        print(f"  Result: {hits}/{len(questions)} hits ({100*hit_rate:.1f}%), avg {avg_latency:.2f}s")

    print(f"\n{'='*50}")
    print("ABLATION SUMMARY")
    print(f"{'Config':<25} {'Hits':>5} {'Rate':>7} {'Latency':>8}")
    print("-" * 50)
    for r in sorted(all_results, key=lambda x: x.hit_rate, reverse=True):
        print(f"{r.config_name:<25} {r.retrieval_hits:>5} {100*r.hit_rate:>6.1f}% {r.avg_latency:>7.2f}s")

    return all_results


def save_ablation(results: list[AblationResult], output_path: Path = None):
    if output_path is None:
        output_path = config.DATA_DIR / "financebench" / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    results = run_ablation()
    save_ablation(results)
