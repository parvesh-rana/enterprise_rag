"""`python -m evaluation` (a.k.a. `make eval`).

Runs the full pipeline against `qa_set.jsonl`, computes retrieval metrics
and LLM-as-judge faithfulness, and writes one CSV + one Markdown report into
evaluation/runs/<timestamp>/.

CLI flags:
  --no-judge          skip LLM faithfulness (fast retrieval-only run)
  --no-reranker       run without the cross-encoder (ablation)
  --limit N           run only the first N examples
  --top-k K           override final top-k
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from core.config import get_settings
from core.logging import configure_logging, get_logger
from evaluation.dataset import QAExample, load_qa_set
from evaluation.judge import JudgeResult, judge_answer
from evaluation.metrics import RetrievalScores, aggregate, recall_at_k, reciprocal_rank
from generation.answer import answer as run_answer
from generation.llm import get_llm_client
from retrieval.dense import RetrievedChunk
from retrieval.filters import RetrievalFilter
from retrieval.pipeline import retrieve

log = get_logger(__name__)


def _run_one(
    example: QAExample,
    *,
    use_reranker: bool,
    final_top_k: int,
    pool_top_k: int,
) -> tuple[list[RetrievedChunk], dict[str, float], float, str, str]:
    filt = RetrievalFilter(
        company=example.filter.company,
        year=example.filter.year,
        item=example.filter.item,
    )
    t0 = time.perf_counter()
    retrieved = retrieve(
        example.question,
        filt=filt,
        final_top_k=pool_top_k,        # we want top-10 for Recall@10 measurement
        use_reranker=use_reranker,
    )
    retrieval_s = time.perf_counter() - t0

    per = {
        "recall@5": recall_at_k(retrieved, example, k=5),
        "recall@10": recall_at_k(retrieved, example, k=10),
        "mrr": reciprocal_rank(retrieved, example),
    }

    # Generate using only top-final_top_k as context (the production behavior).
    t1 = time.perf_counter()
    ans = run_answer(
        example.question,
        retrieved[:final_top_k],
        request_id=f"eval-{example.id}",
    )
    gen_s = time.perf_counter() - t1
    return retrieved, per, retrieval_s + gen_s, ans.text, ans.model


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    settings_dump: dict,
    n: int,
    scores: RetrievalScores,
    faithfulness: float | None,
    by_kind: dict[str, dict[str, float]],
    elapsed_s: float,
) -> None:
    lines = [
        "# Eval run",
        "",
        f"- **Date:** {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- **Examples:** {n}",
        f"- **Elapsed:** {elapsed_s:.1f}s",
        f"- **Embedding model:** `{settings_dump['embedding_model']}`",
        f"- **Reranker:** `{settings_dump['reranker_model']}` (enabled={settings_dump['use_reranker']})",
        f"- **LLM:** `{settings_dump['llm_provider']}` / `{settings_dump['llm_model']}`",
        "",
        "## Aggregate scores",
        "",
        "| Metric        | Value |",
        "|---------------|------:|",
        f"| Recall@5      | {scores.recall_at_5:.3f} |",
        f"| Recall@10     | {scores.recall_at_10:.3f} |",
        f"| MRR           | {scores.mrr:.3f} |",
    ]
    if faithfulness is not None:
        lines.append(f"| Faithfulness  | {faithfulness:.3f} |")
    lines.append("")

    if by_kind:
        lines += [
            "## By question kind",
            "",
            "| Kind | n | Recall@5 | Recall@10 | MRR |",
            "|------|--:|---------:|----------:|----:|",
        ]
        for kind, agg in sorted(by_kind.items()):
            lines.append(
                f"| {kind} | {int(agg['n'])} | {agg['recall@5']:.3f} | "
                f"{agg['recall@10']:.3f} | {agg['mrr']:.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="evaluation")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM faithfulness scoring.")
    parser.add_argument("--no-reranker", action="store_true", help="Disable the cross-encoder.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N examples.")
    parser.add_argument("--top-k", type=int, default=None, help="Final top-k for generation.")
    args = parser.parse_args()

    settings = get_settings()
    qa_path = Path(__file__).parent / "qa_set.jsonl"
    examples = load_qa_set(qa_path)
    if args.limit:
        examples = examples[: args.limit]

    final_top_k = args.top_k or settings.final_top_k
    use_reranker = not args.no_reranker

    log.info(
        "eval.start",
        n=len(examples),
        use_reranker=use_reranker,
        final_top_k=final_top_k,
    )

    judge_client = None if args.no_judge else get_llm_client()

    rows: list[dict] = []
    judge_scores: list[float] = []
    t_start = time.perf_counter()

    for ex in tqdm(examples, desc="eval"):
        retrieved, per, latency_s, answer_text, model = _run_one(
            ex,
            use_reranker=use_reranker,
            final_top_k=final_top_k,
            pool_top_k=max(10, final_top_k),
        )

        verdict_label = ""
        verdict_score: float | None = None
        verdict_reason = ""
        if not args.no_judge:
            jr: JudgeResult = judge_answer(
                example=ex,
                answer_text=answer_text,
                cited_chunks=retrieved[:final_top_k],
                client=judge_client,
            )
            verdict_label = jr.verdict
            verdict_score = jr.score
            verdict_reason = jr.reason
            judge_scores.append(jr.score)

        rows.append(
            {
                "id": ex.id,
                "kind": ex.kind,
                "question": ex.question,
                "recall@5": per["recall@5"],
                "recall@10": per["recall@10"],
                "mrr": per["mrr"],
                "verdict": verdict_label,
                "faithfulness": "" if verdict_score is None else f"{verdict_score:.2f}",
                "latency_s": f"{latency_s:.2f}",
                "model": model,
                "answer": answer_text.replace("\n", " ").strip(),
                "reason": verdict_reason,
            }
        )

    elapsed = time.perf_counter() - t_start
    scores = aggregate([{k: r[k] for k in ("recall@5", "recall@10", "mrr")} for r in rows])

    by_kind: dict[str, dict[str, float]] = {}
    for r in rows:
        bucket = by_kind.setdefault(
            r["kind"], {"n": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
        )
        bucket["n"] += 1
        bucket["recall@5"] += r["recall@5"]
        bucket["recall@10"] += r["recall@10"]
        bucket["mrr"] += r["mrr"]
    for k, agg in by_kind.items():
        n = agg["n"] or 1.0
        agg["recall@5"] /= n
        agg["recall@10"] /= n
        agg["mrr"] /= n

    faithfulness = mean(judge_scores) if judge_scores else None

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(__file__).parent / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "results.csv", rows)
    _write_markdown(
        out_dir / "report.md",
        settings_dump={
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "use_reranker": use_reranker,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
        },
        n=len(rows),
        scores=scores,
        faithfulness=faithfulness,
        by_kind=by_kind,
        elapsed_s=elapsed,
    )

    print(json.dumps({
        "examples": len(rows),
        "recall@5": scores.recall_at_5,
        "recall@10": scores.recall_at_10,
        "mrr": scores.mrr,
        "faithfulness": faithfulness,
        "report_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
