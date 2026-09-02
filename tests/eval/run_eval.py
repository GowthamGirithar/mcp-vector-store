"""Offline retrieval eval for `hybrid_search`, against the golden dataset
built from tests/fixtures/attention.pdf.

For each golden question, runs `hybrid_search` and computes:
  - hit_rate: whether any returned chunk matches a ground-truth chunk index
  - MRR: reciprocal rank of the first matching chunk
  - (unless --no-judge) Ragas context_precision / context_recall, scored by
    an OpenAI judge against `ground_truth_answer`

Negative-control questions (empty `ground_truth_chunk_indices`) are excluded
from hit_rate/MRR/Ragas aggregates — there is no ground-truth chunk to score
against — and are reported separately so a human can eyeball whether
hybrid_search confidently returned an irrelevant chunk for an unanswerable
question.

Usage:
    .venv/bin/python -m tests.eval.run_eval [--no-judge]
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.eval.setup_services import init_eval_services
from mcp_vectordb.services import get_vector_db
from mcp_vectordb.tools.document_embedding import generate_document_embedding
from mcp_vectordb.tools.search import hybrid_search

EVAL_COLLECTION = "eval_attention"
PDF_PATH = "tests/fixtures/attention.pdf"
GOLDEN_PATH = Path("tests/eval/golden/attention_qa.jsonl")
REPORTS_DIR = Path("tests/eval/reports")


def load_golden(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def ensure_ingested() -> None:
    """Ingest attention.pdf into EVAL_COLLECTION if not already present.

    force=False: a prior ingestion (matched by content hash) is left as-is,
    so chunk_ids/chunk_index assignments stay stable across eval runs rather
    than churning on every invocation.
    """
    result = await generate_document_embedding(
        file_path=PDF_PATH, collection=EVAL_COLLECTION, force=False, ctx=None
    )
    print(result)


async def build_text_to_chunk_index() -> Dict[str, int]:
    """Map each stored chunk's exact text to its chunk_index.

    hybrid_search returns bare text strings, not ids, and chunk_ids are
    regenerated on every ingestion — chunk_index (deterministic parse order)
    is the only stable key the golden dataset can reference, so results are
    joined back to ground truth via an exact text match against this map.
    """
    vector_db = get_vector_db()
    docs = await vector_db.get_all_documents(EVAL_COLLECTION)
    return {doc.text: doc.metadata.get("chunk_index") for doc in docs}


def compute_hit_rate_and_mrr(
    retrieved_texts: List[str],
    ground_truth_indices: List[int],
    text_to_index: Dict[str, int],
) -> Dict[str, float]:
    ground_truth_set = set(ground_truth_indices)
    rank_of_first_hit: Optional[int] = None
    for rank, text in enumerate(retrieved_texts, start=1):
        chunk_index = text_to_index.get(text)
        if chunk_index in ground_truth_set:
            rank_of_first_hit = rank
            break
    return {
        "hit": 1.0 if rank_of_first_hit is not None else 0.0,
        "mrr": 1.0 / rank_of_first_hit if rank_of_first_hit is not None else 0.0,
    }


async def score_with_ragas(
    question: str, retrieved_texts: List[str], reference: str, judge_llm
) -> Dict[str, float]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=retrieved_texts,
        reference=reference,
    )
    precision_metric = LLMContextPrecisionWithReference(llm=judge_llm)
    recall_metric = LLMContextRecall(llm=judge_llm)
    precision = await precision_metric.single_turn_ascore(sample)
    recall = await recall_metric.single_turn_ascore(sample)
    return {"context_precision": precision, "context_recall": recall}


async def run(use_judge: bool) -> Dict[str, Any]:
    await init_eval_services()
    await ensure_ingested()
    text_to_index = await build_text_to_chunk_index()

    judge_llm = None
    if use_judge:
        from tests.eval.judge import get_judge_llm

        judge_llm = get_judge_llm()

    golden = load_golden(GOLDEN_PATH)
    per_question_results = []

    for item in golden:
        retrieved_texts = await hybrid_search(
            query=item["question"], collection=EVAL_COLLECTION, filters=None, ctx=None
        )

        row: Dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "difficulty": item["difficulty"],
            "ground_truth_chunk_indices": item["ground_truth_chunk_indices"],
            "retrieved_count": len(retrieved_texts),
            "retrieved_preview": [t[:200] for t in retrieved_texts[:5]],
        }

        is_negative = not item["ground_truth_chunk_indices"]
        if not is_negative:
            row.update(
                compute_hit_rate_and_mrr(
                    retrieved_texts, item["ground_truth_chunk_indices"], text_to_index
                )
            )
            if judge_llm is not None:
                row.update(
                    await score_with_ragas(
                        item["question"], retrieved_texts, item["ground_truth_answer"], judge_llm
                    )
                )

        per_question_results.append(row)
        print(f"[{item['id']}] {item['difficulty']:10s} hit={row.get('hit')} mrr={row.get('mrr')}")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "used_judge": use_judge,
        "results": per_question_results,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    scored = [r for r in results if r["difficulty"] != "negative"]
    if not scored:
        return {}
    agg = {
        "hit_rate": sum(r["hit"] for r in scored) / len(scored),
        "mrr": sum(r["mrr"] for r in scored) / len(scored),
    }
    if "context_precision" in scored[0]:
        agg["context_precision"] = sum(r["context_precision"] for r in scored) / len(scored)
        agg["context_recall"] = sum(r["context_recall"] for r in scored) / len(scored)
    return agg


def write_report(report: Dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORTS_DIR / f"hybrid_search_eval_{stamp}.json"
    md_path = REPORTS_DIR / f"hybrid_search_eval_{stamp}.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    agg = aggregate(report["results"])
    lines = [
        "# hybrid_search retrieval eval — attention.pdf golden set",
        "",
        f"Generated: {report['generated_at']}  |  Judge used: {report['used_judge']}",
        "",
        "## Aggregate (excludes negative-control questions)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in agg.items():
        lines.append(f"| {k} | {v:.3f} |")

    lines += ["", "## Per-question", "", "| id | difficulty | hit | mrr" + (
        " | ctx_precision | ctx_recall |" if report["used_judge"] else " |"
    ), "|---|---|---|---" + ("|---|---|" if report["used_judge"] else "|")]
    for r in report["results"]:
        base = f"| {r['id']} | {r['difficulty']} | {r.get('hit', '-')} | {r.get('mrr', '-')}"
        if report["used_judge"]:
            base += f" | {r.get('context_precision', '-')} | {r.get('context_recall', '-')} |"
        else:
            base += " |"
        lines.append(base)

    lines += ["", "## Negative controls (manual review)", ""]
    for r in report["results"]:
        if r["difficulty"] == "negative":
            lines.append(f"- **{r['id']}**: {r['question']}")
            for preview in r["retrieved_preview"][:2]:
                lines.append(f"  - retrieved: {preview}...")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="Skip Ragas/OpenAI judge scoring; hit-rate/MRR only")
    args = parser.parse_args()

    report = asyncio.run(run(use_judge=not args.no_judge))
    md_path = write_report(report)
    agg = aggregate(report["results"])
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"{k}: {v:.3f}")
    print(f"\nReport written to {md_path}")


if __name__ == "__main__":
    main()
