import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.search import load_chunks, search_chunks


def load_questions(path: Path) -> List[Dict[str, Any]]:
    questions = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def is_relevant(result_chunk: Dict[str, Any], question_item: Dict[str, Any]) -> bool:
    """
    v0에서는 relevant_section 문자열이 section에 포함되는지로 평가한다.
    나중에는 relevant_chunk_ids 기반으로 바꾸면 된다.
    """
    relevant_section: Optional[str] = question_item.get("relevant_section")
    relevant_chunk_ids = question_item.get("relevant_chunk_ids")

    if relevant_chunk_ids is not None:
        return result_chunk["chunk_id"] in set(relevant_chunk_ids)

    if relevant_section is not None:
        return relevant_section.lower() == result_chunk["section"].lower()

    raise ValueError("Each question must have relevant_section or relevant_chunk_ids")


def evaluate(
    questions: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, float]:
    hit_at_1 = 0
    hit_at_k = 0
    mrr_sum = 0.0

    for q in questions:
        query = q["question"]
        results = search_chunks(query, chunks, top_k=top_k)

        first_relevant_rank = None

        for rank, item in enumerate(results, start=1):
            result_chunk = item["chunk"]

            if is_relevant(result_chunk, q):
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            hit_at_k += 1
            mrr_sum += 1.0 / first_relevant_rank

            if first_relevant_rank == 1:
                hit_at_1 += 1

    n = len(questions)

    return {
        "num_questions": n,
        "hit@1": hit_at_1 / n if n > 0 else 0.0,
        f"hit@{top_k}": hit_at_k / n if n > 0 else 0.0,
        f"mrr@{top_k}": mrr_sum / n if n > 0 else 0.0,
    }


def print_failure_cases(
    questions: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> None:
    print()
    print("Failure Cases")
    print("=" * 80)

    has_failure = False

    for q in questions:
        query = q["question"]
        results = search_chunks(query, chunks, top_k=top_k)

        if any(is_relevant(item["chunk"], q) for item in results):
            continue

        has_failure = True
        print(f"Query: {query}")
        print(f"Expected section: {q.get('relevant_section')}")
        print("Retrieved:")

        for rank, item in enumerate(results, start=1):
            chunk = item["chunk"]
            print(f"  [{rank}] score={item['score']:.2f} section={chunk['section']}")

        print("-" * 80)

    if not has_failure:
        print("No failure cases.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="eval/questions.jsonl")
    parser.add_argument("--chunks", type=str, default="data/chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    chunks = load_chunks(Path(args.chunks))

    metrics = evaluate(questions, chunks, top_k=args.top_k)

    print("Retrieval Evaluation")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print_failure_cases(questions, chunks, top_k=args.top_k)


if __name__ == "__main__":
    main()