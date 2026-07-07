import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.search import load_chunks, search_chunks
from rag.bm25 import BM25Index
from rag.dense import DenseIndex, DEFAULT_EMBEDDINGS_PATH, DEFAULT_META_PATH
from rag.hybrid import HybridRetriever

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
    retriever: str,
    candidate_k: int = 10,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0
) -> Dict[str, float]:
    hit_at_1 = 0
    hit_at_k = 0
    mrr_sum = 0.0

    bm25_index = None
    dense_index = None
    hybrid_retriever = None

    if retriever == "bm25":
        bm25_index = BM25Index(chunks)
    elif retriever == "dense":
        dense_index = DenseIndex.load(
            chunks=chunks,
            embeddings_path=Path(DEFAULT_EMBEDDINGS_PATH),
            meta_path=Path(DEFAULT_META_PATH),
        )
    elif retriever == "hybrid":
        hybrid_retriever = HybridRetriever(chunks, bm25_weight=bm25_weight, dense_weight=dense_weight)

    for q in questions:
        query = q["question"]

        if retriever == "keyword":
            results = search_chunks(query, chunks, top_k=top_k)
        elif retriever == "bm25":
            results = bm25_index.search(query, top_k=top_k)
        elif retriever == "dense":
            results = dense_index.search(query, top_k=top_k)    
        elif retriever == "hybrid":
            results = hybrid_retriever.search(
                query,
                top_k=top_k,
                candidate_k=max(candidate_k, top_k)
            )
        else:
            raise ValueError(f"Unknown retriever: {retriever}")

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
        "retriever": retriever,
        "hit@1": hit_at_1 / n if n > 0 else 0.0,
        f"hit@{top_k}": hit_at_k / n if n > 0 else 0.0,
        f"mrr@{top_k}": mrr_sum / n if n > 0 else 0.0,
    }


def print_failure_cases(
    questions: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    top_k: int,
    retriever: str,
    candidate_k: int = 10,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> None:
    print()
    print("Failure Cases")
    print("=" * 80)

    has_failure = False

    bm25_index = None
    dense_index = None
    hybrid_retriever = None

    if retriever == "bm25":
        bm25_index = BM25Index(chunks)
    elif retriever == "dense":
        dense_index = DenseIndex.load(
            chunks=chunks,
            embeddings_path=Path(DEFAULT_EMBEDDINGS_PATH),
            meta_path=Path(DEFAULT_META_PATH),
        )
    elif retriever == "hybrid":
        hybrid_retriever = HybridRetriever(
            chunks,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    for q in questions:
        query = q["question"]

        if retriever == "keyword":
            results = search_chunks(query, chunks, top_k=top_k)
        elif retriever == "bm25":
            results = bm25_index.search(query, top_k=top_k)
        elif retriever == "dense":
            results = dense_index.search(query, top_k=top_k)
        elif retriever == "hybrid":
            results = hybrid_retriever.search(
                query,
                top_k=top_k,
                candidate_k=max(candidate_k, top_k),
            )
        else:
            raise ValueError(f"Unknown retriever: {retriever}")

        if any(is_relevant(item["chunk"], q) for item in results):
            continue

        has_failure = True
        print(f"Query: {query}")
        print(f"Expected section: {q.get('relevant_section')}")
        print("Retrieved:")

        for rank, item in enumerate(results, start=1):
            chunk = item["chunk"]

            ranks = item.get("ranks")
            if ranks is not None:
                bm25_rank = ranks.get("bm25_rank", "-")
                dense_rank = ranks.get("dense_rank", "-")
                print(
                    f"  [{rank}] score={item['score']:.4f} "
                    f"section={chunk['section']} "
                    f"bm25_rank={bm25_rank} dense_rank={dense_rank}"
                )
            else:
                print(
                    f"  [{rank}] score={item['score']:.4f} "
                    f"section={chunk['section']}"
                )

        print("-" * 80)

    if not has_failure:
        print("No failure cases.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="eval/questions.jsonl")
    parser.add_argument("--chunks", type=str, default="data/chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        type=str,
        default="keyword",
        choices=["keyword", "bm25", "dense", "hybrid"],
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    chunks = load_chunks(Path(args.chunks))

    metrics = evaluate(questions, chunks, top_k=args.top_k, retriever=args.retriever, candidate_k=args.candidate_k, bm25_weight=args.bm25_weight, dense_weight=args.dense_weight)

    print("Retrieval Evaluation")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print_failure_cases(
        questions,
        chunks,
        top_k=args.top_k,
        retriever=args.retriever,
        candidate_k=args.candidate_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )

if __name__ == "__main__":
    main()