import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.bm25 import BM25Index
from rag.dense import DenseIndex, DEFAULT_EMBEDDINGS_PATH, DEFAULT_META_PATH
from rag.search import load_chunks


class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        bm25_index: Optional[BM25Index] = None,
        dense_index: Optional[DenseIndex] = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        self.chunks = chunks
        self.bm25_index = bm25_index or BM25Index(chunks)
        self.dense_index = dense_index
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        if self.dense_index is None:
            self.dense_index = DenseIndex.load(
                chunks=chunks,
                embeddings_path=Path(DEFAULT_EMBEDDINGS_PATH),
                meta_path=Path(DEFAULT_META_PATH),
            )

    def _add_rrf_scores(
        self,
        scores: Dict[int, float],
        source_ranks: Dict[int, Dict[str, int]],
        results: List[Dict[str, Any]],
        source_name: str,
        weight: float,
    ) -> None:
        for rank, item in enumerate(results, start=1):
            chunk = item["chunk"]
            chunk_id = chunk["chunk_id"]

            scores[chunk_id] += weight * (1.0 / (self.rrf_k + rank))
            source_ranks[chunk_id][source_name] = rank

    def search(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        candidate_k:
            BM25와 dense에서 각각 몇 개의 후보를 가져올지 결정한다.
            final top_k보다 크게 잡아야 fusion 효과가 있다.
        """
        bm25_results = self.bm25_index.search(query, top_k=candidate_k)
        dense_results = self.dense_index.search(query, top_k=candidate_k)

        scores: Dict[int, float] = defaultdict(float)
        source_ranks: Dict[int, Dict[str, int]] = defaultdict(dict)

        self._add_rrf_scores(
            scores=scores,
            source_ranks=source_ranks,
            results=bm25_results,
            source_name="bm25_rank",
            weight=self.bm25_weight,
        )

        self._add_rrf_scores(
            scores=scores,
            source_ranks=source_ranks,
            results=dense_results,
            source_name="dense_rank",
            weight=self.dense_weight,
        )

        chunk_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for chunk_id, score in ranked[:top_k]:
            final_results.append(
                {
                    "score": score,
                    "chunk": chunk_by_id[chunk_id],
                    "ranks": source_ranks[chunk_id],
                }
            )

        return final_results


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print(f"Query: {query}")
    print()
    print(f"Top-{len(results)} Hybrid Results")
    print("=" * 80)

    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        ranks = item.get("ranks", {})

        bm25_rank = ranks.get("bm25_rank", "-")
        dense_rank = ranks.get("dense_rank", "-")

        print(f"[{rank}] score={item['score']:.6f}")
        print(f"bm25_rank: {bm25_rank}, dense_rank: {dense_rank}")
        print(f"source: {chunk['source_path']}")
        print(f"section: {chunk['section']}")
        print(f"lines: {chunk['start_line']}-{chunk['end_line']}")
        print()
        print(chunk["text"][:700])
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--chunks", type=str, default="data/chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))

    retriever = HybridRetriever(
        chunks=chunks,
        rrf_k=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    print_results(args.query, results)


if __name__ == "__main__":
    main()