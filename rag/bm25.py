import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag.tokenizer import tokenize
from rag.search import load_chunks


class BM25Index:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        section_boost: int = 3,
    ):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.section_boost = section_boost

        self.num_docs = len(chunks)
        self.doc_lens: Dict[int, int] = {}
        self.avgdl = 0.0

        # term -> list of (doc_idx, term_frequency)
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        # term -> document frequency
        self.df: Dict[str, int] = {}

        # term -> idf
        self.idf: Dict[str, float] = {}

        self._build()

    def _chunk_tokens(self, chunk: Dict[str, Any]) -> List[str]:
        """
        기술 문서에서는 section title이 중요하다.
        그래서 section token을 여러 번 넣어서 boost를 준다.
        """
        section_tokens = tokenize(chunk["section"])
        text_tokens = tokenize(chunk["text"])

        return section_tokens * self.section_boost + text_tokens

    def _build(self) -> None:
        total_len = 0

        for doc_idx, chunk in enumerate(self.chunks):
            tokens = self._chunk_tokens(chunk)
            tf = Counter(tokens)

            doc_len = len(tokens)
            self.doc_lens[doc_idx] = doc_len
            total_len += doc_len

            for term, freq in tf.items():
                self.postings[term].append((doc_idx, freq))

        self.avgdl = total_len / self.num_docs if self.num_docs > 0 else 0.0

        for term, posting_list in self.postings.items():
            df = len(posting_list)
            self.df[term] = df

            # BM25 commonly uses this smoothed IDF.
            # 희귀한 term일수록 idf가 커진다.
            self.idf[term] = math.log(
                1.0 + (self.num_docs - df + 0.5) / (df + 0.5)
            )

    def _score_term(self, term: str, doc_idx: int, tf: int) -> float:
        idf = self.idf.get(term, 0.0)
        doc_len = self.doc_lens[doc_idx]

        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * (
            1.0 - self.b + self.b * doc_len / self.avgdl
        )

        return idf * numerator / denominator

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)

        scores = defaultdict(float)

        for term in query_tokens:
            posting_list = self.postings.get(term, [])

            for doc_idx, tf in posting_list:
                scores[doc_idx] += self._score_term(term, doc_idx, tf)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in ranked[:top_k]:
            results.append(
                {
                    "score": score,
                    "chunk": self.chunks[doc_idx],
                }
            )

        return results


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print(f"Query: {query}")
    print()
    print(f"Top-{len(results)} BM25 Results")
    print("=" * 80)

    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]

        print(f"[{rank}] score={item['score']:.4f}")
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
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))
    index = BM25Index(chunks, k1=args.k1, b=args.b)

    results = index.search(args.query, top_k=args.top_k)
    print_results(args.query, results)


if __name__ == "__main__":
    main()