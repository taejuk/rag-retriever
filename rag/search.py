import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

from rag.tokenizer import tokenize, normalize


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def score_chunk(query: str, chunk: Dict[str, Any]) -> float:
    query_tokens = tokenize(query)

    section = chunk["section"]
    text = chunk["text"]

    chunk_tokens = tokenize(section + "\n" + text)
    tf = Counter(chunk_tokens)

    score = 0.0

    # 1. token overlap score
    for token in query_tokens:
        score += tf[token]

    # 2. section title에 query token이 있으면 boost
    normalized_section = normalize(section)
    for token in set(query_tokens):
        if token in normalized_section:
            score += 2.0

    # 3. 원문 substring match boost
    normalized_text = normalize(text)
    for token in set(query_tokens):
        if token in normalized_text:
            score += 0.5

    return score


def search_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    scored = []

    for chunk in chunks:
        score = score_chunk(query, chunk)
        if score > 0:
            scored.append(
                {
                    "score": score,
                    "chunk": chunk,
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print(f"Query: {query}")
    print()
    print(f"Top-{len(results)} Results")
    print("=" * 80)

    for rank, item in enumerate(results, start=1):
        score = item["score"]
        chunk = item["chunk"]

        print(f"[{rank}] score={score:.2f}")
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
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))
    results = search_chunks(args.query, chunks, top_k=args.top_k)
    print_results(args.query, results)


if __name__ == "__main__":
    main()