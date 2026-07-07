import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rag.search import load_chunks


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDINGS_PATH = "data/embeddings.npy"
DEFAULT_META_PATH = "data/dense_meta.json"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Cosine similarity를 dot product로 계산하기 위해 L2 normalize한다.
    """
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / max(norm, eps)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def make_embedding_text(chunk: Dict[str, Any]) -> str:
    """
    Dense retrieval에서도 section 정보가 중요하다.
    따라서 section과 text를 함께 embedding한다.
    """
    return f"Section: {chunk['section']}\nText:\n{chunk['text']}"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        embeddings = embeddings.astype(np.float32)
        return l2_normalize(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )[0]

        embedding = embedding.astype(np.float32)
        return l2_normalize(embedding)


class DenseIndex:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks and embeddings size mismatch: "
                f"{len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )

        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.model_name = model_name
        self.embedder: Optional[SentenceTransformerEmbedder] = None

    @classmethod
    def build(
        cls,
        chunks: List[Dict[str, Any]],
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
    ) -> "DenseIndex":
        embedder = SentenceTransformerEmbedder(model_name)

        texts = [make_embedding_text(chunk) for chunk in chunks]
        embeddings = embedder.encode_texts(texts, batch_size=batch_size)

        index = cls(
            chunks=chunks,
            embeddings=embeddings,
            model_name=model_name,
        )
        index.embedder = embedder
        return index

    @classmethod
    def load(
        cls,
        chunks: List[Dict[str, Any]],
        embeddings_path: Path,
        meta_path: Path,
    ) -> "DenseIndex":
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}\n"
                f"Run: python3 -m rag.dense build"
            )

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Dense metadata file not found: {meta_path}\n"
                f"Run: python3 -m rag.dense build"
            )

        embeddings = np.load(embeddings_path)

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        model_name = meta["model_name"]
        saved_chunk_ids = meta["chunk_ids"]
        current_chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        if saved_chunk_ids != current_chunk_ids:
            raise ValueError(
                "Dense index is stale: chunk ids do not match.\n"
                "Rebuild embeddings with:\n"
                "python3 -m rag.dense build --chunks data/chunks.jsonl"
            )

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            model_name=model_name,
        )

    def save(self, embeddings_path: Path, meta_path: Path) -> None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(embeddings_path, self.embeddings)

        meta = {
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "dim": int(self.embeddings.shape[1]),
            "chunk_ids": [chunk["chunk_id"] for chunk in self.chunks],
        }

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _get_embedder(self) -> SentenceTransformerEmbedder:
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(self.model_name)
        return self.embedder

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        embedder = self._get_embedder()
        query_vec = embedder.encode_query(query)

        # embeddings는 이미 normalize되어 있고, query_vec도 normalize되어 있으므로
        # dot product == cosine similarity
        scores = self.embeddings @ query_vec

        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            # full sort보다 빠른 top-k 후보 추출
            candidate_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        results = []
        for idx in top_indices[:top_k]:
            results.append(
                {
                    "score": float(scores[idx]),
                    "chunk": self.chunks[int(idx)],
                }
            )

        return results


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print(f"Query: {query}")
    print()
    print(f"Top-{len(results)} Dense Results")
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


def build_command(args: argparse.Namespace) -> None:
    chunks = load_chunks(Path(args.chunks))

    index = DenseIndex.build(
        chunks=chunks,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    index.save(
        embeddings_path=Path(args.embeddings),
        meta_path=Path(args.meta),
    )

    print(f"Built dense index")
    print(f"chunks: {len(chunks)}")
    print(f"dim: {index.embeddings.shape[1]}")
    print(f"embeddings: {args.embeddings}")
    print(f"metadata: {args.meta}")


def search_command(args: argparse.Namespace) -> None:
    chunks = load_chunks(Path(args.chunks))

    index = DenseIndex.load(
        chunks=chunks,
        embeddings_path=Path(args.embeddings),
        meta_path=Path(args.meta),
    )

    results = index.search(args.query, top_k=args.top_k)
    print_results(args.query, results)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--chunks", type=str, default="data/chunks.jsonl")
    build_parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS_PATH)
    build_parser.add_argument("--meta", type=str, default=DEFAULT_META_PATH)
    build_parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    build_parser.add_argument("--batch-size", type=int, default=32)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--chunks", type=str, default="data/chunks.jsonl")
    search_parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS_PATH)
    search_parser.add_argument("--meta", type=str, default=DEFAULT_META_PATH)
    search_parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "build":
        build_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()