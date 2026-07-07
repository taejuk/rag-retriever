import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from rag.eval import load_questions, evaluate
from rag.search import load_chunks


DEFAULT_DATASETS = {
    "simple": "eval/questions_simple.jsonl",
    "paraphrase": "eval/questions_paraphrase.jsonl",
}


DEFAULT_RUNS: List[Dict[str, Any]] = [
    {
        "name": "keyword",
        "retriever": "keyword",
        "candidate_k": 10,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
    },
    {
        "name": "bm25",
        "retriever": "bm25",
        "candidate_k": 10,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
    },
    {
        "name": "dense",
        "retriever": "dense",
        "candidate_k": 10,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
    },
    {
        "name": "hybrid_1_1",
        "retriever": "hybrid",
        "candidate_k": 10,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
    },
    {
        "name": "hybrid_2_1",
        "retriever": "hybrid",
        "candidate_k": 10,
        "bm25_weight": 2.0,
        "dense_weight": 1.0,
    },
    {
        "name": "hybrid_3_1",
        "retriever": "hybrid",
        "candidate_k": 10,
        "bm25_weight": 3.0,
        "dense_weight": 1.0,
    },
    {
        "name": "hybrid_2_0.5",
        "retriever": "hybrid",
        "candidate_k": 10,
        "bm25_weight": 2.0,
        "dense_weight": 0.5,
    },
]


def format_float(value: float) -> str:
    return f"{value:.4f}"


def run_benchmark(
    chunks_path: Path,
    datasets: Dict[str, Path],
    runs: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    chunks = load_chunks(chunks_path)
    rows: List[Dict[str, Any]] = []

    for dataset_name, questions_path in datasets.items():
        if not questions_path.exists():
            print(f"[skip] dataset not found: {questions_path}")
            continue

        questions = load_questions(questions_path)

        for run in runs:
            metrics = evaluate(
                questions=questions,
                chunks=chunks,
                top_k=top_k,
                retriever=run["retriever"],
                candidate_k=run["candidate_k"],
                bm25_weight=run["bm25_weight"],
                dense_weight=run["dense_weight"],
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "retriever": run["name"],
                    "retriever_type": run["retriever"],
                    "candidate_k": run["candidate_k"],
                    "bm25_weight": run["bm25_weight"],
                    "dense_weight": run["dense_weight"],
                    "num_questions": metrics["num_questions"],
                    "hit@1": metrics["hit@1"],
                    f"hit@{top_k}": metrics[f"hit@{top_k}"],
                    f"mrr@{top_k}": metrics[f"mrr@{top_k}"],
                }
            )

    return rows


def print_table(rows: List[Dict[str, Any]], top_k: int) -> None:
    if not rows:
        print("No benchmark results.")
        return

    current_dataset = None

    for row in rows:
        dataset = row["dataset"]

        if dataset != current_dataset:
            current_dataset = dataset
            print()
            print(f"Dataset: {dataset}")
            print("=" * 110)
            print(
                f"{'Retriever':<18} "
                f"{'Type':<10} "
                f"{'N':>5} "
                f"{'CandK':>7} "
                f"{'BM25_W':>8} "
                f"{'Dense_W':>8} "
                f"{'Hit@1':>10} "
                f"{'Hit@' + str(top_k):>10} "
                f"{'MRR@' + str(top_k):>10}"
            )
            print("-" * 110)

        print(
            f"{row['retriever']:<18} "
            f"{row['retriever_type']:<10} "
            f"{row['num_questions']:>5} "
            f"{row['candidate_k']:>7} "
            f"{row['bm25_weight']:>8.2f} "
            f"{row['dense_weight']:>8.2f} "
            f"{format_float(row['hit@1']):>10} "
            f"{format_float(row[f'hit@{top_k}']):>10} "
            f"{format_float(row[f'mrr@{top_k}']):>10}"
        )


def save_csv(rows: List[Dict[str, Any]], output_path: Path, top_k: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "retriever",
        "retriever_type",
        "candidate_k",
        "bm25_weight",
        "dense_weight",
        "num_questions",
        "hit@1",
        f"hit@{top_k}",
        f"mrr@{top_k}",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_markdown(rows: List[Dict[str, Any]], output_path: Path, top_k: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Retrieval Benchmark")
    lines.append("")
    lines.append(f"Top-k: {top_k}")
    lines.append("")
    lines.append(
        "| Dataset | Retriever | Type | Candidate K | BM25 Weight | Dense Weight | N | Hit@1 | Hit@{} | MRR@{} |".format(
            top_k,
            top_k,
        )
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| {dataset} | {retriever} | {retriever_type} | {candidate_k} | {bm25_weight} | {dense_weight} | {num_questions} | {hit1} | {hitk} | {mrrk} |".format(
                dataset=row["dataset"],
                retriever=row["retriever"],
                retriever_type=row["retriever_type"],
                candidate_k=row["candidate_k"],
                bm25_weight=f"{row['bm25_weight']:.2f}",
                dense_weight=f"{row['dense_weight']:.2f}",
                num_questions=row["num_questions"],
                hit1=format_float(row["hit@1"]),
                hitk=format_float(row[f"hit@{top_k}"]),
                mrrk=format_float(row[f"mrr@{top_k}"]),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `keyword` uses simple token overlap.")
    lines.append("- `bm25` uses sparse lexical retrieval with an inverted index.")
    lines.append("- `dense` uses cosine similarity over sentence embeddings.")
    lines.append("- `hybrid_*` uses Reciprocal Rank Fusion over BM25 and dense retrieval.")
    lines.append("- `hybrid_2_1` means BM25 weight = 2.0 and dense weight = 1.0.")
    lines.append("- `hybrid_2_0.5` means BM25 weight = 2.0 and dense weight = 0.5.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="data/chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    datasets = {
        name: Path(path)
        for name, path in DEFAULT_DATASETS.items()
    }

    rows = run_benchmark(
        chunks_path=Path(args.chunks),
        datasets=datasets,
        runs=DEFAULT_RUNS,
        top_k=args.top_k,
    )

    print_table(rows, top_k=args.top_k)

    out_dir = Path(args.out_dir)
    save_csv(rows, out_dir / "retrieval_benchmark.csv", top_k=args.top_k)
    save_markdown(rows, out_dir / "retrieval_benchmark.md", top_k=args.top_k)

    print()
    print(f"Saved CSV to {out_dir / 'retrieval_benchmark.csv'}")
    print(f"Saved Markdown to {out_dir / 'retrieval_benchmark.md'}")


if __name__ == "__main__":
    main()