import argparse
import csv
from pathlib import Path
from typing import Dict, List

from rag.eval import load_questions, evaluate
from rag.search import load_chunks


DEFAULT_DATASETS = {
    "simple": "eval/questions_simple.jsonl",
    "paraphrase": "eval/questions_paraphrase.jsonl",
}

DEFAULT_RETRIEVERS = ["keyword", "bm25", "dense", "hybrid"]


def format_float(value: float) -> str:
    return f"{value:.4f}"


def run_benchmark(
    chunks_path: Path,
    datasets: Dict[str, Path],
    retrievers: List[str],
    top_k: int,
) -> List[Dict[str, object]]:
    chunks = load_chunks(chunks_path)
    rows: List[Dict[str, object]] = []

    for dataset_name, questions_path in datasets.items():
        if not questions_path.exists():
            print(f"[skip] dataset not found: {questions_path}")
            continue

        questions = load_questions(questions_path)

        for retriever in retrievers:
            metrics = evaluate(
                questions=questions,
                chunks=chunks,
                top_k=top_k,
                retriever=retriever,
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "retriever": retriever,
                    "num_questions": metrics["num_questions"],
                    "hit@1": metrics["hit@1"],
                    f"hit@{top_k}": metrics[f"hit@{top_k}"],
                    f"mrr@{top_k}": metrics[f"mrr@{top_k}"],
                }
            )

    return rows


def print_table(rows: List[Dict[str, object]], top_k: int) -> None:
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
            print("=" * 80)
            print(
                f"{'Retriever':<12} "
                f"{'N':>5} "
                f"{'Hit@1':>10} "
                f"{'Hit@' + str(top_k):>10} "
                f"{'MRR@' + str(top_k):>10}"
            )
            print("-" * 80)

        print(
            f"{row['retriever']:<12} "
            f"{row['num_questions']:>5} "
            f"{format_float(row['hit@1']):>10} "
            f"{format_float(row[f'hit@{top_k}']):>10} "
            f"{format_float(row[f'mrr@{top_k}']):>10}"
        )


def save_csv(rows: List[Dict[str, object]], output_path: Path, top_k: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "retriever",
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


def save_markdown(rows: List[Dict[str, object]], output_path: Path, top_k: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Retrieval Benchmark")
    lines.append("")
    lines.append(f"Top-k: {top_k}")
    lines.append("")
    lines.append("| Dataset | Retriever | N | Hit@1 | Hit@{} | MRR@{} |".format(top_k, top_k))
    lines.append("|---|---|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| {dataset} | {retriever} | {num_questions} | {hit1} | {hitk} | {mrrk} |".format(
                dataset=row["dataset"],
                retriever=row["retriever"],
                num_questions=row["num_questions"],
                hit1=format_float(row["hit@1"]),
                hitk=format_float(row[f"hit@{top_k}"]),
                mrrk=format_float(row[f"mrr@{top_k}"]),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        retrievers=DEFAULT_RETRIEVERS,
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