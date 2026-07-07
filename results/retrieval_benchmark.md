# Retrieval Benchmark

Top-k: 3

| Dataset | Retriever | Type | Candidate K | BM25 Weight | Dense Weight | N | Hit@1 | Hit@3 | MRR@3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| simple | keyword | keyword | 10 | 1.00 | 1.00 | 9 | 0.8889 | 1.0000 | 0.9444 |
| simple | bm25 | bm25 | 10 | 1.00 | 1.00 | 9 | 1.0000 | 1.0000 | 1.0000 |
| simple | dense | dense | 10 | 1.00 | 1.00 | 9 | 0.7778 | 0.8889 | 0.8333 |
| simple | hybrid_1_1 | hybrid | 10 | 1.00 | 1.00 | 9 | 1.0000 | 1.0000 | 1.0000 |
| simple | hybrid_2_1 | hybrid | 10 | 2.00 | 1.00 | 9 | 1.0000 | 1.0000 | 1.0000 |
| simple | hybrid_3_1 | hybrid | 10 | 3.00 | 1.00 | 9 | 1.0000 | 1.0000 | 1.0000 |
| simple | hybrid_2_0.5 | hybrid | 10 | 2.00 | 0.50 | 9 | 1.0000 | 1.0000 | 1.0000 |
| paraphrase | keyword | keyword | 10 | 1.00 | 1.00 | 5 | 0.6000 | 1.0000 | 0.8000 |
| paraphrase | bm25 | bm25 | 10 | 1.00 | 1.00 | 5 | 0.8000 | 0.8000 | 0.8000 |
| paraphrase | dense | dense | 10 | 1.00 | 1.00 | 5 | 0.4000 | 0.8000 | 0.5667 |
| paraphrase | hybrid_1_1 | hybrid | 10 | 1.00 | 1.00 | 5 | 0.6000 | 0.8000 | 0.7000 |
| paraphrase | hybrid_2_1 | hybrid | 10 | 2.00 | 1.00 | 5 | 0.8000 | 0.8000 | 0.8000 |
| paraphrase | hybrid_3_1 | hybrid | 10 | 3.00 | 1.00 | 5 | 0.8000 | 0.8000 | 0.8000 |
| paraphrase | hybrid_2_0.5 | hybrid | 10 | 2.00 | 0.50 | 5 | 0.8000 | 0.8000 | 0.8000 |

## Notes

- `keyword` uses simple token overlap.
- `bm25` uses sparse lexical retrieval with an inverted index.
- `dense` uses cosine similarity over sentence embeddings.
- `hybrid_*` uses Reciprocal Rank Fusion over BM25 and dense retrieval.
- `hybrid_2_1` means BM25 weight = 2.0 and dense weight = 1.0.
- `hybrid_2_0.5` means BM25 weight = 2.0 and dense weight = 0.5.
