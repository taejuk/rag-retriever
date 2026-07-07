# Retrieval Benchmark

Top-k: 3

| Dataset | Retriever | N | Hit@1 | Hit@3 | MRR@3 |
|---|---|---:|---:|---:|---:|
| simple | keyword | 9 | 0.8889 | 1.0000 | 0.9444 |
| simple | bm25 | 9 | 1.0000 | 1.0000 | 1.0000 |
| simple | dense | 9 | 0.7778 | 0.8889 | 0.8333 |
| paraphrase | keyword | 5 | 0.6000 | 1.0000 | 0.8000 |
| paraphrase | bm25 | 5 | 0.8000 | 0.8000 | 0.8000 |
| paraphrase | dense | 5 | 0.4000 | 0.8000 | 0.5667 |
