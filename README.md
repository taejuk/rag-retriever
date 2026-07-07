# rag-retriever

Implemented a minimal retrieval pipeline for technical RAG:

- Markdown ingestion
- Heading-based chunking
- Source metadata preservation
- Keyword-based retrieval
- Hit@k and MRR evaluation

Initial evaluation on 9 simple queries achieved Hit@1 = 1.0 and MRR@3 = 1.0.
This indicates that the v0 pipeline works, but the benchmark is too easy due to lexical overlap between queries and section titles.
