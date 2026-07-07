# rag-retriever

A minimal retrieval engine for technical RAG systems.

This project focuses on the retrieval part of RAG, especially for technical documents such as LLM serving notes, CUDA kernel notes, and scheduler documentation.

## Goal

The goal is to compare different retrieval strategies for technical RAG:

- Keyword overlap retrieval
- BM25 sparse retrieval
- Dense retrieval
- Hybrid retrieval

Current version implements:

- Markdown ingestion
- Heading-based chunking
- Source metadata preservation
- Keyword-based retrieval
- BM25 retrieval with inverted index
- Hit@k and MRR evaluation
- Failure case analysis

## Pipeline

```text
docs/*.md
  -> heading-based chunking
  -> chunks.jsonl
  -> retriever
      -> keyword overlap
      -> BM25
  -> top-k chunks
  -> Hit@k / MRR evaluation
```
