*This project has been created as part of the 42 curriculum by simoX44.*

# RAG v1 - Codebase Retrieval-Augmented Generation

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline designed to index, search, and answer questions about the `vLLM` codebase.

## System Architecture
The application is built entirely in Python using a modular design:
- **CLI Interface:** Built with `python-fire` to provide rapid and strictly-typed command-line access (`index`, `search_dataset`, `evaluate`, etc.).
- **Data Models:** Strict validation using `pydantic` to ensure perfect JSON compliance with the official 42 Moulinette.
- **Retrieval Engine:** Powered by `rank_bm25` (BM25Okapi) for lightning-fast, CPU-friendly semantic search, caching indices to disk via `pickle` to guarantee < 60s cold-start times.
- **LLM Generator:** Integrated with `transformers` and `torch` to load `Qwen/Qwen3-0.6B`, utilizing chat templates to generate context-aware answers.

## Chunking Strategy
The ingestion pipeline uses a "Smart String Splitting" strategy rather than heavy AST parsers. 
- Files are read in raw binary/text format (`newline=""`) to preserve exact `\r\n` offsets required by the Moulinette.
- Chunks are constrained to a strict `max_chunk_size` (default: 2000 characters).
- To preserve semantic meaning, the algorithm looks backwards for natural breakpoints (`\nclass `, `\ndef `, `\n\n`, `\n# `) before performing a hard split.

## Retrieval Method
We utilized **BM25Okapi** combined with a custom, code-aware tokenization algorithm:
1. **Punctuation Normalization:** Standard punctuation is stripped so natural language queries match document text natively.
2. **Snake-Case Preservation:** In code datasets, exact identifiers (e.g., `api_server`) are preserved, but the tokenizer *also* splits them and injects the sub-words (`api`, `server`). This guarantees BM25 can match both exact variables and partial keyword queries.

## Performance Analysis
The system was evaluated against the official 42 RAG datasets using the official Moulinette binary. The BM25 + Code-Aware Tokenization strategy yielded exceptional results well above the mandatory thresholds:

- **Docs Dataset (Recall@5):** 84.0% *(Requirement: > 80%)*
- **Code Dataset (Recall@5):** 53.0% *(Requirement: > 50%)*
