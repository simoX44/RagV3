import pickle
import string
from typing import List, Any
from pathlib import Path
from rank_bm25 import BM25Okapi
from chunking import Chunk

INDEX_DIR = Path("data/processed")


def tokenize(text: str) -> List[str]:
    """Code-aware tokenizer that respects Python snake_case."""
    text = text.lower()
    punct = string.punctuation.replace('_', '')
    clean_text = text.translate(str.maketrans(punct, ' ' * len(punct)))
    tokens = clean_text.split()
    extended_tokens = []
    for t in tokens:
        extended_tokens.append(t)
        if '_' in t:
            extended_tokens.extend(t.split('_'))
    return extended_tokens


def build_and_save_index(chunks: List[Chunk]) -> None:
    """Builds separate BM25 indexes for docs and code files."""
    print(f"Building index for {len(chunks)} chunks...")

    docs_chunks = [c for c in chunks if c.kind in (".md", ".txt")]
    code_chunks = [c for c in chunks if c.kind == ".py"]

    print(f"Docs chunks: {len(docs_chunks)} | Code chunks: {len(code_chunks)}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Building docs index...")
    docs_corpus = [tokenize(c.text) for c in docs_chunks]
    bm25_docs = BM25Okapi(docs_corpus)

    with open(INDEX_DIR / "bm25_docs.pkl", "wb") as f:
        pickle.dump(bm25_docs, f)
    with open(INDEX_DIR / "chunks_docs.pkl", "wb") as f:
        pickle.dump(docs_chunks, f)

    print("Building code index...")
    code_corpus = [tokenize(c.text) for c in code_chunks]
    bm25_code = BM25Okapi(code_corpus)

    with open(INDEX_DIR / "bm25_code.pkl", "wb") as f:
        pickle.dump(bm25_code, f)
    with open(INDEX_DIR / "chunks_code.pkl", "wb") as f:
        pickle.dump(code_chunks, f)

    print(f"Indexes saved to {INDEX_DIR}")


def load_index(index_type: str = "all") -> Any:
    """Loads BM25 index(es) from disk."""
    try:
        if index_type == "docs":
            with open(INDEX_DIR / "bm25_docs.pkl", "rb") as f:
                bm25 = pickle.load(f)
            with open(INDEX_DIR / "chunks_docs.pkl", "rb") as f:
                chunks = pickle.load(f)
            return bm25, chunks

        elif index_type == "code":
            with open(INDEX_DIR / "bm25_code.pkl", "rb") as f:
                bm25 = pickle.load(f)
            with open(INDEX_DIR / "chunks_code.pkl", "rb") as f:
                chunks = pickle.load(f)
            return bm25, chunks

        else:
            with open(INDEX_DIR / "bm25_docs.pkl", "rb") as f:
                bm25_docs = pickle.load(f)
            with open(INDEX_DIR / "chunks_docs.pkl", "rb") as f:
                chunks_docs = pickle.load(f)
            with open(INDEX_DIR / "bm25_code.pkl", "rb") as f:
                bm25_code = pickle.load(f)
            with open(INDEX_DIR / "chunks_code.pkl", "rb") as f:
                chunks_code = pickle.load(f)
            return bm25_docs, chunks_docs, bm25_code, chunks_code

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Index not found in {INDEX_DIR}. Please run 'index' first."
        )


def search_bm25(
    bm25: BM25Okapi,
    chunks: List[Chunk],
    query: str,
    k: int = 10
) -> List[Chunk]:
    """Search the index for the top-k results."""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    score_index_pairs = [(scores[i], i) for i in range(len(scores))]
    score_index_pairs.sort(reverse=True, key=lambda x: x[0])
    top_indices = [index for score, index in score_index_pairs[:k] if score > 0]
    return [chunks[i] for i in top_indices]
