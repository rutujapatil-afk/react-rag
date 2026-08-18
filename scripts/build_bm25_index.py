from pathlib import Path

from react_rag.retrieval.corpus import load_corpus
from react_rag.retrieval.index import build_bm25_index, save_bm25_index


CORPUS_PATH = Path("data/processed/hotpotqa/corpus.jsonl")
INDEX_PATH = Path("data/processed/hotpotqa/bm25.pkl")


def main() -> None:
    print("Loading corpus...")
    documents = load_corpus(CORPUS_PATH)

    print(f"Documents: {len(documents):,}")
    print("Building BM25 index...")

    index = build_bm25_index(documents)

    print("Saving index...")
    save_bm25_index(index, INDEX_PATH)

    print(f"Saved: {INDEX_PATH}")


if __name__ == "__main__":
    main()