import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


def build_bm25_index(documents: list[dict]) -> BM25Okapi:
    tokenized_documents = [
        document["text"].lower().split()
        for document in documents
    ]

    return BM25Okapi(tokenized_documents)


def save_bm25_index(
    index: BM25Okapi,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(index, file)


def load_bm25_index(path: str | Path) -> BM25Okapi:
    path = Path(path)

    with path.open("rb") as file:
        return pickle.load(file)