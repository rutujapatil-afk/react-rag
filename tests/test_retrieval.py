from pathlib import Path

from react_rag.retrieval.bm25 import BM25Retriever
from react_rag.retrieval.corpus import load_corpus
from react_rag.retrieval.index import load_bm25_index


CORPUS_PATH = Path("data/processed/hotpotqa/corpus.jsonl")
INDEX_PATH = Path("data/processed/hotpotqa/bm25.pkl")


def test_bm25_retrieval():
    documents = load_corpus(CORPUS_PATH)
    index = load_bm25_index(INDEX_PATH)

    retriever = BM25Retriever(
        documents,
        index,
    )

    results = retriever.retrieve(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        top_k=5,
    )

    assert len(results) == 5
    assert results[0].title
    assert results[0].text