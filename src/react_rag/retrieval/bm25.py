from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class RetrievedDocument:
    title: str
    text: str
    score: float


class BM25Retriever:
    """BM25 retriever over a pre-indexed document collection."""

    def __init__(
        self,
        documents: list[dict],
        index: BM25Okapi,
    ) -> None:
        if len(documents) != len(index.doc_freqs):
            raise ValueError(
                "Document count does not match BM25 index."
            )

        self.documents = documents
        self.index = index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        tokens = query.lower().split()

        scores = self.index.get_scores(tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            RetrievedDocument(
                title=self.documents[index]["title"],
                text=self.documents[index]["text"],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]