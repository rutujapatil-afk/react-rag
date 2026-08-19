from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedDocument:
    title: str
    text: str
    score: float


class DenseRetriever:
    """Dense vector retriever using Sentence Transformers + FAISS."""

    def __init__(
        self,
        documents: list[dict],
        model: SentenceTransformer,
        index: faiss.Index,
    ) -> None:
        if index.ntotal != len(documents):
            raise ValueError(
                "Document count does not match FAISS index."
            )

        self.documents = documents
        self.model = model
        self.index = index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            document = self.documents[index]

            results.append(
                RetrievedDocument(
                    title=document["title"],
                    text=document["text"],
                    score=float(score),
                )
            )

        return results