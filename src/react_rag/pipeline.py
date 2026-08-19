from dataclasses import dataclass

from react_rag.generation.ollama_client import OllamaGenerator
from react_rag.retrieval.dense import DenseRetriever


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[dict]


class StandardRAG:
    """Standard retrieval-augmented generation baseline."""

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: OllamaGenerator,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> RAGResult:
        retrieved = self.retriever.retrieve(
            question,
            top_k=self.top_k,
        )

        context_parts = []
        sources = []

        for index, document in enumerate(retrieved, start=1):
            context_parts.append(
                f"[Source {index}]\n"
                f"Title: {document.title}\n"
                f"Text: {document.text}"
            )

            sources.append(
                {
                    "title": document.title,
                    "score": document.score,
                }
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are a factual question-answering system.

Answer the question using the provided sources.

The sources may contain information about multiple entities.
Identify the relevant evidence and combine it when necessary.

Do not use outside knowledge.

If the provided sources contain enough evidence to answer the question,
give the answer directly.

Only answer "Insufficient evidence" if the sources genuinely do not
contain enough information to determine the answer.

Question:
{question}

Sources:
{context}

Answer:
"""

        result = self.generator.generate(prompt)

        return RAGResult(
            question=question,
            answer=result.answer,
            sources=sources,
        )