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

        prompt = f"""You are a question-answering system.

Answer the question using the information in the sources below.

IMPORTANT:
- The answer is contained in the provided sources.
- Find the relevant source(s).
- Perform the necessary reasoning across sources.
- Give the shortest direct answer possible.
- Do not explain your reasoning.
- Do not say "Insufficient evidence" unless the sources truly contain
  no information relevant to the question.
- Do not use outside knowledge.

Question:
{question}

Sources:
{context}

Final answer:
"""

        result = self.generator.generate(prompt)

        return RAGResult(
            question=question,
            answer=result.answer,
            sources=sources,
        )