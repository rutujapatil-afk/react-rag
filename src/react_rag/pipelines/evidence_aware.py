from dataclasses import dataclass

from react_rag.generation.ollama_client import OllamaGenerator
from react_rag.retrieval.dense import DenseRetriever
from react_rag.verification.nli_verifier import (
    NLIVerifier,
)


@dataclass
class EvidenceResult:
    claim: str
    label: str
    score: float
    evidence_title: str


@dataclass
class EvidenceAwareResult:
    question: str
    answer: str
    sources: list[dict]
    evidence: list[EvidenceResult]


class EvidenceAwareRAG:
    """RAG pipeline with post-generation evidence verification."""

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: OllamaGenerator,
        verifier: NLIVerifier,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.top_k = top_k

    def answer(
        self,
        question: str,
    ) -> EvidenceAwareResult:
        retrieved = self.retriever.retrieve(
            question,
            top_k=self.top_k,
        )

        context_parts = []
        sources = []

        for index, document in enumerate(
            retrieved,
            start=1,
        ):
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

Answer the question using ONLY the provided sources.

Identify the relevant evidence and reason across sources when necessary.

Give the shortest direct answer possible.

Do not explain your reasoning.

Do not use outside knowledge.

Question:
{question}

Sources:
{context}

Answer:
"""

        result = self.generator.generate(prompt)

        answer = result.answer.strip()

        evidence_results = []

        # Treat the generated answer as the claim to verify.
        for document in retrieved:
            verification = self.verifier.verify(
                answer,
                document.text,
            )

            evidence_results.append(
                EvidenceResult(
                    claim=answer,
                    label=verification.label,
                    score=verification.score,
                    evidence_title=document.title,
                )
            )

        return EvidenceAwareResult(
            question=question,
            answer=answer,
            sources=sources,
            evidence=evidence_results,
        )