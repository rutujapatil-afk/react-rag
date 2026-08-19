import json
import re
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from react_rag.generation.ollama_client import OllamaGenerator
from react_rag.pipeline import StandardRAG
from react_rag.retrieval.corpus import load_corpus
from react_rag.retrieval.dense import DenseRetriever


CORPUS_PATH = "data/processed/hotpotqa/dev_corpus.jsonl"
INDEX_PATH = "data/processed/hotpotqa/dev_dense.faiss"
VALIDATION_PATH = "data/raw/hotpotqa/validation.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "qwen2.5:3b"

NUM_EXAMPLES = 100
TOP_K = 5

OUTPUT_PATH = Path(
    "experiments/results/standard_rag_100.jsonl"
)


def normalize(text: str) -> str:
    """Normalize text for simple answer matching."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def answer_contains_gold(
    answer: str,
    gold: str,
) -> bool:
    """Simple normalized substring matching."""

    answer_norm = normalize(answer)
    gold_norm = normalize(gold)

    return gold_norm in answer_norm


def load_validation_examples():
    examples = []

    with open(
        VALIDATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            examples.append(json.loads(line))

    return examples[:NUM_EXAMPLES]


def main() -> None:
    print("Loading validation examples...")
    examples = load_validation_examples()

    print(f"Examples: {len(examples)}")

    print("Loading corpus...")
    documents = load_corpus(CORPUS_PATH)

    print(f"Documents: {len(documents)}")

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    retriever = DenseRetriever(
        documents=documents,
        model=embedding_model,
        index=index,
    )

    generator = OllamaGenerator(
        model=LLM_MODEL,
    )

    rag = StandardRAG(
        retriever=retriever,
        generator=generator,
        top_k=TOP_K,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_correct = 0
    total_retrieval_hits = 0
    total_latency = 0.0

    print()
    print("Starting evaluation...")
    print()

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for index_number, example in enumerate(
            examples,
            start=1,
        ):
            question = example["question"]
            gold_answer = example["answer"]

            gold_titles = set(
                example["supporting_facts"]["title"]
            )

            start_time = time.perf_counter()

            result = rag.answer(question)

            latency = time.perf_counter() - start_time

            retrieved_titles = [
                source["title"]
                for source in result.sources
            ]

            retrieved_set = set(retrieved_titles)

            retrieval_hit = bool(
                gold_titles & retrieved_set
            )

            correct = answer_contains_gold(
                result.answer,
                gold_answer,
            )

            if retrieval_hit:
                total_retrieval_hits += 1

            if correct:
                total_correct += 1

            total_latency += latency

            record = {
                "id": example["id"],
                "question": question,
                "gold_answer": gold_answer,
                "prediction": result.answer,
                "gold_supporting_titles": list(
                    gold_titles
                ),
                "retrieved_titles": retrieved_titles,
                "retrieval_hit": retrieval_hit,
                "correct": correct,
                "latency_seconds": round(
                    latency,
                    3,
                ),
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            print(
                f"[{index_number:03d}/{len(examples):03d}] "
                f"correct={correct} "
                f"retrieval={retrieval_hit} "
                f"time={latency:.1f}s"
            )

    accuracy = total_correct / len(examples)
    retrieval_rate = (
        total_retrieval_hits / len(examples)
    )
    average_latency = (
        total_latency / len(examples)
    )

    print()
    print("=" * 60)
    print("STANDARD RAG EVALUATION")
    print("=" * 60)
    print(f"Examples:          {len(examples)}")
    print(f"Answer accuracy:   {accuracy:.4f}")
    print(f"Retrieval hit:     {retrieval_rate:.4f}")
    print(
        f"Average latency:   "
        f"{average_latency:.2f}s"
    )
    print("=" * 60)
    print()
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()