import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from react_rag.evaluation.retrieval import (
    recall_at_k,
    reciprocal_rank,
)
from react_rag.retrieval.corpus import load_corpus
from react_rag.retrieval.dense import DenseRetriever


DATASET_PATH = Path("data/raw/hotpotqa/validation.json")
CORPUS_PATH = Path("data/processed/hotpotqa/dev_corpus.jsonl")
INDEX_PATH = Path("data/processed/hotpotqa/dev_dense.faiss")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

NUM_EXAMPLES = 100
TOP_K = 10


def load_examples(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= limit:
                break

            yield json.loads(line)


def main() -> None:
    print("Loading corpus...")
    documents = load_corpus(CORPUS_PATH)

    print(f"Documents: {len(documents):,}")

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print(f"FAISS documents: {index.ntotal:,}")

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    retriever = DenseRetriever(
        documents=documents,
        model=model,
        index=index,
    )

    examples = list(
        load_examples(
            DATASET_PATH,
            NUM_EXAMPLES,
        )
    )

    recalls_at_1 = []
    recalls_at_5 = []
    recalls_at_10 = []
    reciprocal_ranks = []

    print(f"Evaluating {len(examples)} examples...")

    for example in examples:
        results = retriever.retrieve(
            example["question"],
            top_k=TOP_K,
        )

        retrieved_titles = [
            result.title
            for result in results
        ]

        gold_titles = example["supporting_facts"]["title"]

        recalls_at_1.append(
            recall_at_k(
                retrieved_titles,
                gold_titles,
                1,
            )
        )

        recalls_at_5.append(
            recall_at_k(
                retrieved_titles,
                gold_titles,
                5,
            )
        )

        recalls_at_10.append(
            recall_at_k(
                retrieved_titles,
                gold_titles,
                10,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved_titles,
                gold_titles,
            )
        )

    print()
    print("=" * 40)
    print("Dense Retrieval Evaluation")
    print("=" * 40)
    print(f"Examples:  {len(examples)}")
    print(
        f"Recall@1:  "
        f"{sum(recalls_at_1) / len(recalls_at_1):.4f}"
    )
    print(
        f"Recall@5:  "
        f"{sum(recalls_at_5) / len(recalls_at_5):.4f}"
    )
    print(
        f"Recall@10: "
        f"{sum(recalls_at_10) / len(recalls_at_10):.4f}"
    )
    print(
        f"MRR:       "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}"
    )
    print("=" * 40)


if __name__ == "__main__":
    main()