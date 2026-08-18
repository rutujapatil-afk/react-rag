import json
from pathlib import Path

from react_rag.evaluation.retrieval import recall_at_k, reciprocal_rank
from react_rag.retrieval.bm25 import BM25Retriever
from react_rag.retrieval.corpus import load_corpus


DATASET_PATH = Path("data/raw/hotpotqa/validation.json")
CORPUS_PATH = Path("data/processed/hotpotqa/corpus.jsonl")

NUM_EXAMPLES = 100
TOP_K = 10


def load_examples(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= limit:
                break

            yield json.loads(line)


def main() -> None:
    documents = load_corpus(CORPUS_PATH)
    retriever = BM25Retriever(documents)

    recalls_at_1 = []
    recalls_at_5 = []
    recalls_at_10 = []
    reciprocal_ranks = []

    examples = list(load_examples(DATASET_PATH, NUM_EXAMPLES))

    for example in examples:
        results = retriever.retrieve(
            example["question"],
            top_k=TOP_K,
        )

        retrieved_titles = [result.title for result in results]

        gold_titles = example["supporting_facts"]["title"]

        recalls_at_1.append(
            recall_at_k(retrieved_titles, gold_titles, 1)
        )

        recalls_at_5.append(
            recall_at_k(retrieved_titles, gold_titles, 5)
        )

        recalls_at_10.append(
            recall_at_k(retrieved_titles, gold_titles, 10)
        )

        reciprocal_ranks.append(
            reciprocal_rank(retrieved_titles, gold_titles)
        )

    print(f"Examples: {len(examples)}")
    print(f"Recall@1:  {sum(recalls_at_1) / len(recalls_at_1):.4f}")
    print(f"Recall@5:  {sum(recalls_at_5) / len(recalls_at_5):.4f}")
    print(f"Recall@10: {sum(recalls_at_10) / len(recalls_at_10):.4f}")
    print(f"MRR:       {sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}")


if __name__ == "__main__":
    main()