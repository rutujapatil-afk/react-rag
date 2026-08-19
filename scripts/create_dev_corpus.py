import json
import random
from pathlib import Path


SOURCE = Path("data/processed/hotpotqa/corpus.jsonl")
VALIDATION = Path("data/raw/hotpotqa/validation.json")
TARGET = Path("data/processed/hotpotqa/dev_corpus.jsonl")

NUM_DOCUMENTS = 50_000
SEED = 42


def load_corpus(path: Path) -> dict[str, dict]:
    documents = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)
            documents[document["title"]] = document

    return documents


def load_validation_gold_titles(path: Path) -> set[str]:
    titles = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            example = json.loads(line)

            titles.update(
                example["supporting_facts"]["title"]
            )

    return titles


def main() -> None:
    random.seed(SEED)

    print("Loading full corpus...")
    documents = load_corpus(SOURCE)

    print(f"Full corpus: {len(documents):,}")

    print("Collecting validation gold documents...")
    gold_titles = load_validation_gold_titles(VALIDATION)

    print(
        f"Validation gold documents: "
        f"{len(gold_titles):,}"
    )

    missing = gold_titles - documents.keys()

    if missing:
        raise ValueError(
            f"{len(missing)} validation gold documents "
            "are missing from the corpus."
        )

    selected_titles = set(gold_titles)

    remaining_titles = list(
        documents.keys() - selected_titles
    )

    random.shuffle(remaining_titles)

    remaining_needed = NUM_DOCUMENTS - len(selected_titles)

    if remaining_needed < 0:
        raise ValueError(
            "Number of gold documents exceeds "
            "development corpus size."
        )

    selected_titles.update(
        remaining_titles[:remaining_needed]
    )

    selected_documents = [
        documents[title]
        for title in selected_titles
    ]

    random.shuffle(selected_documents)

    TARGET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TARGET.open("w", encoding="utf-8") as file:
        for document in selected_documents:
            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Saved {len(selected_documents):,} documents "
        f"to {TARGET}"
    )


if __name__ == "__main__":
    main()