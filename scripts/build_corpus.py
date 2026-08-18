from pathlib import Path
import json


INPUT_DIR = Path("data/raw/hotpotqa")
OUTPUT_DIR = Path("data/processed/hotpotqa")
OUTPUT_FILE = OUTPUT_DIR / "corpus.jsonl"


def load_split(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = {}

    for split in ("train", "validation"):
        path = INPUT_DIR / f"{split}.json"

        for example in load_split(path):
            context = example["context"]

            for title, sentences in zip(
                context["title"],
                context["sentences"],
            ):
                if title not in documents:
                    documents[title] = {
                        "title": title,
                        "sentences": sentences,
                        "text": " ".join(sentences),
                    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for document in documents.values():
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    print(f"Documents: {len(documents):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()