import json
from pathlib import Path


def load_corpus(path: str | Path) -> list[dict]:
    """Load a JSONL document corpus."""
    path = Path(path)

    documents = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            documents.append(json.loads(line))

    return documents