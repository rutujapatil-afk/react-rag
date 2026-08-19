from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from react_rag.retrieval.corpus import load_corpus


CORPUS_PATH = Path("data/processed/hotpotqa/dev_corpus.jsonl")
INDEX_PATH = Path("data/processed/hotpotqa/dev_dense.faiss")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 128


def main() -> None:
    print("Loading corpus...")
    documents = load_corpus(CORPUS_PATH)

    print(f"Documents: {len(documents):,}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [
        document["text"]
        for document in documents
    ]

    print("Encoding documents...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    print(f"Embedding shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    print("Building FAISS index...")
    index.add(embeddings)

    print(f"FAISS documents: {index.ntotal:,}")

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    print(f"Saved index: {INDEX_PATH}")


if __name__ == "__main__":
    main()