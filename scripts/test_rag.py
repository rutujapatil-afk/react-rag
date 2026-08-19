import faiss
from sentence_transformers import SentenceTransformer

from react_rag.generation.ollama_client import OllamaGenerator
from react_rag.pipeline import StandardRAG
from react_rag.retrieval.corpus import load_corpus
from react_rag.retrieval.dense import DenseRetriever


CORPUS_PATH = "data/processed/hotpotqa/dev_corpus.jsonl"
INDEX_PATH = "data/processed/hotpotqa/dev_dense.faiss"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> None:
    print("Loading corpus...")
    documents = load_corpus(CORPUS_PATH)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    retriever = DenseRetriever(
        documents=documents,
        model=embedding_model,
        index=index,
    )

    generator = OllamaGenerator(
        model="qwen2.5:3b",
    )

    rag = StandardRAG(
        retriever=retriever,
        generator=generator,
        top_k=5,
    )
    question = (
    "Were Scott Derrickson and Ed Wood of the same nationality?"
    )


    print()
    print("Question:", question)
    print()
    print("Generating answer...")

    result = rag.answer(question)

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.answer)

    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result.sources:
        print(
            f"- {source['title']} "
            f"(score={source['score']:.4f})"
        )

    print()
    print("=" * 60)
    print("RETRIEVED TEXT")
    print("=" * 60)

    retrieved = retriever.retrieve(
        question,
        top_k=5,
    )

    for index, document in enumerate(retrieved, start=1):
        print()
        print(f"[Source {index}]")
        print(f"Title: {document.title}")
        print(document.text)


if __name__ == "__main__":
    main()