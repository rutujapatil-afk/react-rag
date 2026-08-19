from react_rag.generation.ollama_client import OllamaGenerator


def main() -> None:
    generator = OllamaGenerator()

    result = generator.generate(
        "What is the capital of France? "
        "Answer in one short sentence."
    )

    print("Model:", result.model)
    print("Answer:", result.answer)


if __name__ == "__main__":
    main()