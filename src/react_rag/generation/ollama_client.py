from dataclasses import dataclass

import ollama


@dataclass
class GenerationResult:
    answer: str
    model: str


class OllamaGenerator:
    """Local LLM generator backed by Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
    ) -> None:
        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> GenerationResult:
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response["message"]["content"].strip()

        return GenerationResult(
            answer=answer,
            model=self.model,
        )