from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


@dataclass
class VerificationResult:
    label: str
    score: float


class NLIVerifier:
    """Local natural-language-inference evidence verifier."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.device = torch.device("cpu")

        print(f"Loading NLI model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def verify(
        self,
        claim: str,
        evidence: str,
    ) -> VerificationResult:
        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        label_id = int(
            torch.argmax(probabilities).item()
        )

        label = self.model.config.id2label[label_id]

        return VerificationResult(
            label=label.lower(),
            score=float(probabilities[label_id]),
        )