from react_rag.verification.nli_verifier import NLIVerifier


def main() -> None:
    verifier = NLIVerifier()

    tests = [
        (
            "Scott Derrickson is American.",
            "Scott Derrickson is an American director, "
            "screenwriter and producer.",
        ),
        (
            "Ed Wood was American.",
            "Edward Davis Wood Jr. was an American filmmaker, "
            "actor, writer, producer, and director.",
        ),
        (
            "Scott Derrickson was born in France.",
            "Scott Derrickson is an American director, "
            "screenwriter and producer.",
        ),
    ]

    for claim, evidence in tests:
        result = verifier.verify(
            claim,
            evidence,
        )

        print()
        print("CLAIM:")
        print(claim)

        print("EVIDENCE:")
        print(evidence)

        print("LABEL:", result.label)
        print("SCORE:", round(result.score, 4))


if __name__ == "__main__":
    main()