from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path("data/raw/hotpotqa")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    for split, data in dataset.items():
        output_path = OUTPUT_DIR / f"{split}.json"

        data.to_json(
            output_path,
            orient="records",
            lines=True,
        )

        print(f"Saved {split}: {len(data)} examples -> {output_path}")


if __name__ == "__main__":
    main()