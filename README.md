# REACT-RAG

**Reliability-Aware Retrieval-Augmented Generation**

A research project investigating whether evidence consistency can predict factual reliability in Retrieval-Augmented Generation (RAG) systems and whether this signal can be used for adaptive retrieval and calibrated abstention.

## Research Goal

Standard RAG retrieves evidence and passes it to a language model, but retrieval alone does not guarantee that generated claims are adequately supported.

REACT-RAG investigates an evidence-aware pipeline that:

1. Retrieves relevant evidence.
2. Extracts claims from generated answers.
3. Measures supporting and contradicting evidence.
4. Estimates evidence reliability.
5. Retrieves additional evidence when reliability is insufficient.
6. Abstains when evidence remains inadequate or contradictory.

## Status

🚧 Research and development — initial setup.

## Repository Structure

```text
configs/       Configuration files
data/          Dataset management
docs/          Research documentation
experiments/   Experimental runs
notebooks/     Exploratory analysis
scripts/       Utility and experiment scripts
src/           Core implementation
tests/         Automated tests