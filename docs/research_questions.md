# REACT-RAG Research Questions

## Working Title

REACT-RAG: Reliability-Aware Evidence-State Modeling for Adaptive Retrieval and Calibrated Abstention

## Research Problem

Retrieval-Augmented Generation provides external evidence to language models, but retrieved evidence does not necessarily imply that a generated claim is factually supported.

A RAG system must therefore determine whether the available evidence is sufficient to answer, whether additional retrieval is warranted, or whether the claim should be withheld.

## Central Research Question

Can the evidence state surrounding a generated claim be transformed into a calibrated estimate of factual correctness, and can this estimate improve answer/retrieve/abstain decisions in Retrieval-Augmented Generation?

## Research Questions

### RQ1

Which evidence-state signals best predict the factual correctness of generated claims?

Candidate signals include:

- evidence support
- evidence contradiction
- evidence coverage
- cross-document agreement
- source diversity
- retrieval relevance
- number of independent supporting sources
- number of independent contradicting sources

### RQ2

Can a combination of evidence-state signals produce a calibrated estimate of claim correctness?

### RQ3

Can calibrated reliability estimates improve the decision of whether a RAG system should answer, retrieve additional evidence, or abstain?

### RQ4

How robust are evidence-reliability estimates under increasing evidence conflict and evidence insufficiency?

## Hypotheses

### H1

Evidence-state features predict factual correctness beyond retrieval similarity alone.

### H2

Combining support, contradiction, coverage, and agreement produces better-calibrated correctness estimates than individual evidence signals.

### H3

Using calibrated reliability to control answer/retrieve/abstain decisions reduces factual errors at comparable answer coverage relative to standard RAG and strong adaptive-RAG baselines.

### H4

Reliability estimates remain informative under controlled increases in evidence conflict and evidence insufficiency.

## Primary Contributions

1. An evidence-state representation for claim-level RAG reliability analysis.
2. A calibrated claim-level reliability estimator.
3. A reliability-aware answer/retrieve/abstain decision policy.
4. A controlled evidence-conflict and evidence-insufficiency evaluation protocol.
5. An open-source reproducible implementation and evaluation framework.

## Important Non-Claims

The project does not claim to solve hallucination completely.

The project does not assume that any individual evidence signal is inherently reliable.

The reliability score will only be interpreted probabilistically after empirical calibration demonstrates that interpretation is justified.