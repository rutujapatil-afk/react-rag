from collections.abc import Sequence


def recall_at_k(
    retrieved_titles: Sequence[str],
    gold_titles: Sequence[str],
    k: int,
) -> float:
    """Return 1 if any gold title appears in the top-k results."""
    retrieved = set(retrieved_titles[:k])
    gold = set(gold_titles)

    if not gold:
        return 0.0

    return float(bool(retrieved & gold))


def reciprocal_rank(
    retrieved_titles: Sequence[str],
    gold_titles: Sequence[str],
) -> float:
    """Return reciprocal rank of the first retrieved gold document."""
    gold = set(gold_titles)

    for rank, title in enumerate(retrieved_titles, start=1):
        if title in gold:
            return 1.0 / rank

    return 0.0