from __future__ import annotations

import math


def reciprocal_rank(ranked_relevant: list[bool]) -> float:
    return next((1.0 / rank for rank, value in enumerate(ranked_relevant, 1) if value), 0.0)


def ndcg(ranked_relevant: list[bool], relevant_total: int) -> float:
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, value in enumerate(ranked_relevant, 1) if value)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(relevant_total, len(ranked_relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]
