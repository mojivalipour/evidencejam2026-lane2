"""Deterministic rubric scoring. The verifier judges criteria; Python computes the score.

Scores are a provisional confidence estimate, not a calibrated probability.
They are not derived from certainty, citation count, retrieval score, or
agreement between repeated LLM calls.
"""

from __future__ import annotations

from collections.abc import Mapping

Judgement = str  # "pass" | "irrelevant" | "fail"

# Points per judgement. Weighted sums lie in [-1, 1]: a fail costs its full weight.
JUDGEMENT_VALUE: dict[str, float] = {"pass": 1.0, "irrelevant": 0.0, "fail": -1.0}

ANSWERED_WEIGHTS: dict[str, float] = {
    "evidence_match": 0.30,
    "direction_and_causal_language": 0.30,
    "citation_support": 0.25,
    "search_coverage": 0.15,
}

NO_EVIDENCE_WEIGHTS: dict[str, float] = {
    "interpretation": 0.30,
    "search_coverage": 0.40,
    "candidates_checked": 0.30,
}


def weighted_score(judgements: Mapping[str, Judgement], weights: Mapping[str, float]) -> float:
    missing = set(weights) - set(judgements)
    if missing:
        raise ValueError(f"missing rubric criteria: {sorted(missing)}")
    unknown = set(judgements) - set(weights)
    if unknown:
        raise ValueError(f"unknown rubric criteria: {sorted(unknown)}")
    for name, value in judgements.items():
        if value not in JUDGEMENT_VALUE:
            raise ValueError(f"invalid judgement for {name}: {value!r}")
    total = sum(weights[name] * JUDGEMENT_VALUE[judgements[name]] for name in weights)
    return round(total, 4)


if __name__ == "__main__":
    # Demo: score an answered entry where the citations pass and the search is irrelevant.
    judgements = {
        "evidence_match": "pass",
        "direction_and_causal_language": "pass",
        "citation_support": "pass",
        "search_coverage": "irrelevant",
    }
    print(f"answered score: {weighted_score(judgements, ANSWERED_WEIGHTS)}")
