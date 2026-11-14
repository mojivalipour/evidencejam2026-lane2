"""Rubric questions and deterministic confidence scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

Judgement = Literal["pass", "irrelevant", "fail"]

JUDGEMENT_VALUE: dict[Judgement, float] = {
    "pass": 1.0,
    "irrelevant": 0.0,
    "fail": -1.0,
}

ANSWERED_WEIGHTS: dict[str, float] = {
    "evidence_match": 0.30,
    "direction_and_causal_language": 0.30,
    "citation_support": 0.25,
    "search_coverage": 0.15,
}

ANSWERED_QUESTIONS: dict[str, dict[str, str]] = {
    "evidence_match": {
        "A1": "Does the cited evidence address the intervention or condition in the question?",
        "A2": "Does it address the specific outcome variable requested?",
        "A3": "Is the subject-to-object direction consistent with the question?",
        "A4": "Does it fit the requested species, ecosystem, and comparator context?",
    },
    "direction_and_causal_language": {
        "B1": "Does the effect direction match the evidence, node qualifiers, and comparator?",
        "B2": "Is a tested null result distinguished from absence of evidence?",
        "B3": "Are causal claims, associations, and uncertain causal language distinguished?",
        "B4": "Are agreement, disagreement, or insufficient evidence represented accurately?",
    },
    "citation_support": {
        "C1": "Do all cited edge IDs exist in the supplied graph?",
        "C2": "Does each cited edge support the claim for which it is used?",
        "C3": "Are all substantive factual claims supported by cited evidence?",
        "C4": "Does the answer avoid unsupported numbers, study details, mechanisms, or causal conclusions?",
    },
    "search_coverage": {
        "D1": "Were relevant alternative terms, synonyms, and abbreviations considered?",
        "D2": "Were semantically equivalent intervention and outcome nodes considered?",
        "D3": "Were potentially conflicting findings and null results included?",
        "D4": "Is evidence coverage sufficient to justify the answer?",
    },
}

NO_EVIDENCE_WEIGHTS: dict[str, float] = {
    "interpretation": 0.30,
    "search_coverage": 0.40,
    "no_evidence_justification": 0.30,
}

NO_EVIDENCE_QUESTIONS: dict[str, dict[str, str]] = {
    "interpretation": {
        "N1": "Was the requested intervention or condition correctly identified?",
        "N2": "Was the requested outcome, including measured attribute and target, correctly identified?",
    },
    "search_coverage": {
        "N3": "Were relevant synonyms, alternative names, and abbreviations searched?",
        "N4": "Were semantically equivalent nodes and their directed edges examined?",
        "N5": "Did the search fail to find evidence directly addressing the requested relationship?",
    },
    "no_evidence_justification": {
        "N6": "Does the answer avoid claiming that an effect does not exist merely because evidence was not found?",
    },
}

# Search-dependent criteria are kept in the rubric but excluded from the score:
# the verifier sees cited records, not the full search history.
ANSWERED_SCORED_CATEGORIES = (
    "evidence_match",
    "direction_and_causal_language",
    "citation_support",
)
NO_EVIDENCE_SCORED_CATEGORIES = (
    "interpretation",
    "no_evidence_justification",
)


def _question_ids(
    questions: Mapping[str, Mapping[str, str]],
    categories: Sequence[str],
) -> set[str]:
    unknown_categories = set(categories) - set(questions)
    if unknown_categories:
        raise ValueError(f"unknown rubric categories: {sorted(unknown_categories)}")
    return {
        question_id
        for category in categories
        for question_id in questions[category]
    }


def weighted_score(
    judgements: Mapping[str, Judgement],
    weights: Mapping[str, float],
    questions: Mapping[str, Mapping[str, str]],
    scored_categories: Sequence[str],
) -> float:
    """Return a [0, 1] score, renormalizing weights for evaluated categories.

    Individual judgments are -1 (fail), 0 (irrelevant), or +1 (pass). The
    weighted result is shifted from [-1, 1] to [0, 1].
    """
    expected = _question_ids(questions, scored_categories)
    missing = expected - set(judgements)
    if missing:
        raise ValueError(f"missing rubric questions: {sorted(missing)}")
    unknown = set(judgements) - expected
    if unknown:
        raise ValueError(f"unknown or unscored rubric questions: {sorted(unknown)}")
    invalid = {
        question_id: judgement
        for question_id, judgement in judgements.items()
        if judgement not in JUDGEMENT_VALUE
    }
    if invalid:
        raise ValueError(f"invalid rubric judgements: {invalid}")
    if not scored_categories:
        raise ValueError("at least one rubric category must be scored")
    missing_weights = set(scored_categories) - set(weights)
    if missing_weights:
        raise ValueError(f"missing rubric category weights: {sorted(missing_weights)}")

    total_weight = sum(weights[category] for category in scored_categories)
    if total_weight <= 0:
        raise ValueError("scored rubric categories must have positive total weight")

    raw_score = 0.0
    for category in scored_categories:
        category_questions = questions[category]
        if not category_questions:
            raise ValueError(f"rubric category {category!r} has no questions")
        category_score = sum(
            JUDGEMENT_VALUE[judgements[question_id]]
            for question_id in category_questions
        ) / len(category_questions)
        raw_score += weights[category] / total_weight * category_score

    return round((raw_score + 1.0) / 2.0, 4)
