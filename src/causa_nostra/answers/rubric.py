"""Single source of truth for the verification rubric.

Sub-items, category weights, gates, and scoring live here. The verifier builds its
prompt and parses its output from these definitions. The LLM judges sub-items only;
Python derives category judgements, gate results, and the score.

Scores are a provisional confidence estimate, not a calibrated probability.
They are not derived from certainty, citation count, retrieval score, or
agreement between repeated LLM calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

Judgement = str  # "pass" | "partial" | "fail" | "irrelevant"

# Scored judgements. "irrelevant" is not scored: the item does not apply to this question.
JUDGEMENT_VALUE: dict[str, float] = {"pass": 1.0, "partial": 0.5, "fail": 0.0}
IRRELEVANT: Judgement = "irrelevant"
JUDGEMENT_CHOICES: tuple[str, ...] = (*JUDGEMENT_VALUE, IRRELEVANT)
IRRELEVANT_GUIDANCE = (
    '"irrelevant" means the item does not apply to this question (for example, no species or '
    "context was requested). Use it sparingly. Citation items always apply and cannot be irrelevant."
)

# Answered drafts: sub-items grouped by scored category.
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

ANSWERED_WEIGHTS: dict[str, float] = {
    "evidence_match": 0.30,
    "direction_and_causal_language": 0.30,
    "citation_support": 0.25,
    "search_coverage": 0.15,
}

# A fail on any sub-item in these categories sends the draft to correction,
# regardless of its weighted score.
ANSWERED_GATE_CATEGORIES: tuple[str, ...] = (
    "evidence_match",
    "direction_and_causal_language",
    "citation_support",
)
ANSWERED_GATE_QUESTIONS: dict[str, str] = {
    item: text
    for category in ANSWERED_GATE_CATEGORIES
    for item, text in ANSWERED_QUESTIONS[category].items()
}

# No-evidence answers: scored sub-items grouped by category (N1-N5).
NO_EVIDENCE_QUESTIONS: dict[str, dict[str, str]] = {
    "interpretation": {
        "N1": "Was the requested intervention or condition correctly identified?",
        "N2": "Was the requested outcome, including measured attribute and target, correctly identified?",
    },
    "search_coverage": {
        "N3": "Were relevant synonyms, alternative names, and abbreviations searched?",
    },
    "candidates_checked": {
        "N4": "Were semantically equivalent nodes and their directed edges examined?",
        "N5": "Did the search fail to find evidence directly addressing the requested relationship?",
    },
}

NO_EVIDENCE_WEIGHTS: dict[str, float] = {
    "interpretation": 0.30,
    "search_coverage": 0.40,
    "candidates_checked": 0.30,
}

# Gate only: a fail means the answer claims absence of an effect from absence of evidence.
NO_EVIDENCE_GATE_QUESTIONS: dict[str, str] = {
    "N6": "Does the answer avoid claiming that an effect does not exist merely because evidence was not found?",
}


def weighted_score(judgements: Mapping[str, Judgement], weights: Mapping[str, float]) -> float:
    """Weighted score over applicable categories. Irrelevant categories are dropped and the
    remaining weights are renormalised; if nothing applies, the score is 0.0."""
    missing = set(weights) - set(judgements)
    if missing:
        raise ValueError(f"missing rubric criteria: {sorted(missing)}")
    unknown = set(judgements) - set(weights)
    if unknown:
        raise ValueError(f"unknown rubric criteria: {sorted(unknown)}")
    for name, value in judgements.items():
        if value not in JUDGEMENT_CHOICES:
            raise ValueError(f"invalid judgement for {name}: {value!r}")
    applicable = [name for name in weights if judgements[name] != IRRELEVANT]
    applicable_weight = sum(weights[name] for name in applicable)
    if applicable_weight == 0:
        return 0.0
    total = sum(weights[name] * JUDGEMENT_VALUE[judgements[name]] for name in applicable)
    return round(total / applicable_weight, 4)


CITATION_ITEMS: frozenset[str] = frozenset(ANSWERED_QUESTIONS["citation_support"])


def normalise(value: object, item: str = "") -> Judgement:
    """Lower-case a judgement. Anything not in JUDGEMENT_CHOICES counts as fail,
    and citation items cannot be marked irrelevant."""
    text = str(value).strip().lower()
    if text == IRRELEVANT and item in CITATION_ITEMS:
        return "fail"
    return text if text in JUDGEMENT_CHOICES else "fail"


def category_judgement(item_judgements: list[Judgement]) -> Judgement:
    """Worst-case over applicable items: any fail gives fail, then any partial gives partial.
    Irrelevant items are ignored; a category with no applicable items is irrelevant."""
    applicable = [j for j in item_judgements if j != IRRELEVANT]
    if not applicable:
        return IRRELEVANT
    if "fail" in applicable:
        return "fail"
    if "partial" in applicable:
        return "partial"
    return "pass"


@dataclass(frozen=True)
class RubricResult:
    items: dict[str, Judgement]
    criteria: dict[str, Judgement]
    score: float
    failed_gates: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failed_gates


def evaluate(
    items: Mapping[str, object],
    questions: Mapping[str, Mapping[str, str]],
    weights: Mapping[str, float],
    gate_questions: Mapping[str, str],
) -> RubricResult:
    """Score one entry. Missing or invalid sub-item judgements count as fail."""
    judged: dict[str, Judgement] = {
        item: normalise(items.get(item, "fail"), item)
        for category_items in questions.values()
        for item in category_items
    }
    for item in gate_questions:
        judged.setdefault(item, normalise(items.get(item, "fail"), item))
    criteria = {
        category: category_judgement([judged[item] for item in category_items])
        for category, category_items in questions.items()
    }
    failed_gates = tuple(item for item in gate_questions if judged[item] == "fail")
    return RubricResult(
        items=judged,
        criteria=criteria,
        score=weighted_score(criteria, weights),
        failed_gates=failed_gates,
    )


def evaluate_answered(items: Mapping[str, object]) -> RubricResult:
    return evaluate(items, ANSWERED_QUESTIONS, ANSWERED_WEIGHTS, ANSWERED_GATE_QUESTIONS)


def evaluate_no_evidence(items: Mapping[str, object]) -> RubricResult:
    return evaluate(items, NO_EVIDENCE_QUESTIONS, NO_EVIDENCE_WEIGHTS, NO_EVIDENCE_GATE_QUESTIONS)


def rubric_score(criteria: Mapping[str, object]) -> float:
    """Answered weighted score from category judgements. Missing categories count as fail."""
    judgements = {name: normalise(criteria.get(name, "fail")) for name in ANSWERED_WEIGHTS}
    return weighted_score(judgements, ANSWERED_WEIGHTS)


def _check_config(questions: Mapping[str, object], weights: Mapping[str, float]) -> None:
    if set(questions) != set(weights):
        raise ValueError(f"rubric categories and weights differ: {sorted(set(questions) ^ set(weights))}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("rubric weights must sum to 1")


_check_config(ANSWERED_QUESTIONS, ANSWERED_WEIGHTS)
_check_config(NO_EVIDENCE_QUESTIONS, NO_EVIDENCE_WEIGHTS)


if __name__ == "__main__":
    # Demo: all answered sub-items pass except D4, which is partial.
    items = {item: "pass" for category in ANSWERED_QUESTIONS.values() for item in category}
    items["D4"] = "partial"
    result = evaluate_answered(items)
    print(f"answered score: {result.score}, passed: {result.passed}")
