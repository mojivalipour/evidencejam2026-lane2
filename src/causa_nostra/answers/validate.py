"""Deterministic checks that tie an answer file to the graph and question set.

These run after every LLM stage. They do not judge meaning; that is the
verifier's job. Problems are returned as strings so they can be logged in the
evidence trace.
"""

from __future__ import annotations

from collections.abc import Iterable

from causa_nostra.answers.models import AnswerEntry, AnswersFile
from causa_nostra.graph.store import GraphStore


def check_entry(entry: AnswerEntry, store: GraphStore) -> list[str]:
    return [
        f"{entry.id}: citation {edge_id} does not exist in the graph"
        for edge_id in entry.edge_ids
        if not store.has_edge(edge_id)
    ]


def check_file(
    answers: AnswersFile,
    store: GraphStore,
    question_ids: Iterable[str],
) -> list[str]:
    expected = list(question_ids)
    expected_set = set(expected)
    returned = [entry.id for entry in answers.answers]
    returned_set = set(returned)

    problems = [f"missing answer for question {qid}" for qid in expected if qid not in returned_set]
    problems += [f"answer for unknown question {aid}" for aid in returned_set - expected_set]
    for entry in answers.answers:
        problems.extend(check_entry(entry, store))
    return problems


if __name__ == "__main__":
    # Demo: a bad citation and a missing question are both reported.
    from causa_nostra.graph.store import GraphStore

    store = GraphStore.from_json("causal_graph.json")
    sample = AnswersFile(
        answers=[
            AnswerEntry(
                id="q1",
                status="answered",
                answer="Demo.",
                direction="unclear",
                confidence=0.4,
                certainty="very_low",
                edge_ids=["camo:edge_does_not_exist"],
            )
        ]
    )
    for problem in check_file(sample, store, ["q1", "q2"]):
        print(problem)
