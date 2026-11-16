# TODOs

## Hybrid retrieval (next)

- [ ] Measure the baseline on `questions.yaml`: record which practice questions get `no_evidence` and why (the `outputs/real_run.json` cases: b2, c3, c7, b4, b5).
- [ ] Write the per-signal ranks (`retrieval_ranks`) into the verifier's search log.
- [ ] Update the reasoner and verifier prompts so they check that both the intervention and the outcome are covered, since one-sided edges can now appear.
- [ ] Compare the baseline and fused results on the practice questions before keeping the change.

## Synonym expansion (after hybrid retrieval)

- [ ] Ask `QuestionProcessor` for several synonyms per side, including measurement terms the graph uses (e.g. "emissions" → "methane flux").
- [ ] Keep synonyms as extra inputs to the same BM25 signals, not as a separate stage. Label them as search candidates, not equivalences.
- [ ] Check generated synonyms against the graph vocabulary and log which ones match nothing.
- [ ] Add tests for synonym-driven recall (e.g. b2 "methane emissions" reaches "methane flux" edges).

## Graph-neighbour signal (optional, only if measured)

- [ ] Only if evaluation shows same-entity findings are missed: link question terms to graph nodes and add one-hop neighbour edges as an extra RRF signal.
- [ ] Tag neighbour edges in the trace so the reasoner knows they are not direct matches.
- [ ] Do not chain edges to infer a new effect. Count edges from one document as one study.

## Graph vocabulary (deferred)

- [ ] Measure the baseline first: do the current search terms miss relevant graph labels on the practice questions?
- [ ] If they do, build a term-frequency / IDF index from `causal_graph.json` and use it to expand BM25 queries (retrieval side, no extra LLM tokens).
- [ ] Only if retrieval expansion is insufficient, consider passing top graph candidates to `QuestionProcessor`. Keep the original question terms alongside them.
- [ ] Compare results with and without the hint before keeping it.

## Reasoner output normalization (deferred)

- [ ] Add a normalizer between raw model JSON and `AnswerEntry` in `agents/reasoner.py`.
- [ ] Trim whitespace and lowercase exact enum matches only. Do not fuzzy-map values such as "decreased" or "no effect".
- [ ] Reject unknown keys with a warning log and a count, rather than silently dropping them.
- [ ] Reject `no_evidence` entries that also carry a direction, certainty, or citations. Do not null them silently.
- [ ] Reject `answered` entries with any citation not in the supplied records. Do not prune citations silently.
- [ ] Reject confidence outside [0, 1]. Do not rescale.
- [ ] Use the actual search log for `searched_for`. Fail if it is empty.
- [ ] Convert all rejections to `AgentOutputError` so the existing retry path handles them.
- [ ] Log rejection and retry rates across the batch to decide whether the prompt needs fixing.
- [ ] Add tests for each branch in `tests/test_agents.py`.
