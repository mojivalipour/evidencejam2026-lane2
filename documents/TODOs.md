# TODOs

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
