# TODOs

## Graph vocabulary (deferred)

- [ ] Measure the baseline first: do the current search terms miss relevant graph labels on the practice questions?
- [ ] If they do, build a term-frequency / IDF index from `causal_graph.json` and use it to expand BM25 queries (retrieval side, no extra LLM tokens).
- [ ] Only if retrieval expansion is insufficient, consider passing top graph candidates to `QuestionProcessor`. Keep the original question terms alongside them.
- [ ] Compare results with and without the hint before keeping it.
