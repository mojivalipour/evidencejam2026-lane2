# TEAM: CAUSA NOSTRA
## Lane 2 Proposal: Evidence-Grounded Reasoning Assistant

### Objective

Build an assistant that answers questions from the supplied CAMO graph, cites its evidence, and clearly states when evidence is missing.

### Workflow

| Step | What happens |
|---|---|
| Understand | Qwen identifies the intervention, outcome, context, and alternative search terms. |
| Retrieve | Haystack searches graph records. Python collects relevant edges, node details, and source sentences. |
| Reason | Qwen compares the evidence and drafts an answer, distinguishing effects, null results, disagreements, and missing evidence. |
| Verify | Python validates citations and output structure. A verifier agent checks the answer against the evidence using a fixed rubric. |
| Respond | Produce the answer, direction, evidence certainty, confidence, and supporting edge IDs. |

Allow one broader search when evidence coverage is insufficient. Revise answers when verification finds incorrect interpretations or unsupported claims.

### Verification and Confidence

The verifier returns `pass`, `partial`, or `fail` for each criterion, with a short explanation and supporting edge IDs.

| Criterion | Weight |
|---|---:|
| Evidence matches the intervention, outcome, and context | 30% |
| Direction, null findings, and causal language are interpreted correctly | 30% |
| Every factual claim is supported by its citations | 25% |
| Search covers alternative terms and potentially conflicting evidence | 15% |

Python maps `pass = 1`, `partial = 0.5`, and `fail = 0`, then calculates the weighted score.

Invalid citations and unsupported claims require correction; they cannot be offset by passing other criteria.

For `no_evidence` answers, use a separate rubric assessing concept matching, search coverage, and whether relevant evidence was overlooked. Empty retrieval alone is insufficient.

The rubric score is a provisional confidence estimate, not a calibrated probability. Use independently checked examples to improve the mapping from rubric scores to correctness.

Keep `certainty` separate: it describes evidence strength, while `confidence` describes confidence that our answer correctly represents the graph.

### Technology

- **Haystack:** retrieval and workflow orchestration.
- **Qwen:** question interpretation, evidence reasoning, and rubric verification.
- **Pydantic:** answer format validation.
- **Python CAMO module:** graph lookup and schema-specific checks.

Use the supplied LLM endpoint through `BRINE_URL`, `BRINE_MODEL`, and `BRINE_API_KEY`. Pin dependencies and avoid other runtime network calls.

### Deliverables

- One command: `./run.sh questions.yaml answers.json`.
- Valid `answers.json` with `"team": "CAUSA NOSTRA"` and one entry per question.
- A simple evidence view showing cited claims, source sentences, and verification results.
- Tests covering direction errors, null results, conflicting evidence, misleading matches, and missing evidence.

### Scope

Reason over the supplied evidence. Do not estimate new causal effects or simulate counterfactuals.