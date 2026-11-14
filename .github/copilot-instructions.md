# Copilot Instructions for CAUSA NOSTRA — Lane 2

## Project Objective

Build a simple, evidence-grounded reasoning assistant for Evidence Jam 2026.

Answer: “According to the supplied graph, what does intervention or condition X do to outcome Y?”

Prioritize:
- Correct interpretation of CAMO.
- Relevant, traceable citations.
- Honest uncertainty and missing-evidence answers.
- Low energy consumption.
- Reproducible execution from a clean checkout.

The supplied graph is the source of truth for this competition. Do not replace its findings with external knowledge.

## Project References

- `causal_graph.json` / `causal_graph.yaml`: supplied evidence.
- `schema/causalmosaic.yaml`: authoritative definitions of graph fields.
- `questions.yaml`: practice questions without reference answers.
- `documents/PROBLEM.md`: project scope.
- `documents/PROPOSAL.md`: proposed implementation.
- `assets/reference_ui.jpg`: illustrative evidence-view reference.

Read schema descriptions before implementing interpretation rules. UI examples are not evidence.

## Technology and Design

- **Haystack:** retrieval and workflow orchestration.
- **Qwen:** question interpretation, evidence reasoning, and verification.
- **Pydantic:** structured intermediate results and final output validation.
- **Small Python CAMO module:** graph lookup, provenance, and deterministic checks.

Keep components modular and independently testable. Prefer existing permissively licensed libraries compatible with the MIT repository, preserving required notices.

Avoid unnecessary services, agent frameworks, databases, MCP servers, or model training.

## Pipeline

### 1. Understand

Extract:
- Intervention or condition.
- Outcome and measured attribute.
- Explicit context constraints.
- Alternative search terms.

Prioritize the hidden-set format: one broad intervention or condition paired with one broad outcome.

Treat generated synonyms as search candidates, not established equivalences. Do not invent context restrictions.

### 2. Retrieve

Start with Haystack BM25 retrieval over readable graph records.

Include:
- Subject and object labels and `embedding_text`.
- Structured node attributes and qualifiers.
- Predicate and causal language.
- Original source sentence.
- Relevant context and provenance.

Use Python graph lookups to inspect matching nodes and connected edges.

Search for supporting, conflicting, and null findings. An empty top-k result does not establish that the graph has no evidence.

Allow one broader search when coverage is insufficient. Add more retrieval methods only when evaluation shows a clear benefit.

### 3. Reason

Give Qwen the question and retrieved evidence records.

Require it to:
- Match both the intervention and the outcome.
- Interpret direction using CAMO semantics.
- Distinguish effects, null findings, disagreements, and unclear findings.
- Preserve relevant context and comparisons.
- Cite the edges supporting each factual claim.

Do not treat related evidence as a direct answer. Do not infer a new causal effect by chaining graph edges.

### 4. Verify

Use deterministic Python checks for:
- Citation IDs exist.
- Required fields and enum values are valid.
- Output fields are consistent with the answer status.
- Schema-defined invariants, where applicable.

Use a verifier agent for semantic checks. Provide the question, draft answer, evidence records, and search log.

The verifier must return structured judgments, short explanations, and supporting edge IDs. Do not ask it to invent a confidence percentage.

### 5. Respond

Write one validated answer per input question.

Keep the answer short and grounded in the graph. Put detailed evidence and verification results in optional extra outputs.

## CAMO Interpretation Rules

- Read node qualifiers, entities, measured attributes, and edge semantics together.
- Do not infer outcome direction from the predicate alone.
- Do not confuse an unchanged node state with a tested null relationship.
- Distinguish `no_effect` from `no_evidence`.
- Do not turn associations or correlations into causal claims.
- Do not confuse species-specific findings with broader community outcomes.
- Do not treat multiple edges from one document as independent studies.
- Preserve conflicting findings; do not hide them through majority voting.
- Do not silently overwrite graph annotations using external ecological knowledge.
- Flag ambiguities or inconsistencies in the evidence trace.

## Verification Rubric

For answered questions:

| Criterion | Weight |
|---|---:|
| Evidence matches intervention, outcome, and requested context | 30% |
| Direction, null findings, and causal language are interpreted correctly | 30% |
| Every factual claim is supported by its citations | 25% |
| Search covers alternative terms and potentially conflicting evidence | 15% |

Map judgments to:
- `pass`: 1
- `partial`: 0.5
- `fail`: 0

Calculate the weighted score in Python.

Invalid citations, incorrect interpretations, and unsupported claims require correction. Passing other criteria cannot compensate for these failures.

Allow one revision after verification. Keep retries bounded and record unresolved limitations. Verification failure alone must not be relabeled as `no_evidence`.

### Missing-Evidence Rubric

For `no_evidence`, use a separate rubric:

| Criterion | Weight |
|---|---:|
| Intervention and outcome were interpreted correctly | 30% |
| Search covered plausible labels, attributes, and alternatives | 40% |
| Candidate edges were checked and none answers the question | 30% |

Record what was searched and why nearby evidence did not answer the question.

## Confidence and Certainty

- `confidence`: estimated probability that the answer correctly represents the graph.
- `certainty`: strength of the underlying evidence.

Calculate the rubric score deterministically. Initially use it as a provisional confidence estimate; do not describe it as empirically calibrated.

Use independently checked examples to improve the mapping between rubric scores and actual correctness.

Do not derive confidence directly from evidence certainty, citation count, retrieval score, or agreement between repeated LLM calls.

Grade certainty using graph evidence metadata and a documented aggregation rule. Do not invent missing study-quality information.

## Required Output Contract

The output must match the competition format exactly:

{
  "team": "CAUSA NOSTRA",
  "answers": [
    {
      "id": "H-001",
      "status": "answered",
      "answer": "...",
      "direction": "decrease",
      "confidence": 0.8,
      "certainty": "moderate",
      "edge_ids": ["existing-edge-id"],
      "searched_for": null,
      "extras": ["outputs/H-001_evidence.html"]
    },
    {
      "id": "H-002",
      "status": "no_evidence",
      "answer": "The graph has no evidence answering this intervention–outcome question.",
      "direction": null,
      "confidence": 0.7,
      "certainty": null,
      "edge_ids": [],
      "searched_for": "Describe the actual nodes, attributes, and relationships searched."
    }
  ]
}

The values above illustrate structure only; never copy placeholder citations or scores into real outputs.

Rules:
- `team` belongs at the top level.
- Use `id`, not `question_id`.
- Use `edge_ids`, not `supporting_edges`.
- `status`: `answered` or `no_evidence`.
- `direction`: `increase`, `decrease`, `no_effect`, `mixed`, or `unclear`.
- `certainty`: `high`, `moderate`, `low`, or `very_low`.
- `confidence`: a number between 0 and 1.
- For `no_evidence`, direction and certainty are null, edge IDs are empty, and `searched_for` is required.
- `extras` is optional and contains existing repo-relative output paths.
- Include every input question exactly once.
- Keep detailed verification explanations in extras rather than changing the required contract.

## Execution Constraints

Provide one command:

    ./run.sh questions.yaml answers.json

The input path may point to hidden questions with only `id` and `question`. Do not hardcode practice questions, categories, or answers.

Read the LLM configuration from:
- `BRINE_URL`
- `BRINE_MODEL`
- `BRINE_API_KEY`

Use the provided Qwen endpoint for judging. Never hardcode credentials or commit secrets.

Support the judging environment: ARM64/aarch64 Linux on NVIDIA DGX Spark.

Pin dependencies and document installation. During pipeline execution, the only permitted network calls are to the LLM endpoint. Package any required retrieval resources beforehand.

Keep model calls, prompts, and retries small. Log token usage where available.

## Testing

Prioritize tests for:
- Reversed directions and qualifier interpretation.
- Null findings versus missing evidence.
- Conflicting findings.
- Associations incorrectly described as causal.
- Correct intervention but wrong outcome, and vice versa.
- Misleading species, attribute, or context matches.
- Invalid citations and malformed outputs.
- Search expansion that discovers previously missed evidence.
- A complete run through the required command.

Use explicit graph cases for deterministic checks. Do not treat LLM-generated labels as verified ground truth.

Keep paraphrases of the same underlying question together when splitting development and evaluation examples.

## Evidence View

Keep the evidence view simple. Show:
- The question and answer.
- Cited edges and source sentences.
- Source document references.
- Direction, certainty, and confidence.
- Rubric judgments and search summary.

Build the reliable batch pipeline before polishing the interface.

## Scope

Reason over existing graph findings, explain disagreements, and identify missing evidence.

Do not estimate new ecological causal effects, simulate counterfactuals, or introduce external evidence.

Confidence estimation about answer correctness is in scope.

If conversational support is added, reuse this pipeline. Previous assistant responses are not evidence, and independent batch questions must not share conversation history.