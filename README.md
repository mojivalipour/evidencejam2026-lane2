# CAUSA NOSTRA

**Evidence Jam 2026 · Lane 2 — Reasoning**

CAUSA NOSTRA answers ecology questions using the supplied Causal Mosaic (CAMO) graph. Instead of giving a general answer from a language model, it looks for relevant claims in `causal_graph.json`, uses those claims to write an answer, and checks whether the answer is supported.

The program does not search the web. It can only answer from evidence found in the supplied graph.

## Run the project

From the project folder, run:

```bash
./run.sh questions.yaml answers.json
```

On Windows, use PowerShell:

```powershell
py -m uv run --locked python -m causa_nostra.cli run .\questions.yaml .\answers.json --graph .\causal_graph.json
```

The script accepts a different question file in the same format. This includes the hidden question file used during judging.

The results are written to `answers.json`. The program saves the file after each question, so an interrupted run may leave a partial set of results.

### Set up the model connection

Set these environment variables before running the program:

| Variable | What it is |
| --- | --- |
| `BRINE_URL` | URL of the OpenAI-compatible model endpoint |
| `BRINE_MODEL` | Model name (`Qwen3.8-27B` for judging) |
| `BRINE_API_KEY` | API key for the endpoint |

For example, in PowerShell:

```powershell
$env:BRINE_URL = "your endpoint URL"
$env:BRINE_MODEL = "your model name"
$env:BRINE_API_KEY = "your API key"
```

You can also put these settings in a local `.env` file for `run.sh`. Never commit `.env` or API keys to the repository.

Dependencies are pinned in `uv.lock` and the run script uses `uv run --locked`. Install `uv` and prepare the locked environment before running if needed. The answering pipeline makes no web requests; its external calls go to the configured model endpoint. Make sure dependencies are available locally when running under the competition's no-download rule.

## Questions and answers

Questions are supplied in a YAML file. Every question needs a unique `id` and a `question`. The optional `kind` field is for organizing questions; it does not currently change how they are processed.

```yaml
questions:
  - id: Q01
    question: Does fertilization affect seedling growth?
  - id: Q02
    question: Does deer browsing affect seedling survival?
```

Questions that ask about one intervention or condition and one outcome are easiest to check. The program may have difficulty with broader questions that combine several effects.

Each output entry contains a short answer and the information needed to check it:

| Field | Meaning |
| --- | --- |
| `status` | `answered` or `no_evidence` |
| `answer` | The answer in plain language, or why no supported answer was produced |
| `direction` | `increase`, `decrease`, `no_effect`, `mixed`, or `unclear`; null for `no_evidence` |
| `certainty` | Evidence strength: `high`, `moderate`, `low`, or `very_low` |
| `confidence` | The pipeline's confidence score, between 0 and 1 |
| `edge_ids` | IDs of the graph claims used as evidence |
| `searched_for` | What was searched when no evidence was found |
| `extras` | Optional paths to additional output files |

`no_effect` means evidence reports a null result. It is different from `no_evidence`, which means the program could not establish an answer from the graph. A processing error can also produce a fallback `no_evidence` entry; the accompanying explanation should make that clear.

## How it works

1. **Read the question.** A model identifies the intervention or condition, the outcome, any stated context, and alternative search terms. These alternative terms help with wording differences but are not treated as automatically equivalent.
2. **Find evidence.** The program loads the local CAMO graph, turns edges into readable records, and searches subject and object indexes separately. It looks for claims that connect the requested intervention to the requested outcome, not just records that happen to mention similar words. By default, it passes up to 20 ranked records to the next stage.
3. **Write an answer.** The reasoning model uses the retrieved records to draft a short response. It cites edge IDs and should not add outside facts or join separate edges into an unsupported causal claim.
4. **Verify the draft.** The program checks cited edge IDs, then asks a verifier to review the draft against the evidence and rubric. If the review fails, the reasoner can revise its answer once by default.
5. **Save the result.** Python applies the rubric rules and checks the final output. Invalid citations and file references are removed. If an answer has no valid supporting citation, it becomes a fallback entry. Questions are saved in their original order.

### What the verification step checks

The verifier receives the original question, the draft answer, its cited edges, the other retrieved edges, and a summary of the search. It reviews the answer using questions based on the project's verification criteria and the judging rubric.

**When the program has an answer:**

- Does the evidence actually connect the intervention or condition in the question to the outcome being asked about?
- Does the evidence match the question's context, or is the answer making a broader claim than the graph supports?
- Is the direction correct: increase, decrease, no effect, mixed, or unclear? Have the graph's predicates and node qualifiers been read correctly?
- Is a tested null result being distinguished from a missing result?
- Does the wording reflect the source? For example, does it avoid calling an association a proven causal effect?
- Does every cited edge ID exist, and does each cited edge support the statement it is attached to?
- Are conflicting findings, limitations, and uncertainty represented fairly? Were important retrieved claims overlooked?

**When the program reports `no_evidence`:**

- Were both sides of the question searched, including relevant alternative terms?
- Is there truly no matching claim, rather than a claim that reports no effect?
- Does `searched_for` explain which intervention, outcome, attributes, or relationships were checked?
- Is this a genuine lack of evidence, or did the model endpoint or processing step fail?

The verifier judges the rubric items, and Python applies the weights, rules, and acceptance checks. For an accepted answer, the resulting rubric score is used as `confidence`; a processing fallback gets a score of zero. This score is a useful quality check, **not a calibrated probability**. The competition rubric, however, evaluates confidence as a probability of being correct, so calibration still needs attention.

The verification step checks whether the answer follows the *graph*. It does not establish whether the original research claim or the graph itself is correct.

## Checks and troubleshooting

To check the graph against the CAMO schema and project rules, and to run unit tests:

```bash
uv run --locked causa-nostra check-graph
uv run --locked pytest
```

`check-graph` reports problems but does not change the graph.

If the program fails, check the three `BRINE_*` variables, YAML indentation, unique question IDs, and availability of `uv`. If only one question fails, the batch runner can continue and record a fallback answer for that question.

## Project files

| Path | Purpose |
| --- | --- |
| `causal_graph.json` | Supplied graph of ecological claims |
| `questions.yaml` | Practice questions |
| `schema/causalmosaic.yaml` | CAMO schema |
| `src/causa_nostra/cli.py` | Command-line entry point |
| `src/causa_nostra/graph/` | Graph loading and indexing |
| `src/causa_nostra/retrieval/` | Evidence records and two-sided search (Haystack BM25) |
| `src/causa_nostra/agents/` | Question interpretation, reasoning, and verification |
| `src/causa_nostra/answers/` | Output format, rubric, and citation checks |
| `src/causa_nostra/pipeline/` | Single-question processing and batch runs |
| `src/causa_nostra/llm/` | Connection to the model endpoint |
| `src/causa_nostra/camo_rules/` | CAMO interpretation and graph validation rules |

## Limits and current status

The graph is a fixed dataset and may be incomplete or contain mistakes. Different common names, scientific names, abbreviations, and ecological terms can also affect retrieval. Finding no matching record is not proof that an ecological relationship does not exist. Readers should check cited edges, especially for mixed, null, and context-specific findings.

The question-processing, reasoning, and verification stages are connected to the configured model endpoint. A question may still receive a fallback answer if processing fails or verification cannot be completed.

## Solution architecture diagram

*Space reserved for the final solution architecture diagram.*

<!-- Add the final architecture image or Mermaid diagram here. -->



## Contributors

| Name | Title |
| --- | --- |
| Andrii Kramarenko | Biologist/Ecologist, Lab Tech|
| Antonina | Data Scientist |
| Mojtaba Valipour, PhD | AI/ML Research Engineer |
| Panos Vlachos | PhD Student in Civil Engineering and Environmental Science |

## License

See `LICENSE` (MIT License).
