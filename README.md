# CAUSA NOSTRA — Lane 2 (Evidence Jam 2026)

Evidence-grounded reasoning over the supplied CAMO graph (`causal_graph.json`).

## Run

From a clean checkout:

```bash
./run.sh questions.yaml answers.json
```

The hidden question file has the same shape (`id` and `question` per entry) and
can be passed in place of `questions.yaml`.

Output: `answers.json` in the competition format, plus any extra outputs listed in each entry's `extras`.

## Configuration

The LLM endpoint is read from the environment. Nothing is hardcoded.

| Variable | Meaning |
| --- | --- |
| `BRINE_URL` | LLM endpoint URL (OpenAI-compatible) |
| `BRINE_MODEL` | Model name (Qwen3.8-27B for judging) |
| `BRINE_API_KEY` | API key |

These may be set in the shell or in a local `.env` file. Do not commit `.env` or any key.

## Dependencies

Dependencies are pinned in `uv.lock`. `run.sh` uses `uv run --locked`, so the
lock file must be up to date. Install uv, then run `./run.sh`. The first run
installs the locked environment from the package index; no other network calls
are made at run time.

## Checks

```bash
uv run --locked causa-nostra check-graph   # LinkML schema and graph invariants
uv run --locked pytest                     # unit tests
```

Graph loading logs warnings for missing essential node/edge fields and unknown
subject/object node IDs, but continues loading so the graph can be inspected.
`check-graph` reports these warnings and exits with a non-zero status.

## Layout

- `src/causa_nostra/camo_rules/`: CAMO interpretation rules and graph invariants.
- `src/causa_nostra/graph/`: graph loading and indexed lookups.
- `src/causa_nostra/retrieval/`: Haystack BM25 retrieval over edge records.
- `src/causa_nostra/answers/`: output contract and deterministic citation checks.
- `src/causa_nostra/pipeline/`: understand → retrieve → reason → verify, one question at a time.
- `schema/causalmosaic.yaml`: CAMO schema, the source of truth for graph fields.

## Status

The reasoning and verification stages are not yet connected to the LLM. The
current run writes a valid `answers.json` with an entry for every question, but
the answers are placeholders.
