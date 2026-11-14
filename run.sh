#!/usr/bin/env bash
# Answer a question file against the CAMO causal graph.
#
#   ./run.sh questions.yaml answers.json
#
# Reads BRINE_URL, BRINE_MODEL, BRINE_API_KEY from the environment or from .env.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: ./run.sh questions.yaml answers.json" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve user paths before changing directory, so relative paths keep working.
QUESTIONS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUTPUT_DIR="$(mkdir -p "$(dirname "$2")" && cd "$(dirname "$2")" && pwd)"
OUTPUT="$OUTPUT_DIR/$(basename "$2")"

if [ ! -f "$QUESTIONS" ]; then
    echo "error: questions file not found: $1" >&2
    exit 1
fi

cd "$ROOT"

if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
fi

GRAPH="$ROOT/causal_graph.json"

if command -v uv >/dev/null 2>&1; then
    exec uv run --locked --quiet causa-nostra run "$QUESTIONS" "$OUTPUT" --graph "$GRAPH"
fi

exec python3 -m causa_nostra.cli run "$QUESTIONS" "$OUTPUT" --graph "$GRAPH"
