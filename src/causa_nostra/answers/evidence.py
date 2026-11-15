"""Per-question evidence view: the cited edges, their source sentences and documents.

One HTML file is written for each answered entry. The entry lists it in `extras`,
so a reader can trace every claim to a graph edge and its source text.
"""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote

from causa_nostra.answers.models import AnswerEntry
from causa_nostra.graph.store import GraphStore
from causa_nostra.retrieval.records import EvidenceRecord, build_record, curie_labels


def render_evidence_html(entry: AnswerEntry, records: list[EvidenceRecord]) -> str:
    """HTML page showing the answer, its direction and certainty, and each cited edge."""
    esc = html.escape
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td><code>{esc(record.edge_id)}</code></td>"
            f"<td>{esc(record.subject_label)}</td>"
            f"<td>{esc(('NULL RESULT: ' if record.negated else '') + record.predicate)}</td>"
            f"<td>{esc(record.object_label)}</td>"
            f"<td>{esc(record.relation_sign or '')}</td>"
            f"<td>{esc(record.source_document or '')}</td>"
            f"<td>{esc(record.original_sentence or '')}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Edge</th><th>Intervention</th><th>Predicate</th>"
        "<th>Outcome</th><th>Sign</th><th>Source</th><th>Source sentence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else "<p>No cited edges.</p>"
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{esc(entry.id)} evidence</title>"
        "<style>body{font-family:sans-serif;margin:2em}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}"
        "table{border-collapse:collapse}</style></head><body>"
        f"<h1>{esc(entry.id)}</h1>"
        f"<p><b>Status:</b> {esc(entry.status)} &nbsp; <b>Direction:</b> {esc(entry.direction or '—')} "
        f"&nbsp; <b>Certainty:</b> {esc(entry.certainty or '—')} &nbsp; <b>Confidence:</b> {entry.confidence:.3f}</p>"
        f"<h2>Answer</h2><p>{esc(entry.answer)}</p>"
        f"<h2>Cited edges</h2>{table}"
        "</body></html>\n"
    )


def write_evidence(entry: AnswerEntry, store: GraphStore, out_dir: str | Path, repo_root: Path) -> str | None:
    """Write the evidence page for an answered entry. Returns its repo-relative path, or None.

    `no_evidence` entries have no citations and get no page.
    """
    if entry.status != "answered" or not entry.edge_ids:
        return None
    labels = curie_labels(store)
    records = [build_record(store.edge(edge_id), store, labels) for edge_id in entry.edge_ids]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{quote(entry.id, safe='')}_evidence.html"
    path.write_text(render_evidence_html(entry, records), encoding="utf-8")
    try:
        # POSIX separators, so a file written on Windows still resolves for graders on Linux.
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None  # outside the repo: cannot be listed as a repo-relative extra


if __name__ == "__main__":
    # Demo: write the evidence page for a made-up answer citing two real edges, into a temp directory.
    import tempfile

    store = GraphStore.from_json("causal_graph.json")
    edge_ids = [edge.id for edge in store.graph.edges[:2]]
    entry = AnswerEntry(
        id="demo", status="answered", answer="Demo answer.", direction="mixed",
        confidence=0.5, certainty="low", edge_ids=edge_ids,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = write_evidence(entry, store, Path(tmp), repo_root=Path(tmp))
        print(f"wrote {path} for {len(edge_ids)} edge(s)")
