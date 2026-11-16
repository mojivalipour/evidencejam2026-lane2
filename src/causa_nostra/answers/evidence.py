"""Per-question evidence view: the cited edges, their source sentences and documents.

One HTML file is written for each answered entry. The entry lists it in `extras`,
so a reader can trace every claim to a graph edge and its source text.
"""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote

from causa_nostra.answers.models import AnswerEntry
from causa_nostra.answers.rubric import item_texts
from causa_nostra.graph.store import GraphStore
from causa_nostra.pipeline.types import Verification
from causa_nostra.retrieval.records import EvidenceRecord, build_record, curie_labels

NOT_APPLICABLE = "not_applicable"


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
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return None  # outside the repo: cannot be listed as a repo-relative extra


def _item_reason(entry: AnswerEntry, verification: Verification, item: str) -> str:
    """Verifier's reason for an item it judged. Items outside this draft's rubric say so."""
    if item in verification.items:
        return verification.item_reasons.get(item, "no reason returned")
    return f"Not part of the rubric for a {entry.status} draft; not scored."


def render_rubric_html(entry: AnswerEntry, verification: Verification) -> str:
    """HTML page listing all 22 rubric sub-items (A1-D4 and N1-N6) with the judgement for each, and the category totals.

    Items the verifier judged are shown with its judgement. Items outside this draft's rubric are shown as not_applicable.
    """
    esc = html.escape
    texts = {**item_texts(True), **item_texts(False)}
    criteria_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{esc(value)}</td></tr>" for name, value in verification.criteria.items()
    )
    item_rows = "".join(
        f"<tr><td><code>{esc(item)}</code></td><td>{esc(text)}</td>"
        f"<td>{esc(verification.items.get(item, NOT_APPLICABLE))}</td>"
        f"<td>{esc(_item_reason(entry, verification, item))}</td></tr>"
        for item, text in texts.items()
    )
    items_table = (
        "<table><thead><tr><th>Item</th><th>Question</th><th>Judgement</th><th>Reason</th></tr></thead>"
        f"<tbody>{item_rows}</tbody></table>"
        if item_rows
        else "<p>No sub-item judgements.</p>"
    )
    problems = "".join(f"<li>{esc(p)}</li>" for p in verification.problems) or "<li>None</li>"
    supporting = ", ".join(verification.supporting_edge_ids) or "none"
    score = "n/a" if verification.score is None else f"{verification.score:.3f}"
    # The same fields for every status, so answered and no_evidence pages are read the same way.
    draft_fields = {
        "Status": entry.status,
        "Answer": entry.answer,
        "Direction": entry.direction or "(null)",
        "Certainty": entry.certainty or "(null)",
        "Confidence (rubric score)": f"{entry.confidence:.2f}",
        "Cited edge_ids": ", ".join(entry.edge_ids) or "none",
        "Searched for": entry.searched_for or "(null)",
    }
    draft_table = (
        "<table><tbody>"
        + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in draft_fields.items())
        + "</tbody></table>"
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{esc(entry.id)} rubric</title>"
        "<style>body{font-family:sans-serif;margin:2em}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}"
        "table{border-collapse:collapse}</style></head><body>"
        f"<h1>{esc(entry.id)} rubric</h1>"
        f"<p><b>Status:</b> {esc(entry.status)} &nbsp; <b>Verifier passed:</b> {verification.passed} "
        f"&nbsp; <b>Rubric score:</b> {score}</p>"
        "<h2>Returned answer</h2>"
        f"{draft_table}"
        "<h2>Sub-item judgements</h2>"
        f"{items_table}"
        "<h2>Category totals</h2>"
        f"<table><thead><tr><th>Category</th><th>Judgement</th></tr></thead><tbody>{criteria_rows}</tbody></table>"
        f"<h2>Problems</h2><ul>{problems}</ul>"
        f"<h2>Supporting edges named by the verifier</h2><p>{esc(supporting)}</p>"
        "</body></html>\n"
    )


def write_rubric(
    entry: AnswerEntry, verification: Verification, out_dir: str | Path, repo_root: Path
) -> str | None:
    """Write the rubric page for a returned entry. Returns its repo-relative path, or None."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{quote(entry.id, safe='')}_rubric.html"
    path.write_text(render_rubric_html(entry, verification), encoding="utf-8")
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return None


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
