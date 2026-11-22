"""Write the three JSON files that the web demo reads.

* `typologies.json`: the reference documents.
* `cases.json`: 12 synthetic cases with facts, a sample of transactions, the
  retrieval results (closest chunks, the typology ranking, the typology the
  narrative was written for and the reference ids the writer was allowed to
  cite), the drafted narrative and the verifier report.
* `eval.json`: the metrics from one evaluation run.

The cases are not hand-picked. They are the first 12 of a seeded, balanced
sequence, so all eight typologies appear and the first four appear twice.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .cases import Case, generate_cases
from .evaluate import DEFAULT_N, DEFAULT_SEED, evaluate, retrieval_ablation
from .generate import draft
from .retrieval import Retriever
from .typologies import load_typologies
from .verify import verify

EXPORT_CASES = 12
EXPORT_CASE_SEED = 42
MAX_TRANSACTION_ROWS = 25


def _transaction_rows(case: Case) -> list[dict]:
    """At most 25 rows: the largest transactions, shown in date order.

    The full count stays available in the 'Transactions in the review period' fact.
    """
    indexed = list(enumerate(case.transactions))
    keep = sorted(indexed, key=lambda pair: -pair[1].amount)[:MAX_TRANSACTION_ROWS]
    return [
        {"date": t.date, "type": t.type, "amount": t.amount, "counterparty": t.counterparty,
         "channel": t.channel, "location": t.location}
        for _, t in sorted(keep, key=lambda pair: pair[0])
    ]


def export_case(case: Case, retriever: Retriever, number: int, titles: dict[str, str]) -> dict:
    d = draft(case, retriever)
    report = verify(d.narrative, case, d.references)
    return {
        "id": case.id,
        "title": f"Synthetic case {number:02d}: {titles[case.typology_id]} ({case.subject['kind']} subject)",
        "typology_id": case.typology_id,
        "subject": {"name": case.subject["name"], "kind": case.subject["kind"], "detail": case.subject["detail"]},
        "period": dict(case.period),
        "facts": [{"id": f.id, "label": f.label, "value": f.value} for f in case.facts],
        "transactions": _transaction_rows(case),
        "query": d.query.text,
        "retrieved": [hit.to_dict() for hit in d.hits],
        "ranking": [{"typology_id": tid, "score": round(score, 4)} for tid, score in d.ranking],
        "written_for": d.typology_id,
        "references": [chunk.id for chunk in d.references],
        "narrative": [sentence.to_dict() for sentence in d.narrative.sentences],
        "verification": report.to_dict(),
    }


def export_web(out_dir: str | Path = "web_export", n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
               case_seed: int = EXPORT_CASE_SEED, with_neural: bool = False) -> dict[str, Path]:
    """Generate the three JSON files and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    retriever = Retriever.from_directory()
    typologies = load_typologies()
    titles = {t.id: t.title for t in typologies}

    cases = generate_cases(EXPORT_CASES, seed=case_seed, balanced=True)
    metrics = evaluate(n=n, seed=seed, retriever=retriever)
    metrics["retrieval_ablation"] = retrieval_ablation(n=n, seed=seed, with_neural=with_neural)
    payloads = {
        "typologies.json": [t.to_dict() for t in typologies],
        "cases.json": [export_case(case, retriever, i, titles) for i, case in enumerate(cases, start=1)],
        "eval.json": {
            "n_cases": n,
            "seed": seed,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data": "synthetic",
            **metrics,
        },
    }
    paths = {}
    for name, payload in payloads.items():
        path = out / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths[name] = path
    return paths
