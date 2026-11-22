"""Evaluation harness: retrieval, narrative coverage and verifier behaviour.

For N seeded cases (balanced across the eight typologies) it reports:

* retrieval hit@1 and hit@3 against the labelled typology,
* citation coverage (share of sentences with a citation that exists),
* fact inclusion rate (share of the labelled typology's key facts that the
  narrative cites),
* verifier pass rate on the unmodified drafts,
* corruption catch rate: one known error of each type is injected into each
  draft and the verifier is run again.

Results are broken down by typology, and retrieval misses are listed as
label -> prediction pairs.

The corruption list deliberately includes two error types that I expect the
verifier to miss (two values swapped inside one sentence, and a flipped
direction word). Leaving them out would make the catch rate look better than
the verifier is.
"""

from __future__ import annotations

import copy
import random
import re
from collections import Counter
from datetime import date, timedelta

from .cases import Case, generate_cases
from .generate import Draft, draft
from .narrative import Narrative
from .retrieval import (QUERY_MODES, Retriever, SentenceTransformerBackend, TfidfBackend, background_corpus,
                        build_chunks, build_query)
from .typologies import TYPOLOGY_IDS, load_typologies
from .verify import verify

DEFAULT_N = 200
DEFAULT_SEED = 1000

# Names that the case generator never produces.
OUTSIDE_PEOPLE = ["Gideon Harrowgate", "Ottoline Brisk", "Leopold Varnham", "Marguerite Sandover"]
OUTSIDE_COMPANIES = ["Vexley Marine Ltd", "Corriander Freight LLC", "Ostrava Peak Holdings LLC"]

_MONEY_RE = re.compile(r"\$(\d{1,3}(?:,\d{3})*)(\.\d{2})?")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_COUNT_RE = re.compile(r"(?<![\w$.,/-])(\d{1,3})(?!,\d)(?!\.\d)(?![\w%/-])")
_DIRECTION_SWAPS = [("deposits", "withdrawals"), ("withdrawals", "deposits"), ("incoming", "outgoing"),
                    ("outgoing", "incoming"), ("credits", "debits"), ("debits", "credits"),
                    ("received", "sent"), ("sent", "received")]

CORRUPTIONS = {
    "amount_changed": "a dollar amount is altered",
    "date_changed": "a date is moved by a few days",
    "count_changed": "a count is raised by 1 to 3",
    "name_invented": "a person or company name is replaced with one that is not in the case",
    "citation_removed": "all citations are stripped from one sentence",
    "citation_fabricated": "a sentence cites an id that does not exist",
    "citation_swapped": "a sentence cites a different, real fact",
    "values_swapped": "two amounts or two counts trade places inside one sentence",
    "direction_flipped": "a direction word is reversed, for example credits to debits",
}
EXPECTED_BLIND_SPOTS = ("values_swapped", "direction_flipped")


def _fmt_money_like(original: re.Match, value: int) -> str:
    cents = original.group(2) or ""
    return f"${value:,}{cents}"


def corrupt(narrative: Narrative, kind: str, case: Case, rng: random.Random) -> tuple[Narrative, int] | None:
    """Inject one error of the given kind. Returns the new narrative and the
    index of the sentence that was changed, or None if the kind does not apply."""
    out = copy.deepcopy(narrative)
    indices = list(range(len(out.sentences)))
    rng.shuffle(indices)

    if kind == "amount_changed":
        for i in indices:
            matches = list(_MONEY_RE.finditer(out.sentences[i].text))
            if matches:
                m = rng.choice(matches)
                old = int(m.group(1).replace(",", ""))
                new = old + rng.choice([100, 500, 1000, 10000])
                text = out.sentences[i].text
                out.sentences[i].text = text[:m.start()] + _fmt_money_like(m, new) + text[m.end():]
                return out, i

    elif kind == "date_changed":
        for i in indices:
            matches = list(_DATE_RE.finditer(out.sentences[i].text))
            if matches:
                m = rng.choice(matches)
                shifted = date.fromisoformat(m.group(0)) + timedelta(days=rng.choice([-1, 1]) * rng.randint(1, 9))
                text = out.sentences[i].text
                out.sentences[i].text = text[:m.start()] + shifted.isoformat() + text[m.end():]
                return out, i

    elif kind == "count_changed":
        for i in indices:
            text = _DATE_RE.sub(lambda m: " " * len(m.group(0)), out.sentences[i].text)
            matches = list(_COUNT_RE.finditer(text))
            if matches:
                m = rng.choice(matches)
                new = int(m.group(1)) + rng.randint(1, 3)
                original = out.sentences[i].text
                out.sentences[i].text = original[:m.start(1)] + str(new) + original[m.end(1):]
                return out, i

    elif kind == "name_invented":
        known = [(case.subject["name"], case.subject["kind"])]
        known += [(name, cp.get("kind", "")) for name, cp in case.counterparties.items()
                  if cp.get("kind") in ("individual", "business", "exchange")]
        for i in indices:
            present = [(name, k) for name, k in known if name in out.sentences[i].text]
            if present:
                name, k = rng.choice(present)
                replacement = rng.choice(OUTSIDE_PEOPLE if k == "individual" else OUTSIDE_COMPANIES)
                out.sentences[i].text = out.sentences[i].text.replace(name, replacement)
                return out, i

    elif kind == "citation_removed":
        i = indices[0]
        out.sentences[i].citations = []
        return out, i

    elif kind == "citation_fabricated":
        i = indices[0]
        out.sentences[i].citations = [f"F{len(case.facts) + rng.randint(3, 20)}"]
        return out, i

    elif kind == "citation_swapped":
        fact_ids = [fact.id for fact in case.facts]
        for i in indices:
            cited = out.sentences[i].citations
            if cited and all(c in fact_ids for c in cited):
                others = [f for f in fact_ids if f not in cited]
                if others:
                    out.sentences[i].citations = [rng.choice(others)]
                    return out, i

    elif kind == "values_swapped":
        for pattern in (_MONEY_RE, _COUNT_RE):
            for i in indices:
                text = out.sentences[i].text
                scan = text if pattern is _MONEY_RE else _DATE_RE.sub(lambda m: " " * len(m.group(0)), text)
                matches = list(pattern.finditer(scan))
                distinct = {m.group(0) for m in matches}
                if len(distinct) >= 2:
                    first = matches[0]
                    second = next(m for m in matches if m.group(0) != first.group(0))
                    out.sentences[i].text = (text[:first.start()] + second.group(0) + text[first.end():second.start()]
                                             + first.group(0) + text[second.end():])
                    return out, i

    elif kind == "direction_flipped":
        for i in indices:
            for old, new in _DIRECTION_SWAPS:
                pattern = re.compile(rf"\b{old}\b")
                if pattern.search(out.sentences[i].text):
                    out.sentences[i].text = pattern.sub(new, out.sentences[i].text, count=1)
                    return out, i
    else:
        raise ValueError(f"unknown corruption {kind!r}")
    return None


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


class _Tally:
    def __init__(self) -> None:
        self.n = 0
        self.hit1 = 0
        self.hit3 = 0
        self.sentences = 0
        self.cited = 0
        self.key_facts = 0
        self.key_cited = 0
        self.passed = 0

    def add(self, d: Draft, report) -> None:
        label = d.case.typology_id
        order = [tid for tid, _ in d.ranking]
        self.n += 1
        self.hit1 += order[0] == label
        self.hit3 += label in order[:3]
        self.sentences += report.n_sentences
        self.cited += report.n_cited
        cited_ids = d.narrative.cited_ids()
        key_ids = [fact.id for fact in d.case.facts if fact.is_key]
        self.key_facts += len(key_ids)
        self.key_cited += sum(1 for fid in key_ids if fid in cited_ids)
        self.passed += report.passed

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n,
            "retrieval_hit_at_1": _rate(self.hit1, self.n),
            "retrieval_hit_at_3": _rate(self.hit3, self.n),
            "citation_coverage": _rate(self.cited, self.sentences),
            "fact_inclusion_rate": _rate(self.key_cited, self.key_facts),
            "verifier_pass_rate": _rate(self.passed, self.n),
            "mean_sentences": round(self.sentences / self.n, 2) if self.n else None,
        }


def evaluate(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, retriever: Retriever | None = None,
             query_mode: str = "both") -> dict:
    """Run the full evaluation with the default pipeline and return the metrics."""
    retriever = retriever or Retriever.from_directory()
    cases = generate_cases(n, seed=seed, balanced=True)
    overall = _Tally()
    per_typology = {tid: _Tally() for tid in TYPOLOGY_IDS}
    misses: Counter = Counter()
    corruption = {kind: {"applicable": 0, "caught": 0} for kind in CORRUPTIONS}

    for case in cases:
        d = draft(case, retriever, query_mode=query_mode)
        report = verify(d.narrative, case, d.references)
        overall.add(d, report)
        per_typology[case.typology_id].add(d, report)
        predicted = d.ranking[0][0]
        if predicted != case.typology_id:
            misses[(case.typology_id, predicted)] += 1

        baseline = {(f.sentence_index, f.kind, f.detail) for f in report.findings}
        rng = random.Random(f"corrupt-{case.seed}")
        for kind in CORRUPTIONS:
            result = corrupt(d.narrative, kind, case, rng)
            if result is None:
                continue
            corrupted, index = result
            corruption[kind]["applicable"] += 1
            after = verify(corrupted, case, d.references)
            new = [f for f in after.findings
                   if f.sentence_index == index and (f.sentence_index, f.kind, f.detail) not in baseline]
            corruption[kind]["caught"] += bool(new)

    for kind, row in corruption.items():
        row["catch_rate"] = _rate(row["caught"], row["applicable"])
        row["description"] = CORRUPTIONS[kind]
        row["expected_blind_spot"] = kind in EXPECTED_BLIND_SPOTS

    return {
        "config": {
            "n_cases": n, "seed": seed, "balanced": True, "generator": "template",
            "backend": getattr(retriever.backend, "name", type(retriever.backend).__name__),
            "query_mode": query_mode, "ranking": retriever.ranking,
        },
        "overall": overall.to_dict(),
        "per_typology": {tid: tally.to_dict() for tid, tally in per_typology.items()},
        "retrieval_misses": [
            {"label": label, "predicted": predicted, "count": count}
            for (label, predicted), count in sorted(misses.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "corruption": corruption,
    }


# --------------------------------------------------------------------------
# Retrieval ablation
# --------------------------------------------------------------------------

ABLATION_CONFIGS = [
    # (name, backend, query mode, ranking, background idf)
    ("one unweighted string of all facts", "tfidf", "naive", "chunks", False),
    ("behaviour description only", "tfidf", "behaviour", "chunks", False),
    ("weighted facts only", "tfidf", "facts", "chunks", False),
    ("weighted facts + behaviour", "tfidf", "both", "chunks", False),
    ("  + rank whole documents", "tfidf", "both", "document", False),
    ("  + IDF from unlabelled cases (default)", "tfidf", "both", "document", True),
    ("default without stemming", "tfidf-nostem", "both", "document", True),
]
NEURAL_CONFIGS = [
    ("sentence-transformers, behaviour only", "sentence-transformers", "behaviour", "document", False),
    ("sentence-transformers, facts + behaviour", "sentence-transformers", "both", "document", False),
]


def retrieval_ablation(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, with_neural: bool = False) -> list[dict]:
    """Retrieval hit@1 and hit@3 for several configurations on the same cases."""
    cases = generate_cases(n, seed=seed, balanced=True)
    chunks = build_chunks(load_typologies())
    configs = ABLATION_CONFIGS + (NEURAL_CONFIGS if with_neural else [])
    neural_backend = SentenceTransformerBackend() if with_neural else None
    rows = []
    for name, backend_name, mode, ranking, background in configs:
        assert mode in QUERY_MODES
        if backend_name == "sentence-transformers":
            backend = neural_backend
        else:
            backend = TfidfBackend(stem=backend_name != "tfidf-nostem")
        retriever = Retriever(chunks, backend, idf_corpus=background_corpus() if background else None,
                              ranking=ranking)
        hit1 = hit3 = 0
        for case in cases:
            order = [tid for tid, _ in retriever.rank_typologies(build_query(case, mode))]
            hit1 += order[0] == case.typology_id
            hit3 += case.typology_id in order[:3]
        rows.append({"name": name.strip(), "backend": backend_name, "query_mode": mode, "ranking": ranking,
                     "background_idf": background, "retrieval_hit_at_1": _rate(hit1, n),
                     "retrieval_hit_at_3": _rate(hit3, n)})
    return rows


# --------------------------------------------------------------------------
# Plain-text report
# --------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def format_report(metrics: dict, ablation: list[dict] | None = None) -> str:
    cfg, overall = metrics["config"], metrics["overall"]
    lines = [
        f"Evaluation on {cfg['n_cases']} synthetic cases (seed {cfg['seed']}, backend {cfg['backend']}, "
        f"query mode {cfg['query_mode']}, ranking {cfg['ranking']})",
        "",
        "Overall",
        f"  retrieval hit@1        {_pct(overall['retrieval_hit_at_1'])}",
        f"  retrieval hit@3        {_pct(overall['retrieval_hit_at_3'])}",
        f"  citation coverage      {_pct(overall['citation_coverage'])}",
        f"  fact inclusion rate    {_pct(overall['fact_inclusion_rate'])}",
        f"  verifier pass rate     {_pct(overall['verifier_pass_rate'])}",
        f"  mean sentences/draft   {overall['mean_sentences']}",
        "",
        "Per typology",
        f"  {'id':<5}{'n':>4}{'hit@1':>9}{'hit@3':>9}{'facts':>9}{'cited':>9}{'pass':>9}",
    ]
    for tid, row in metrics["per_typology"].items():
        lines.append(
            f"  {tid:<5}{row['n_cases']:>4}{_pct(row['retrieval_hit_at_1']):>9}{_pct(row['retrieval_hit_at_3']):>9}"
            f"{_pct(row['fact_inclusion_rate']):>9}{_pct(row['citation_coverage']):>9}"
            f"{_pct(row['verifier_pass_rate']):>9}")
    lines += ["", "Retrieval misses (label -> top prediction)"]
    if metrics["retrieval_misses"]:
        for miss in metrics["retrieval_misses"]:
            lines.append(f"  {miss['label']} -> {miss['predicted']}: {miss['count']}")
    else:
        lines.append("  none")
    lines += ["", "Corruption catch rate", f"  {'error type':<22}{'injected':>9}{'caught':>8}{'rate':>9}"]
    for kind, row in metrics["corruption"].items():
        flag = "  (expected blind spot)" if row["expected_blind_spot"] else ""
        lines.append(f"  {kind:<22}{row['applicable']:>9}{row['caught']:>8}{_pct(row['catch_rate']):>9}{flag}")
    if ablation:
        lines += ["", "Retrieval ablation (same cases)", f"  {'configuration':<46}{'hit@1':>8}{'hit@3':>8}"]
        for row in ablation:
            lines.append(f"  {row['name']:<46}{_pct(row['retrieval_hit_at_1']):>8}{_pct(row['retrieval_hit_at_3']):>8}")
    return "\n".join(lines)
