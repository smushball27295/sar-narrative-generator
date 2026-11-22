"""Check a drafted narrative against the case it claims to describe.

The verifier does not judge whether the narrative is well written or whether
the conclusion is right. It checks something narrower that can be checked
mechanically:

1. every sentence cites at least one fact or reference indicator,
2. every cited id exists for this case,
3. every money amount, date, count, percentage, identifier and proper name in
   the sentence appears in the facts or indicators that the sentence cites.

A value that is absent from the cited sources but present elsewhere in the
case is reported separately from a value that appears nowhere, because the
first is usually a citation slip and the second is an invention.

The same extraction code runs on the sentence and on the cited sources, so a
formatting quirk affects both sides equally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .cases import Case
from .narrative import CITATION_RE, Narrative, parse_narrative
from .retrieval import Chunk

MONTHS = {name: i for i, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
MONTHS.update({name[:3]: i for name, i in list(MONTHS.items())})
MONTHS["sept"] = 9
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}  # "one" is left out: it is far more often a pronoun than a count

_IDENTIFIER_RE = re.compile(r"\b[A-Z]{2,5}-\d{2,}\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MDY_DATE_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)
_DMY_DATE_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.IGNORECASE)
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_MONEY_RE = re.compile(rf"(?:\$|USD\s?)\s?({_NUMBER})(\s?(?:million|thousand|k|m)\b)?", re.IGNORECASE)
_MONEY_WORD_RE = re.compile(rf"(?<![\w$.,-])({_NUMBER})(\s?(?:million|thousand))?\s?(?:dollars|USD)\b", re.IGNORECASE)
_PERCENT_RE = re.compile(rf"(?<![\w$.,-])({_NUMBER})\s?(?:%|percent\b|per cent\b)", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b")
_COUNT_RE = re.compile(rf"(?<![\w$.,-])({_NUMBER})(?!\w)")
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
_NAME_RE = re.compile(r"[A-Z][A-Za-z'’-]*(?: [A-Z][A-Za-z'’-]*)*")

# Capitalised words that are not names. A sentence-initial word is only
# treated as a possible name if it is not in this list, so an unfamiliar
# opener produces a finding. That errs on the side of refusing.
OPENERS = {
    "the", "a", "an", "in", "on", "at", "of", "for", "from", "to", "by", "with", "during", "between", "over",
    "after", "before", "this", "that", "these", "those", "it", "its", "there", "no", "all", "some", "most",
    "many", "several", "each", "both", "cash", "money", "funds", "payroll", "debits", "credits", "wire",
    "wires", "individuals", "account", "customer", "subject", "activity", "review", "largest", "median",
    "virtual-currency", "however", "additionally", "overall", "based", "according", "total", "deposits",
    "withdrawals", "transfers", "payments", "transactions", "incoming", "outgoing", "records", "analysis",
    "subsequently", "further", "finally", "also", "as", "when", "while", "although", "because", "since",
    "within", "approximately", "following", "given", "one", "none", "nothing", "he", "she", "they", "we",
    "payees", "money", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "trade",
    "invoices", "invoice", "gift", "savings", "and", "but", "if", "such", "taken", "together", "in",
    "out", "their", "his", "her", "our", "notes", "note", "reference", "indicators", "matching", "red",
}
IGNORED_TOKENS = {
    "SAR", "ATM", "ATMs", "ACH", "USD", "CTR", "KYC", "AML", "ID", "IDs", "US", "U.S", "I", "United", "States",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
} | {m.capitalize() for m in MONTHS}

FINDING_KINDS = (
    "missing_citation", "unknown_citation", "value_not_in_cited_source",
    "unsupported_amount", "unsupported_date", "unsupported_count", "unsupported_percentage",
    "unsupported_identifier", "unsupported_name",
)


@dataclass
class Claims:
    """The checkable values found in a piece of text."""

    amounts: set[int] = field(default_factory=set)  # cents
    dates: set[str] = field(default_factory=set)  # ISO dates
    months: set[str] = field(default_factory=set)  # YYYY-MM, from "March 2025"
    percentages: set[float] = field(default_factory=set)
    counts: set[float] = field(default_factory=set)
    identifiers: set[str] = field(default_factory=set)
    names: list[str] = field(default_factory=list)


def _to_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def _safe_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def extract_claims(text: str, with_names: bool = True) -> Claims:
    """Pull amounts, dates, percentages, counts, identifiers and names out of text.

    Each pattern masks what it matched so that later patterns do not read the
    same characters again (the year in a date is not also a count).
    """
    claims = Claims()
    work = CITATION_RE.sub(" ", text)

    def mask(pattern: re.Pattern, handler) -> None:
        nonlocal work

        def replace(match: re.Match) -> str:
            handler(match)
            return " " * (match.end() - match.start())

        work = pattern.sub(replace, work)

    mask(_IDENTIFIER_RE, lambda m: claims.identifiers.add(m.group(0)))

    def iso(m):
        value = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if value:
            claims.dates.add(value)

    def mdy(m):
        value = _safe_date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        if value:
            claims.dates.add(value)

    def dmy(m):
        value = _safe_date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
        if value:
            claims.dates.add(value)

    def slash(m):  # read as month/day/year, the US convention
        value = _safe_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        if value:
            claims.dates.add(value)

    mask(_ISO_DATE_RE, iso)
    mask(_MDY_DATE_RE, mdy)
    mask(_DMY_DATE_RE, dmy)
    mask(_SLASH_DATE_RE, slash)
    mask(_MONTH_YEAR_RE, lambda m: claims.months.add(f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}"))

    def money(m):
        value = _to_decimal(m.group(1))
        if value is None:
            return
        scale = (m.group(2) or "").strip().lower()
        if scale in ("million", "m"):
            value *= 1_000_000
        elif scale in ("thousand", "k"):
            value *= 1_000
        claims.amounts.add(int((value * 100).to_integral_value()))

    mask(_MONEY_RE, money)
    mask(_MONEY_WORD_RE, money)

    def percent(m):
        value = _to_decimal(m.group(1))
        if value is not None:
            claims.percentages.add(float(value))

    mask(_PERCENT_RE, percent)
    mask(_ORDINAL_RE, lambda m: None)

    def count(m):
        value = _to_decimal(m.group(1))
        if value is not None:
            claims.counts.add(float(value))

    mask(_COUNT_RE, count)
    for match in _NUMBER_WORD_RE.finditer(work):
        claims.counts.add(float(NUMBER_WORDS[match.group(1).lower()]))

    if with_names:
        claims.names = _extract_names(work)
    return claims


def _extract_names(masked: str) -> list[str]:
    names: list[str] = []
    first_alpha = re.search(r"[A-Za-z]", masked)
    for match in _NAME_RE.finditer(masked):
        tokens = match.group(0).split(" ")
        if first_alpha and match.start() == first_alpha.start():
            while tokens and tokens[0].lower() in OPENERS:
                tokens.pop(0)
        tokens = [t.rstrip("'’-") for t in tokens]
        while tokens and tokens[0] in IGNORED_TOKENS:
            tokens.pop(0)
        while tokens and tokens[-1] in IGNORED_TOKENS:
            tokens.pop()
        tokens = [t[:-2] if t.endswith(("'s", "’s")) else t for t in tokens]
        if tokens and any(len(t) > 1 for t in tokens):
            names.append(" ".join(tokens))
    return names


def _contains_name(name: str, text: str) -> bool:
    pattern = r"(?<![A-Za-z])" + r"\s+".join(re.escape(part) for part in name.split()) + r"(?![A-Za-z])"
    return re.search(pattern, text, re.IGNORECASE) is not None


@dataclass
class Finding:
    sentence_index: int
    kind: str
    detail: str

    def to_dict(self) -> dict:
        return {"sentence_index": self.sentence_index, "kind": self.kind, "detail": self.detail}


@dataclass
class VerificationReport:
    passed: bool
    findings: list[Finding]
    n_sentences: int
    n_cited: int  # sentences with at least one citation that exists

    @property
    def citation_coverage(self) -> float:
        return self.n_cited / self.n_sentences if self.n_sentences else 0.0

    def to_dict(self) -> dict:
        return {"passed": self.passed, "findings": [f.to_dict() for f in self.findings]}


class _Source:
    """Text and extracted values of a set of facts and reference chunks."""

    def __init__(self, texts: Iterable[str]):
        self.text = "\n".join(texts)
        self.claims = extract_claims(self.text, with_names=False)

    def has_amount(self, value: int) -> bool:
        return value in self.claims.amounts

    def has_date(self, value: str) -> bool:
        return value in self.claims.dates

    def has_month(self, value: str) -> bool:
        return value in self.claims.months or any(d.startswith(value) for d in self.claims.dates)

    def has_percentage(self, value: float) -> bool:
        return any(abs(value - v) < 1e-9 for v in self.claims.percentages)

    def has_count(self, value: float) -> bool:
        return any(abs(value - v) < 1e-9 for v in self.claims.counts)

    def has_identifier(self, value: str) -> bool:
        return value in self.claims.identifiers

    def has_name(self, value: str) -> bool:
        return _contains_name(value, self.text)


def _chunk_text(chunk: Chunk) -> str:
    return f"{chunk.title}. {chunk.text}"


def verify(narrative: Narrative | str, case: Case, references: Iterable[Chunk] = ()) -> VerificationReport:
    """Verify a narrative against a case and the reference chunks it was given.

    `references` are the typology chunks that were available to the generator
    for this case. A citation to any other indicator id counts as unknown.
    """
    text = narrative.to_text() if isinstance(narrative, Narrative) else narrative
    parsed = parse_narrative(text)
    fact_text = {fact.id: fact.text() for fact in case.facts}
    chunk_text = {chunk.id: _chunk_text(chunk) for chunk in references}
    whole_case = _Source(list(fact_text.values()) + list(chunk_text.values()))

    findings: list[Finding] = []
    n_cited = 0
    for index, sentence in enumerate(parsed.sentences):
        known = [c for c in sentence.citations if c in fact_text or c in chunk_text]
        unknown = [c for c in sentence.citations if c not in fact_text and c not in chunk_text]
        if not sentence.citations:
            findings.append(Finding(index, "missing_citation", "sentence has no citation"))
        for citation in unknown:
            findings.append(Finding(index, "unknown_citation",
                                    f"[{citation}] is not a fact of this case or a reference retrieved for it"))
        if known:
            n_cited += 1

        cited = _Source([fact_text.get(c) or chunk_text[c] for c in known])
        claims = extract_claims(sentence.text)
        checks = [
            ("amount", sorted(claims.amounts), cited.has_amount, whole_case.has_amount,
             lambda v: f"${v / 100:,.2f}"),
            ("date", sorted(claims.dates), cited.has_date, whole_case.has_date, str),
            ("date", sorted(claims.months), cited.has_month, whole_case.has_month, str),
            ("percentage", sorted(claims.percentages), cited.has_percentage, whole_case.has_percentage,
             lambda v: f"{v:g}%"),
            ("count", sorted(claims.counts), cited.has_count, whole_case.has_count, lambda v: f"{v:g}"),
            ("identifier", sorted(claims.identifiers), cited.has_identifier, whole_case.has_identifier, str),
            ("name", list(dict.fromkeys(claims.names)), cited.has_name, whole_case.has_name, str),
        ]
        for kind, values, in_cited, in_case, show in checks:
            for value in values:
                if in_cited(value):
                    continue
                if in_case(value):
                    findings.append(Finding(index, "value_not_in_cited_source",
                                            f"{kind} {show(value)} is in the case but not in the sources cited here"))
                else:
                    findings.append(Finding(index, f"unsupported_{kind}",
                                            f"{kind} {show(value)} does not appear anywhere in the case"))
    return VerificationReport(passed=not findings, findings=findings,
                              n_sentences=len(parsed.sentences), n_cited=n_cited)
