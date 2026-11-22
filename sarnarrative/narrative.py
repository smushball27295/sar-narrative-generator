"""Narrative data structures and the text format shared by generator and verifier.

A narrative is a list of sentences, each carrying the ids it cites. In text
form the citations sit in square brackets before the full stop:

    The account received 14 cash deposits totalling $128,300.00 [F9].

The verifier always works from the text form, so that a narrative from the
template generator and one typed by a language model go through exactly the
same parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# F12 is a fact id, STR-03 an indicator id, STR-00 a typology description.
CITATION_RE = re.compile(r"\[([A-Z]{1,4}-?\d{1,3})\]")
_TRAILING_CITATIONS_RE = re.compile(r"([.!?])((?:\s*\[[A-Z]{1,4}-?\d{1,3}\])+)")
_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')]*\s+(?=[A-Z0-9$\[(\"'])")
_BULLET_RE = re.compile(r"^(?:[-*•]\s+|\d{1,2}[.)]\s+)")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "inc", "ltd", "co", "corp", "no", "st", "vs", "etc", "approx",
                  "e.g", "i.e", "u.s"}

# The order FinCEN's narrative guidance lists the questions in.
SECTIONS = ("who", "what", "when", "where", "why", "how")


@dataclass
class Sentence:
    text: str  # without citation markers and without the final full stop
    citations: list[str] = field(default_factory=list)
    section: str = ""

    def render(self) -> str:
        marks = "".join(f"[{c}]" for c in self.citations)
        return f"{self.text} {marks}." if marks else f"{self.text}."

    def to_dict(self) -> dict:
        return {"text": self.text + ".", "citations": list(self.citations)}


@dataclass
class Narrative:
    sentences: list[Sentence]
    generator: str = "template"
    typology_id: str | None = None

    def to_text(self) -> str:
        return " ".join(sentence.render() for sentence in self.sentences)

    def cited_ids(self) -> set[str]:
        return {c for sentence in self.sentences for c in sentence.citations}


def split_sentences(text: str) -> list[str]:
    """Split narrative text into sentences, keeping citation markers attached.

    Handles the two layouts a model tends to produce: citations before the
    full stop and citations after it. Markdown headings are dropped and list
    markers are stripped. A full stop after a common abbreviation does not
    end a sentence.
    """
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = _BULLET_RE.sub("", line)
        line = _TRAILING_CITATIONS_RE.sub(lambda m: f"{m.group(2)}{m.group(1)}", line)
        pieces: list[str] = []
        for piece in _BOUNDARY_RE.split(line):
            piece = piece.strip()
            if not piece:
                continue
            previous_words = pieces[-1].rstrip(".").split() if pieces else []
            if previous_words and previous_words[-1].lower() in _ABBREVIATIONS and pieces[-1].endswith("."):
                pieces[-1] = f"{pieces[-1]} {piece}"
            else:
                pieces.append(piece)
        sentences.extend(pieces)
    return sentences


def parse_sentence(raw: str) -> Sentence:
    citations = list(dict.fromkeys(CITATION_RE.findall(raw)))
    text = CITATION_RE.sub("", raw)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = text.rstrip(".!? ").strip()
    return Sentence(text=text, citations=citations)


def parse_narrative(text: str, generator: str = "text") -> Narrative:
    """Parse narrative text into sentences with their citations."""
    sentences = [parse_sentence(raw) for raw in split_sentences(text)]
    return Narrative([s for s in sentences if s.text], generator=generator)
