"""Load the typology reference documents from the `typologies/` directory.

Each document is a small markdown file with front matter (id, title, source),
a description, a numbered list of red-flag indicators with stable ids, and a
list of points the narrative should establish. The parser is deliberately
strict: a malformed file raises instead of being skipped, because a silently
missing indicator would change retrieval results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TYPOLOGY_DIR = Path(__file__).resolve().parent.parent / "typologies"

# Order matters in two places: balanced evaluation sets cycle through this
# list, and the web export takes the first 12 cases of that cycle, so the four
# typologies listed first (the two confusable pairs) appear twice there.
TYPOLOGY_IDS = ["STR", "FUN", "RPM", "MUL", "TBM", "SHL", "EFE", "VCO"]

_INDICATOR_RE = re.compile(r"^\d+\.\s+\*\*([A-Z]{3}-\d{2})\*\*\s+(.+)$")


@dataclass(frozen=True)
class Indicator:
    id: str
    text: str


@dataclass(frozen=True)
class Typology:
    id: str
    title: str
    source: str
    description: str
    indicators: tuple[Indicator, ...]
    narrative_should_establish: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "indicators": [{"id": i.id, "text": i.text} for i in self.indicators],
            "narrative_should_establish": list(self.narrative_should_establish),
        }


def parse_typology(text: str) -> Typology:
    """Parse one typology markdown document."""
    if not text.startswith("---"):
        raise ValueError("typology file must start with front matter")
    _, front, body = text.split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    for required in ("id", "title", "source"):
        if not meta.get(required):
            raise ValueError(f"front matter is missing '{required}'")

    sections: dict[str, list[str]] = {}
    current = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line.strip())

    description = " ".join(sections.get("description", []))
    indicators = []
    for line in sections.get("red-flag indicators", []):
        match = _INDICATOR_RE.match(line)
        if not match:
            raise ValueError(f"cannot parse indicator line: {line!r}")
        indicators.append(Indicator(match.group(1), match.group(2).strip()))
    establish = [
        line[2:].strip()
        for line in sections.get("what the narrative should establish", [])
        if line.startswith("- ")
    ]
    if not description or not indicators or not establish:
        raise ValueError(f"typology {meta['id']} is missing a section")
    for indicator in indicators:
        if not indicator.id.startswith(meta["id"] + "-"):
            raise ValueError(f"indicator {indicator.id} does not belong to {meta['id']}")
    return Typology(
        id=meta["id"],
        title=meta["title"],
        source=meta["source"],
        description=description,
        indicators=tuple(indicators),
        narrative_should_establish=tuple(establish),
    )


def load_typologies(directory: Path | str | None = None) -> list[Typology]:
    """Load every typology document, returned in the order of TYPOLOGY_IDS."""
    directory = Path(directory) if directory else TYPOLOGY_DIR
    loaded = {}
    for path in sorted(directory.glob("*.md")):
        typology = parse_typology(path.read_text(encoding="utf-8"))
        if typology.id in loaded:
            raise ValueError(f"duplicate typology id {typology.id}")
        loaded[typology.id] = typology
    missing = [tid for tid in TYPOLOGY_IDS if tid not in loaded]
    if missing:
        raise ValueError(f"missing typology documents for: {missing}")
    return [loaded[tid] for tid in TYPOLOGY_IDS]
