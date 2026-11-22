"""Tests for chunking, the encoder interface and retrieval."""

import numpy as np
import pytest

from sarnarrative.cases import generate_case
from sarnarrative.retrieval import (Query, Retriever, build_chunks, build_query, default_retriever,
                                    describe_behaviour)
from sarnarrative.typologies import TYPOLOGY_IDS, load_typologies


def test_typology_documents_and_chunking():
    typologies = load_typologies()
    assert [t.id for t in typologies] == TYPOLOGY_IDS
    for typology in typologies:
        assert 5 <= len(typology.indicators) <= 7
        assert [i.id for i in typology.indicators] == [
            f"{typology.id}-{n:02d}" for n in range(1, len(typology.indicators) + 1)]
        assert typology.description and typology.source and typology.narrative_should_establish

    # one chunk per description and one per indicator, with unique ids
    chunks = build_chunks(typologies)
    assert len(chunks) == sum(1 + len(t.indicators) for t in typologies)
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.id for c in chunks if c.kind == "description"] == [f"{t.id}-00" for t in typologies]
    assert all(c.id.startswith(c.typology_id + "-") for c in chunks)


@pytest.mark.parametrize("text, expected", [
    ("many cash deposits just under $10,000 at several branches on the same day", "STR"),
    ("invoices for goods were paid by third parties not named on the trade documents", "TBM"),
    ("an older customer suddenly bought gift cards and wired money to new payees", "EFE"),
    ("payouts from virtual-currency exchanges followed by cash withdrawals", "VCO"),
])
def test_retrieval_finds_the_obvious_typology(text, expected):
    retriever = default_retriever()
    assert retriever.rank_typologies(text)[0][0] == expected
    assert retriever.search(text, k=1)[0].chunk.typology_id == expected


def test_query_does_not_depend_on_the_label():
    case = generate_case(21, "MUL")
    before = build_query(case)
    case.typology_id = "STR"  # relabel without touching the transactions
    after = build_query(case)
    assert before == after
    assert all(0 < w <= 1 for w in before.weights)
    assert describe_behaviour(case)[0].startswith("The account holder")

    hits = default_retriever().search(before, k=5)
    scores = [h.score for h in hits]
    assert len(hits) == 5 and scores == sorted(scores, reverse=True)
    assert all(-1e-9 <= score <= 1 + 1e-9 for score in scores)


def test_backend_can_be_swapped():
    """Any object with fit/encode works, which is what a neural backend relies on."""

    class LengthBackend:
        def fit(self, texts, idf_corpus=None):
            self.fitted = len(texts)

        def encode(self, texts):
            return np.array([[len(t), t.count("cash") + 1.0] for t in texts], dtype=float)

    backend = LengthBackend()
    retriever = Retriever(build_chunks(load_typologies()), backend)
    assert backend.fitted == len(retriever.chunks)
    assert len(retriever.rank_typologies(Query.from_text("cash cash cash"))) == 8
