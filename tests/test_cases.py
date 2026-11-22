"""Tests for the synthetic case generator and the derived facts."""

from datetime import date

import pytest

from sarnarrative.cases import Case, generate_case, generate_cases
from sarnarrative.facts import EVIDENCE_KEYS
from sarnarrative.typologies import TYPOLOGY_IDS


def test_generator_is_deterministic():
    first = generate_case(123, "FUN").to_dict()
    assert generate_case(123, "FUN").to_dict() == first
    assert generate_case(124, "FUN").to_dict() != first
    # without a typology the label is drawn from the seed, also deterministically
    assert generate_case(7).typology_id == generate_case(7).typology_id
    # a balanced batch cycles through the typologies, and a case survives a JSON round trip
    cases = generate_cases(16, seed=300)
    assert [c.typology_id for c in cases] == TYPOLOGY_IDS * 2
    assert Case.from_dict(cases[4].to_dict()).to_dict() == cases[4].to_dict()


@pytest.mark.parametrize("typology", TYPOLOGY_IDS)
def test_case_structure(typology):
    case = generate_case(5, typology)
    assert case.typology_id == typology
    assert case.subject["name"] and case.subject["kind"] in ("individual", "business")
    assert case.account["id"].startswith("SYN-")

    start, end = date.fromisoformat(case.period["start"]), date.fromisoformat(case.period["end"])
    dates = [date.fromisoformat(t.date) for t in case.transactions]
    assert dates == sorted(dates)
    assert all(start <= d <= end for d in dates)
    assert all(t.amount > 0 for t in case.transactions)

    assert [f.id for f in case.facts] == [f"F{i}" for i in range(1, len(case.facts) + 1)]
    count_fact = case.fact("txn_count")
    assert count_fact is not None and count_fact.value == str(len(case.transactions))
    key_facts = [f for f in case.facts if f.is_key]
    assert key_facts, "the labelled pattern should leave at least one key fact"
    assert {f.key for f in key_facts} <= set(EVIDENCE_KEYS[typology])
