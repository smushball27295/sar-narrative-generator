"""Derive a behaviour profile and a list of citable facts from a case.

`compute_profile` turns the raw transactions and the customer record into a
flat dictionary of numbers. `derive_facts` turns that profile into the facts
(F1, F2, ...) that narratives cite. Neither function reads the case label,
with one exception that is kept visibly separate: after the facts are built,
`derive_facts` marks which of them are "key" for the labelled typology so that
the evaluation harness can ask whether a narrative included them.

A fact is only emitted when it has something to say (for example there is no
"cash deposits" fact on an account with no cash deposits), so the fact ids
are sequential within a case rather than fixed across cases.
"""

from __future__ import annotations

from datetime import date
from statistics import median

from .cases import IN_TYPES, OUT_TYPES, TYPE_LABELS, Case, Fact

BAND_LOW, BAND_HIGH = 8000.0, 10000.0
EVERYDAY_KINDS = {"merchant", "rent", "utility", "payroll", "employer", "settlement"}

# The facts that carry the pattern for each typology. This one table is used
# twice: `derive_facts` uses it to mark the key facts of the labelled typology,
# and the template generator uses it to choose which facts to write up for the
# retrieved typology. When retrieval picks the wrong typology the two sets
# differ, and the fact inclusion rate in the evaluation shows the cost.
EVIDENCE_KEYS = {
    "STR": ["cash_in", "cash_in_band", "cash_in_sites", "cash_multi", "cash_out_band", "expected"],
    "FUN": ["cash_in", "cash_in_away", "cash_in_sites", "cash_in_third_party", "cash_out", "wires_out", "turnaround"],
    "RPM": ["wires_in", "wires_out", "wire_parties", "foreign", "turnaround", "flow_through", "round_wires"],
    "MUL": ["account_age", "person_in", "cash_out", "wires_out", "person_out", "turnaround", "flow_through"],
    "TBM": ["invoices", "invoice_payments", "invoice_third_party", "goods_line", "wires_out", "foreign"],
    "SHL": ["wires_in", "wires_out", "everyday", "shared_address", "returned", "round_wires", "foreign"],
    "EFE": ["prior_avg", "new_payees", "savings_in", "gift_cards", "cash_out", "exchange_out"],
    "VCO": ["exchange_in", "exchange_out", "person_in", "cash_out", "wires_out", "turnaround", "account_age"],
}


def fmt_money(amount: float) -> str:
    return f"${amount:,.2f}"


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {plural or word + 's'}"


def _state(location: str) -> str | None:
    """Two-letter state from 'City, ST', or None for anything else."""
    if ", " in location:
        tail = location.rsplit(", ", 1)[1]
        if len(tail) == 2 and tail.isupper():
            return tail
    return None


def _is_foreign(location: str) -> bool:
    return location not in ("online", "United States") and _state(location) is None


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def compute_profile(case: Case) -> dict:
    """Summarise a case as numbers. No thresholds are applied here."""
    txns = case.transactions
    cps = case.counterparties
    account = case.account
    subject = case.subject
    start, end = case.period["start"], case.period["end"]

    def kind(name: str) -> str:
        return cps.get(name, {}).get("kind", "unknown")

    credits = [t for t in txns if t.type in IN_TYPES]
    debits = [t for t in txns if t.type in OUT_TYPES]
    total_in = round(sum(t.amount for t in credits), 2)
    total_out = round(sum(t.amount for t in debits), 2)

    cash_in = [t for t in txns if t.type == "cash_deposit"]
    cash_out = [t for t in txns if t.type == "cash_withdrawal"]
    wires_in = [t for t in txns if t.type == "wire_in"]
    wires_out = [t for t in txns if t.type == "wire_out"]
    wires = wires_in + wires_out
    person_in = [t for t in credits if kind(t.counterparty) == "individual"]
    person_out = [t for t in debits if kind(t.counterparty) == "individual"]
    exchange_in = [t for t in credits if kind(t.counterparty) == "exchange"]
    exchange_out = [t for t in debits if kind(t.counterparty) == "exchange"]
    gift_cards = [t for t in debits if kind(t.counterparty) == "gift_card"]
    savings_in = [t for t in txns if t.type == "transfer_in"]
    everyday = [t for t in txns if kind(t.counterparty) in EVERYDAY_KINDS]

    sites = sorted({t.location for t in cash_in})
    cash_days: dict[str, int] = {}
    for t in cash_in:
        cash_days[t.date] = cash_days.get(t.date, 0) + 1

    # Time between money arriving and money leaving, for sums that matter.
    lags = []
    big_debits = [t for t in debits if t.amount >= 1000]
    for credit in credits:
        if credit.amount < 1000:
            continue
        following = [t for t in big_debits if t.date >= credit.date]
        if following:
            lags.append(_days(credit.date, following[0].date))

    foreign_places = sorted({t.location for t in wires if _is_foreign(t.location)})

    by_party: dict[str, float] = {}
    for t in wires:
        by_party[t.counterparty] = by_party.get(t.counterparty, 0.0) + t.amount
    wire_parties = [name for name, _ in sorted(by_party.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]

    senders_in = {t.counterparty for t in wires_in}
    returned = [t for t in wires_out if t.counterparty in senders_in
                and any(w.counterparty == t.counterparty and w.date <= t.date for w in wires_in)]

    established = set(account.get("established_payees") or [])
    new_payee_totals: dict[str, float] = {}
    if established:
        for t in debits:
            if t.counterparty in established or t.counterparty == "self":
                continue
            if kind(t.counterparty) in EVERYDAY_KINDS:
                continue
            if t.amount >= 500:
                new_payee_totals[t.counterparty] = new_payee_totals.get(t.counterparty, 0.0) + t.amount

    invoices = [d for d in case.documents if d.get("kind") == "invoice"]
    invoice_numbers = {d["number"] for d in invoices}
    invoice_buyers = sorted({d["buyer"] for d in invoices})
    invoice_payments = [t for t in wires_in if any(number in t.memo for number in invoice_numbers)]
    third_party_payments = [t for t in invoice_payments if t.counterparty not in invoice_buyers]
    invoice_goods = sorted({d["goods"] for d in invoices})
    goods_mismatch = bool(invoices) and any(g not in subject.get("goods", []) for g in invoice_goods)

    address_id = subject.get("address_id")
    shared_address = sorted(name for name, cp in cps.items() if address_id and cp.get("address_id") == address_id)

    largest = max(txns, key=lambda t: t.amount) if txns else None
    months = max(1.0, (_days(start, end) + 1) / 30.0)

    return {
        "days": _days(start, end) + 1,
        "months": months,
        "n_txns": len(txns),
        "n_credits": len(credits), "total_in": total_in,
        "n_debits": len(debits), "total_out": total_out,
        "largest": largest,
        "expected_monthly": account.get("expected_monthly_activity"),
        "prior_avg_debits": account.get("prior_avg_monthly_debits"),
        "account_age_days": _days(account["opened"], start),
        "company_age_days": _days(subject["registered"], start) if subject.get("registered") else None,
        "cash_in_n": len(cash_in), "cash_in_total": round(sum(t.amount for t in cash_in), 2),
        "cash_in_band_n": sum(1 for t in cash_in if BAND_LOW <= t.amount < BAND_HIGH),
        "cash_in_sites": sites,
        "cash_in_states": sorted({s for s in (_state(x) for x in sites) if s}),
        "cash_in_away_n": sum(1 for t in cash_in if _state(t.location) != subject.get("home_state")),
        "cash_in_third_party_n": sum(1 for t in cash_in if t.counterparty != "self"),
        "cash_multi_days": sum(1 for n in cash_days.values() if n >= 2), "cash_days": len(cash_days),
        "cash_out_n": len(cash_out), "cash_out_total": round(sum(t.amount for t in cash_out), 2),
        "cash_out_band_n": sum(1 for t in cash_out if BAND_LOW <= t.amount < BAND_HIGH),
        "wires_in_n": len(wires_in), "wires_in_total": round(sum(t.amount for t in wires_in), 2),
        "wires_in_parties": len({t.counterparty for t in wires_in}),
        "wires_out_n": len(wires_out), "wires_out_total": round(sum(t.amount for t in wires_out), 2),
        "wires_out_parties": len({t.counterparty for t in wires_out}),
        "wire_parties": wire_parties,
        "foreign_places": foreign_places,
        "foreign_wire_total": round(sum(t.amount for t in wires if _is_foreign(t.location)), 2),
        "wires_total": round(sum(t.amount for t in wires), 2),
        "wire_party_count": len({t.counterparty for t in wires}),
        "round_wires_n": sum(1 for t in wires if t.amount % 1000 == 0), "wires_n": len(wires),
        "person_in_n": len(person_in), "person_in_total": round(sum(t.amount for t in person_in), 2),
        "person_in_parties": len({t.counterparty for t in person_in}),
        "person_out_n": len(person_out), "person_out_total": round(sum(t.amount for t in person_out), 2),
        "person_out_parties": sorted({t.counterparty for t in person_out}),
        "exchange_in_n": len(exchange_in), "exchange_in_total": round(sum(t.amount for t in exchange_in), 2),
        "exchange_in_names": sorted({t.counterparty for t in exchange_in}),
        "exchange_out_n": len(exchange_out), "exchange_out_total": round(sum(t.amount for t in exchange_out), 2),
        "gift_cards_n": len(gift_cards), "gift_cards_total": round(sum(t.amount for t in gift_cards), 2),
        "savings_in_n": len(savings_in), "savings_in_total": round(sum(t.amount for t in savings_in), 2),
        "everyday_n": len(everyday),
        "lag_n": len(lags), "lag_median": int(median(lags)) if lags else None,
        "flow_through_pct": int(round(100 * total_out / total_in)) if total_in > 0 else None,
        "returned_n": len(returned), "returned_total": round(sum(t.amount for t in returned), 2),
        "returned_parties": sorted({t.counterparty for t in returned}),
        "new_payees": sorted(new_payee_totals), "new_payees_total": round(sum(new_payee_totals.values()), 2),
        "invoices_n": len(invoices), "invoices_total": round(sum(d["value"] for d in invoices), 2),
        "invoice_goods": invoice_goods, "invoice_buyers": invoice_buyers,
        "invoice_payments_n": len(invoice_payments),
        "invoice_payments_total": round(sum(t.amount for t in invoice_payments), 2),
        "invoice_payers": len({t.counterparty for t in invoice_payments}),
        "third_party_payments_n": len(third_party_payments),
        "third_party_payers": sorted({t.counterparty for t in third_party_payments}),
        "goods_mismatch": goods_mismatch,
        "shared_address": shared_address,
    }


def _share(part: float, whole: float) -> float:
    return min(1.0, part / whole) if whole else 0.0


def derive_facts(case: Case) -> list[Fact]:
    """Build the citable facts for a case, numbered F1, F2, ... in a fixed order.

    Each fact also gets a salience weight between 0 and 1: roughly the share of
    the period's money, or of the relevant transactions, that the fact
    describes. Identity facts (who, which account, which dates) get 0 because
    they say nothing about behaviour. The weights only affect retrieval.
    """
    p = compute_profile(case)
    subject, account = case.subject, case.account
    rows: list[tuple[str, str, str, float]] = []  # (key, label, value, weight)

    def add(key: str, label: str, value: str, weight: float = 0.0) -> None:
        rows.append((key, label, value, round(max(0.0, min(1.0, weight)), 3)))

    total_in, total_out = p["total_in"], p["total_out"]
    turnover = total_in + total_out

    add("subject", "Subject", f"{subject['name']}, {subject['kind']}, {subject['detail']}")
    add("account", "Account under review",
        f"{account['id']}, {account['type']}, opened {account['opened']} at the {account['branch']} branch")
    add("period", "Review period", f"{case.period['start']} to {case.period['end']} ({p['days']} days)")
    add("txn_count", "Transactions in the review period", str(p["n_txns"]))
    add("credits", "Credits in the review period",
        f"{_plural(p['n_credits'], 'credit')} totalling {fmt_money(total_in)}")
    add("debits", "Debits in the review period",
        f"{_plural(p['n_debits'], 'debit')} totalling {fmt_money(total_out)}")
    largest = p["largest"]
    if largest is not None:
        add("largest", "Largest single transaction",
            f"{fmt_money(largest.amount)} ({TYPE_LABELS[largest.type]} on {largest.date}"
            + (f", counterparty {largest.counterparty})" if largest.counterparty != "self" else ")"))
    if p["expected_monthly"]:
        # Ten times the stated activity or more counts as fully salient.
        add("expected", "Expected monthly activity stated at account opening", fmt_money(p["expected_monthly"]),
            total_in / (p["expected_monthly"] * p["months"]) / 10)
    if p["account_age_days"] < 365:
        add("account_age", "Account age at the start of the review period", _plural(p["account_age_days"], "day"),
            1 - p["account_age_days"] / 365)
    if p["prior_avg_debits"]:
        add("prior_avg", "Average monthly debits in the six months before the review period",
            fmt_money(p["prior_avg_debits"]), total_out / (p["prior_avg_debits"] * p["months"]) / 10)

    if p["cash_in_n"]:
        n = p["cash_in_n"]
        cash_share = _share(p["cash_in_total"], total_in)
        add("cash_in", "Cash deposits", f"{_plural(n, 'cash deposit')} totalling {fmt_money(p['cash_in_total'])}",
            cash_share)
        if p["cash_in_band_n"]:
            add("cash_in_band", f"Cash deposits of at least {fmt_money(BAND_LOW)} and under {fmt_money(BAND_HIGH)}",
                f"{p['cash_in_band_n']} of {n}", cash_share * p["cash_in_band_n"] / n)
        add("cash_in_sites", "Locations of cash deposits",
            f"{_plural(len(p['cash_in_sites']), 'branch', 'branches')} in "
            f"{_plural(len(p['cash_in_states']), 'state')}: {'; '.join(p['cash_in_sites'])}",
            cash_share * min(1.0, (len(p["cash_in_sites"]) - 1) / 3))
        if p["cash_in_away_n"]:
            add("cash_in_away", f"Cash deposits made outside {subject['home_state']}, the home state of the account",
                f"{p['cash_in_away_n']} of {n}", cash_share * p["cash_in_away_n"] / n)
        if p["cash_in_third_party_n"]:
            add("cash_in_third_party", "Cash deposits made by someone other than the account holder",
                f"{p['cash_in_third_party_n']} of {n}", cash_share * p["cash_in_third_party_n"] / n)
        if p["cash_multi_days"]:
            add("cash_multi", "Days with two or more cash deposits", _plural(p["cash_multi_days"], "day"),
                cash_share * p["cash_multi_days"] / p["cash_days"])
    if p["cash_out_n"]:
        cash_out_share = _share(p["cash_out_total"], total_out)
        add("cash_out", "Cash withdrawals",
            f"{_plural(p['cash_out_n'], 'cash withdrawal')} totalling {fmt_money(p['cash_out_total'])}",
            cash_out_share)
        if p["cash_out_band_n"]:
            add("cash_out_band",
                f"Cash withdrawals of at least {fmt_money(BAND_LOW)} and under {fmt_money(BAND_HIGH)}",
                f"{p['cash_out_band_n']} of {p['cash_out_n']}",
                cash_out_share * p["cash_out_band_n"] / p["cash_out_n"])

    wire_share = _share(p["wires_total"], turnover)
    if p["wires_in_n"]:
        add("wires_in", "Incoming wires",
            f"{_plural(p['wires_in_n'], 'incoming wire')} totalling {fmt_money(p['wires_in_total'])} "
            f"from {_plural(p['wires_in_parties'], 'sender')}", _share(p["wires_in_total"], total_in))
    if p["wires_out_n"]:
        add("wires_out", "Outgoing wires",
            f"{_plural(p['wires_out_n'], 'outgoing wire')} totalling {fmt_money(p['wires_out_total'])} "
            f"to {_plural(p['wires_out_parties'], 'beneficiary', 'beneficiaries')}",
            _share(p["wires_out_total"], total_out))
    if p["wire_parties"]:
        add("wire_parties", "Largest wire counterparties by value", "; ".join(p["wire_parties"]))
    if p["foreign_places"]:
        add("foreign", "Wire counterparty locations outside the United States", "; ".join(p["foreign_places"]),
            wire_share * _share(p["foreign_wire_total"], p["wires_total"]))
    if p["wires_n"] >= 2 and p["round_wires_n"]:
        add("round_wires", "Wires for a whole number of thousands of dollars",
            f"{p['round_wires_n']} of {p['wires_n']}", wire_share * p["round_wires_n"] / p["wires_n"])

    if p["person_in_n"]:
        add("person_in", "Credits from individuals",
            f"{_plural(p['person_in_n'], 'credit')} totalling {fmt_money(p['person_in_total'])} "
            f"from {_plural(p['person_in_parties'], 'individual')}", _share(p["person_in_total"], total_in))
    if p["person_out_n"]:
        add("person_out", "Payments to individuals",
            f"{_plural(p['person_out_n'], 'payment')} totalling {fmt_money(p['person_out_total'])} "
            f"to {'; '.join(p['person_out_parties'][:4])}", _share(p["person_out_total"], total_out))
    if p["exchange_in_n"]:
        add("exchange_in", "Credits from virtual-currency exchanges",
            f"{_plural(p['exchange_in_n'], 'credit')} totalling {fmt_money(p['exchange_in_total'])} "
            f"({'; '.join(p['exchange_in_names'])})", _share(p["exchange_in_total"], total_in))
    if p["exchange_out_n"]:
        add("exchange_out", "Payments to virtual-currency exchanges",
            f"{_plural(p['exchange_out_n'], 'payment')} totalling {fmt_money(p['exchange_out_total'])}",
            _share(p["exchange_out_total"], total_out))
    if p["gift_cards_n"]:
        add("gift_cards", "Gift card purchases",
            f"{_plural(p['gift_cards_n'], 'purchase')} totalling {fmt_money(p['gift_cards_total'])}",
            _share(p["gift_cards_total"], total_out))
    if p["savings_in_n"]:
        add("savings_in", "Transfers in from the customer's own savings",
            f"{_plural(p['savings_in_n'], 'transfer')} totalling {fmt_money(p['savings_in_total'])}",
            _share(p["savings_in_total"], total_in))
    if p["new_payees"]:
        add("new_payees", "Payees first paid during the review period",
            f"{_plural(len(p['new_payees']), 'payee')} receiving {fmt_money(p['new_payees_total'])}: "
            f"{'; '.join(p['new_payees'][:5])}", _share(p["new_payees_total"], total_out))

    if p["lag_n"] >= 3:
        add("turnaround",
            f"Median time from a credit of {fmt_money(1000)} or more to the next debit of {fmt_money(1000)} or more",
            _plural(p["lag_median"], "day"), 1 / (1 + p["lag_median"]))
    if p["flow_through_pct"] is not None:
        add("flow_through", "Debits as a share of credits in the review period", f"{p['flow_through_pct']}%")
    add("everyday", "Payroll, rent, utility and card transactions in the review period",
        _plural(p["everyday_n"], "transaction"),
        1.0 if (subject["kind"] == "business" and p["everyday_n"] == 0) else 0.0)

    if p["invoices_n"]:
        invoice_share = _share(p["invoice_payments_total"], total_in)
        add("invoices", "Invoices on file",
            f"{_plural(p['invoices_n'], 'invoice')} totalling {fmt_money(p['invoices_total'])} for "
            f"{', '.join(p['invoice_goods'])}, issued to {'; '.join(p['invoice_buyers'])}", invoice_share)
        if p["invoice_payments_n"]:
            add("invoice_payments", "Incoming wires that quote an invoice number",
                f"{_plural(p['invoice_payments_n'], 'wire')} totalling {fmt_money(p['invoice_payments_total'])} "
                f"from {_plural(p['invoice_payers'], 'sender')}", invoice_share)
        if p["third_party_payments_n"]:
            add("invoice_third_party", "Invoice payments from senders not named on any invoice",
                f"{p['third_party_payments_n']} of {p['invoice_payments_n']}, sent by "
                f"{'; '.join(p['third_party_payers'])}",
                invoice_share * p["third_party_payments_n"] / p["invoice_payments_n"])
        if p["goods_mismatch"]:
            add("goods_line", "Goods on the invoices compared with the stated line of business",
                f"the invoices describe {', '.join(p['invoice_goods'])} and the stated line of business is "
                f"{subject.get('line')}", invoice_share)
    if p["shared_address"]:
        add("shared_address", "Counterparties registered at the same address as the subject",
            f"{_plural(len(p['shared_address']), 'counterparty', 'counterparties')}: "
            f"{'; '.join(p['shared_address'])}", wire_share * _share(len(p["shared_address"]), p["wire_party_count"]))
    if p["returned_n"]:
        add("returned", "Outgoing wires to parties that had earlier wired money in",
            f"{_plural(p['returned_n'], 'outgoing wire')} totalling {fmt_money(p['returned_total'])} to "
            f"{'; '.join(p['returned_parties'])}", _share(p["returned_total"], total_out))
    for note in case.notes:
        add("note", "Note on file", note, 1.0)

    # The only use of the label: flag the facts that matter for the ground truth.
    key_set = set(EVIDENCE_KEYS[case.typology_id])
    return [
        Fact(id=f"F{i}", key=key, label=label, value=value, is_key=key in key_set, weight=weight)
        for i, (key, label, value, weight) in enumerate(rows, start=1)
    ]
