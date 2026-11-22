"""Seeded generator of synthetic AML cases.

Everything here is invented: the people, the companies, the account numbers
and the transactions. A case is built in three layers:

1. an ordinary baseline (salary, rent, card spending, or business running costs),
2. the pattern for the labelled typology, with its parameters drawn at random,
3. sometimes a weaker overlay borrowed from a different pattern, so that the
   signals of the eight typologies overlap the way they do in real alerts.

The label (`typology_id`) is ground truth for evaluation only. Facts are
derived afterwards by `facts.derive_facts` from the transactions and the
customer record. The label is used there for one thing, marking which facts
count as "key" when the evaluation scores a narrative.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from .typologies import TYPOLOGY_IDS

# --------------------------------------------------------------------------
# Vocabulary for invented names. The surnames are whimsical on purpose so that
# nobody mistakes a subject for a real person.
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Tessa", "Dorian", "Mirela", "Caspian", "Odette", "Lucan", "Anneliese", "Thaddeus",
    "Ysolde", "Barnaby", "Linnea", "Cormac", "Saskia", "Emrys", "Petra", "Alaric",
    "Wilhelmina", "Osgood", "Fenella", "Ignatius",
]
LAST_NAMES = [
    "Quillfeather", "Thistlewood", "Emberly", "Hollowbrook", "Stonecastle", "Wrenfield",
    "Marrowby", "Dunmoor", "Skerrit", "Oakhaven", "Pennywhistle", "Fennimore",
    "Thornbury", "Ashgrove", "Winterbourne", "Brackenridge",
]
COMPANY_STEMS = [
    "Bluefinch", "Copperline", "Saltmarsh", "Kestrel Ridge", "Ironwillow", "Halcyon Bay",
    "Tidewater Lantern", "Greymoor", "Sablewood", "Northquay", "Amberfold", "Larchmont Vale",
    "Fernhollow", "Stagmoor", "Quietwater", "Pebblecroft", "Harrowfield", "Sorrel Point",
]
COMPANY_SUFFIXES = [
    "Trading LLC", "Imports Ltd", "Logistics LLC", "Holdings LLC", "Consulting LLC",
    "Metals Ltd", "Textiles Co", "Exports Ltd", "Supply Ltd", "Ventures LLC",
    "Partners LLC", "Group Ltd",
]
EXCHANGES = ["Puddlecoin Exchange", "Gullwing Digital Exchange", "Marrowbit Exchange"]
EMPLOYERS = ["Pinecrest Staffing Co", "Willowmere Diner Group", "Dapplegate Care Homes", "Rookery Lane Schools"]

METROS = {
    "NJ": ["Newark, NJ", "Jersey City, NJ", "Elizabeth, NJ", "Paterson, NJ", "Edison, NJ"],
    "TX": ["Houston, TX", "Pasadena, TX", "Sugar Land, TX", "Katy, TX", "Baytown, TX"],
    "CA": ["San Diego, CA", "Chula Vista, CA", "Escondido, CA", "Oceanside, CA", "El Cajon, CA"],
    "IL": ["Chicago, IL", "Cicero, IL", "Evanston, IL", "Aurora, IL", "Joliet, IL"],
    "GA": ["Atlanta, GA", "Marietta, GA", "Decatur, GA", "Smyrna, GA", "Roswell, GA"],
    "AZ": ["Phoenix, AZ", "Mesa, AZ", "Tempe, AZ", "Chandler, AZ", "Gilbert, AZ"],
    "FL": ["Miami, FL", "Hialeah, FL", "Doral, FL", "Homestead, FL", "Coral Gables, FL"],
    "WA": ["Seattle, WA", "Tacoma, WA", "Bellevue, WA", "Kent, WA", "Renton, WA"],
}
FOREIGN_PLACES = [
    "Hong Kong", "Dubai, UAE", "Panama City, Panama", "Istanbul, Turkey", "Singapore",
    "Limassol, Cyprus", "Riga, Latvia", "Shenzhen, China", "Mexico City, Mexico",
    "Toronto, Canada", "London, UK",
]

INDIVIDUAL_OCCUPATIONS = [
    "restaurant server", "delivery driver", "dental assistant", "warehouse worker",
    "hair stylist", "bookkeeper", "bus mechanic", "retail supervisor",
]
YOUNG_OCCUPATIONS = ["student", "unemployed", "part-time barista", "rideshare driver"]
BUSINESS_LINES = {
    "machine parts wholesale": ["machine parts", "bearings and fasteners"],
    "used clothing export": ["used clothing", "baled textiles"],
    "consumer electronics import": ["mobile phone accessories", "consumer electronics"],
    "scrap metal trading": ["scrap copper", "aluminium scrap"],
    "food products import": ["dried fruit", "canned seafood"],
}
SERVICE_LINES = ["management consulting", "general trading", "freight brokerage", "investment holding"]
CASH_BUSINESS_LINES = ["restaurant", "convenience store", "car wash", "nail salon"]
# Suffixes for the subject's own name, so a nail salon is not called "Metals Ltd".
SUBJECT_SUFFIXES = {
    "restaurant": ["Kitchen LLC", "Grill LLC"],
    "convenience store": ["Market LLC", "Corner Store LLC"],
    "car wash": ["Car Wash LLC"],
    "nail salon": ["Nail Studio LLC"],
    "management consulting": ["Consulting LLC", "Partners LLC"],
    "general trading": ["Trading LLC", "Ventures LLC"],
    "freight brokerage": ["Logistics LLC"],
    "investment holding": ["Holdings LLC", "Group Ltd"],
}
TRADE_SUFFIXES = ["Trading LLC", "Imports Ltd", "Exports Ltd", "Supply Ltd"]

# Transaction types. The type carries the direction; the channel says how the
# transaction reached the bank.
IN_TYPES = {"cash_deposit", "wire_in", "ach_in", "p2p_in", "transfer_in"}
OUT_TYPES = {"cash_withdrawal", "wire_out", "ach_out", "p2p_out", "card_purchase", "cashiers_check"}
DEFAULT_CHANNEL = {
    "cash_deposit": "branch", "cash_withdrawal": "branch", "wire_in": "wire", "wire_out": "wire",
    "ach_in": "ach", "ach_out": "ach", "p2p_in": "p2p", "p2p_out": "p2p",
    "card_purchase": "card", "cashiers_check": "branch", "transfer_in": "online",
}
TYPE_LABELS = {
    "cash_deposit": "cash deposit", "cash_withdrawal": "cash withdrawal", "wire_in": "incoming wire",
    "wire_out": "outgoing wire", "ach_in": "incoming ACH transfer", "ach_out": "outgoing ACH transfer",
    "p2p_in": "incoming person-to-person transfer", "p2p_out": "outgoing person-to-person transfer",
    "card_purchase": "card purchase", "cashiers_check": "cashier's check purchase",
    "transfer_in": "transfer in from the customer's own account",
}


@dataclass
class Transaction:
    date: str  # ISO date
    type: str
    amount: float
    counterparty: str
    channel: str
    location: str
    memo: str = ""

    @property
    def is_credit(self) -> bool:
        return self.type in IN_TYPES


@dataclass
class Fact:
    id: str  # F1, F2, ... stable for a given case
    key: str  # semantic key, e.g. "cash_in"
    label: str
    value: str
    is_key: bool = False  # True if this fact is central to the labelled typology
    weight: float = 0.0  # salience in [0, 1], used to weight the retrieval query

    def text(self) -> str:
        return f"{self.label}: {self.value}"


@dataclass
class Case:
    id: str
    seed: int
    typology_id: str
    subject: dict
    accounts: list[dict]
    counterparties: dict[str, dict]
    transactions: list[Transaction]
    documents: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    period: dict = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)

    @property
    def account(self) -> dict:
        return self.accounts[0]

    def fact(self, key: str) -> Fact | None:
        for fact in self.facts:
            if fact.key == key:
                return fact
        return None

    def fact_by_id(self, fact_id: str) -> Fact | None:
        for fact in self.facts:
            if fact.id == fact_id:
                return fact
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Case":
        data = dict(data)
        data["transactions"] = [Transaction(**t) for t in data["transactions"]]
        data["facts"] = [Fact(**f) for f in data["facts"]]
        return cls(**data)


# --------------------------------------------------------------------------
# Builder: shared state and helpers used by every pattern.
# --------------------------------------------------------------------------


def _round_to(value: float, step: int) -> float:
    return float(max(step, int(round(value / step)) * step))


class _Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.start = date(2025, 1, 6) + timedelta(days=rng.randint(0, 240))
        self.days = rng.randint(28, 70)
        self.end = self.start + timedelta(days=self.days - 1)
        self.home_state = rng.choice(sorted(METROS))
        self.home = rng.choice(METROS[self.home_state])
        self.transactions: list[Transaction] = []
        self.counterparties: dict[str, dict] = {}
        self.documents: list[dict] = []
        self.notes: list[str] = []
        self.subject: dict = {}
        self.accounts: list[dict] = []
        self._used: set[str] = set()

    # ----- names -----

    def _unique(self, make) -> str:
        for _ in range(200):
            name = make()
            if name not in self._used:
                self._used.add(name)
                return name
        raise RuntimeError("ran out of unique names")

    def person_name(self) -> str:
        return self._unique(lambda: f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}")

    def company_name(self, suffixes: list[str] | None = None) -> str:
        pool = suffixes or COMPANY_SUFFIXES
        return self._unique(lambda: f"{self.rng.choice(COMPANY_STEMS)} {self.rng.choice(pool)}")

    def person(self, location: str | None = None) -> str:
        name = self.person_name()
        self.counterparties[name] = {"kind": "individual", "location": location or "United States"}
        return name

    def company(self, location: str | None = None, address_id: str | None = None) -> str:
        name = self.company_name()
        self.counterparties[name] = {"kind": "business", "location": location or self.us_city()}
        if address_id:
            self.counterparties[name]["address_id"] = address_id
        return name

    def exchange(self, name: str) -> str:
        self.counterparties.setdefault(name, {"kind": "exchange", "location": "United States"})
        return name

    def generic(self, name: str, kind: str = "merchant") -> str:
        self.counterparties.setdefault(name, {"kind": kind, "location": self.home})
        return name

    def us_city(self) -> str:
        state = self.rng.choice(sorted(METROS))
        return self.rng.choice(METROS[state])

    # ----- dates and transactions -----

    def day(self, lo: int = 0, hi: int | None = None) -> date:
        hi = self.days - 1 if hi is None else min(hi, self.days - 1)
        lo = min(lo, hi)
        return self.start + timedelta(days=self.rng.randint(lo, hi))

    def clamp(self, when: date) -> date:
        return min(max(when, self.start), self.end)

    def add(self, when: date, type_: str, amount: float, counterparty: str, location: str,
            memo: str = "", channel: str | None = None) -> Transaction:
        txn = Transaction(
            date=self.clamp(when).isoformat(),
            type=type_,
            amount=round(float(amount), 2),
            counterparty=counterparty,
            channel=channel or DEFAULT_CHANNEL[type_],
            location=location,
            memo=memo,
        )
        self.transactions.append(txn)
        return txn

    def cash_out(self, when: date, total: float, location: str) -> None:
        """Withdraw `total` in cash. Sums above $9,900 are split across withdrawals."""
        remaining = _round_to(total, 20)
        i = 0
        while remaining > 0:
            chunk = remaining if remaining <= 9900 else _round_to(self.rng.uniform(5000, 9800), 100)
            channel = "atm" if chunk <= 800 else "branch"
            self.add(when + timedelta(days=i // 2), "cash_withdrawal", chunk, "self", location, channel=channel)
            remaining = round(remaining - chunk, 2)
            i += 1


# --------------------------------------------------------------------------
# Subjects and baselines
# --------------------------------------------------------------------------


def _account_id(rng: random.Random) -> str:
    return f"SYN-{rng.randint(100000, 999999)}"


def _individual(b: _Builder, age_range=(24, 62), occupations=None, opened_days_before=(400, 4000),
                salary: bool = True) -> None:
    rng = b.rng
    name = b.person_name()
    age = rng.randint(*age_range)
    occupation = rng.choice(occupations or INDIVIDUAL_OCCUPATIONS)
    monthly_income = _round_to(rng.uniform(2200, 5200), 50) if salary else _round_to(rng.uniform(300, 1200), 50)
    opened = b.start - timedelta(days=rng.randint(*opened_days_before))
    b.subject = {
        "name": name, "kind": "individual", "age": age, "occupation": occupation,
        "home": b.home, "home_state": b.home_state,
        "detail": f"age {age}, {occupation}, resident of {b.home}",
    }
    established = (opened < b.start - timedelta(days=200))
    b.accounts = [{
        "id": _account_id(rng), "type": "personal checking", "opened": opened.isoformat(),
        "branch": b.home, "expected_monthly_activity": _round_to(monthly_income * rng.uniform(1.1, 1.6), 100),
        "prior_avg_monthly_debits": _round_to(monthly_income * rng.uniform(0.75, 0.95), 10) if established else None,
        "established_payees": ["Landlord", "Utility company"] if established else [],
    }]
    _individual_baseline(b, monthly_income, salary)


def _individual_baseline(b: _Builder, monthly_income: float, salary: bool) -> None:
    rng = b.rng
    if salary:
        employer = b.generic(rng.choice(EMPLOYERS), kind="employer")
        pay = round(monthly_income / 2 * rng.uniform(0.97, 1.03), 2)
        when = b.start + timedelta(days=rng.randint(0, 13))
        while when <= b.end:
            b.add(when, "ach_in", pay, employer, "online", memo="payroll")
            when += timedelta(days=14)
    landlord = b.generic("Landlord", kind="rent")
    utility = b.generic("Utility company", kind="utility")
    rent = _round_to(rng.uniform(700, 1900), 5)
    when = b.start + timedelta(days=rng.randint(0, 27))
    while when <= b.end:
        b.add(when, "ach_out", rent, landlord, "online", memo="rent")
        b.add(when + timedelta(days=rng.randint(1, 5)), "ach_out", round(rng.uniform(60, 220), 2),
              utility, "online", memo="utilities")
        when += timedelta(days=30)
    for _ in range(rng.randint(6, 14)):
        merchant = b.generic(rng.choice(["Grocery store", "Fuel station", "Pharmacy", "Coffee shop", "Hardware store"]))
        b.add(b.day(), "card_purchase", round(rng.uniform(8, 190), 2), merchant, b.home)


def _business(b: _Builder, line: str, goods: list[str] | None = None,
              registered_days_before=(700, 4000), operating: bool = True,
              address_id: str | None = None) -> None:
    rng = b.rng
    name = b.company_name(SUBJECT_SUFFIXES.get(line, TRADE_SUFFIXES))
    registered = b.start - timedelta(days=rng.randint(*registered_days_before))
    opened = registered + timedelta(days=rng.randint(3, 25))
    b.subject = {
        "name": name, "kind": "business", "line": line, "goods": goods or [],
        "registered": registered.isoformat(), "home": b.home, "home_state": b.home_state,
        "detail": f"{line}, registered {registered.isoformat()} in {b.home}",
    }
    if address_id:
        b.subject["address_id"] = address_id
    monthly = _round_to(rng.uniform(25000, 90000), 1000)
    established = opened < b.start - timedelta(days=200)
    b.accounts = [{
        "id": _account_id(rng), "type": "business checking", "opened": opened.isoformat(),
        "branch": b.home, "expected_monthly_activity": monthly,
        "prior_avg_monthly_debits": _round_to(monthly * rng.uniform(0.7, 0.95), 100) if (established and operating) else None,
        "established_payees": ["Payroll service", "Commercial landlord"] if (established and operating) else [],
    }]
    if operating:
        _business_baseline(b, monthly)


def _business_baseline(b: _Builder, monthly: float) -> None:
    rng = b.rng
    payroll = b.generic("Payroll service", kind="payroll")
    landlord = b.generic("Commercial landlord", kind="rent")
    when = b.start + timedelta(days=rng.randint(0, 13))
    while when <= b.end:
        b.add(when, "ach_out", round(monthly * rng.uniform(0.08, 0.14), 2), payroll, "online", memo="payroll")
        when += timedelta(days=14)
    when = b.start + timedelta(days=rng.randint(0, 27))
    while when <= b.end:
        b.add(when, "ach_out", _round_to(monthly * rng.uniform(0.05, 0.09), 50), landlord, "online", memo="rent")
        when += timedelta(days=30)
    for _ in range(rng.randint(5, 12)):
        merchant = b.generic(rng.choice(["Office supplies store", "Fuel station", "Courier service", "Telecom provider"]))
        b.add(b.day(), "card_purchase", round(rng.uniform(20, 600), 2), merchant, b.home)
    for _ in range(rng.randint(2, 5)):
        b.add(b.day(), "ach_in", round(rng.uniform(1500, 9000), 2), b.generic("Card settlement", kind="settlement"),
              "online", memo="card sales settlement")


# --------------------------------------------------------------------------
# Typology patterns
# --------------------------------------------------------------------------


def _pattern_structuring(b: _Builder) -> None:
    rng = b.rng
    if rng.random() < 0.3:
        _business(b, rng.choice(CASH_BUSINESS_LINES))
    else:
        _individual(b)
    local = [c for c in METROS[b.home_state] if c != b.home]
    branches = [b.home] + rng.sample(local, rng.randint(0, 3))
    if rng.random() < 0.25:  # a second state, which is what a funnel account also shows
        other_state = rng.choice([s for s in sorted(METROS) if s != b.home_state])
        branches += rng.sample(METROS[other_state], rng.randint(1, 2))
    n = rng.randint(8, 18)
    when = b.day(0, 6)
    last_branch = None
    for i in range(n):
        if rng.random() < 0.8:
            amount = _round_to(rng.uniform(8000, 9900), rng.choice([20, 50, 100]))
        else:
            amount = _round_to(rng.uniform(3500, 7900), 100)
        choices = [c for c in branches if c != last_branch] or branches
        branch = rng.choice(choices)
        b.add(when, "cash_deposit", amount, "self", branch)
        same_day_next = rng.random() < 0.3 and len(branches) > 1
        if same_day_next:
            last_branch = branch  # the next deposit, on the same day, goes to a different branch
        else:
            when = when + timedelta(days=rng.randint(1, max(1, (b.days - 8) // n + 1)))
            last_branch = None
    roll = rng.random()
    if roll < 0.35:
        dealer = b.company(b.home)
        for _ in range(rng.randint(1, 2)):
            b.add(b.day(b.days * 2 // 3), "cashiers_check", _round_to(rng.uniform(18000, 60000), 500),
                  dealer, b.home, memo="cashier's check")
    elif roll < 0.5:
        for _ in range(rng.randint(3, 6)):
            b.add(b.day(b.days // 2), "cash_withdrawal", _round_to(rng.uniform(8000, 9800), 100), "self",
                  rng.choice(branches))
    if rng.random() < 0.5:
        b.notes.append(rng.choice([
            f"Teller note {b.day().isoformat()}: the customer asked how much cash could be paid in before a form has to be completed",
            f"Teller note {b.day().isoformat()}: the customer took back part of a cash deposit after being asked for identification",
        ]))


def _pattern_funnel(b: _Builder) -> None:
    rng = b.rng
    if rng.random() < 0.3:
        _business(b, rng.choice(SERVICE_LINES + CASH_BUSINESS_LINES))
    else:
        _individual(b)
    away_states = rng.sample([s for s in sorted(METROS) if s != b.home_state], rng.randint(2, 5))
    away = [city for s in away_states for city in rng.sample(METROS[s], rng.randint(1, 3))]
    band_focus = rng.random() < 0.35
    n = rng.randint(10, 24)
    mode = rng.choices(["cash", "wire", "mixed"], weights=[5, 3, 2])[0]
    exporter = b.company(rng.choice(FOREIGN_PLACES + [b.us_city()])) if mode != "cash" else None
    third_party_rate = rng.uniform(0.5, 1.0)
    pending = 0.0
    when = b.day(0, 5)
    for i in range(n):
        location = rng.choice(away) if rng.random() < 0.85 else b.home
        if band_focus and rng.random() < 0.7:
            amount = _round_to(rng.uniform(8000, 9900), 100)
        else:
            amount = _round_to(rng.uniform(1500, 9800), rng.choice([20, 100]))
        depositor = "unidentified third party" if rng.random() < third_party_rate else "self"
        b.add(when, "cash_deposit", amount, depositor, location)
        pending += amount
        if rng.random() < 0.45 or i == n - 1:
            out_day = when + timedelta(days=rng.randint(0, 2))
            out_total = pending * rng.uniform(0.9, 0.99)
            use_wire = mode == "wire" or (mode == "mixed" and rng.random() < 0.5)
            if use_wire:
                b.add(out_day, "wire_out", _round_to(out_total, 100), exporter,
                      b.counterparties[exporter]["location"], memo="payment for goods")
            else:
                b.cash_out(out_day, out_total, b.home)
            pending = 0.0
        if rng.random() < 0.65:
            when = when + timedelta(days=rng.randint(1, max(1, (b.days - 8) // n + 1)))
    if rng.random() < 0.5:
        b.notes.append(
            f"Branch note {b.day().isoformat()}: cash was paid in by a person who is not named on the account "
            "and who gave only the account number")


def _pattern_rapid_movement(b: _Builder) -> None:
    rng = b.rng
    if rng.random() < 0.6:
        _business(b, rng.choice(SERVICE_LINES), operating=rng.random() < 0.5)
    else:
        _individual(b)
    senders = [b.company(rng.choice(FOREIGN_PLACES) if rng.random() < 0.4 else None) for _ in range(rng.randint(2, 4))]
    foreign_out = rng.random() < 0.5
    receivers = [b.company(rng.choice(FOREIGN_PLACES) if foreign_out and rng.random() < 0.8 else None)
                 for _ in range(rng.randint(2, 4))]
    cycles = rng.randint(4, 9)
    for _ in range(cycles):
        when = b.day(0, b.days - 3)
        if rng.random() < 0.7:
            amount = _round_to(rng.uniform(20000, 250000), 1000)
        else:
            amount = round(rng.uniform(20000, 250000), 2)
        sender = rng.choice(senders)
        b.add(when, "wire_in", amount, sender, b.counterparties[sender]["location"],
              memo=rng.choice(["", "payment", "transfer", "invoice"]))
        out_day = when + timedelta(days=rng.randint(0, 1))
        out_amount = round(amount * (1 - rng.choice([0, 0, 0.002, 0.005, 0.01, 0.015])), 2)
        receiver = rng.choice(receivers)
        if rng.random() < 0.2:
            first = round(out_amount * rng.uniform(0.4, 0.6), 2)
            other = rng.choice(receivers)
            b.add(out_day, "wire_out", first, receiver, b.counterparties[receiver]["location"], memo="transfer")
            b.add(out_day, "wire_out", round(out_amount - first, 2), other,
                  b.counterparties[other]["location"], memo="transfer")
        else:
            b.add(out_day, "wire_out", out_amount, receiver, b.counterparties[receiver]["location"], memo="transfer")
    if rng.random() < 0.5:
        b.notes.append(
            f"Analyst note {b.day().isoformat()}: the customer was asked for contracts or invoices behind the "
            "transfers and did not provide any")


def _pattern_mule(b: _Builder) -> None:
    rng = b.rng
    if rng.random() < 0.7:
        _individual(b, age_range=(18, 27), occupations=YOUNG_OCCUPATIONS, opened_days_before=(10, 80), salary=False)
    else:
        _individual(b, opened_days_before=(10, 400))
    n = rng.randint(8, 22)
    senders = [b.person() for _ in range(max(4, int(n * rng.uniform(0.6, 1.0))))]
    handlers = [b.person(rng.choice(FOREIGN_PLACES) if rng.random() < 0.3 else None) for _ in range(rng.randint(1, 2))]
    exchange = b.exchange(rng.choice(EXCHANGES)) if rng.random() < 0.2 else None
    out_modes = rng.sample(["cash", "p2p", "wire"], rng.randint(1, 2)) + (["exchange"] if exchange else [])
    keep = rng.uniform(0.05, 0.12)
    pending = 0.0
    when = b.day(0, 6)
    for i in range(n):
        kind = rng.choices(["p2p_in", "ach_in", "wire_in"], weights=[5, 3, 2])[0]
        amount = round(rng.uniform(400, 9500), 2) if rng.random() < 0.5 else _round_to(rng.uniform(400, 9500), 50)
        sender = senders[i] if i < len(senders) else rng.choice(senders)
        b.add(when, kind, amount, sender, "online" if kind != "wire_in" else b.counterparties[sender]["location"])
        pending += amount
        if rng.random() < 0.6 or i == n - 1:
            out_day = when + timedelta(days=rng.randint(0, 2))
            out_total = pending * (1 - keep)
            mode = rng.choice(out_modes)
            if mode == "cash":
                b.cash_out(out_day, out_total, b.home)
            elif mode == "p2p":
                b.add(out_day, "p2p_out", round(out_total, 2), rng.choice(handlers), "online")
            elif mode == "wire":
                handler = rng.choice(handlers)
                b.add(out_day, "wire_out", round(out_total, 2), handler, b.counterparties[handler]["location"])
            else:
                b.add(out_day, "ach_out", round(out_total, 2), exchange, "online", memo="exchange purchase")
            pending = 0.0
        if rng.random() < 0.75:
            when = when + timedelta(days=rng.randint(1, max(1, (b.days - 8) // n + 1)))
    if rng.random() < 0.5:
        b.notes.append(rng.choice([
            f"Analyst note {b.day().isoformat()}: the account holder said the incoming payments were for a "
            "work-from-home job found online",
            f"Analyst note {b.day().isoformat()}: another bank asked for one of the incoming payments to be sent "
            "back, saying its customer had not agreed to it",
        ]))


def _pattern_trade(b: _Builder) -> None:
    rng = b.rng
    line = rng.choice(sorted(BUSINESS_LINES))
    _business(b, line, goods=BUSINESS_LINES[line])
    mismatch = rng.random() < 0.5
    if mismatch:
        other_line = rng.choice([l for l in sorted(BUSINESS_LINES) if l != line])
        goods = rng.choice(BUSINESS_LINES[other_line])
    else:
        goods = rng.choice(BUSINESS_LINES[line])
    buyers = [b.company(rng.choice(FOREIGN_PLACES)) for _ in range(rng.randint(1, 2))]
    third_parties = [b.company(rng.choice(FOREIGN_PLACES)) for _ in range(rng.randint(2, 4))]
    suppliers = [b.company(rng.choice(FOREIGN_PLACES)) for _ in range(rng.randint(1, 3))]
    third_party_rate = rng.uniform(0.5, 1.0)
    received = 0.0
    for i in range(rng.randint(3, 6)):
        number = f"INV-{rng.randint(1000, 9999)}"
        value = round(rng.uniform(40000, 220000), 2)
        issued = b.day(0, b.days // 2)
        buyer = rng.choice(buyers)
        b.documents.append({"kind": "invoice", "number": number, "date": issued.isoformat(), "goods": goods,
                            "value": value, "buyer": buyer})
        target = value * rng.uniform(0.8, 1.6)
        parts = rng.randint(1, 3)
        for _ in range(parts):
            payer = rng.choice(third_parties) if rng.random() < third_party_rate else buyer
            amount = _round_to(target / parts, 5000)
            b.add(issued + timedelta(days=rng.randint(2, 20)), "wire_in", amount, payer,
                  b.counterparties[payer]["location"], memo=f"payment {number}")
            received += amount
    paid_out = received * rng.uniform(0.6, 0.9)
    parts = rng.randint(2, 5)
    for _ in range(parts):
        supplier = rng.choice(suppliers)
        b.add(b.day(b.days // 3), "wire_out", _round_to(paid_out / parts * rng.uniform(0.8, 1.2), 500), supplier,
              b.counterparties[supplier]["location"], memo="supplier payment")
    if rng.random() < 0.5:
        b.notes.append(
            f"Analyst note {b.day().isoformat()}: the customer could not produce shipping papers for "
            f"{rng.randint(1, 3)} of the invoices")


def _pattern_shell(b: _Builder) -> None:
    rng = b.rng
    address_id = f"ADDR-{rng.randint(1000, 9999)}"
    _business(b, rng.choice(["management consulting", "investment holding", "general trading"]),
              registered_days_before=(30, 240), operating=False, address_id=address_id)
    entities = []
    shared = rng.randint(1, 3)
    for i in range(rng.randint(4, 7)):
        place = rng.choice(FOREIGN_PLACES) if rng.random() < 0.4 else None
        entities.append(b.company(place, address_id=address_id if i < shared else None))
    memos = ["consulting services", "loan", "services agreement", "advisory fee", "intercompany transfer"]
    senders = []
    for _ in range(rng.randint(5, 10)):
        sender = rng.choice(entities)
        senders.append(sender)
        when = b.day(0, b.days - 10)
        b.add(when, "wire_in", _round_to(rng.uniform(25000, 400000), rng.choice([5000, 10000, 25000])), sender,
              b.counterparties[sender]["location"], memo=rng.choice(memos))
    total_in = sum(t.amount for t in b.transactions if t.type == "wire_in")
    n_out = rng.randint(5, 10)
    circular = rng.randint(1, 3) if rng.random() < 0.7 else 0
    for i in range(n_out):
        receiver = rng.choice(senders) if i < circular else rng.choice([e for e in entities if e not in senders] or entities)
        b.add(b.day(6), "wire_out", _round_to(total_in * rng.uniform(0.85, 0.99) / n_out * rng.uniform(0.7, 1.3), 5000),
              receiver, b.counterparties[receiver]["location"], memo=rng.choice(memos))
    if rng.random() < 0.5:
        b.notes.append(
            f"Analyst note {b.day().isoformat()}: no website, staff or premises could be found for the company")


def _pattern_elder(b: _Builder) -> None:
    rng = b.rng
    _individual(b, age_range=(68, 91), occupations=["retired teacher", "retired nurse", "retired machinist",
                                                     "retired postal worker"],
                opened_days_before=(3000, 9000))
    account = b.accounts[0]
    savings_id = _account_id(rng)
    b.accounts.append({"id": savings_id, "type": "savings", "opened": account["opened"], "branch": b.home})
    own = b.generic("Own savings account", kind="self")
    onset = rng.randint(3, b.days // 2)
    payees = [b.person(rng.choice(FOREIGN_PLACES) if rng.random() < 0.3 else None) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.4:
        payees.append(b.company())
    channels = rng.sample(["wire", "check", "gift", "cash", "exchange"], rng.randint(2, 4))
    spent = 0.0
    for _ in range(rng.randint(3, 8)):
        when = b.day(onset)
        mode = rng.choice(channels)
        if mode == "wire":
            payee = rng.choice(payees)
            amount = _round_to(rng.uniform(4000, 45000), 500)
            b.add(when, "wire_out", amount, payee, b.counterparties[payee]["location"])
        elif mode == "check":
            amount = _round_to(rng.uniform(3000, 30000), 500)
            b.add(when, "cashiers_check", amount, rng.choice(payees), b.home, memo="cashier's check")
        elif mode == "gift":
            amount = _round_to(rng.uniform(500, 2000), 100)
            b.add(when, "card_purchase", amount, b.generic("Gift card kiosk", kind="gift_card"), b.home,
                  memo="gift cards")
        elif mode == "cash":
            amount = _round_to(rng.uniform(2500, 9500), 100)
            b.add(when, "cash_withdrawal", amount, "self", b.home)
        else:
            amount = _round_to(rng.uniform(2000, 20000), 100)
            b.add(when, "ach_out", amount, b.exchange(rng.choice(EXCHANGES)), "online", memo="exchange purchase")
        spent += amount
    for _ in range(rng.randint(1, 3)):
        b.add(b.day(max(0, onset - 3)), "transfer_in", _round_to(spent * rng.uniform(0.3, 0.6), 500), own, "online",
              memo="from savings")
    if rng.random() < 0.6:
        b.notes.append(rng.choice([
            f"Teller note {b.day(onset).isoformat()}: the customer said the money was for a friend met online and "
            "asked that relatives not be told",
            f"Teller note {b.day(onset).isoformat()}: the customer was on the phone throughout and read the payment "
            "details from a handwritten slip",
            f"Teller note {b.day(onset).isoformat()}: a person the branch had not seen before came in with the "
            "customer and answered the questions put to the customer",
        ]))


def _pattern_virtual_currency(b: _Builder) -> None:
    rng = b.rng
    new_account = rng.random() < 0.4
    _individual(b, age_range=(21, 55), opened_days_before=(15, 110) if new_account else (400, 3000))
    exchanges = [b.exchange(name) for name in rng.sample(EXCHANGES, rng.randint(1, 3))]
    out_modes = rng.sample(["cash", "wire"], rng.randint(1, 2))
    receivers = [b.person(rng.choice(FOREIGN_PLACES)) if rng.random() < 0.6 else b.company(rng.choice(FOREIGN_PLACES))
                 for _ in range(rng.randint(1, 3))]
    for _ in range(rng.randint(4, 12)):
        when = b.day(0, b.days - 4)
        mode = rng.choice(out_modes)
        # Payouts that leave as cash are kept smaller so the number of withdrawals stays plausible.
        amount = round(rng.uniform(3000, 24000 if mode == "cash" else 60000), 2)
        b.add(when, rng.choice(["ach_in", "ach_in", "wire_in"]), amount, rng.choice(exchanges), "online",
              memo="exchange payout")
        out_day = when + timedelta(days=rng.randint(0, 2))
        out_total = amount * rng.uniform(0.95, 1.0)
        if mode == "cash":
            b.cash_out(out_day, out_total, b.home)
        else:
            receiver = rng.choice(receivers)
            b.add(out_day, "wire_out", round(out_total, 2), receiver, b.counterparties[receiver]["location"])
    if rng.random() < 0.3:  # also trades for other people
        for _ in range(rng.randint(10, 25)):
            b.add(b.day(), "p2p_in", round(rng.uniform(100, 1500), 2), b.person(), "online")
        for _ in range(rng.randint(3, 6)):
            b.add(b.day(), "ach_out", _round_to(rng.uniform(1500, 8000), 100), rng.choice(exchanges), "online",
                  memo="exchange purchase")
    if rng.random() < 0.5:
        b.notes.append(
            f"Analyst note {b.day().isoformat()}: the customer would not say where the digital currency came from")


PATTERNS = {
    "STR": _pattern_structuring,
    "FUN": _pattern_funnel,
    "RPM": _pattern_rapid_movement,
    "MUL": _pattern_mule,
    "TBM": _pattern_trade,
    "SHL": _pattern_shell,
    "EFE": _pattern_elder,
    "VCO": _pattern_virtual_currency,
}


# --------------------------------------------------------------------------
# Overlays: weak signals from other patterns
# --------------------------------------------------------------------------


def _overlay(b: _Builder) -> None:
    rng = b.rng
    options = ["cash_small", "wire_pair", "people_in", "exchange_small", "cash_out"]
    choice = rng.choice(options)
    if choice == "cash_small":
        for _ in range(rng.randint(2, 4)):
            b.add(b.day(), "cash_deposit", _round_to(rng.uniform(800, 4500), 20), "self", b.home)
    elif choice == "wire_pair":
        sender, receiver = b.company(), b.company()
        when = b.day(0, b.days - 10)
        amount = _round_to(rng.uniform(5000, 30000), 100)
        b.add(when, "wire_in", amount, sender, b.counterparties[sender]["location"])
        b.add(when + timedelta(days=rng.randint(3, 9)), "wire_out", _round_to(amount * rng.uniform(0.5, 0.9), 100),
              receiver, b.counterparties[receiver]["location"])
    elif choice == "people_in":
        for _ in range(rng.randint(3, 5)):
            b.add(b.day(), "p2p_in", round(rng.uniform(40, 600), 2), b.person(), "online")
    elif choice == "exchange_small":
        for _ in range(rng.randint(1, 2)):
            b.add(b.day(), "ach_in", round(rng.uniform(150, 2500), 2), b.exchange(rng.choice(EXCHANGES)), "online",
                  memo="exchange payout")
    else:
        for _ in range(rng.randint(2, 4)):
            b.add(b.day(), "cash_withdrawal", _round_to(rng.uniform(200, 800), 20), "self", b.home, channel="atm")


OVERLAY_RATE = 0.4


def generate_case(seed: int, typology: str | None = None) -> Case:
    """Generate one synthetic case. The same (seed, typology) always gives the same case."""
    from .facts import derive_facts  # local import: facts imports this module

    rng = random.Random(f"sar-case-{seed}-{typology or 'any'}")
    label = typology or rng.choice(TYPOLOGY_IDS)
    if label not in PATTERNS:
        raise ValueError(f"unknown typology {label!r}; expected one of {TYPOLOGY_IDS}")
    b = _Builder(rng)
    PATTERNS[label](b)
    if rng.random() < OVERLAY_RATE:
        _overlay(b)
    order = {id(t): i for i, t in enumerate(b.transactions)}
    b.transactions.sort(key=lambda t: (t.date, order[id(t)]))
    case = Case(
        id=f"SYN-CASE-{seed:05d}-{label}",
        seed=seed,
        typology_id=label,
        subject=b.subject,
        accounts=b.accounts,
        counterparties=b.counterparties,
        transactions=b.transactions,
        documents=b.documents,
        notes=b.notes,
        period={"start": b.start.isoformat(), "end": b.end.isoformat()},
    )
    case.facts = derive_facts(case)
    return case


def generate_cases(n: int, seed: int = 0, balanced: bool = True) -> list[Case]:
    """Generate `n` cases. With `balanced`, typologies are assigned in rotation."""
    cases = []
    for i in range(n):
        typology = TYPOLOGY_IDS[i % len(TYPOLOGY_IDS)] if balanced else None
        cases.append(generate_case(seed + i, typology))
    return cases
