"""Draft a SAR narrative for a case from its facts and the retrieved references.

Two generators share one output format (see `narrative.py`):

* `template_narrative` is deterministic. It writes one sentence per fact from
  fixed templates and tags each sentence with the fact or indicator it rests on.
* `llm_narrative` is optional. It sends the facts and the retrieved indicators
  to a chat-completions endpoint (the common `/chat/completions` request
  format) over plain `urllib` and asks for the same citation format. It only
  runs when `SAR_LLM_API_KEY` is set.

Neither generator sees the case label. The typology written up is the one
retrieval ranked first, unless the caller names one (an investigator who
disagrees with the ranking can override it).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .cases import Case, Fact
from .facts import EVIDENCE_KEYS
from .narrative import SECTIONS, Narrative, Sentence, parse_narrative
from .retrieval import Chunk, Hit, Query, Retriever, build_query

# Facts that every narrative states, whatever the typology.
CORE_KEYS = ["subject", "account", "period", "txn_count", "credits", "debits", "largest", "expected"]

# One sentence template per fact key: (section, template). `{v}` is the fact
# value, copied verbatim so the verifier can match it.
FACT_SENTENCES = {
    "subject": ("who", "The subject of this report is {v}"),
    "account": ("where", "The activity took place in account {v}"),
    "largest": ("what", "The largest single transaction was {v}"),
    "expected": ("why", "At account opening the customer stated expected monthly activity of {v}"),
    "account_age": ("when", "The account had been open for {v} when the review period began"),
    "prior_avg": ("why", "In the six months before the review period, debits averaged {v} a month"),
    "cash_in": ("how", "Cash came in through {v}"),
    "cash_in_band": ("how", "Of the cash deposits, {v} were for at least $8,000.00 and under $10,000.00"),
    "cash_in_sites": ("where", "The cash deposits were made at {v}"),
    "cash_in_away": ("where", "Of the cash deposits, {v} were made outside {home_state}, the home state of the account"),
    "cash_in_third_party": ("who", "Of the cash deposits, {v} were made by someone other than the account holder"),
    "cash_multi": ("how", "Two or more cash deposits were made on the same day on {v}"),
    "cash_out": ("how", "Cash left the account through {v}"),
    "cash_out_band": ("how", "Of the cash withdrawals, {v} were for at least $8,000.00 and under $10,000.00"),
    "wires_in": ("how", "The account received {v}"),
    "wires_out": ("how", "The account sent {v}"),
    "wire_parties": ("who", "The largest wire counterparties by value were {v}"),
    "foreign": ("where", "Wire counterparties outside the United States were located in {v}"),
    "round_wires": ("how", "Of the wires, {v} were for a whole number of thousands of dollars"),
    "person_in": ("how", "From individuals the account received {v}"),
    "person_out": ("how", "The account made {v}"),
    "exchange_in": ("how", "Virtual-currency exchanges sent the account {v}"),
    "exchange_out": ("how", "The account sent virtual-currency exchanges {v}"),
    "gift_cards": ("how", "The customer bought gift cards in {v}"),
    "savings_in": ("how", "The account was funded by {v} from the customer's own savings"),
    "new_payees": ("who", "During the review period the account paid, for the first time, {v}"),
    "turnaround": ("how", "The median time from a credit of $1,000.00 or more to the next debit of $1,000.00 or more was {v}"),
    "flow_through": ("how", "Debits in the review period came to {v} of credits"),
    "everyday": ("how", "Payroll, rent, utility and card activity in the review period amounted to {v}"),
    "invoices": ("how", "The trade documents on file are {v}"),
    "invoice_payments": ("how", "The account received {v} that quote an invoice number"),
    "invoice_third_party": ("who", "Of the invoice payments, {v}, came from senders not named on any invoice"),
    "goods_line": ("why", "A comparison of the documents with the customer profile shows that {v}"),
    "shared_address": ("who", "The registered address of the subject is also the registered address of {v}"),
    "returned": ("how", "Money went back to earlier senders through {v}"),
    "note": ("why", "A note on file reads: {v}"),
}

INDICATORS_PER_NARRATIVE = 3


@dataclass
class Draft:
    """Everything produced for one case: retrieval results and the narrative."""

    case: Case
    query: Query
    ranking: list[tuple[str, float]]  # typologies, best first
    hits: list[Hit]  # top chunks across all typologies
    typology_id: str  # the typology the narrative was written for
    references: list[Chunk]  # chunks the generator was allowed to cite
    narrative: Narrative


def _lower_first(text: str) -> str:
    first_word = text.split(" ", 1)[0]
    if first_word.isupper() and len(first_word) > 1:  # an acronym such as ATM
        return text
    return text[0].lower() + text[1:]


def select_references(retriever: Retriever, query: Query, typology_id: str) -> list[Chunk]:
    """The chunks a generator may cite: the description and the best indicators."""
    description = retriever.chunk(f"{typology_id}-00")
    indicators = [hit.chunk for hit in retriever.indicators_for(query, typology_id, INDICATORS_PER_NARRATIVE)
                  if hit.score > 0]
    return ([description] if description else []) + indicators


def template_narrative(case: Case, typology_id: str, references: list[Chunk]) -> Narrative:
    """Write the narrative from templates. Deterministic for a given input."""
    by_section: dict[str, list[Sentence]] = {section: [] for section in SECTIONS}

    def say(section: str, text: str, citations: list[str]) -> None:
        by_section[section].append(Sentence(text=text, citations=citations, section=section))

    def fact_sentence(fact: Fact) -> None:
        section, template = FACT_SENTENCES[fact.key]
        say(section, template.format(v=fact.value, home_state=case.subject.get("home_state", "")), [fact.id])

    used: set[str] = set()

    def use(key: str) -> None:
        for fact in case.facts:
            if fact.key == key and fact.id not in used:
                used.add(fact.id)
                fact_sentence(fact)

    use("subject")
    use("account")
    period, count = case.fact("period"), case.fact("txn_count")
    if period and count:
        say("when", f"The review covers {period.value}, during which the account recorded {count.value} transactions",
            [period.id, count.id])
    credits, debits = case.fact("credits"), case.fact("debits")
    if credits and debits:
        say("what", f"In that period the account received {credits.value} and paid out {debits.value}",
            [credits.id, debits.id])
    use("largest")
    for key in EVIDENCE_KEYS.get(typology_id, []):
        if key != "expected":
            use(key)
    use("expected")

    description = next((c for c in references if c.kind == "description"), None)
    if description:
        say("why", f"The activity in this account resembles the reference typology {description.title}",
            [description.id])
    for chunk in references:
        if chunk.kind == "indicator":
            say("why", f"A red flag from that reference that may apply is: {_lower_first(chunk.text)}", [chunk.id])
    use("note")

    ordered = [sentence for section in SECTIONS for sentence in by_section[section]]
    return Narrative(ordered, generator="template", typology_id=typology_id)


# --------------------------------------------------------------------------
# Optional LLM mode
# --------------------------------------------------------------------------


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0


def llm_config_from_env(env: dict | None = None) -> LLMConfig | None:
    """Read the LLM settings from the environment. None means the mode is off."""
    env = os.environ if env is None else env
    api_key = env.get("SAR_LLM_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = env.get("SAR_LLM_BASE_URL", "").strip()
    model = env.get("SAR_LLM_MODEL", "").strip()
    if not base_url or not model:
        raise LLMError("SAR_LLM_API_KEY is set, so SAR_LLM_BASE_URL and SAR_LLM_MODEL must be set as well")
    return LLMConfig(api_key=api_key, base_url=base_url.rstrip("/"), model=model)


SYSTEM_PROMPT = (
    "You draft the narrative section of a suspicious activity report for an anti-money-laundering "
    "investigator. Use only the numbered facts and reference indicators you are given. Write plain prose "
    "that covers, in this order: who, what, when, where, why and how. End every sentence with the ids of "
    "the facts or indicators that support it, in square brackets before the full stop, for example "
    "'... totalling $12,300.00 [F5].' or '... [F7][STR-03].' Do not state any amount, count, date or name "
    "that is not in a fact you cite in the same sentence. Copy amounts, dates and names exactly as given. "
    "Do not add headings, bullet points, opinions about guilt or recommendations."
)


def build_prompt(case: Case, references: list[Chunk]) -> list[dict]:
    """Chat messages for the LLM. Only facts and retrieved references go in."""
    lines = ["Facts:"]
    lines += [f"[{fact.id}] {fact.label}: {fact.value}" for fact in case.facts]
    lines.append("")
    lines.append("Reference indicators retrieved for this case:")
    for chunk in references:
        prefix = f"{chunk.title} (description)" if chunk.kind == "description" else chunk.title
        lines.append(f"[{chunk.id}] {prefix}: {chunk.text}")
    lines.append("")
    lines.append("Write the narrative now.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


def call_chat_completions(config: LLMConfig, messages: list[dict]) -> str:
    """POST to `{base_url}/chat/completions` and return the reply text."""
    body = json.dumps({"model": config.model, "messages": messages, "temperature": 0}).encode("utf-8")
    request = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMError(f"chat-completions request failed: {exc}") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("chat-completions response has no choices[0].message.content") from exc


def llm_narrative(case: Case, typology_id: str, references: list[Chunk], config: LLMConfig) -> Narrative:
    """Ask the configured model for a narrative and parse it into sentences."""
    text = call_chat_completions(config, build_prompt(case, references))
    narrative = parse_narrative(text, generator="llm")
    narrative.typology_id = typology_id
    return narrative


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def draft(case: Case, retriever: Retriever, typology_id: str | None = None,
          llm: LLMConfig | None = None, query_mode: str = "both") -> Draft:
    """Run retrieval and generation for one case."""
    query = build_query(case, query_mode)
    ranking = retriever.rank_typologies(query)
    chosen = typology_id or ranking[0][0]
    references = select_references(retriever, query, chosen)
    if llm is not None:
        narrative = llm_narrative(case, chosen, references, llm)
    else:
        narrative = template_narrative(case, chosen, references)
    return Draft(case=case, query=query, ranking=ranking, hits=retriever.search(query, 5),
                 typology_id=chosen, references=references, narrative=narrative)
