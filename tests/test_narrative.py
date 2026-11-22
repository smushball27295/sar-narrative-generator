"""Tests for the generators and the verifier."""

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sarnarrative.cases import generate_case
from sarnarrative.evaluate import corrupt
from sarnarrative.generate import LLMConfig, LLMError, draft, llm_config_from_env
from sarnarrative.narrative import parse_narrative, split_sentences
from sarnarrative.retrieval import default_retriever
from sarnarrative.typologies import TYPOLOGY_IDS
from sarnarrative.verify import extract_claims, verify


@pytest.fixture(scope="module")
def retriever():
    return default_retriever()


@pytest.mark.parametrize("typology", TYPOLOGY_IDS)
def test_template_narrative_is_fully_cited_and_passes(retriever, typology):
    case = generate_case(40, typology)
    d = draft(case, retriever)
    assert all(sentence.citations for sentence in d.narrative.sentences)
    assert {s.section for s in d.narrative.sentences} == {"who", "what", "when", "where", "why", "how"}
    report = verify(d.narrative, case, d.references)
    assert report.passed, [f.to_dict() for f in report.findings]
    assert report.n_cited == report.n_sentences == len(d.narrative.sentences)
    # the text form parses back to the same sentences and citations
    parsed = parse_narrative(d.narrative.to_text())
    assert [(s.text, s.citations) for s in parsed.sentences] == [(s.text, s.citations) for s in d.narrative.sentences]


@pytest.mark.parametrize("kind, expected_kinds", [
    ("amount_changed", {"unsupported_amount", "value_not_in_cited_source"}),
    ("date_changed", {"unsupported_date", "value_not_in_cited_source"}),
    ("name_invented", {"unsupported_name"}),
    ("citation_removed", {"missing_citation"}),
    ("citation_fabricated", {"unknown_citation"}),
])
def test_verifier_catches_injected_errors(retriever, kind, expected_kinds):
    for seed in range(60, 66):
        case = generate_case(seed)
        d = draft(case, retriever)
        corrupted, index = corrupt(d.narrative, kind, case, random.Random(seed))
        report = verify(corrupted, case, d.references)
        assert not report.passed
        kinds_at_sentence = {f.kind for f in report.findings if f.sentence_index == index}
        assert kinds_at_sentence & expected_kinds, (kind, seed, [f.to_dict() for f in report.findings])


def test_verifier_on_handwritten_sentences(retriever):
    case = generate_case(7)
    d = draft(case, retriever)
    subject, credits, period = case.fact("subject"), case.fact("credits"), case.fact("period")
    name = case.subject["name"]
    text = (
        f"The subject is {name} [{subject.id}]. "                              # 0 fine
        f"The account received {credits.value} [{credits.id}]. "              # 1 fine
        f"The account received {credits.value} [{period.id}]. "               # 2 right value, wrong source
        f"A further $1,234,567.89 arrived from Zebulon Quartermain [{credits.id}]. "  # 3 invented amount and name
        "The customer could not explain the activity. "                       # 4 no citation
        f"The pattern continued for 4,321 days [{period.id}][F999]."          # 5 unknown citation, invented count
    )
    report = verify(text, case, d.references)
    by_sentence = {}
    for finding in report.findings:
        by_sentence.setdefault(finding.sentence_index, set()).add(finding.kind)
    assert not report.passed
    assert 0 not in by_sentence and 1 not in by_sentence
    assert by_sentence[2] == {"value_not_in_cited_source"}
    assert by_sentence[3] == {"unsupported_amount", "unsupported_name"}
    assert by_sentence[4] == {"missing_citation"}
    assert {"unknown_citation", "unsupported_count"} <= by_sentence[5]


def test_sentence_splitting_and_claim_extraction():
    text = ("Payment went to Acme Co. Ltd. on 2025-03-04 for $9,400.00 [F2]. It was repeated. [F3] "
            "A third transfer followed on March 9, 2025 [F4][STR-01].")
    sentences = parse_narrative(text).sentences
    assert len(split_sentences(text)) == 3
    assert [s.citations for s in sentences] == [["F2"], ["F3"], ["F4", "STR-01"]]

    claims = extract_claims("On 3 March 2025 account SYN-123456 received 14 deposits totalling $128,300.50, "
                            "which is 92% of credits, from Tessa Quillfeather in Newark")
    assert claims.amounts == {12830050}
    assert claims.dates == {"2025-03-03"}
    assert claims.counts == {14.0}
    assert claims.percentages == {92.0}
    assert claims.identifiers == {"SYN-123456"}
    assert claims.names == ["Tessa Quillfeather", "Newark"]


def test_llm_mode_is_off_without_a_key():
    assert llm_config_from_env({}) is None
    assert llm_config_from_env({"SAR_LLM_API_KEY": "  "}) is None
    with pytest.raises(LLMError):
        llm_config_from_env({"SAR_LLM_API_KEY": "k"})  # key without endpoint and model
    config = llm_config_from_env({"SAR_LLM_API_KEY": "k", "SAR_LLM_BASE_URL": "http://host/v1/", "SAR_LLM_MODEL": "m"})
    assert (config.base_url, config.model) == ("http://host/v1", "m")


def test_llm_mode_against_a_local_stub(retriever, monkeypatch):
    """The LLM path, exercised against a stub server that returns one good and one bad sentence."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    case = generate_case(7)
    subject = case.fact("subject")
    reply = (f"The subject of this report is {subject.value} [{subject.id}]. "
             f"The subject also wired $999,999.00 to Zebulon Quartermain [{subject.id}].")
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(path=self.path, auth=self.headers.get("Authorization"), body=body)
            payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = LLMConfig(api_key="test-key", base_url=f"http://127.0.0.1:{server.server_port}/v1", model="stub")
        d = draft(case, retriever, llm=config)
    finally:
        server.shutdown()
        server.server_close()

    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "stub"
    prompt = seen["body"]["messages"][1]["content"]
    assert f"[{subject.id}]" in prompt and case.typology_id not in seen["body"]["messages"][0]["content"]
    assert d.narrative.generator == "llm" and len(d.narrative.sentences) == 2

    report = verify(d.narrative, case, d.references)
    assert not report.passed
    assert {f.sentence_index for f in report.findings} == {1}
    assert {f.kind for f in report.findings} == {"unsupported_amount", "unsupported_name"}
