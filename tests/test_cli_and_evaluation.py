"""Smoke tests for the CLI, the evaluation harness and the web export."""

import json
import os
import subprocess
import sys
from pathlib import Path

from sarnarrative.evaluate import CORRUPTIONS, evaluate
from sarnarrative.export import MAX_TRANSACTION_ROWS, export_web
from sarnarrative.typologies import TYPOLOGY_IDS

ROOT = Path(__file__).resolve().parent.parent


def _run(*args):
    return subprocess.run([sys.executable, "-m", "sarnarrative", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=120, env=_env_without_llm())


def _env_without_llm():
    return {k: v for k, v in os.environ.items() if not k.startswith("SAR_LLM_")}


def test_cli_draft_and_generate_case():
    result = _run("draft", "--seed", "7")
    assert result.returncode == 0, result.stderr
    assert "Narrative (template generator" in result.stdout
    assert "Verifier: PASS" in result.stdout
    assert "[F1]" in result.stdout

    result = _run("generate-case", "--seed", "7", "--typology", "SHL")
    assert result.returncode == 0, result.stderr
    case = json.loads(result.stdout)
    assert case["typology_id"] == "SHL" and case["transactions"] and case["facts"][0]["id"] == "F1"

    result = _run("draft", "--seed", "7", "--generator", "llm", "--json")
    assert result.returncode == 0, result.stderr
    assert "SAR_LLM_API_KEY is not set" in result.stderr
    assert json.loads(result.stdout)["generator"] == "template"


def test_evaluation_runs_on_a_small_sample():
    metrics = evaluate(n=16, seed=77)
    overall = metrics["overall"]
    assert overall["n_cases"] == 16
    for key in ("retrieval_hit_at_1", "retrieval_hit_at_3", "citation_coverage", "fact_inclusion_rate",
                "verifier_pass_rate"):
        assert 0.0 <= overall[key] <= 1.0
    assert overall["retrieval_hit_at_3"] >= overall["retrieval_hit_at_1"]
    assert overall["citation_coverage"] == 1.0 and overall["verifier_pass_rate"] == 1.0
    assert set(metrics["per_typology"]) == set(TYPOLOGY_IDS)
    assert sum(row["n_cases"] for row in metrics["per_typology"].values()) == 16
    assert set(metrics["corruption"]) == set(CORRUPTIONS)
    assert metrics["corruption"]["citation_removed"]["catch_rate"] == 1.0
    misses = sum(m["count"] for m in metrics["retrieval_misses"])
    assert misses == round(16 * (1 - overall["retrieval_hit_at_1"]))


def test_export_web_writes_the_three_files(tmp_path):
    paths = export_web(tmp_path, n=16, seed=77)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cases.json", "eval.json", "typologies.json"]

    typologies = json.loads(paths["typologies.json"].read_text())
    assert [t["id"] for t in typologies] == TYPOLOGY_IDS
    assert set(typologies[0]) == {"id", "title", "description", "source", "indicators", "narrative_should_establish"}

    cases = json.loads(paths["cases.json"].read_text())
    assert len(cases) == 12 and {c["typology_id"] for c in cases} == set(TYPOLOGY_IDS)
    for case in cases:
        assert set(case) == {"id", "title", "typology_id", "subject", "period", "facts", "transactions", "query",
                             "retrieved", "ranking", "written_for", "references", "narrative", "verification"}
        assert case["title"].startswith("Synthetic case")
        assert len(case["transactions"]) <= MAX_TRANSACTION_ROWS and len(case["retrieved"]) == 5
        assert set(case["transactions"][0]) == {"date", "type", "amount", "counterparty", "channel", "location"}
        assert all(set(s) == {"text", "citations"} and s["citations"] for s in case["narrative"])
        assert isinstance(case["verification"]["passed"], bool)
        assert case["ranking"][0]["typology_id"] == case["written_for"]
        assert case["references"][0] == f"{case['written_for']}-00"
        cited = {c for sentence in case["narrative"] for c in sentence["citations"] if "-" in c}
        assert cited <= set(case["references"])

    metrics = json.loads(paths["eval.json"].read_text())
    assert metrics["n_cases"] == 16 and metrics["seed"] == 77 and "generated_at" in metrics
    assert "overall" in metrics and "per_typology" in metrics and "corruption" in metrics
