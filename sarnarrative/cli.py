"""Command line interface: `python -m sarnarrative <command>`.

Commands:
    generate-case   print one synthetic case as JSON
    draft           case -> retrieval -> narrative with citations -> verifier report
    evaluate        run the evaluation harness
    export-web      write the JSON files used by the web demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cases import Case, generate_case
from .evaluate import DEFAULT_N, DEFAULT_SEED, evaluate, format_report, retrieval_ablation
from .export import export_web
from .generate import LLMError, draft, llm_config_from_env
from .narrative import SECTIONS
from .retrieval import default_retriever
from .typologies import TYPOLOGY_IDS, load_typologies
from .verify import verify


def _load_case(args: argparse.Namespace) -> Case:
    if getattr(args, "case", None):
        return Case.from_dict(json.loads(Path(args.case).read_text(encoding="utf-8")))
    return generate_case(args.seed, args.typology)


def cmd_generate_case(args: argparse.Namespace) -> int:
    case = generate_case(args.seed, args.typology)
    text = case.to_json()
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    case = _load_case(args)
    retriever = default_retriever()
    titles = {t.id: t.title for t in load_typologies()}

    llm = None
    if args.generator in ("auto", "llm"):
        try:
            llm = llm_config_from_env()
        except LLMError as exc:
            print(f"LLM mode not used: {exc}", file=sys.stderr)
        if llm is None and args.generator == "llm":
            print("LLM mode skipped: SAR_LLM_API_KEY is not set. Using the template generator.", file=sys.stderr)
    try:
        d = draft(case, retriever, typology_id=args.write_up, llm=llm)
    except LLMError as exc:
        print(f"LLM call failed ({exc}). Using the template generator.", file=sys.stderr)
        d = draft(case, retriever, typology_id=args.write_up)
    report = verify(d.narrative, case, d.references)

    if args.json:
        print(json.dumps({
            "case_id": case.id, "label": case.typology_id, "written_for": d.typology_id,
            "ranking": [{"typology_id": tid, "score": round(score, 4)} for tid, score in d.ranking],
            "retrieved": [hit.to_dict() for hit in d.hits],
            "generator": d.narrative.generator,
            "narrative": [s.to_dict() for s in d.narrative.sentences],
            "verification": report.to_dict(),
        }, indent=2))
        return 0

    print(f"Case {case.id} (synthetic data). Ground-truth label, not shown to the pipeline: {case.typology_id}")
    print()
    print("Facts")
    for fact in case.facts:
        print(f"  {fact.id:<4} {fact.label}: {fact.value}")
    print()
    print("Typology ranking from retrieval")
    for rank, (tid, score) in enumerate(d.ranking[:3], start=1):
        print(f"  {rank}. {tid}  {titles[tid]}  ({score:.3f})")
    print()
    print("Closest reference chunks")
    for hit in d.hits:
        text = hit.chunk.text if len(hit.chunk.text) <= 110 else hit.chunk.text[:107] + "..."
        print(f"  {hit.chunk.id}  {hit.score:.3f}  {text}")
    print()
    print(f"Narrative ({d.narrative.generator} generator, written for {d.typology_id})")
    if any(s.section for s in d.narrative.sentences):
        for section in SECTIONS:
            sentences = [s for s in d.narrative.sentences if s.section == section]
            if sentences:
                print(f"  {section.upper()}")
                for s in sentences:
                    print(f"    {s.render()}")
    else:
        for s in d.narrative.sentences:
            print(f"  {s.render()}")
    print()
    status = "PASS" if report.passed else "FAIL"
    print(f"Verifier: {status} ({report.n_sentences} sentences, {report.n_cited} with a valid citation, "
          f"{len(report.findings)} findings)")
    for finding in report.findings:
        print(f"  sentence {finding.sentence_index + 1}: {finding.kind}: {finding.detail}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    metrics = evaluate(n=args.n, seed=args.seed)
    ablation = retrieval_ablation(args.n, args.seed, with_neural=args.with_neural) if args.ablation else None
    if ablation:
        metrics["retrieval_ablation"] = ablation
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_report(metrics, ablation))
    return 0


def cmd_export_web(args: argparse.Namespace) -> int:
    paths = export_web(args.out, n=args.n, seed=args.seed, with_neural=args.with_neural)
    for name, path in paths.items():
        print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sarnarrative", description="Draft and verify SAR narratives on synthetic cases.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-case", help="print one synthetic case as JSON")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--typology", choices=TYPOLOGY_IDS, help="pattern to generate (default: drawn from the seed)")
    p.add_argument("--out", help="write the JSON to this file instead of stdout")
    p.set_defaults(func=cmd_generate_case)

    p = sub.add_parser("draft", help="draft a narrative for a case and verify it")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--typology", choices=TYPOLOGY_IDS, help="pattern to generate (default: drawn from the seed)")
    p.add_argument("--case", help="read the case from a JSON file written by generate-case")
    p.add_argument("--write-up", choices=TYPOLOGY_IDS, dest="write_up",
                   help="write the narrative for this typology instead of the top retrieval result")
    p.add_argument("--generator", choices=["auto", "template", "llm"], default="auto",
                   help="auto uses the LLM only if SAR_LLM_API_KEY is set")
    p.add_argument("--json", action="store_true", help="print machine-readable output")
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("evaluate", help="run the evaluation harness")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--ablation", action="store_true", help="also compare retrieval configurations")
    p.add_argument("--with-neural", action="store_true",
                   help="include the sentence-transformers backend in the ablation (optional package)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", help="also write the metrics JSON to this file")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("export-web", help="write typologies.json, cases.json and eval.json")
    p.add_argument("--out", default="web_export")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--with-neural", action="store_true")
    p.set_defaults(func=cmd_export_web)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
