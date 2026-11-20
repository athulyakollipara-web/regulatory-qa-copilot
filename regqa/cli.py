"""Command line.

    python -m regqa.corpus.ecfr --out data/corpus.jsonl     # download the regulations (needs internet)
    python -m regqa.cli ask "Can my bank charge overdraft fees on debit card purchases?" --trace
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap


def main(argv: list[str] | None = None) -> int:
    from regqa.runtime import build_runtime

    p = argparse.ArgumentParser(prog="regqa")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--trace", action="store_true")
    args = p.parse_args(argv)

    r = build_runtime().ask(args.question)
    print("\nANSWER  (" + str(r["mode"]) + ")")
    print(textwrap.fill(r["answer"], 100, initial_indent="  ", subsequent_indent="  "))
    if r["found"]:
        print("\nSOURCES")
        for s in r["sources"][:3]:
            print(f"  {s['cite']:<18} {s['regulation']}: {s['heading']}  (score {s['score']:.2f})")
            print(f"  {'':<18} {s['url']}")
        c = r["citations"] or {}
        print(f"\nCITATION CHECK  {'PASS' if c.get('ok') else 'FAIL'}: {c.get('cited_sentences')}/{c.get('sentences')} sentences cited, "
              f"{len(c.get('unknown_citations', []))} unknown citations, {len(c.get('ungrounded_figures', []))} ungrounded figures")
    if args.trace:
        print("\nTRACE\n" + json.dumps(r["trace"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
