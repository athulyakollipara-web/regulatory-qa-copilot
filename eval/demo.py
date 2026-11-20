"""Ask a few real questions and write the answers to the GitHub Actions run summary (used for the README demo)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regqa.runtime import build_runtime  # noqa: E402

QUESTIONS = [
    "How much can a consumer lose if their debit card is stolen and they tell the bank within two business days?",
    "When must funds from a cash deposit be made available?",
    "What is the capital gains tax rate on cryptocurrency?",
]

rt = build_runtime()
lines = ["## Demo answers", ""]
for q in QUESTIONS:
    r = rt.ask(q)
    c = r.get("citations") or {}
    lines += [f"**Q:** {q}", "", f"**A** ({r['mode']}): {r['answer']}", ""]
    if r["found"]:
        lines += [f"Sources: " + ", ".join(f"[{s['cite']}]({s['url']})" for s in r["sources"][:3]),
                  f"Citation check: {'PASS' if c.get('ok') else 'FAIL'}", ""]
    lines.append("---")
text = "\n".join(lines) + "\n"
print(text)
if os.getenv("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
        fh.write(text)
