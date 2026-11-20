"""Evaluate the copilot on 50 questions over the real eCFR corpus, logged to MLflow.

40 answerable questions (gold = the section that answers them) across
Regulations B, E, P, DD and CC, plus 10 questions the corpus cannot answer.

Measures:
* retrieval recall@1 / recall@5 - is the gold section in the top results?
* answer correctness - does the final answer cite the gold section?
* citation validity - does every answer pass the citation check?
* refusal - are the 10 out-of-scope questions answered "not found"?
* a threshold sweep for the "not found" gate, so it is tuned on data, not guessed

    python eval/run_eval.py                               # offline: hashing embeddings, lexical reranker, extractive answers
    REGQA_LLM=bedrock REGQA_RERANKER=bedrock REGQA_EMBEDDINGS=titan python eval/run_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(HERE.parent))

from regqa.config import get_settings  # noqa: E402
from regqa.runtime import build_runtime  # noqa: E402


def main() -> int:
    s = get_settings()
    rt = build_runtime(s)
    items = [json.loads(l) for l in (HERE / "questions.jsonl").read_text().splitlines() if l.strip()]

    rows = []
    for it in items:
        t = time.perf_counter()
        candidates = rt.retriever.search(it["question"], k=5)
        r = rt.ask(it["question"])
        cited = (r.get("citations") or {}).get("cited_sections", [])
        rows.append({
            "id": it["id"], "reg": it["reg"], "answerable": bool(it["gold"]), "gold": it["gold"],
            "top_sections": [h.chunk.section for h in candidates],
            "support": round(candidates[0].score, 4) if candidates else 0.0,
            "found": r["found"], "mode": r["mode"],
            "citations_ok": (r.get("citations") or {}).get("ok"),
            "cites_gold": bool(set(cited) & set(it["gold"])),
            "latency_s": round(time.perf_counter() - t, 3),
        })

    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    found = [r for r in ans if r["found"]]

    def rate(xs, key):
        return round(sum(1 for x in xs if key(x)) / max(1, len(xs)), 4)

    summary = {
        "config": {"llm": s.llm, "reranker": s.reranker, "embeddings": s.embeddings, "support_threshold": s.support_threshold,
                   "chunks": len(rt.chunks), "sections": len({c.section for c in rt.chunks})},
        "retrieval_recall_at_1": rate(ans, lambda r: r["top_sections"][:1] and r["top_sections"][0] in r["gold"]),
        "retrieval_recall_at_5": rate(ans, lambda r: set(r["top_sections"]) & set(r["gold"])),
        "answered_rate_on_answerable": rate(ans, lambda r: r["found"]),
        "answer_cites_gold_section": rate(ans, lambda r: r["cites_gold"]),
        "citation_check_pass_rate": rate(found, lambda r: r["citations_ok"]),
        "correct_refusals_on_out_of_scope": rate(neg, lambda r: not r["found"]),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3),
    }

    sweep = []
    for i in range(10, 80, 5):
        th = i / 100
        answered = sum(1 for r in ans if r["support"] >= th) / len(ans)
        refused = sum(1 for r in neg if r["support"] < th) / len(neg)
        sweep.append({"threshold": th, "answerable_answered": round(answered, 3), "out_of_scope_refused": round(refused, 3),
                      "balanced": round((answered + refused) / 2, 3)})
    best = max(sweep, key=lambda x: (x["balanced"], x["answerable_answered"]))
    summary["suggested_threshold"] = best

    by_reg = {}
    for reg in sorted({r["reg"] for r in ans}):
        sub = [r for r in ans if r["reg"] == reg]
        by_reg[reg] = {"n": len(sub), "recall@5": rate(sub, lambda r: set(r["top_sections"]) & set(r["gold"])),
                       "cites_gold": rate(sub, lambda r: r["cites_gold"])}
    summary["by_regulation"] = by_reg

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    tag = f"{s.llm}-{s.reranker}-{s.embeddings}"
    (out / f"{tag}.json").write_text(json.dumps({"summary": summary, "sweep": sweep, "rows": rows}, indent=2))

    md = [f"## Evaluation: {tag}", "",
          f"Corpus: {summary['config']['sections']} sections, {summary['config']['chunks']} chunks. 40 answerable + 10 out-of-scope questions.", "",
          "| Metric | Value |", "|---|---|"]
    for k in ("retrieval_recall_at_1", "retrieval_recall_at_5", "answered_rate_on_answerable", "answer_cites_gold_section",
              "citation_check_pass_rate", "correct_refusals_on_out_of_scope", "mean_latency_s"):
        v = summary[k]
        md.append(f"| {k} | {v:.1%} |" if k != "mean_latency_s" else f"| {k} | {v} |")
    md += ["", f"Support threshold used: {s.support_threshold}. Best on this set: {best['threshold']} "
           f"(answers {best['answerable_answered']:.0%} of answerable, refuses {best['out_of_scope_refused']:.0%} of out-of-scope).", "",
           "| Regulation | Questions | Recall@5 | Answer cites gold |", "|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['recall@5']:.0%} | {v['cites_gold']:.0%} |" for k, v in by_reg.items()]
    report = "\n".join(md) + "\n"
    (out / f"{tag}.md").write_text(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(report)

    try:
        import mlflow

        mlflow.set_experiment("regulatory-qa-copilot")
        with mlflow.start_run(run_name=tag):
            mlflow.log_params(summary["config"])
            mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(out / f"{tag}.json"))
    except ImportError:
        pass
    print(report)
    print(json.dumps(sweep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
