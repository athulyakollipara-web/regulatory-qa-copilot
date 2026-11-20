"""The question-answering workflow as a LangGraph state machine.

    screen -> rewrite -> retrieve -> support gate --(weak)--> not found
                                          |
                                        answer -> check citations --(fail, 1 retry)--> answer
                                                         |                       \\--(still failing)--> quote the rule (extractive)
                                                   grounding check (Guardrails) -> done

Two independent checks guard every answer: a deterministic citation check
(every sentence cited, every citation retrieved, every figure present in the
cited paragraph) and, in AWS mode, a model-based contextual grounding check.
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from regqa.answer import extractive
from regqa.answer.citations import check
from regqa.llm import prompts
from regqa.retrieval.hybrid import HybridRetriever

NOT_FOUND = "NOT_FOUND"
NOT_FOUND_MESSAGE = ("I couldn't find this in the regulations I search (12 CFR Parts 1002, 1005, 1016, 1030 and 229), "
                     "so I won't guess. Try rephrasing, or check with your compliance team.")


class QAState(TypedDict, total=False):
    question: str
    blocked: bool
    query: str
    hits: list
    support: float
    found: bool
    answer: str
    answer_mode: str
    attempts: int
    feedback: list[str]
    citations: dict
    grounding: dict
    trace: list[dict]


def _t(state, step, started, **info: Any) -> list[dict]:
    return state.get("trace", []) + [{"step": step, "ms": round((time.perf_counter() - started) * 1000, 1), **info}]


def build_graph(retriever: HybridRetriever, screen: Callable, llm=None, guardrail=None,
                top_k: int = 5, support_threshold: float = 0.42, max_attempts: int = 2):

    def screen_node(s: QAState) -> QAState:
        t = time.perf_counter()
        r = screen(s["question"])
        return {"question": r.text, "blocked": not r.allowed, "attempts": 0, "feedback": [],
                "trace": _t(s, "screen", t, allowed=r.allowed, reasons=r.reasons)}

    def rewrite(s: QAState) -> QAState:
        t = time.perf_counter()
        query = s["question"]
        if llm is not None:
            try:
                extra = llm.complete("rewrite", prompts.REWRITE_SYSTEM, s["question"], max_tokens=60).text.strip()
                query = f"{s['question']} {extra}"
            except Exception:  # retrieval still works on the raw question
                pass
        return {"query": query, "trace": _t(s, "rewrite", t, query=query)}

    def retrieve(s: QAState) -> QAState:
        t = time.perf_counter()
        hits = retriever.search(s["query"], k=top_k)
        support = hits[0].score if hits else 0.0
        return {"hits": hits, "support": support, "found": support >= support_threshold,
                "trace": _t(s, "retrieve", t, reranker=retriever.reranker.name, support=support,
                            hits=[(h.chunk.chunk_id, h.score, h.sources) for h in hits])}

    def generate(s: QAState) -> QAState:
        t = time.perf_counter()
        attempts = s["attempts"] + 1
        if llm is None:
            text, mode = extractive.answer(s["question"], s["hits"]), "extractive"
        else:
            resp = llm.complete("answer", prompts.ANSWER_SYSTEM, prompts.answer_prompt(s["question"], s["hits"], s.get("feedback")), max_tokens=500)
            text, mode = resp.text.strip(), resp.model_id
        return {"answer": text, "answer_mode": mode, "attempts": attempts, "trace": _t(s, "answer", t, attempt=attempts, mode=mode)}

    def verify(s: QAState) -> QAState:
        t = time.perf_counter()
        if s["answer"].strip() == NOT_FOUND:
            return {"found": False, "feedback": [], "trace": _t(s, "verify", t, model_said_not_found=True)}
        report = check(s["answer"], [h.chunk for h in s["hits"]])
        return {"citations": {**report.__dict__, "ok": report.ok}, "feedback": report.problems,
                "trace": _t(s, "verify", t, ok=report.ok, problems=report.problems)}

    def fallback(s: QAState) -> QAState:
        t = time.perf_counter()
        text = extractive.answer(s["question"], s["hits"])
        report = check(text, [h.chunk for h in s["hits"]])
        return {"answer": text, "answer_mode": f"extractive (fallback after {s['answer_mode']} failed checks)",
                "citations": {**report.__dict__, "ok": report.ok}, "trace": _t(s, "fallback", t)}

    def ground(s: QAState) -> QAState:
        t = time.perf_counter()
        if guardrail is None or s["answer_mode"].startswith("extractive"):
            return {"trace": _t(s, "grounding", t, skipped=True)}
        result = guardrail.grounding(s["question"], prompts.sources_block(s["hits"]), s["answer"])
        update: QAState = {"grounding": result, "trace": _t(s, "grounding", t, **result)}
        if not result["passed"]:
            text = extractive.answer(s["question"], s["hits"])
            update.update(answer=text, answer_mode="extractive (model answer failed the grounding check)",
                          citations={**check(text, [h.chunk for h in s["hits"]]).__dict__})
        return update

    def finish(s: QAState) -> QAState:
        if s.get("blocked"):
            return {"answer": "This request was blocked by the safety screen.", "answer_mode": "blocked"}
        if not s.get("found"):
            return {"answer": NOT_FOUND_MESSAGE, "answer_mode": "not_found"}
        return {}

    g = StateGraph(QAState)
    for name, fn in [("screen", screen_node), ("rewrite", rewrite), ("retrieve", retrieve), ("generate", generate),
                     ("verify", verify), ("fallback", fallback), ("ground", ground), ("finish", finish)]:
        g.add_node(name, fn)
    g.set_entry_point("screen")
    g.add_conditional_edges("screen", lambda s: "finish" if s.get("blocked") else "rewrite", {"finish": "finish", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", lambda s: "generate" if s["found"] else "finish", {"generate": "generate", "finish": "finish"})
    g.add_edge("generate", "verify")

    def after_verify(s: QAState) -> str:
        if not s.get("found"):
            return "finish"
        if not s["feedback"]:
            return "ground"
        return "generate" if s["attempts"] < max_attempts and llm is not None else "fallback"

    g.add_conditional_edges("verify", after_verify, {"finish": "finish", "ground": "ground", "generate": "generate", "fallback": "fallback"})
    g.add_edge("fallback", "finish")
    g.add_edge("ground", "finish")
    g.add_edge("finish", END)
    return g.compile()
