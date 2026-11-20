"""FastAPI service. Interactive docs at http://localhost:8000/docs."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from regqa.runtime import default_runtime

app = FastAPI(
    title="Regulatory Q&A Copilot",
    description="Answers questions about US banking regulations (12 CFR Parts 1002, 1005, 1016, 1030, 229). "
                "Every sentence cites the paragraph it comes from; unsupported questions get 'not found'.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=600,
                          examples=["How much can a consumer lose if their debit card is stolen and they report it within two business days?"])
    include_trace: bool = False


@app.get("/health")
def health() -> dict:
    rt = default_runtime()
    return {"status": "ok", "llm": rt.settings.llm, "reranker": rt.settings.reranker,
            "sections": len({c.section for c in rt.chunks}), "chunks": len(rt.chunks)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    out = default_runtime().ask(req.question)
    if not req.include_trace:
        out.pop("trace")
    return out


@app.get("/sections/{section}")
def section(section: str) -> dict:
    """Full text of a section, so a reader can check a citation in one click."""
    chunks = [c for c in default_runtime().chunks if c.section == section]
    if not chunks:
        return {"section": section, "found": False}
    c = chunks[0]
    return {"section": section, "found": True, "heading": c.heading, "regulation": c.regulation, "url": c.url,
            "paragraphs": [{"label": p.label, "text": p.text} for ch in chunks for p in ch.paragraphs]}
