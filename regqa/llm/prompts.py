ANSWER_SYSTEM = """You answer questions about US banking regulations for a bank's compliance team.

Rules:
- Use ONLY the numbered source paragraphs provided. Never rely on memory of the regulations.
- End EVERY sentence with at least one citation in exactly this form: [§ 1005.6(b)(1)].
  Cite the most specific paragraph label shown in the sources.
- Copy figures (dollar amounts, days, percentages) exactly as they appear in the cited paragraph.
- If the sources do not answer the question, reply with exactly: NOT_FOUND
- Plain English, 2-5 sentences. No legal advice, no speculation, no text outside the answer."""

REWRITE_SYSTEM = """Rewrite the user's question as a short search query in the vocabulary of US banking regulations
(for example "lost debit card" -> "unauthorized electronic fund transfer access device liability").
Return only the query, no explanation."""


def sources_block(hits) -> str:
    lines = []
    for h in hits:
        c = h.chunk
        lines.append(f"## {c.regulation}, § {c.section} {c.heading}")
        for p in c.paragraphs:
            lines.append(f"[§ {p.label}] {p.text}")
        lines.append("")
    return "\n".join(lines)


def answer_prompt(question: str, hits, feedback: list[str] | None = None) -> str:
    out = f"<SOURCES>\n{sources_block(hits)}</SOURCES>\n\n<QUESTION>\n{question}\n</QUESTION>"
    if feedback:
        out += "\n\nYour previous answer failed these checks. Fix them:\n- " + "\n- ".join(feedback)
    return out
