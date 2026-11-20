from regqa.answer.citations import check
from regqa.corpus.chunks import build_chunks


def _chunks(sections):
    return build_chunks(sections)


def test_valid_answer_passes(sections):
    a = "If notice is given within 2 business days, liability is capped at $50 [§ 9999.2(b)(1)]. Later notice raises the cap to $500 [§ 9999.2(b)(2)]."
    r = check(a, _chunks(sections))
    assert r.ok and r.cited_sentences == 2 and r.cited_sections == ["9999.2"]


def test_wrong_figure_is_caught(sections):
    r = check("Liability is capped at $500 if notice is given within 2 business days [§ 9999.2(b)(1)].", _chunks(sections))
    assert not r.ok and "$500" in r.ungrounded_figures


def test_uncited_sentence_is_caught(sections):
    r = check("Liability is capped at $50 [§ 9999.2(b)(1)]. Banks usually waive it anyway.", _chunks(sections))
    assert not r.ok and len(r.uncited_sentences) == 1


def test_citation_outside_sources_is_caught(sections):
    chunks = [c for c in _chunks(sections) if c.section == "9999.3"]
    r = check("Liability is capped at $50 [§ 9999.2(b)(1)].", chunks)
    assert not r.ok and r.unknown_citations == ["§ 9999.2(b)(1)"]


def test_nonexistent_paragraph_is_caught(sections):
    r = check("The bank must investigate within 10 business days [§ 9999.2(q)].", _chunks(sections))
    assert r.unknown_citations == ["§ 9999.2(q)"]


def test_extractive_answers_always_pass_the_checker(runtime):
    for q in ["How much is a consumer liable if they report an unauthorized withdrawal within 2 business days?",
              "When must cash deposits be available?", "When must a bank give a privacy notice?"]:
        r = runtime.ask(q)
        assert r["found"] and r["citations"]["ok"], (q, r["answer"])
