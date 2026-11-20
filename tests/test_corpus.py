from regqa.corpus.chunks import build_chunks
from regqa.corpus.ecfr import label_paragraphs, load, save


def test_parser_reads_sections_and_skips_reserved(sections):
    assert [s.section for s in sections] == ["9999.1", "9999.2", "9999.3", "9999.5"]
    assert sections[1].heading == "Notice of unauthorized withdrawals."
    assert sections[1].url.endswith("/title-12/section-9999.2")


def test_paragraph_labels_follow_nesting(sections):
    labels = [p.label for p in sections[1].paragraphs]
    assert labels == ["9999.2(a)", "9999.2(b)(1)", "9999.2(b)(2)", "9999.2(b)(2)(i)", "9999.2(b)(2)(ii)", "9999.2(c)"]


def test_letter_i_after_h_is_not_a_roman_numeral(sections):
    assert [p.label for p in sections[2].paragraphs][-2:] == ["9999.3(h)", "9999.3(i)"]


def test_uppercase_level_four():
    labels = [p.label for p in label_paragraphs("1.1", ["(a) Rule.", "(1) One.", "(i) First.", "(A) Detail.", "(B) More.", "(ii) Second."])]
    assert labels == ["1.1(a)", "1.1(a)(1)", "1.1(a)(1)(i)", "1.1(a)(1)(i)(A)", "1.1(a)(1)(i)(B)", "1.1(a)(1)(ii)"]


def test_chunks_group_by_top_level_paragraph(sections):
    chunks = build_chunks(sections)
    b = next(c for c in chunks if c.chunk_id == "9999.2(b)")
    assert b.labels == {"9999.2(b)(1)", "9999.2(b)(2)", "9999.2(b)(2)(i)", "9999.2(b)(2)(ii)"}
    assert "Notice of unauthorized withdrawals" in b.search_text


def test_save_and_load_roundtrip(sections, tmp_path):
    save(sections, tmp_path / "c.jsonl")
    assert load(tmp_path / "c.jsonl") == sections


def test_ecfr_download_requests_and_decodes_gzip(monkeypatch):
    """eCFR answers HTTP 406 unless the request accepts a compressed response."""
    import gzip
    import urllib.request

    from regqa.corpus import ecfr

    payload = b"<ECFR/>"
    seen = {}

    class Resp:
        def __init__(self, body, enc):
            self.body, self.headers = body, ({"Content-Encoding": enc} if enc else {})

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(req, timeout):
        seen["accept_encoding"] = req.get_header("Accept-encoding")
        return Resp(gzip.compress(payload), "gzip")

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    assert ecfr._get("https://www.ecfr.gov/api/versioner/v1/titles.json") == payload
    assert seen["accept_encoding"] == "gzip"

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: Resp(payload, None))
    assert ecfr._get("https://www.ecfr.gov/x") == payload  # an uncompressed reply still works
