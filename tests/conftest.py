import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "part_9999.xml"


@pytest.fixture(scope="session")
def sections():
    from regqa.corpus.ecfr import parse_part_xml

    return parse_part_xml(FIXTURE.read_bytes(), "9999", "Example Regulation", as_of="2026-01-01")


@pytest.fixture(scope="session")
def corpus_path(sections, tmp_path_factory) -> Path:
    from regqa.corpus.ecfr import save

    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    save(sections, path)
    os.environ["REGQA_CORPUS"] = str(path)
    for k in ("REGQA_LLM", "REGQA_RERANKER", "REGQA_EMBEDDINGS", "BEDROCK_GUARDRAIL_ID", "REGQA_SUPPORT_THRESHOLD"):
        os.environ.pop(k, None)
    return path


@pytest.fixture(scope="session")
def runtime(corpus_path):
    from regqa.runtime import build_runtime

    return build_runtime()
