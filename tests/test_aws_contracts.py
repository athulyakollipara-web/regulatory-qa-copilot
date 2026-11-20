"""AWS request shapes checked against botocore's official API models, without calling AWS.

Stubber rejects any parameter the real service would reject, so a typo in a rerank
or guardrail request fails here instead of in a paid run.
"""
import pytest
from botocore.stub import Stubber

from regqa.corpus.chunks import build_chunks
from regqa.retrieval.hybrid import BedrockReranker, Hit
from regqa.safety import BLOCKED_SENTINEL, BedrockGuardrail

ARN = "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
USAGE = {k: 0 for k in ("topicPolicyUnits", "contentPolicyUnits", "wordPolicyUnits", "sensitiveInformationPolicyUnits",
                        "sensitiveInformationPolicyFreeUnits", "contextualGroundingPolicyUnits")}


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


def test_bedrock_rerank_request_and_reordering(sections):
    chunks = build_chunks(sections)[:3]
    hits = [Hit(c, 0.5, ("bm25",)) for c in chunks]
    rr = BedrockReranker(ARN, "us-west-2")
    expected = {
        "queries": [{"type": "TEXT", "textQuery": {"text": "q"}}],
        "sources": [{"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": c.search_text}}} for c in chunks],
        "rerankingConfiguration": {"type": "BEDROCK_RERANKING_MODEL", "bedrockRerankingConfiguration": {
            "modelConfiguration": {"modelArn": ARN}, "numberOfResults": 2}},
    }
    with Stubber(rr.client) as stub:
        stub.add_response("rerank", {"results": [{"index": 2, "relevanceScore": 0.91}, {"index": 0, "relevanceScore": 0.40}]}, expected)
        out = rr.rerank("q", hits, top_k=2)
    assert [h.chunk.chunk_id for h in out] == [chunks[2].chunk_id, chunks[0].chunk_id] and out[0].score == 0.91


def test_guardrail_input_block_and_pii_mask():
    g = BedrockGuardrail("gr-123", "DRAFT", "us-east-1")
    with Stubber(g.client) as stub:
        params = {"guardrailIdentifier": "gr-123", "guardrailVersion": "DRAFT", "source": "INPUT", "content": [{"text": {"text": "x"}}]}
        stub.add_response("apply_guardrail", {"usage": USAGE, "action": "GUARDRAIL_INTERVENED", "outputs": [{"text": BLOCKED_SENTINEL}], "assessments": []}, params)
        stub.add_response("apply_guardrail", {"usage": USAGE, "action": "GUARDRAIL_INTERVENED", "outputs": [{"text": "my SSN is {SSN}"}], "assessments": []}, params)
        assert not g.screen_input("x").allowed
        masked = g.screen_input("x")
    assert masked.allowed and masked.text == "my SSN is {SSN}"


def test_guardrail_contextual_grounding_request():
    g = BedrockGuardrail("gr-123", "DRAFT", "us-east-1")
    params = {"guardrailIdentifier": "gr-123", "guardrailVersion": "DRAFT", "source": "OUTPUT", "content": [
        {"text": {"text": "SOURCES", "qualifiers": ["grounding_source"]}},
        {"text": {"text": "Q", "qualifiers": ["query"]}},
        {"text": {"text": "A", "qualifiers": ["guard_content"]}}]}
    resp = {"usage": USAGE, "action": "GUARDRAIL_INTERVENED", "outputs": [], "assessments": [{"contextualGroundingPolicy": {"filters": [
        {"type": "GROUNDING", "threshold": 0.75, "score": 0.42, "action": "BLOCKED"},
        {"type": "RELEVANCE", "threshold": 0.5, "score": 0.88, "action": "NONE"}]}}]}
    with Stubber(g.client) as stub:
        stub.add_response("apply_guardrail", resp, params)
        out = g.grounding("Q", "SOURCES", "A")
    assert out == {"passed": False, "grounding": 0.42, "relevance": 0.88}
