import json
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(runtime):
    import regqa.runtime as rt_module
    from regqa.api.app import app

    rt_module.default_runtime.cache_clear()
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["sections"] == 4


def test_ask_returns_cited_answer_and_sources(client):
    body = client.post("/ask", json={"question": "When must cash deposits be available?"}).json()
    assert body["found"] and "[§ 9999.3(a)]" in body["answer"]
    assert body["sources"][0]["url"].startswith("https://www.ecfr.gov/")
    assert "trace" not in body


def test_section_lookup(client):
    body = client.get("/sections/9999.2").json()
    assert body["found"] and body["paragraphs"][1]["label"] == "9999.2(b)(1)"


def test_lambda_handler(runtime):
    from regqa import lambda_handler

    r = lambda_handler.handler({"routeKey": "POST /ask", "body": json.dumps({"question": "When must a bank give a privacy notice?"})}, None)
    assert r["statusCode"] == 200 and json.loads(r["body"])["found"]
    assert lambda_handler.handler({"routeKey": "POST /ask", "body": "{}"}, None)["statusCode"] == 400


def test_cdk_stack():
    pytest.importorskip("aws_cdk")
    if not shutil.which("node"):
        pytest.skip("needs Node.js")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))
    import aws_cdk as cdk
    from aws_cdk.assertions import Match, Template
    from stack import RegQaStack

    t = Template.from_stack(RegQaStack(cdk.App(), "T"))
    t.has_resource_properties("AWS::Bedrock::Guardrail", {"ContextualGroundingPolicyConfig": {"FiltersConfig": Match.array_with([
        Match.object_like({"Type": "GROUNDING", "Threshold": 0.75})])}})
    t.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey": "POST /ask", "AuthorizationType": "JWT"})
