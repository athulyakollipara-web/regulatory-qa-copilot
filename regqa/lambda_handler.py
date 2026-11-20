"""AWS Lambda entry point (API Gateway HTTP API, payload v2).

On cold start the corpus JSONL is copied from S3 (``CORPUS_S3_URI``) to /tmp and
indexed in memory: five regulations are a few MB, so this takes seconds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _ensure_corpus() -> None:
    uri = os.getenv("CORPUS_S3_URI")
    local = Path(os.getenv("REGQA_CORPUS", "/tmp/corpus.jsonl"))
    if uri and not local.exists():
        import boto3

        bucket, _, key = uri[5:].partition("/")
        boto3.client("s3").download_file(bucket, key, str(local))


def _resp(status: int, body) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body, default=str)}


def handler(event: dict, context) -> dict:
    _ensure_corpus()
    from regqa.runtime import default_runtime

    route = event.get("routeKey", "")
    if route == "GET /health":
        return _resp(200, {"status": "ok"})
    if route == "POST /ask":
        try:
            body = json.loads(event.get("body") or "{}")
        except json.JSONDecodeError:
            return _resp(400, {"error": "Body must be JSON"})
        q = (body.get("question") or "").strip()
        if not 3 <= len(q) <= 600:
            return _resp(400, {"error": "question must be 3-600 characters"})
        out = default_runtime().ask(q)
        out.pop("trace", None)
        return _resp(200, out)
    return _resp(404, {"error": f"unknown route {route}"})
