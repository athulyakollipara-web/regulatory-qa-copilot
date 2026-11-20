"""Amazon Bedrock client with task-aware routing, fallback and prompt caching.

* Routing: each step can use a different model list. Answer writing, where
  citation discipline matters, goes to the strongest model; query rewriting uses
  a cheaper one.
* Fallback: if a model is throttled or unavailable, the next model in the list is
  tried, and the model that actually answered is recorded in the trace.
* Prompt caching: the long, static system prompt (citation rules) is marked
  with a cache point so repeated questions don't pay for it every time. Turn it
  off with ``BEDROCK_PROMPT_CACHE=0`` for models that don't support it.
"""
from __future__ import annotations

import os

from regqa.llm.base import LLMResponse

RETRYABLE = {
    "ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
    "ModelTimeoutException", "InternalServerException", "ModelErrorException",
}


class BedrockLLM:
    def __init__(self, primary: str, fallback: str, region: str, routes: dict[str, list[str]] | None = None):
        import boto3
        from botocore.config import Config

        self.client = boto3.client("bedrock-runtime", region_name=region,
                                   config=Config(retries={"max_attempts": 2, "mode": "standard"}))
        self.routes = routes or {
            "rewrite": [fallback, primary],
            "answer": [primary, fallback],
        }
        self.default = [primary, fallback]
        self.cache = os.getenv("BEDROCK_PROMPT_CACHE", "1") == "1"

    def complete(self, task: str, system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        from botocore.exceptions import ClientError

        system_blocks: list[dict] = [{"text": system}]
        if self.cache:
            system_blocks.append({"cachePoint": {"type": "default"}})
        last_error: Exception | None = None
        for model_id in self.routes.get(task, self.default):
            try:
                resp = self.client.converse(
                    modelId=model_id,
                    system=system_blocks,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in RETRYABLE or (code == "ValidationException" and "cach" in str(e).lower()):
                    last_error = e
                    continue
                raise
            text = "".join(block.get("text", "") for block in resp["output"]["message"]["content"])
            usage = resp.get("usage", {})
            return LLMResponse(text, model_id, usage.get("inputTokens", 0), usage.get("outputTokens", 0))
        raise RuntimeError(f"All models failed for task '{task}': {last_error}")
