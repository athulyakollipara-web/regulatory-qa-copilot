"""Safety checks around the model.

Input: Bedrock Guardrails (prompt-attack filter + PII masking) in AWS mode, a
rule-based screen locally.

Output: Bedrock Guardrails *contextual grounding* check. The guardrail scores
the answer against the retrieved paragraphs (grounding) and the question
(relevance); below the thresholds set in the CDK stack, the answer is replaced.
This is a second, model-based check on top of the deterministic citation check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

BLOCKED_SENTINEL = "BLOCKED_BY_GUARDRAIL"
INJECTION = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|rules)",
    r"disregard (the )?(system|previous) prompt",
    r"reveal (your|the) (system )?prompt",
    r"you are now",
]
PII = {"SSN": r"\b\d{3}-\d{2}-\d{4}\b", "ACCOUNT_NUMBER": r"\b\d{10,17}\b", "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"}


@dataclass
class Screen:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)


def local_input_screen(text: str) -> Screen:
    reasons = [f"prompt_attack:{p}" for p in INJECTION if re.search(p, text, re.I)]
    for label, pattern in PII.items():
        if re.search(pattern, text):
            reasons.append(f"pii_masked:{label}")
            text = re.sub(pattern, f"[{label}]", text)
    return Screen(not any(r.startswith("prompt_attack") for r in reasons), text, reasons)


class BedrockGuardrail:
    def __init__(self, guardrail_id: str, version: str, region: str):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.id, self.version = guardrail_id, version

    def screen_input(self, text: str) -> Screen:
        r = self.client.apply_guardrail(guardrailIdentifier=self.id, guardrailVersion=self.version, source="INPUT",
                                        content=[{"text": {"text": text}}])
        if r["action"] != "GUARDRAIL_INTERVENED":
            return Screen(True, text)
        out = (r.get("outputs") or [{"text": ""}])[0].get("text", "")
        if out.strip() == BLOCKED_SENTINEL:
            return Screen(False, text, ["guardrail:blocked"])
        return Screen(True, out, ["guardrail:pii_masked"])

    def grounding(self, question: str, sources_text: str, answer: str) -> dict:
        """Contextual grounding check: returns the GROUNDING and RELEVANCE scores and whether it passed."""
        r = self.client.apply_guardrail(
            guardrailIdentifier=self.id, guardrailVersion=self.version, source="OUTPUT",
            content=[
                {"text": {"text": sources_text[:90000], "qualifiers": ["grounding_source"]}},
                {"text": {"text": question, "qualifiers": ["query"]}},
                {"text": {"text": answer, "qualifiers": ["guard_content"]}},
            ],
        )
        scores = {}
        for a in r.get("assessments", []):
            for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
                scores[f["type"].lower()] = round(float(f["score"]), 3)
        return {"passed": r["action"] != "GUARDRAIL_INTERVENED", **scores}
