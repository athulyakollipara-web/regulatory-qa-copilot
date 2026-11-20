from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def complete(self, task: str, system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """``task`` names the agent step (rewrite / answer) for routing and tracing."""
        ...
