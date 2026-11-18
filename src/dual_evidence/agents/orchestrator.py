"""Optional LLM coordinator; all numerical tools remain Python-owned."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Callable


class LLMProvider:
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        raise NotImplementedError


@dataclass
class OpenAICompatibleProvider(LLMProvider):
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60

    @classmethod
    def from_environment(cls):
        key = os.getenv("LLM_API_KEY")
        if not key:
            raise ValueError("LLM_API_KEY is required to enable remote LLM coordination")
        return cls(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), key,
                   os.getenv("LLM_MODEL", "gpt-4o-mini"))

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode())


class ResearchCoordinator:
    """Expose approved Python tools; LLM output can never override numeric policy."""
    def __init__(self, tools: dict[str, Callable], provider: LLMProvider | None = None):
        self.tools = tools
        self.provider = provider

    def run_tool(self, name: str, *args, **arguments):
        if name not in self.tools:
            raise KeyError(f"Unapproved research tool: {name}")
        return self.tools[name](*args, **arguments)

    def summarize(self, results: dict) -> str | None:
        if self.provider is None:
            return None
        response = self.provider.complete([
            {"role": "system", "content": (
                "Summarize the supplied experimental results faithfully. Do not recalculate metrics, "
                "choose thresholds, or change controller decisions. State missing model runs." )},
            {"role": "user", "content": json.dumps(results, allow_nan=False)},
        ])
        return response["choices"][0]["message"].get("content", "")


def configured_coordinator(tools: dict[str, Callable]):
    """Use an LLM only when configured; offline workflow remains fully supported."""
    provider = OpenAICompatibleProvider.from_environment() if os.getenv("LLM_API_KEY") else None
    return ResearchCoordinator(tools, provider)
