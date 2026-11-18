"""Optional LLM coordinator; all numerical tools remain Python-owned."""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)
_TOOL_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class LLMProvider:
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        raise NotImplementedError


@dataclass
class OpenAICompatibleProvider(LLMProvider):
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 180
    disable_thinking: bool = True

    @classmethod
    def from_environment(cls):
        base = os.getenv("LLM_BASE_URL")
        key = os.getenv("LLM_API_KEY")
        if not base and not key:
            raise ValueError("LLM_BASE_URL or LLM_API_KEY is required to enable LLM coordination")
        return cls(base or "https://api.openai.com/v1",
                   key or "local",
                   os.getenv("LLM_MODEL", "gpt-4o-mini"))

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": 0,
                   "parallel_tool_calls": False}
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_effort"] = "none"
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


def extract_tool_calls(message: dict) -> list[dict]:
    """Prefer native OpenAI tool_calls; fall back to Qwen-style <tool_call> JSON."""
    requested = list(message.get("tool_calls") or [])
    if requested:
        return requested
    content = message.get("content") or ""
    parsed = []
    for index, match in enumerate(_TOOL_JSON.finditer(content), start=1):
        body = json.loads(match.group(1))
        parsed.append({
            "id": f"parsed_{index}",
            "type": "function",
            "function": {
                "name": body.get("name"),
                "arguments": json.dumps(body.get("arguments") or {}),
            },
        })
    return parsed


class ResearchCoordinator:
    """Expose approved Python tools; LLM output can never override numeric policy."""
    def __init__(self, tools: dict[str, Callable], provider: LLMProvider | None = None):
        self.tools = tools
        self.provider = provider

    def run_tool(self, name: str, *args, **arguments):
        if name not in self.tools:
            raise KeyError(f"Unapproved research tool: {name}")
        return self.tools[name](*args, **arguments)

    def orchestrate(self, messages: list[dict], tool_schemas: list[dict],
                    max_tool_calls: int = 12) -> dict:
        """Let the LLM choose approved workflow tools, then execute them in Python.

        Tool names are checked against the Python allowlist, arguments must be JSON
        objects, and the call count is bounded. Numerical policy remains inside the
        registered deterministic functions.
        """
        if self.provider is None:
            raise RuntimeError("LLM orchestration requires a configured provider")
        transcript = list(messages)
        trace = []
        call_count = 0
        model_turns = 0
        while True:
            response = self.provider.complete(transcript, tools=tool_schemas)
            model_turns += 1
            raw_message = response["choices"][0]["message"]
            requested = extract_tool_calls(raw_message)
            LOGGER.info("coordinator turn %s: %s", model_turns,
                        [call.get("function", {}).get("name") for call in requested] or "no-tool")
            if not requested:
                return {"content": raw_message.get("content", ""), "tool_calls": trace,
                        "messages": transcript, "model_turns": model_turns}
            if len(requested) != 1:
                raise RuntimeError("Coordinator must request one workflow tool at a time")
            transcript.append(raw_message)
            call = requested[0]
            call_count += 1
            if call_count > max_tool_calls:
                raise RuntimeError(f"LLM exceeded the {max_tool_calls}-call limit")
            function = call.get("function", {})
            name = function.get("name")
            if name not in self.tools:
                raise KeyError(f"Unapproved research tool requested by LLM: {name}")
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                raw_arguments = json.loads(raw_arguments or "{}")
            if not isinstance(raw_arguments, dict):
                raise TypeError(f"Arguments for {name} must be a JSON object")
            try:
                result = self.run_tool(name, **raw_arguments)
                outcome = {"status": "ok", "result": result}
            except Exception as exc:
                LOGGER.exception("Approved tool %s failed", name)
                outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            trace.append({"invocation": call_count, "tool": name,
                          "arguments": raw_arguments, "outcome": outcome})
            transcript.append({"role": "tool",
                "tool_call_id": call.get("id") or f"call_{call_count}",
                "name": name,
                "content": json.dumps(outcome, ensure_ascii=False, default=str)})
            if outcome["status"] == "error":
                raise RuntimeError(f"Tool {name} failed: {outcome['error']}")

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
    if not os.getenv("LLM_API_KEY") and not os.getenv("LLM_BASE_URL"):
        return ResearchCoordinator(tools, None)
    return ResearchCoordinator(tools, OpenAICompatibleProvider.from_environment())
