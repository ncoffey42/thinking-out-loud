"""
Thin LLM client that normalizes reasoning/thinking tokens across providers.
- Ollama: hits native /api/chat (returns `thinking` field)
- OpenAI-compatible (Kimi, OpenRouter, etc.): hits /v1/chat/completions (returns `reasoning_content`)
"""

import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    reasoning: Optional[str] = None


@dataclass
class ModelConfig:
    provider: str          # "ollama" or "moonshot"
    model: str             # "gpt-oss:20b", "kimi-k2.5"
    base_url: str          # "http://localhost:11434", "https://api.moonshot.cn/v1"
    api_key: str = ""      # only needed for cloud providers
    think: str = "high"    # reasoning effort: "low", "medium", "high" (ollama/gpt-oss)


def chat(config: ModelConfig, messages: list[dict], system_prompt: str = "") -> LLMResponse:
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    if config.provider == "ollama":
        return _ollama_chat(config, full_messages)
    else:
        return _openai_chat(config, full_messages)


def _ollama_chat(config: ModelConfig, messages: list[dict]) -> LLMResponse:
    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
    }
    if config.think:
        payload["think"] = config.think
    resp = requests.post(
        f"{config.base_url}/api/chat",
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {})
    return LLMResponse(
        content=msg.get("content", ""),
        reasoning=msg.get("thinking"),
    )


def _openai_chat(config: ModelConfig, messages: list[dict]) -> LLMResponse:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": config.model,
        "messages": messages,
        "max_tokens": 16000,
    }
    if config.think:
        # Many OpenAI-compatible endpoints (like o1/o3 or Kimi 2.5) accept reasoning_effort
        payload["reasoning_effort"] = config.think

    resp = requests.post(
        f"{config.base_url}/chat/completions",
        headers=headers,
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    return LLMResponse(
        content=msg.get("content", ""),
        reasoning=msg.get("reasoning_content"),
    )