"""
Thin LLM client that normalizes reasoning/thinking tokens across providers.

Local llama.cpp behavior:
- Manages exactly one local llama-server process at a time.
- If a different local model is requested, the old llama-server is stopped and a new one is started.
- Calls llama.cpp through its OpenAI-compatible /v1/chat/completions endpoint.
- Supports per-model server launch settings such as --reasoning and --reasoning-budget.

Cloud/OpenAI-compatible behavior:
- Sends requests directly to the configured OpenAI-compatible base_url.
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests


@dataclass
class LLMResponse:
    content: str
    reasoning: Optional[str] = None


@dataclass
class ModelConfig:
    provider: str                  # "llamacpp", "openai", or "openrouter"
    model: str                     # "gpt20b", "qwen2b", "kimi-k2.5", etc.
    base_url: str                  # "http://127.0.0.1:65419", "https://api.moonshot.ai/v1"
    api_key: str = ""              # only needed for cloud providers

    # Openrouter reasoning control
    openrouter_reasoning_tokens: int = 0
    include_reasoning: bool = True

    # Cloud/OpenAI-compatible reasoning control.
    think: str = ""                # reasoning_effort for openai-compatible providers only

    # Kept for compatibility with your existing scen1_negotiation.py.
    # For llama.cpp, this client mainly uses server-side --reasoning-budget.
    thinking_budget_tokens: int = 0

    # Optional explicit local llama.cpp settings.
    # If model_path is blank, this client infers it from env vars based on model name.
    model_path: str = ""
    # 0 lets llama.cpp use the context length from the model metadata.
    ctx_size: int = 0
    n_gpu_layers: int = 999
    max_tokens: int = 1024

    # llama.cpp reasoning server flags.
    # reasoning: "auto", "on", "off", or "" to omit.
    reasoning: str = "auto"

    # -1 means omit --reasoning-budget.
    #  0 means immediate reasoning end if supported by your llama.cpp build/model template.
    # >0 means token budget.
    reasoning_budget: int = -1

    # Extra llama-server CLI flags, e.g. ("--temp", "0.7")
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    # Kimi K2.x supports thinking mode, but dialogue agents need stable content.
    disable_thinking: bool = False


_ACTIVE_LLAMA_PROCESS: subprocess.Popen | None = None
_ACTIVE_LLAMA_MODEL_KEY: str | None = None
_ACTIVE_LLAMA_LOG_FILE = None


def _llama_verbose() -> bool:
    return os.getenv("LLAMA_CLIENT_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def chat(config: ModelConfig, messages: list[dict], system_prompt: str = "") -> LLMResponse:
    """
    Main entry point used by scen1_negotiation.py.

    For llama.cpp:
      - ensure the correct local llama-server is running
      - call /v1/chat/completions

    For cloud/OpenAI-compatible:
      - call the configured /chat/completions endpoint
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    if config.provider == "llamacpp":
        resolved_config = _ensure_llama_server(config)
        return _llamacpp_chat(resolved_config, full_messages)

    if config.provider == "openrouter":
        return _openrouter_chat(config, full_messages)

    return _openai_chat(config, full_messages)


def _llamacpp_chat(config: ModelConfig, messages: list[dict]) -> LLMResponse:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": _canonical_local_model_name(config.model),
        "messages": messages,
        "stream": False,
        "max_tokens": config.max_tokens,
    }

    resp = requests.post(
        f"{config.base_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"llama.cpp chat request failed with HTTP {resp.status_code}: {resp.text}"
        ) from e

    data = resp.json()
    msg = data["choices"][0]["message"]

    content = msg.get("content", "") or ""
    reasoning = (
        msg.get("reasoning_content")
        or msg.get("reasoning")
        or None
    )

    # Fallback for Qwen/DeepSeek-style outputs that include <think>...</think>
    # inside content instead of returning a separate reasoning_content field.
    if not reasoning and content and "<think>" in content:
        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip()
            content = (content[: m.start()] + content[m.end():]).strip()

    return LLMResponse(content=content.strip(), reasoning=reasoning)


def _openai_chat(config: ModelConfig, messages: list[dict]) -> LLMResponse:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": config.model,
        "messages": messages,
        "max_tokens": int(os.getenv("OPENAI_COMPAT_MAX_TOKENS", "32768")),
    }

    if _is_kimi_k2_model(config.model):
        if config.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
    elif config.think:
        payload["reasoning_effort"] = config.think

    resp = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"OpenAI-compatible chat request failed with HTTP {resp.status_code}: {resp.text}"
        ) from e

    data = resp.json()
    msg = data["choices"][0]["message"]

    content = msg.get("content", "") or ""
    reasoning = (
        msg.get("reasoning_content")
        or msg.get("reasoning")
        or None
    )

    if not content and reasoning and _is_kimi_k2_model(config.model) and not config.disable_thinking:
        raise RuntimeError(
            "Kimi returned reasoning_content but no final content. "
            "Disable thinking for dialogue output or increase OPENAI_COMPAT_MAX_TOKENS."
        )

    return LLMResponse(content=content.strip(), reasoning=reasoning)


def _openrouter_chat(config: ModelConfig, messages: list[dict]) -> LLMResponse:
    messages = _with_openrouter_reasoning_instruction(messages, config.openrouter_reasoning_tokens)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "UTK SWE Negotiation").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "max_tokens": config.max_tokens,
    }

    reasoning_payload = _openrouter_reasoning_payload(config)
    if reasoning_payload:
        payload["reasoning"] = reasoning_payload

    data = _post_openrouter_chat(config, headers, payload)
    choice = data["choices"][0]
    msg = choice["message"]

    content = msg.get("content", "") or ""
    reasoning = (
        msg.get("reasoning")
        or msg.get("reasoning_content")
        or None
    )

    if not content.strip() and reasoning and _openrouter_retry_empty_content_enabled():
        retry_payload = dict(payload)
        retry_payload["messages"] = _with_final_only_instruction(messages)
        retry_payload["reasoning"] = _openrouter_reasoning_payload(
            config,
            include_reasoning=False,
            fallback_budget=config.openrouter_reasoning_tokens,
        )
        retry_payload["max_tokens"] = int(
            os.getenv("OPENROUTER_EMPTY_CONTENT_RETRY_MAX_TOKENS", str(config.max_tokens))
        )
        retry_data = _post_openrouter_chat(config, headers, retry_payload)
        retry_msg = retry_data["choices"][0]["message"]
        content = retry_msg.get("content", "") or ""

    if (
        reasoning
        and config.openrouter_reasoning_tokens > 0
        and _reasoning_trim_enabled()
    ):
        reasoning = _trim_to_approx_tokens(reasoning, config.openrouter_reasoning_tokens)

    return LLMResponse(content=content.strip(), reasoning=reasoning)


def _openrouter_reasoning_payload(
    config: ModelConfig,
    include_reasoning: bool | None = None,
    fallback_budget: int = 0,
) -> dict:
    include = config.include_reasoning if include_reasoning is None else include_reasoning
    budget = config.openrouter_reasoning_tokens or fallback_budget

    if budget > 0:
        return {"max_tokens": budget, "exclude": not include}

    if not include:
        if _openrouter_requires_reasoning(config.model):
            return {
                "max_tokens": int(os.getenv("OPENROUTER_HIDDEN_REASONING_TOKENS", "100")),
                "exclude": True,
            }
        return {"effort": "none", "exclude": True}

    return {}


def _openrouter_requires_reasoning(model_name: str) -> bool:
    return "thinking" in model_name.lower()


def _post_openrouter_chat(config: ModelConfig, headers: dict, payload: dict) -> dict:
    resp = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"OpenRouter chat request failed with HTTP {resp.status_code}: {resp.text}"
        ) from e
    return resp.json()


def _openrouter_retry_empty_content_enabled() -> bool:
    return os.getenv("OPENROUTER_RETRY_EMPTY_CONTENT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _reasoning_trim_enabled() -> bool:
    return os.getenv("TRUNCATE_REASONING_TO_BUDGET", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _with_openrouter_reasoning_instruction(messages: list[dict], token_budget: int) -> list[dict]:
    if token_budget <= 0:
        return messages

    instruction = (
        f"Keep any internal reasoning concise and under roughly {token_budget} tokens. "
        "Return only the final answer in the message content."
    )

    adjusted = [dict(message) for message in messages]
    if adjusted and adjusted[0].get("role") == "system":
        adjusted[0]["content"] = f"{adjusted[0].get('content', '')}\n\n{instruction}"
        return adjusted

    return [{"role": "system", "content": instruction}, *adjusted]


def _with_final_only_instruction(messages: list[dict]) -> list[dict]:
    instruction = (
        "The previous attempt produced internal reasoning but no final answer. "
        "Do not include reasoning. Return only the final response content now."
    )

    adjusted = [dict(message) for message in messages]
    if adjusted and adjusted[0].get("role") == "system":
        adjusted[0]["content"] = f"{adjusted[0].get('content', '')}\n\n{instruction}"
        return adjusted

    return [{"role": "system", "content": instruction}, *adjusted]


def _trim_to_approx_tokens(text: str, token_budget: int) -> str:
    """Clamp returned reasoning when a provider ignores a requested reasoning budget.

    This is intentionally conservative and tokenizer-free. It treats words,
    punctuation runs, and whitespace-preserved chunks as approximate tokens,
    which is good enough for keeping logs/context from ballooning.
    """
    if token_budget <= 0:
        return ""

    chunks = re.findall(r"\S+\s*", text)
    if len(chunks) <= token_budget:
        return text

    trimmed = "".join(chunks[:token_budget]).rstrip()
    omitted = len(chunks) - token_budget
    return (
        f"{trimmed}\n\n"
        f"[Reasoning truncated locally to approximately {token_budget} tokens; "
        f"provider returned about {omitted} additional tokens despite the requested budget.]"
    )


def _ensure_llama_server(config: ModelConfig) -> ModelConfig:
    """
    Ensure exactly one llama-server is running for the requested local model.

    If the active server already matches the requested model/settings, reuse it.
    Returns the resolved config that should be used for the OpenAI-compatible
    request, including env-based host/port overrides.
    Otherwise, stop the old server and start a new one.
    """
    global _ACTIVE_LLAMA_PROCESS, _ACTIVE_LLAMA_MODEL_KEY, _ACTIVE_LLAMA_LOG_FILE

    resolved = _resolve_llamacpp_config(config)
    desired_key = _llama_model_key(resolved)

    if (
        _ACTIVE_LLAMA_PROCESS is not None
        and _ACTIVE_LLAMA_PROCESS.poll() is None
        and _ACTIVE_LLAMA_MODEL_KEY == desired_key
    ):
        return resolved

    _stop_active_llama_server()

    llama_server_bin = os.getenv("LLAMA_SERVER_BIN", "llama-server")
    host = os.getenv("LLAMA_CPP_HOST", "127.0.0.1")
    port = str(_base_url_port(resolved.base_url))

    cmd = [
        llama_server_bin,
        "--host", host,
        "--port", port,
        "-m", resolved.model_path,
        "-c", str(resolved.ctx_size),
        "-ngl", str(resolved.n_gpu_layers),
    ]

    if resolved.reasoning:
        cmd.extend(["--reasoning", resolved.reasoning])

    if resolved.reasoning_budget >= 0:
        cmd.extend(["--reasoning-budget", str(resolved.reasoning_budget)])

    if resolved.model == "qwen2b" and not _has_chat_template_arg(resolved.extra_args):
        template_file = _qwen_chat_template_file()
        cmd.extend(["--chat-template-file", str(template_file)])

    cmd.extend(list(resolved.extra_args))

    log_dir = Path(os.getenv("LLAMA_CLIENT_LOG_DIR", "llama_client_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = _canonical_local_model_name(resolved.model).replace("/", "_").replace(":", "_")
    log_path = log_dir / f"{safe_model_name}.log"

    _ACTIVE_LLAMA_LOG_FILE = open(log_path, "a", encoding="utf-8")

    if _llama_verbose():
        print(f"[llama.cpp] Starting local model: {_canonical_local_model_name(resolved.model)}")
        print(f"[llama.cpp] Model path: {resolved.model_path}")
        print(f"[llama.cpp] Log file: {log_path}")
        print(f"[llama.cpp] Command: {' '.join(cmd)}")

    _ACTIVE_LLAMA_PROCESS = subprocess.Popen(
        cmd,
        stdout=_ACTIVE_LLAMA_LOG_FILE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _ACTIVE_LLAMA_MODEL_KEY = desired_key

    try:
        _wait_for_llama_server(resolved.base_url)
    except Exception:
        _stop_active_llama_server()
        raise

    return resolved


def _stop_active_llama_server() -> None:
    """Stop the currently active local llama-server process, if this client started one."""
    global _ACTIVE_LLAMA_PROCESS, _ACTIVE_LLAMA_MODEL_KEY, _ACTIVE_LLAMA_LOG_FILE

    if _ACTIVE_LLAMA_PROCESS is not None:
        proc = _ACTIVE_LLAMA_PROCESS

        if proc.poll() is None:
            if _llama_verbose():
                print("[llama.cpp] Stopping active local model server...")

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print("[llama.cpp] Server did not exit after SIGTERM; killing...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)

    _ACTIVE_LLAMA_PROCESS = None
    _ACTIVE_LLAMA_MODEL_KEY = None

    if _ACTIVE_LLAMA_LOG_FILE is not None:
        try:
            _ACTIVE_LLAMA_LOG_FILE.close()
        except Exception:
            pass
        _ACTIVE_LLAMA_LOG_FILE = None


atexit.register(_stop_active_llama_server)


def _wait_for_llama_server(base_url: str, timeout_seconds: int = 300) -> None:
    """
    Wait until llama-server has finished loading the model.

    Modern llama.cpp exposes /health before and after model load:
      - 503 means the HTTP server is up but the model is still loading.
      - 200 {"status": "ok"} means chat/completion requests can be served.
    """
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=2)
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError:
                    body = {}
                if body.get("status") in ("ok", None):
                    if _llama_verbose():
                        print(f"[llama.cpp] Server ready at {base_url}")
                    return

            if resp.status_code != 503:
                last_error = RuntimeError(f"/health returned {resp.status_code}: {resp.text}")

        except Exception as e:
            last_error = e

        # Fallback for older or customized llama.cpp servers that lack /health.
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=2)
            if resp.status_code == 200:
                if _llama_verbose():
                    print(f"[llama.cpp] Server ready at {base_url}")
                return
        except Exception:
            pass

        time.sleep(0.75)

    raise RuntimeError(
        f"llama-server did not become ready at {base_url} within {timeout_seconds}s. "
        f"Last error: {last_error}"
    )


def _resolve_llamacpp_config(config: ModelConfig) -> ModelConfig:
    """
    Fill in llama.cpp model_path/reasoning settings from env vars if scen1_negotiation.py
    did not provide them.

    This keeps your existing call style working:
      python -u scen1_negotiation.py gpt20b kimi qwen2b
    """
    model_name = _canonical_local_model_name(config.model)

    model_path = config.model_path or _infer_model_path_from_env(model_name)
    if not model_path:
        raise ValueError(
            f"No GGUF path found for local model '{config.model}'.\n"
            f"Set one of these in your .env:\n"
            f"  GPT20B_GGUF=/path/to/gpt-oss-20b.gguf\n"
            f"  QWEN2B_GGUF=/path/to/qwen.gguf\n"
            f"  LLAMA8B_GGUF=/path/to/llama8b.gguf\n"
            f"or pass model_path in ModelConfig."
        )

    if not Path(model_path).expanduser().exists():
        raise FileNotFoundError(
            f"GGUF file for model '{config.model}' does not exist:\n{model_path}"
        )

    # Use the port from config.base_url unless overridden globally.
    base_url = _normalize_local_base_url(config.base_url)

    reasoning_budget = config.reasoning_budget

    # Backward compatibility:
    # Your scen1_negotiation.py currently sets thinking_budget_tokens for gpt20b/qwen2b.
    # If reasoning_budget was not explicitly configured, reuse that value as the server budget.
    if reasoning_budget < 0 and config.thinking_budget_tokens > 0:
        reasoning_budget = config.thinking_budget_tokens

    # Sensible defaults for your three local names.
    ctx_size = config.ctx_size
    if model_name == "qwen2b":
        ctx_size = int(os.getenv("QWEN2B_CTX_SIZE", str(ctx_size)))
    elif model_name == "gpt20b":
        ctx_size = int(os.getenv("GPT20B_CTX_SIZE", str(ctx_size)))
    elif model_name == "llama8b":
        ctx_size = int(os.getenv("LLAMA8B_CTX_SIZE", str(ctx_size)))

    n_gpu_layers = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", str(config.n_gpu_layers)))
    max_tokens = int(os.getenv("LLAMA_CPP_MAX_TOKENS", str(config.max_tokens)))
    extra_args = tuple(config.extra_args)

    if model_name == "qwen2b":
        template_override = os.getenv("QWEN2B_CHAT_TEMPLATE_FILE", "").strip()
        if template_override:
            extra_args = ("--chat-template-file", template_override, *extra_args)

    return ModelConfig(
        provider=config.provider,
        model=model_name,
        base_url=base_url,
        api_key=config.api_key,
        think=config.think,
        thinking_budget_tokens=config.thinking_budget_tokens,
        model_path=str(Path(model_path).expanduser()),
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        max_tokens=max_tokens,
        reasoning=config.reasoning,
        reasoning_budget=reasoning_budget,
        extra_args=extra_args,
        disable_thinking=config.disable_thinking,
    )


def _infer_model_path_from_env(model_name: str) -> str:
    env_map = {
        "gpt20b": "GPT20B_GGUF",
        "gpt-oss:20b": "GPT20B_GGUF",
        "gpt-oss-20b": "GPT20B_GGUF",

        "qwen2b": "QWEN2B_GGUF",
        "qwen3.5:2b": "QWEN2B_GGUF",
        "qwen3.5-2b": "QWEN2B_GGUF",

        "llama8b": "LLAMA8B_GGUF",
        "llama3.1:8b-instruct-q8_0": "LLAMA8B_GGUF",
        "llama-8b": "LLAMA8B_GGUF",
    }

    env_name = env_map.get(model_name)
    if not env_name:
        return ""

    return os.getenv(env_name, "")


def _is_kimi_k2_model(model_name: str) -> bool:
    return model_name.startswith("kimi-k2")


def _canonical_local_model_name(model_name: str) -> str:
    name = model_name.strip()

    aliases = {
        "gpt-oss:20b": "gpt20b",
        "gpt-oss-20b": "gpt20b",

        "qwen3.5:2b": "qwen2b",
        "qwen3.5-2b": "qwen2b",

        "llama3.1:8b-instruct-q8_0": "llama8b",
        "llama-8b": "llama8b",
    }

    return aliases.get(name, name)


def _normalize_local_base_url(base_url: str) -> str:
    """
    Keep using the base_url passed by scen1_negotiation.py, but allow env override.

    For your current setup, all local models should use LLAMA_CPP_PORT,
    currently 65419 in .env.
    """
    override_port = os.getenv("LLAMA_CPP_PORT", "").strip()
    override_host = os.getenv("LLAMA_CPP_HOST", "").strip()

    if not override_port and not override_host:
        return base_url.rstrip("/")

    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = override_host or parsed.hostname or "127.0.0.1"
    port = override_port or str(parsed.port or 65419)

    return f"{scheme}://{host}:{port}"


def _base_url_port(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port is None:
        raise ValueError(f"base_url must include a port, got: {base_url}")
    return int(parsed.port)


def _llama_model_key(config: ModelConfig) -> str:
    """
    Key that determines whether the currently running server matches the requested config.
    If any launch-relevant setting changes, the server restarts.
    """
    return "|".join(
        [
            _canonical_local_model_name(config.model),
            str(Path(config.model_path).expanduser()),
            str(config.ctx_size),
            str(config.n_gpu_layers),
            str(config.max_tokens),
            config.reasoning,
            str(config.reasoning_budget),
            " ".join(config.extra_args),
            config.base_url,
        ]
    )


def _has_chat_template_arg(extra_args: tuple[str, ...]) -> bool:
    return any(arg in {"--chat-template", "--chat-template-file"} for arg in extra_args)


def _qwen_chat_template_file() -> Path:
    return Path(__file__).with_name("chat_templates") / "qwen_chatml.jinja"
