"""
LLM provider router — round-robin between Anthropic, DeepSeek, OpenRouter, and Ollama.

DeepSeek, OpenRouter, and Ollama are OpenAI-compatible, so we use the openai SDK for all three.
Anthropic uses its own SDK.

Usage:
    from llm.router import call_llm

    text = await call_llm(
        system="You are a helpful assistant.",
        user="Write a cover letter for...",
        max_tokens=800,
    )
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Optional

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_CODES = {429, 500, 503, 529}
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt

LLM_TIMEOUT = 120  # seconds; hung API calls are cancelled and re-raise TimeoutError
# CPU llama.cpp inference is slow (long prompts can take 2+ min); give it a wider
# timeout than cloud providers so legitimate calls aren't killed.
LLAMA_CPP_TIMEOUT = 300

from config.settings import settings

# ── Build the provider cycle ───────────────────────────────────────────────────

def _build_cycle() -> itertools.cycle:
    """
    Build a round-robin cycle from whichever providers are configured.
    If LLM_PROVIDER is pinned to a specific provider, always use that one.
    """
    if settings.llm_provider == "anthropic":
        return itertools.cycle(["anthropic"])
    if settings.llm_provider == "deepseek":
        return itertools.cycle(["deepseek"])
    if settings.llm_provider == "openrouter":
        return itertools.cycle(["openrouter"])
    if settings.llm_provider == "ollama":
        return itertools.cycle(["ollama"])
    if settings.llm_provider == "llama_cpp":
        return itertools.cycle(["llama_cpp"])

    # "auto" — include only providers that have a key set
    available: list[str] = []
    if settings.anthropic_api_key:
        available.append("anthropic")
    if settings.deepseek_api_key:
        available.append("deepseek")
    if settings.openrouter_api_key:
        available.append("openrouter")
    if settings.ollama_base_url:
        available.append("ollama")
    if settings.llama_cpp_base_url:
        available.append("llama_cpp")

    if not available:
        raise RuntimeError(
            "No LLM provider configured. Set ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, "
            "OPENROUTER_API_KEY, OLLAMA_BASE_URL, or LLAMA_CPP_BASE_URL."
        )

    return itertools.cycle(available)


_provider_cycle = _build_cycle()


def _next_provider() -> str:
    return next(_provider_cycle)


def llama_cpp_active() -> bool:
    """True when llama.cpp will serve LLM calls — either the explicit provider or
    part of the `auto` round-robin. Callers that must serialize local inference
    (a single-slot CPU server) use this to decide whether to cap concurrency."""
    if settings.llm_provider == "llama_cpp":
        return True
    return settings.llm_provider == "auto" and bool(settings.llama_cpp_base_url)


# ── Anthropic call ─────────────────────────────────────────────────────────────

async def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    messages = [{"role": "user", "content": user}]
    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    # Models with adaptive/extended thinking on by default (e.g. Claude Sonnet
    # 5) prepend a ThinkingBlock to response.content, which has no .text
    # attribute — filter to the actual text block(s) instead of assuming
    # content[0] is text.
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "".join(text_blocks).strip()


# ── OpenAI-compatible calls (DeepSeek, OpenRouter, Ollama, llama.cpp) ─────────

async def _openai_chat(
    *, api_key: str, base_url: str, model: str, max_tokens: int,
    system: str, user: str,
    headers: dict | None = None,
    think: bool | None = None,
) -> str:
    """Shared OpenAI-compatible chat call. `think` toggles `reasoning_effort` for
    local thinking models; pass None (the default) to omit the param entirely."""
    from openai import AsyncOpenAI

    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if headers:
        client_kwargs["default_headers"] = headers
    client = AsyncOpenAI(**client_kwargs)

    create_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if think is not None:
        create_kwargs["reasoning_effort"] = "high" if think else "none"

    response = await client.chat.completions.create(**create_kwargs)
    return (response.choices[0].message.content or "").strip()


async def _call_deepseek(system: str, user: str, max_tokens: int) -> str:
    return await _openai_chat(
        api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
        model=settings.deepseek_model, max_tokens=max_tokens,
        system=system, user=user,
    )


async def _call_openrouter(system: str, user: str, max_tokens: int) -> str:
    return await _openai_chat(
        api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url,
        model=settings.openrouter_model, max_tokens=max_tokens,
        system=system, user=user,
        headers={
            "HTTP-Referer": "https://github.com/Donvink/swiss-job-hunter",
            "X-Title": "Swiss Job Hunter",
        },
    )


async def _call_ollama(system: str, user: str, max_tokens: int) -> str:
    return await _openai_chat(
        api_key="ollama", base_url=settings.ollama_base_url,
        model=settings.ollama_model, max_tokens=max_tokens,
        system=system, user=user, think=settings.ollama_think,
    )


async def _call_llama_cpp(system: str, user: str, max_tokens: int) -> str:
    # llama.cpp ignores OpenAI's `reasoning_effort`; thinking models (e.g. Qwen3)
    # toggle chain-of-thought via the /no_think token instead. Thinking is on by
    # default, so only append the token when the user has disabled it.
    if not settings.llama_cpp_think:
        user = f"{user}\n/no_think"
    return await _openai_chat(
        api_key="llama_cpp", base_url=settings.llama_cpp_base_url,
        model=settings.llama_cpp_model, max_tokens=max_tokens,
        system=system, user=user,
    )


# ── Public interface ───────────────────────────────────────────────────────────

def _provider_endpoint(p: str) -> str:
    """Best-effort endpoint URL for a provider, for error messages."""
    if p == "anthropic":
        return "https://api.anthropic.com"
    if p == "deepseek":
        return settings.deepseek_base_url
    if p == "openrouter":
        return settings.openrouter_base_url
    if p == "ollama":
        return settings.ollama_base_url or "(OLLAMA_BASE_URL not set)"
    if p == "llama_cpp":
        return settings.llama_cpp_base_url or "(LLAMA_CPP_BASE_URL not set)"
    return "?"


async def call_llm(
    user: str,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 1000,
    provider: Optional[str] = None,  # override round-robin for this call
) -> tuple[str, str]:
    """
    Call the next LLM provider in rotation.

    Returns:
        (response_text, provider_used)  — so callers can log which provider ran.
    """
    p = provider or _next_provider()

    if p == "anthropic":
        coro = _call_anthropic(system, user, max_tokens)
    elif p == "deepseek":
        coro = _call_deepseek(system, user, max_tokens)
    elif p == "openrouter":
        coro = _call_openrouter(system, user, max_tokens)
    elif p == "ollama":
        coro = _call_ollama(system, user, max_tokens)
    elif p == "llama_cpp":
        coro = _call_llama_cpp(system, user, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {p}")

    last_exc: Exception | None = None
    timeout = LLAMA_CPP_TIMEOUT if p == "llama_cpp" else LLM_TIMEOUT
    for attempt in range(_MAX_RETRIES):
        try:
            text = await asyncio.wait_for(coro, timeout=timeout)
            return text, p
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status in _RETRYABLE_HTTP_CODES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                log.warning("LLM %s %s (attempt %d/%d), retrying in %.0fs",
                            p, status, attempt + 1, _MAX_RETRIES, delay)
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    # rebuild coroutine for next attempt
                    if p == "anthropic":
                        coro = _call_anthropic(system, user, max_tokens)
                    elif p == "deepseek":
                        coro = _call_deepseek(system, user, max_tokens)
                    elif p == "openrouter":
                        coro = _call_openrouter(system, user, max_tokens)
                    elif p == "ollama":
                        coro = _call_ollama(system, user, max_tokens)
                    elif p == "llama_cpp":
                        coro = _call_llama_cpp(system, user, max_tokens)
                continue
            raise RuntimeError(f"[{p} @ {_provider_endpoint(p)}] {exc}") from exc
    raise RuntimeError(f"[{p} @ {_provider_endpoint(p)}] {last_exc}") from last_exc
