from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
    timeout: int = 120,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ForecastingBriefRAW",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": bool(stream),
    }
    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        data=json.dumps(payload),
        stream=stream,
        timeout=timeout,
    )
    if response.status_code >= 400:
        try:
            err = response.json()
        except Exception:
            err = {"error": response.text[:2000]}
        raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {err}")

    if not stream:
        return response.json()["choices"][0]["message"]["content"]
    return _stream_chunks(response.iter_lines(decode_unicode=True))


def _stream_chunks(lines: Iterable[str]):
    for line in lines:
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            break
        try:
            content = json.loads(payload)["choices"][0].get("delta", {}).get("content", "")
        except Exception:
            content = ""
        if content:
            yield content


def llm_raw(
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    stream_to: Optional[object] = None,
) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if stream_to is None:
        return openrouter_chat(api_key, model, msgs, temperature, max_tokens, stream=False)

    placeholder = stream_to.empty()
    acc = ""
    for part in openrouter_chat(api_key, model, msgs, temperature, max_tokens, stream=True):
        acc += part
        placeholder.text(acc)
    return acc
