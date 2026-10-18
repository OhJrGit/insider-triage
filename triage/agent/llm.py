"""LLM backends.

  ollama   Ollama's own API (recommended for local runs). Sets the context window per
           request, so long sessions are never silently truncated.
             model e.g. qwen2.5:7b, llama3.1:8b, qwen2.5:14b
  openai   Any OpenAI-compatible endpoint: OpenAI, Groq, LM Studio, vLLM...
           base_url from the provider, key in TRIAGE_LLM_API_KEY

Env vars: TRIAGE_LLM_BACKEND, TRIAGE_LLM_MODEL, TRIAGE_LLM_BASE_URL, TRIAGE_LLM_API_KEY, TRIAGE_LLM_NUM_CTX
Every backend counts tokens in `.usage` so you can report cost/latency per report.
"""
from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request


def _post(url: str, body: dict, headers: dict, timeout: int, retries: int = 1) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (TimeoutError, socket.timeout):
        if retries > 0:
            print(f"    [warn] request timed out after {timeout}s, retrying once")
            return _post(url, body, headers, timeout, retries - 1)
        raise
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {url} ({e.reason}). Is the server running?") from e


class _Usage:
    def reset_usage(self):
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _add(self, prompt: int | None, completion: int | None):
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(prompt or 0)
        self.usage["completion_tokens"] += int(completion or 0)


class OllamaLLM(_Usage):
    def __init__(self, model: str | None = None, base_url: str | None = None, num_ctx: int | None = None,
                 temperature: float = 0.0, timeout: int = 300, max_output_tokens: int = 3000):
        self.model = model or os.environ.get("TRIAGE_LLM_MODEL", "qwen2.5:7b")
        self.base_url = (base_url or os.environ.get("TRIAGE_LLM_BASE_URL", "http://localhost:11434")).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.num_ctx = int(num_ctx or os.environ.get("TRIAGE_LLM_NUM_CTX", 32768))
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens   # stops JSON mode looping on whitespace forever
        self.reset_usage()

    def chat(self, messages: list[dict]) -> str:
        data = _post(f"{self.base_url}/api/chat", {
            "model": self.model, "messages": messages, "stream": False, "format": "json",
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature,
                        "num_predict": self.max_output_tokens},
        }, {}, self.timeout)
        self._add(data.get("prompt_eval_count"), data.get("eval_count"))
        if data.get("prompt_eval_count") and data["prompt_eval_count"] >= self.num_ctx - 64:
            print(f"    [warn] prompt filled the {self.num_ctx}-token context window; raise --num-ctx")
        return data["message"]["content"]

    def describe(self) -> str:
        return f"ollama:{self.model} (num_ctx={self.num_ctx})"


class OpenAICompatLLM(_Usage):
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, timeout: int = 600, json_mode: bool = True):
        self.model = model or os.environ.get("TRIAGE_LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("TRIAGE_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("TRIAGE_LLM_API_KEY", "not-needed")
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode
        self.reset_usage()

    def chat(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        data = _post(f"{self.base_url}/chat/completions", body,
                     {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
        u = data.get("usage") or {}
        self._add(u.get("prompt_tokens"), u.get("completion_tokens"))
        return data["choices"][0]["message"]["content"]

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url}"


def make_llm(backend: str | None = None, model: str | None = None, base_url: str | None = None,
             num_ctx: int | None = None):
    backend = backend or os.environ.get("TRIAGE_LLM_BACKEND", "ollama")
    if backend == "ollama":
        return OllamaLLM(model=model, base_url=base_url, num_ctx=num_ctx)
    if backend == "openai":
        return OpenAICompatLLM(model=model, base_url=base_url)
    raise ValueError(f"unknown backend {backend}")
