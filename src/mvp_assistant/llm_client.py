import json
import os
import urllib.error
import urllib.request
from typing import Any


def _debug(message: str) -> None:
    if os.getenv("LLM_DEBUG") == "1":
        print(f"[llm] {message}")


class LlmClient:
    def complete_json(self, prompt: str) -> dict[str, Any] | None:
        raise NotImplementedError


class NullLlmClient(LlmClient):
    def complete_json(self, prompt: str) -> dict[str, Any] | None:
        del prompt
        return None


class BudgetedLlmClient(LlmClient):
    def __init__(self, wrapped: LlmClient, max_calls: int) -> None:
        self.wrapped = wrapped
        self.max_calls = max_calls
        self.calls = 0
        self.disabled = False

    def complete_json(self, prompt: str) -> dict[str, Any] | None:
        if self.disabled or self.calls >= self.max_calls:
            return None
        self.calls += 1
        result = self.wrapped.complete_json(prompt)
        if result is None and getattr(self.wrapped, "rate_limited", False):
            self.disabled = True
        return result


class OpenAiCompatibleClient(LlmClient):
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "12000"))

    def complete_json(self, prompt: str) -> dict[str, Any] | None:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content)
        except urllib.error.HTTPError as exc:
            _debug(f"openai-compatible http error: {exc.code}")
            return None
        except urllib.error.URLError as exc:
            _debug(f"openai-compatible url error: {exc.reason}")
            return None
        except (KeyError, json.JSONDecodeError) as exc:
            _debug(f"openai-compatible parse error: {type(exc).__name__}")
            return None


class GeminiClient(LlmClient):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.rate_limited = False

    def complete_json(self, prompt: str) -> dict[str, Any] | None:
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")[:300]
            _debug(f"gemini http error: {exc.code} {body}")
            if exc.code == 429:
                self.rate_limited = True
            return None
        except urllib.error.URLError as exc:
            _debug(f"gemini url error: {exc.reason}")
            return None
        except (KeyError, json.JSONDecodeError) as exc:
            _debug(f"gemini parse error: {type(exc).__name__}")
            return None


def build_llm_client() -> LlmClient:
    budget = int(os.getenv("LLM_CALL_BUDGET", "0"))
    if budget <= 0:
        return NullLlmClient()

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return BudgetedLlmClient(
            GeminiClient(api_key=gemini_key, model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
            max_calls=budget,
        )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return NullLlmClient()

    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if os.getenv("OPENROUTER_API_KEY") and not base_url:
        base_url = "https://openrouter.ai/api/v1"
        model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    if os.getenv("GROQ_API_KEY") and not base_url:
        base_url = "https://api.groq.com/openai/v1"
        model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")

    return BudgetedLlmClient(
        OpenAiCompatibleClient(api_key=api_key, model=model, base_url=base_url or "https://api.openai.com/v1"),
        max_calls=budget,
    )
