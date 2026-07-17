"""Provider boundary and standalone OpenAI-compatible client for SGIRA."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Protocol


class StructuredLLM(Protocol):
    def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        payload: Mapping[str, Any],
    ) -> str | Mapping[str, Any]: ...


class ChatClientAdapter:
    """Adapt an object exposing ``chat(system_prompt, user_prompt)``."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        payload: Mapping[str, Any],
    ) -> str:
        del role
        return self.client.chat(
            system_prompt,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            temperature=0,
        )


def _truthy_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_object(text: str) -> dict[str, Any]:
    """Extract one balanced JSON object, tolerating a Markdown fence."""
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    if start < 0:
        raise ValueError("model response does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(value[start:index + 1])
                    if not isinstance(parsed, dict):
                        raise ValueError("model JSON response must be an object")
                    return parsed
    raise ValueError("model response contains incomplete JSON")


class OpenAICompatibleLLM:
    """Standalone client for an OpenAI-compatible company model gateway.

    Configuration is read from ``MA_LLM_*`` environment variables. The
    client deliberately records no API key or prompt in its audit log.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 4.0,
        max_tokens: int = 1024,
        cache_dir: str | Path | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the project dependencies with: pip install -e .") from exc

        self.model = (
            model
            or os.getenv("MA_LLM_MODEL")
            or os.getenv("MA_MODEL_GEMINI3_FLASH")
        )
        if not self.model:
            raise ValueError("Set MA_LLM_MODEL or pass --model")
        self.base_url = base_url or os.getenv("MA_LLM_BASE_URL")
        self.api_key = (
            api_key
            or os.getenv("MA_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Set MA_LLM_API_KEY in .env")

        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.max_tokens = int(os.getenv("MA_LLM_MAX_COMPLETION_TOKENS", max_tokens))
        self.json_mode = _truthy_env("MA_LLM_JSON_MODE", True)
        self.reasoning_effort = os.getenv("MA_LLM_REASONING_EFFORT") or None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.audit_path = Path(audit_path) if audit_path else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

        http_client = None
        if not _truthy_env("MA_LLM_TRUST_ENV", True):
            import httpx
            http_client = httpx.Client(timeout=timeout, trust_env=False)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _audit(self, *, role: str, response: Any, valid: bool) -> None:
        if not self.audit_path:
            return
        usage = getattr(response, "usage", None)
        record = {
            "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "role": role,
            "base_url": self.base_url or "https://api.openai.com/v1",
            "requested_model": self.model,
            "response_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "finish_reason": (
                response.choices[0].finish_reason
                if getattr(response, "choices", None) else None
            ),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "output_valid": valid,
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        user_prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        cache_key = "|".join((
            "sgira-v1", self.base_url or "", self.model, role,
            str(self.max_tokens), str(self.json_mode), system_prompt, user_prompt,
        ))
        cache_path = self._cache_path(cache_key)
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                raise ValueError("cached model response is not a JSON object")
            return cached

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "n": 1,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
            request["max_completion_tokens"] = self.max_tokens
        else:
            request["max_tokens"] = self.max_tokens
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                content = response.choices[0].message.content or ""
                parsed = _json_object(content)
                valid = response.choices[0].finish_reason != "length"
                self._audit(role=role, response=response, valid=valid)
                if not valid:
                    raise ValueError("model response reached the token limit")
                if cache_path:
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(parsed, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8",
                    )
                    temporary.replace(cache_path)
                return parsed
            except Exception as exc:  # provider exceptions vary by gateway
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"model call failed after {self.max_retries} attempts: {last_error}"
        ) from last_error
