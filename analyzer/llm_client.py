import json
import logging
from dataclasses import dataclass
from typing import Any

import requests
from decouple import config

from analyzer.constants import (
    LLM_CLASSIFICATION_PROMPT_VERSION,
    LLM_SUMMARY_PROMPT_VERSION,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMDisabledError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMJSONError(LLMError):
    pass


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int
    num_ctx: int
    batch_news_size: int
    concurrency: int
    think: bool
    json_mode: bool
    classification_prompt_version: str
    summary_prompt_version: str


def get_llm_settings() -> LLMSettings:
    prompt_version = config("LLM_PROMPT_VERSION", default="")
    return LLMSettings(
        enabled=config("LLM_ENABLED", default=False, cast=bool),
        base_url=config("LLM_BASE_URL", default="http://localhost:11434"),
        api_key=config("LLM_API_KEY", default=""),
        model=config("LLM_MODEL", default="gemma4:e2b"),
        timeout_seconds=config("LLM_TIMEOUT_SECONDS", default=120, cast=float),
        temperature=config("LLM_TEMPERATURE", default=0, cast=float),
        max_tokens=config("LLM_MAX_TOKENS", default=2048, cast=int),
        num_ctx=max(config("LLM_NUM_CTX", default=0, cast=int), 0),
        batch_news_size=max(config("LLM_BATCH_NEWS_SIZE", default=1, cast=int), 1),
        concurrency=max(config("LLM_CONCURRENCY", default=1, cast=int), 1),
        think=config("LLM_THINK", default=False, cast=bool),
        json_mode=config("LLM_JSON_MODE", default=True, cast=bool),
        classification_prompt_version=config(
            "LLM_CLASSIFICATION_PROMPT_VERSION",
            default=prompt_version or LLM_CLASSIFICATION_PROMPT_VERSION,
        ),
        summary_prompt_version=config(
            "LLM_SUMMARY_PROMPT_VERSION",
            default=prompt_version or LLM_SUMMARY_PROMPT_VERSION,
        ),
    )


def _strip_json_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def parse_json_object(text: str) -> dict:
    cleaned = _strip_json_fences(text)
    if not cleaned:
        raise LLMJSONError("Empty LLM response")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMJSONError("LLM response does not contain a JSON object")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMJSONError(f"Invalid LLM JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMJSONError("LLM response root must be a JSON object")
    return parsed


class LLMClient:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or get_llm_settings()
        self.last_response_data: dict[str, Any] | None = None

    def complete_json(self, system_prompt: str, user_prompt: str) -> tuple[dict, str]:
        if not self.settings.enabled:
            raise LLMDisabledError("LLM is disabled. Set LLM_ENABLED=true to enable local LLM calls.")

        raw_text = self.complete(system_prompt, user_prompt)
        try:
            return parse_json_object(raw_text), raw_text
        except LLMJSONError as first_error:
            logger.warning("LLM returned invalid JSON, retrying once: %s", first_error)
            retry_prompt = (
                f"{user_prompt}\n\nПредыдущий ответ был невалидным JSON. "
                "Верни только исправленный валидный JSON по той же схеме, без markdown."
            )
            raw_retry = self.complete(system_prompt, retry_prompt)
            try:
                return parse_json_object(raw_retry), raw_retry
            except LLMJSONError as second_error:
                raise LLMJSONError(f"Invalid JSON after retry: {second_error}") from second_error

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = self._endpoint()
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        payload = self._payload(endpoint, system_prompt, user_prompt)
        self.last_response_data = None
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise LLMRequestError(f"LLM request timed out after {self.settings.timeout_seconds:g}s") from exc
        except requests.RequestException as exc:
            raise LLMRequestError(f"LLM request failed: {exc}") from exc

        data = response.json()
        self.last_response_data = data
        content = self._extract_content(endpoint, data)
        if not content:
            raise LLMRequestError("LLM returned an empty response")
        return content

    def _endpoint(self) -> str:
        base_url = self.settings.base_url.rstrip("/")
        if base_url.endswith("/api/chat") or base_url.endswith("/api/generate"):
            return base_url
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/api/chat"

    def _payload(self, endpoint: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if endpoint.endswith("/chat/completions"):
            payload = {
                "model": self.settings.model,
                "messages": messages,
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.max_tokens,
            }
            if self.settings.json_mode:
                payload["response_format"] = {"type": "json_object"}
            return payload
        options = {
            "temperature": self.settings.temperature,
            "num_predict": self.settings.max_tokens,
        }
        if self.settings.num_ctx:
            options["num_ctx"] = self.settings.num_ctx

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": self.settings.think,
            "options": options,
        }
        if self.settings.json_mode:
            payload["format"] = "json"
        return payload

    def _extract_content(self, endpoint: str, data: dict[str, Any]) -> str:
        if endpoint.endswith("/chat/completions"):
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            return (message.get("content") or "").strip()

        message = data.get("message") or {}
        return (message.get("content") or data.get("response") or "").strip()
