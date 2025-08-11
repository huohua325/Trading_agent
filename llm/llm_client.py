from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

import httpx

from trading_agent_v2.utils.io import ensure_dir, sha256_text, canonical_json


@dataclass
class LLMConfig:
    provider: str = "openai-compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 256
    seed: Optional[int] = None
    timeout_sec: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    budget_prompt_tokens: int = 200_000
    budget_completion_tokens: int = 200_000


class LLMClient:
    def __init__(self, api_key_env: str = "OPENAI_API_KEY", cache_dir: Optional[str] = None) -> None:
        self.api_key = os.getenv(api_key_env) or os.getenv("LLM_API_KEY", "")
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "trading_agent_v2", "storage", "cache", "llm")
        ensure_dir(self.cache_dir)
        self._client: Optional[httpx.Client] = None
        self._prompt_tokens_used = 0
        self._completion_tokens_used = 0

    def _get_client(self, base_url: str, timeout_sec: float) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=base_url, timeout=timeout_sec)
        return self._client

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _read_cache(self, key: str, ttl_hours: int) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        st = os.stat(path)
        if (time.time() - st.st_mtime) > ttl_hours * 3600:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass

    def _make_cache_key(self, role: str, cfg: LLMConfig, prompt: str) -> str:
        ident = {
            "role": role,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
            "prompt": prompt,
        }
        return sha256_text(canonical_json(ident))

    @staticmethod
    def _extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            # 直接尝试整体解析
            return json.loads(text), None
        except Exception:
            pass
        try:
            # 兜底：截取第一个 { 到最后一个 }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                sub = text[start : end + 1]
                return json.loads(sub), None
        except Exception as e:
            return None, str(e)
        return None, "no_json_found"

    def remaining_budget_ok(self, cfg: LLMConfig) -> bool:
        return (
            self._prompt_tokens_used < cfg.budget_prompt_tokens
            and self._completion_tokens_used < cfg.budget_completion_tokens
        )

    def get_cached_payload(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """公开读取缓存内容（不校验 TTL）。"""
        try:
            path = self._cache_path(cache_key)
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def generate_json(self, role: str, cfg: LLMConfig, system_prompt: str, user_prompt: str, cache_only: bool = False) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        meta: Dict[str, Any] = {"cached": False, "usage": {}, "latency_ms": 0}
        key = self._make_cache_key(role, cfg, f"{system_prompt}\n\n{user_prompt}")
        meta["cache_key"] = key

        if cfg.cache_enabled:
            cached = self._read_cache(key, ttl_hours=cfg.cache_ttl_hours)
            if cached is not None:
                meta["cached"] = True
                return cached.get("json"), {**meta, "usage": cached.get("usage", {}), "latency_ms": 0}
            if cache_only:
                return None, {**meta, "reason": "cache_only_miss"}

        if not self.api_key or not self.remaining_budget_ok(cfg):
            return None, {**meta, "reason": "no_api_key_or_budget_exceeded"}

        client = self._get_client(cfg.base_url, cfg.timeout_sec)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body: Dict[str, Any] = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.seed is not None:
            body["seed"] = cfg.seed

        backoff = cfg.backoff_factor
        start_ts = time.time()
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = client.post("/chat/completions", headers=headers, json=body)
                if resp.status_code in (429,) or resp.status_code >= 500:
                    # 退避
                    delay = min(30.0, backoff * (2 ** attempt))
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = (
                    (data.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                parsed, err = self._extract_json(content or "")
                usage = data.get("usage", {})
                self._prompt_tokens_used += int(usage.get("prompt_tokens", 0))
                self._completion_tokens_used += int(usage.get("completion_tokens", 0))
                meta["usage"] = usage
                meta["latency_ms"] = int((time.time() - start_ts) * 1000)
                if parsed is not None:
                    if cfg.cache_enabled:
                        self._write_cache(key, {
                            "role": role,
                            "ts_utc": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "json": parsed,
                            "usage": usage,
                            "system": system_prompt,
                            "user": user_prompt,
                            "raw": content
                        })
                    return parsed, meta
                else:
                    return None, {**meta, "reason": f"parse_failed:{err}"}
            except httpx.RequestError:
                delay = min(30.0, backoff * (2 ** attempt))
                time.sleep(delay)
                continue
            except httpx.HTTPStatusError as e:
                return None, {**meta, "reason": f"http_error:{e.response.status_code}"}
            except Exception as e:  # 解析或其它异常
                return None, {**meta, "reason": f"unexpected:{e}"}
        return None, {**meta, "reason": "exhausted_retries"} 