from __future__ import annotations

from typing import Dict, List, Tuple
import os

from trading_agent_v2.core.schemas import AnalyzerOutput, FeatureInput
from trading_agent_v2.llm.llm_client import LLMClient, LLMConfig


def _prompt_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    path = os.path.join(_prompt_dir(), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "系统：你是量化分析归纳器。只使用输入字段，严格输出 JSON。"


def _prompt_version(name: str) -> str:
    base = os.path.splitext(name)[0]
    return base.replace("_", "/")


def _neutral_output() -> Dict:
    out = AnalyzerOutput(tech_score=0.5, sent_score=0.0, event_risk="normal", summary=["neutral"], confidence=0.5)
    return out.model_dump()


def analyze_batch(features_list: List[Dict], cfg: Dict | None = None, enable_llm: bool = False, cache_only: bool = False) -> Dict[str, Dict]:
    """支持 LLM 与降级。返回 {symbol: analyzer_output_dict}；并附加 '__meta__' 汇总指标。
    """
    results: Dict[str, Dict] = {}
    meta_agg: Dict[str, object] = {"calls": 0, "cache_hits": 0, "parse_errors": 0, "latency_ms_sum": 0, "tokens_prompt": 0, "tokens_completion": 0, "prompt_version": None}

    # 若未启用 LLM，直接中性降级
    if not enable_llm:
        for item in features_list:
            fi = FeatureInput(**item)
            results[fi.symbol] = _neutral_output()
        results["__meta__"] = meta_agg
        return results

    # 读取配置
    llm_cfg_raw = (cfg or {}).get("llm", {})
    llm_cfg = LLMConfig(
        base_url=str(llm_cfg_raw.get("base_url", "https://api.openai.com/v1")),
        model=str(llm_cfg_raw.get("analyzer_model", "gpt-4o-mini")),
        temperature=float(llm_cfg_raw.get("temperature", 0.0)),
        max_tokens=int(llm_cfg_raw.get("max_tokens", 256)),
        seed=llm_cfg_raw.get("seed"),
        timeout_sec=float(llm_cfg_raw.get("timeout_sec", 60)),
        max_retries=int(llm_cfg_raw.get("retry", {}).get("max_retries", 3)),
        backoff_factor=float(llm_cfg_raw.get("retry", {}).get("backoff_factor", 0.5)),
        cache_enabled=bool(llm_cfg_raw.get("cache", {}).get("enabled", True)),
        cache_ttl_hours=int(llm_cfg_raw.get("cache", {}).get("ttl_hours", 24)),
        budget_prompt_tokens=int(llm_cfg_raw.get("budget", {}).get("max_prompt_tokens", 200_000)),
        budget_completion_tokens=int(llm_cfg_raw.get("budget", {}).get("max_completion_tokens", 200_000)),
    )
    batch_size = int(llm_cfg_raw.get("batch_size", 16))

    prompt_name = "analyzer_v1.txt"
    system_prompt = _load_prompt(prompt_name)
    meta_agg["prompt_version"] = _prompt_version(os.path.basename(prompt_name))
    client = LLMClient()

    def _render_user_prompt(batch_items: List[FeatureInput]) -> str:
        import json
        payload = [FeatureInput(**it if isinstance(it, dict) else it).model_dump() for it in batch_items]
        return json.dumps(payload, ensure_ascii=False)

    # 切片执行
    for i in range(0, len(features_list), max(1, batch_size)):
        chunk = features_list[i : i + batch_size]
        if not chunk:
            continue
        user_prompt = _render_user_prompt(chunk)
        data, meta = client.generate_json("analyzer", llm_cfg, system_prompt, user_prompt, cache_only=cache_only)
        meta_agg["calls"] = int(meta_agg["calls"]) + 1
        meta_agg["cache_hits"] = int(meta_agg["cache_hits"]) + (1 if meta.get("cached") else 0)
        meta_agg["latency_ms_sum"] = int(meta_agg["latency_ms_sum"]) + int(meta.get("latency_ms", 0))
        usage = meta.get("usage", {})
        meta_agg["tokens_prompt"] = int(meta_agg["tokens_prompt"]) + int(usage.get("prompt_tokens", 0))
        meta_agg["tokens_completion"] = int(meta_agg["tokens_completion"]) + int(usage.get("completion_tokens", 0))

        # 解析失败或未启用→降级
        if not data or not isinstance(data, dict):
            meta_agg["parse_errors"] = int(meta_agg["parse_errors"]) + 1
            for it in chunk:
                fi = FeatureInput(**it)
                results[fi.symbol] = _neutral_output()
            continue
        # 期望返回 {symbol: AnalyzerOutput-like}
        for it in chunk:
            fi = FeatureInput(**it)
            val = data.get(fi.symbol)
            try:
                if isinstance(val, dict):
                    # 严格校验与越界收敛由 pydantic 保障
                    results[fi.symbol] = AnalyzerOutput(**val).model_dump()
                else:
                    meta_agg["parse_errors"] = int(meta_agg["parse_errors"]) + 1
                    results[fi.symbol] = _neutral_output()
            except Exception:
                meta_agg["parse_errors"] = int(meta_agg["parse_errors"]) + 1
                results[fi.symbol] = _neutral_output()

    results["__meta__"] = meta_agg
    return results 