from __future__ import annotations

from typing import Dict, List
import os

from trading_agent_v2.core.schemas import DecisionOutput, FeatureInput
from trading_agent_v2.llm.llm_client import LLMClient, LLMConfig


def _prompt_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    path = os.path.join(_prompt_dir(), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "系统：你是交易决策器。严格输出 JSON。"


def _guarded(curr: float, allowed: List[str], max_pos: float, proposal: Dict | None) -> Dict:
    # 默认 hold
    action = "hold"
    target = curr
    reasons = ["guarded decision"]
    if isinstance(proposal, dict):
        try:
            cand = DecisionOutput(**proposal).model_dump()
            action = cand.get("action", "hold")
            target = float(cand.get("target_pos_pct", curr))
            reasons = cand.get("reasons", reasons)
        except Exception:
            pass
    # Guardrails：收敛
    if action not in allowed:
        action = "hold"
        target = curr
        reasons = ["guardrail: action not allowed"]
    target = max(0.0, min(float(max_pos), float(target)))
    if action == "increase" and target < curr:
        target = curr
        reasons = ["guardrail: increase but target<current"]
    return DecisionOutput(action=action, target_pos_pct=target, reasons=reasons, confidence=0.5).model_dump()


def decide_batch(items: List[Dict], cfg: Dict | None = None, enable_llm: bool = False, cache_only: bool = False) -> Dict[str, Dict]:
    """支持 LLM 与降级。每个元素为 {features, analysis, limits}；返回 {symbol: decision_output}；并附加 '__meta__'。
    """
    results: Dict[str, Dict] = {}
    meta_agg: Dict[str, object] = {"calls": 0, "cache_hits": 0, "parse_errors": 0, "latency_ms_sum": 0, "tokens_prompt": 0, "tokens_completion": 0, "prompt_version": None}

    # 未启用 LLM：延用简单保守规则
    if not enable_llm:
        for item in items:
            features = FeatureInput(**item["features"]) if isinstance(item.get("features"), dict) else FeatureInput(**item)
            limits = item.get("limits", {"allowed": ["hold"], "max_pos_pct": features.position_state.current_position_pct})
            curr = float(features.position_state.current_position_pct)
            allowed = list(limits.get("allowed", ["hold"]))
            max_pos = float(limits.get("max_pos_pct", curr))
            action = "increase" if ("increase" in allowed and curr < max_pos) else "hold"
            target = min(max_pos, curr + 0.02) if action == "increase" else curr
            results[features.symbol] = DecisionOutput(action=action, target_pos_pct=target, reasons=["baseline"], confidence=0.5).model_dump()
        results["__meta__"] = meta_agg
        return results

    # 读取配置
    llm_cfg_raw = (cfg or {}).get("llm", {})
    llm_cfg = LLMConfig(
        base_url=str(llm_cfg_raw.get("base_url", "https://api.openai.com/v1")),
        model=str(llm_cfg_raw.get("decision_model", "gpt-4o-mini")),
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

    prompt_name = "decision_v1.txt"
    system_prompt = _load_prompt(prompt_name)
    from trading_agent_v2.agents.analyzer_llm import _prompt_version
    meta_agg["prompt_version"] = _prompt_version(os.path.basename(prompt_name))
    client = LLMClient()

    import json
    for item in items:
        try:
            features = FeatureInput(**item["features"]) if isinstance(item.get("features"), dict) else FeatureInput(**item)
            analysis = item.get("analysis", {})
            limits = item.get("limits", {})
            curr = float(features.position_state.current_position_pct)
            allowed = list(limits.get("allowed", ["hold"]))
            max_pos = float(limits.get("max_pos_pct", curr))

            user_obj = {"features": features.model_dump(), "analysis": analysis, "limits": limits}
            user_prompt = json.dumps(user_obj, ensure_ascii=False)
            data, meta = client.generate_json("decision", llm_cfg, system_prompt, user_prompt, cache_only=cache_only)

            meta_agg["calls"] = int(meta_agg["calls"]) + 1
            meta_agg["cache_hits"] = int(meta_agg["cache_hits"]) + (1 if meta.get("cached") else 0)
            meta_agg["latency_ms_sum"] = int(meta_agg["latency_ms_sum"]) + int(meta.get("latency_ms", 0))
            usage = meta.get("usage", {})
            meta_agg["tokens_prompt"] = int(meta_agg["tokens_prompt"]) + int(usage.get("prompt_tokens", 0))
            meta_agg["tokens_completion"] = int(meta_agg["tokens_completion"]) + int(usage.get("completion_tokens", 0))

            proposal = data or {}
            decided = _guarded(curr, allowed, max_pos, proposal)
            results[features.symbol] = decided
        except Exception:
            try:
                symbol = item.get("features", {}).get("symbol") or item.get("symbol", "UNKNOWN")
            except Exception:
                symbol = "UNKNOWN"
            results[symbol] = DecisionOutput(action="hold", target_pos_pct=0.0, reasons=["fallback"], confidence=0.0).model_dump()

    results["__meta__"] = meta_agg
    return results 