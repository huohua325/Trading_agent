from __future__ import annotations

from typing import Dict, List
import os

from trading_agent_v2.core.schemas import DecisionOutput
from trading_agent_v2.llm.llm_client import LLMClient, LLMConfig


def _prompt_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    path = os.path.join(_prompt_dir(), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "系统：你是交易决策器。依据输入，输出合规的决策。严格输出 JSON。"


def _prompt_version(name: str) -> str:
    base = os.path.splitext(name)[0]
    return base.replace("_", "/")


def _neutral_decision(current_position_pct: float = 0.0) -> Dict:
    """中性决策：保持当前仓位"""
    out = DecisionOutput(
        action="hold",
        target_pos_pct=current_position_pct,
        confidence=0.5,
        reasons=["降级为中性决策"]
    )
    return out.model_dump()


def _apply_limits(decision: Dict, limits: Dict, current_position_pct: float) -> Dict:
    """对决策进行硬约束：动作合规与目标仓位收敛，并在必要时追加 guard 原因。"""
    try:
        allowed: List[str] = list(limits.get("allowed", ["hold"]))
        max_pos: float = float(limits.get("max_pos_pct", current_position_pct))
    except Exception:
        allowed = ["hold"]
        max_pos = current_position_pct

    action = str(decision.get("action", "hold"))
    reasons = decision.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    # 夹逼 target 到 [0, max_pos]
    try:
        target = float(decision.get("target_pos_pct", current_position_pct))
    except Exception:
        target = current_position_pct
    clamped_target = max(0.0, min(float(max_pos), float(target)))
    if clamped_target != target:
        reasons.append("guard: target_pos_pct clamped to max_pos_pct")
        target = clamped_target

    # increase 需满足 target ≥ current
    if action == "increase" and target < float(current_position_pct):
        reasons.append("guard: increase requires target≥current")
        target = float(current_position_pct)

    # 若动作不在 allowed，回退为 hold 或第一个允许动作
    if action not in allowed:
        if "hold" in allowed:
            reasons.append("guard: action not allowed -> hold")
            action = "hold"
            target = float(current_position_pct)
        else:
            # 选择第一个允许动作，并进行合理的 target 收敛
            fallback = allowed[0] if allowed else "hold"
            reasons.append(f"guard: action not allowed -> {fallback}")
            action = fallback
            if action == "increase":
                target = max(float(current_position_pct), float(target))
            elif action in ("decrease", "close"):
                target = min(float(current_position_pct), float(target))
            else:  # hold 或其它
                target = float(current_position_pct)

    # 置信度夹逼
    try:
        conf = float(decision.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    return DecisionOutput(action=action, target_pos_pct=float(target), reasons=reasons, confidence=conf).model_dump()


def decide_batch(decisions_input: List[Dict], cfg: Dict | None = None, enable_llm: bool = True, cache_only: bool = False) -> Dict[str, Dict]:
    """批量决策。输入为 [{features, analysis, limits}] 列表，返回 {symbol: decision_output_dict}。
    
    参数:
        decisions_input: 输入列表，每项包含 features/analysis/limits
        cfg: 配置字典，包含 llm 子配置
        enable_llm: 是否启用 LLM，若为 False 则降级为中性决策
        cache_only: 是否仅使用缓存，若为 True 且缓存未命中则降级
        
    返回:
        字典 {symbol: decision_dict, "__meta__": meta_dict}
    """
    results: Dict[str, Dict] = {}
    meta_agg: Dict[str, object] = {"calls": 0, "cache_hits": 0, "parse_errors": 0, "latency_ms_sum": 0, "tokens_prompt": 0, "tokens_completion": 0, "prompt_version": None}
    
    # 若未启用 LLM，直接中性降级
    if not enable_llm:
        for item in decisions_input:
            features = item.get("features", {})
            symbol = features.get("symbol", "UNKNOWN")
            current_position_pct = float((features.get("position_state") or {}).get("current_position_pct", 0.0))
            results[symbol] = _neutral_decision(current_position_pct)
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
    batch_size = int(llm_cfg_raw.get("batch_size", 16))
    
    prompt_name = "decision_v1.txt"
    system_prompt = _load_prompt(prompt_name)
    meta_agg["prompt_version"] = _prompt_version(os.path.basename(prompt_name))
    client = LLMClient()
    
    def _render_user_prompt(item: Dict) -> str:
        import json
        return json.dumps(item, ensure_ascii=False)
    
    # 切片执行
    for i in range(0, len(decisions_input), max(1, batch_size)):
        chunk = decisions_input[i : i + batch_size]
        if not chunk:
            continue
        
        # 单条处理（决策通常每标的一次，不像分析可批量）
        for item in chunk:
            features = item.get("features", {})
            limits = item.get("limits", {})
            symbol = features.get("symbol", "UNKNOWN")
            current_position_pct = float((features.get("position_state") or {}).get("current_position_pct", 0.0))
            
            user_prompt = _render_user_prompt(item)
            data, meta = client.generate_json("decision", llm_cfg, system_prompt, user_prompt, cache_only=cache_only)
            meta_agg["calls"] = int(meta_agg["calls"]) + 1
            meta_agg["cache_hits"] = int(meta_agg["cache_hits"]) + (1 if meta.get("cached") else 0)
            meta_agg["latency_ms_sum"] = int(meta_agg["latency_ms_sum"]) + int(meta.get("latency_ms", 0))
            usage = meta.get("usage", {})
            meta_agg["tokens_prompt"] = int(meta_agg["tokens_prompt"]) + int(usage.get("prompt_tokens", 0))
            meta_agg["tokens_completion"] = int(meta_agg["tokens_completion"]) + int(usage.get("completion_tokens", 0))
            
            # 解析失败或未启用→降级
            if not data or not isinstance(data, dict):
                meta_agg["parse_errors"] = int(meta_agg["parse_errors"]) + 1
                results[symbol] = _neutral_decision(current_position_pct)
                continue
            
            try:
                # 初步解析（容忍非严格字段），随后应用硬约束与收敛
                parsed = {
                    "action": data.get("action", "hold"),
                    "target_pos_pct": data.get("target_pos_pct", current_position_pct),
                    "confidence": data.get("confidence", 0.5),
                    "reasons": data.get("reasons", ["无说明"]),
                }
                decided = _apply_limits(parsed, limits, current_position_pct)
                results[symbol] = decided
            except Exception:
                meta_agg["parse_errors"] = int(meta_agg["parse_errors"]) + 1
                results[symbol] = _neutral_decision(current_position_pct)
    
    results["__meta__"] = meta_agg
    return results 