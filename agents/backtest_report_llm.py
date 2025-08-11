from __future__ import annotations

import json
import os
from typing import Dict, Any

from trading_agent_v2.llm.llm_client import LLMClient, LLMConfig


def generate_backtest_report(payload: Dict[str, Any], cfg: Dict | None = None, cache_only: bool = True) -> str:
    llm_cfg_raw = (cfg or {}).get("llm", {})
    llm_cfg = LLMConfig(
        base_url=str(llm_cfg_raw.get("base_url", "https://api.openai.com/v1")),
        model=str(llm_cfg_raw.get("analyzer_model", "gpt-4o-mini")),
        temperature=0.2,
        max_tokens=280,
        seed=llm_cfg_raw.get("seed"),
        timeout_sec=float(llm_cfg_raw.get("timeout_sec", 60)),
        max_retries=int(llm_cfg_raw.get("retry", {}).get("max_retries", 3)),
        backoff_factor=float(llm_cfg_raw.get("retry", {}).get("backoff_factor", 0.5)),
        cache_enabled=False,
    )
    client = LLMClient()
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "backtest_report_v1.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception:
        system_prompt = "系统：你是量化回测结果解读助手。输出 JSON {\"report\": \"...\"}。"

    user_prompt = json.dumps(payload, ensure_ascii=False)
    data, meta = client.generate_json("backtest_report", llm_cfg, system_prompt, user_prompt, cache_only=cache_only)
    if isinstance(data, dict) and isinstance(data.get("report"), str):
        return data["report"].strip()
    # 兜底
    m = payload.get("metrics", {})
    return (
        f"回测摘要：累计收益 {m.get('cum_return', 0):.2%}，最大回撤 {m.get('max_drawdown', 0):.2%}，"
        f"年化波动 {m.get('volatility', 0):.2%}，Sharpe {m.get('sharpe', 0):.2f}。"
    ) 