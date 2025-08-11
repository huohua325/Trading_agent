from __future__ import annotations

import json
import os
from typing import Dict

from trading_agent_v2.llm.llm_client import LLMClient, LLMConfig


def generate_report(features: Dict, analysis: Dict, limits: Dict, decision: Dict, cfg: Dict | None = None, cache_only: bool = False) -> str:
    llm_cfg_raw = (cfg or {}).get("llm", {})
    llm_cfg = LLMConfig(
        base_url=str(llm_cfg_raw.get("base_url", "https://api.openai.com/v1")),
        model=str(llm_cfg_raw.get("analyzer_model", "gpt-4o-mini")),
        temperature=0.2,
        max_tokens=200,
        seed=llm_cfg_raw.get("seed"),
        timeout_sec=float(llm_cfg_raw.get("timeout_sec", 60)),
        max_retries=int(llm_cfg_raw.get("retry", {}).get("max_retries", 3)),
        backoff_factor=float(llm_cfg_raw.get("retry", {}).get("backoff_factor", 0.5)),
        cache_enabled=False,
    )
    client = LLMClient()
    # 读取提示词
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "report_v1.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception:
        system_prompt = "系统：你是交易建议报告撰写助手。输出 JSON {\"report\": \"...\"}。"

    user_payload = {"features": features, "analysis": analysis, "limits": limits, "decision": decision}
    user_prompt = json.dumps(user_payload, ensure_ascii=False)
    data, meta = client.generate_json("report", llm_cfg, system_prompt, user_prompt, cache_only=cache_only)
    if isinstance(data, dict) and isinstance(data.get("report"), str):
        return data["report"].strip()
    # 兜底：构造一段简要文案
    action = decision.get("action", "hold")
    t = decision.get("target_pos_pct", 0)
    return f"建议{action}，目标仓位 {t:.0%}。技术/情绪/基本面综合评估后生成（详情见审计记录）。"


def write_report(symbol: str, report_text: str) -> str:
    base_dir = os.path.join(os.getcwd(), "trading_agent_v2", "storage", "reports", "runs")
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{symbol}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return path 