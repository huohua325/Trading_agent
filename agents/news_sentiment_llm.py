from __future__ import annotations

from typing import Dict, List, Tuple

from trading_agent_v2.llm.llm_client import LLMClient, LLMConfig


def score_news_batch(items: List[Dict], cfg: Dict | None = None, cache_only: bool = False, batch_size: int = 32) -> List[float]:
    """用 LLM 对一批新闻项（title+summary）打分。返回与输入等长的分数列表（缺项用 0 填充）。"""
    if not items:
        return []
    llm_cfg_raw = (cfg or {}).get("llm", {})
    llm_cfg = LLMConfig(
        base_url=str(llm_cfg_raw.get("base_url", "https://api.openai.com/v1")),
        model=str(llm_cfg_raw.get("analyzer_model", "gpt-4o-mini")),
        temperature=0.0,
        max_tokens=64,
        seed=llm_cfg_raw.get("seed"),
        timeout_sec=float(llm_cfg_raw.get("timeout_sec", 30)),
        max_retries=int(llm_cfg_raw.get("retry", {}).get("max_retries", 2)),
        backoff_factor=float(llm_cfg_raw.get("retry", {}).get("backoff_factor", 0.4)),
        cache_enabled=True,
        cache_ttl_hours=int(llm_cfg_raw.get("cache", {}).get("ttl_hours", 48)),
    )
    client = LLMClient()
    system_prompt = (
        "你是金融新闻情绪打分器。对输入的每条新闻（标题+摘要）给出一个介于 -1 到 1 的分数："
        "-1 强烈利空，-0.5 偏空，0 中性，0.5 偏多，1 强烈利多。只考虑对标的基本面/价格影响，忽略钓鱼标题。"
        "严格输出 JSON 形如 {\"scores\": [..]}，scores 与输入顺序一一对应。"
    )

    out: List[float] = [0.0 for _ in items]
    # 分批
    for i in range(0, len(items), max(1, batch_size)):
        batch = items[i : i + batch_size]
        to_score = []
        pos = []
        for j, it in enumerate(batch):
            title = (it.get("title") or "").strip()
            summary = (it.get("description") or it.get("summary") or it.get("article_body") or "").strip()
            if not title and not summary:
                continue
            to_score.append({"title": title[:300], "summary": summary[:600]})
            pos.append(j)
        if not to_score:
            continue
        import json as _json
        user_prompt = _json.dumps({"items": to_score}, ensure_ascii=False)
        data, meta = client.generate_json("news_sentiment", llm_cfg, system_prompt, user_prompt, cache_only=cache_only)
        scores = []
        if isinstance(data, dict):
            scores = data.get("scores") or []
        for rel_k, s in enumerate(scores):
            try:
                v = float(s)
                v = max(-1.0, min(1.0, v))
                out[i + pos[rel_k]] = v
            except Exception:
                continue
    return out 