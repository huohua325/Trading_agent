from __future__ import annotations

from typing import Dict, List

from trading_agent_v2.core.schemas import RiskLimits


def make_limits(analysis: Dict, position_state: Dict, risk_cfg: Dict, market_ctx: Dict) -> Dict:
    """
    规则集合：
    - 基线仓位上限：max_pos_pct
    - 财报窗降杠杆：in_earnings_window → max_pos_pct=min(max_pos_pct, max_pos_pct_earnings_window)
    - 高波动降杠杆：atr_pct > 高分位阈值 → max_pos_pct *= atr_scale_down
    - 日内最大回撤触发：daily_drawdown_pct <= -max_daily_drawdown_pct → 禁止 increase
    - 最小持有期：holding_days < min_holding_days → 禁止 decrease/close
    输出 allowed、max_pos_pct、cooldown，并附上命中规则列表（写入审计由调用方处理）。
    """
    hits: List[str] = []

    max_pos = float(risk_cfg.get("max_pos_pct", 0.10))
    earnings_cap = float(risk_cfg.get("max_pos_pct_earnings_window", max_pos))
    atr_high_q = float(risk_cfg.get("atr_pct_high_quantile", 0.80))
    atr_scale = float(risk_cfg.get("atr_scale_down", 0.7))
    min_hold_days = int(risk_cfg.get("min_holding_days", 2))
    max_dd = float(risk_cfg.get("max_daily_drawdown_pct", 0.03))

    allowed: List[str] = ["increase", "hold", "decrease", "close"]
    cooldown = False

    # 财报窗
    in_earnings = bool(market_ctx.get("in_earnings_window", False) or position_state.get("in_earnings_window", False))
    if in_earnings:
        if earnings_cap < max_pos:
            max_pos = earnings_cap
            hits.append("earnings_window_cap")

    # 高波动
    atr_pct = float(market_ctx.get("atr_pct", 0.0))
    if atr_pct > atr_high_q:
        max_pos = max_pos * atr_scale
        hits.append("atr_high_quantile_scaled")

    # 日内最大回撤
    daily_dd = float(market_ctx.get("daily_drawdown_pct", 0.0))
    if daily_dd <= -max_dd:
        # 禁止加仓
        if "increase" in allowed:
            allowed.remove("increase")
        cooldown = True
        hits.append("intraday_drawdown_limit")

    # 最小持有期
    holding_days = int(position_state.get("holding_days", 0))
    if holding_days < min_hold_days:
        # 禁止减仓/清仓
        for a in ["decrease", "close"]:
            if a in allowed:
                allowed.remove(a)
        hits.append("min_holding_days_enforced")

    limits = RiskLimits(allowed=allowed, max_pos_pct=max_pos, cooldown=cooldown).model_dump()
    limits["hits"] = hits
    return limits 