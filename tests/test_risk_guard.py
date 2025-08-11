from trading_agent_v2.core.risk_guard import make_limits


def test_min_holding_blocks_decrease_close():
    cfg = {"max_pos_pct": 0.1, "min_holding_days": 2}
    limits = make_limits({}, {"holding_days": 0}, cfg, {})
    assert "decrease" not in limits["allowed"]
    assert "close" not in limits["allowed"]
    assert "min_holding_days_enforced" in limits["hits"]


def test_drawdown_blocks_increase():
    cfg = {"max_pos_pct": 0.1, "max_daily_drawdown_pct": 0.03}
    limits = make_limits({}, {"holding_days": 10}, cfg, {"daily_drawdown_pct": -0.05})
    assert "increase" not in limits["allowed"]
    assert "intraday_drawdown_limit" in limits["hits"] 