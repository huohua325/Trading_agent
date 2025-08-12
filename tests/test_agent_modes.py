from typing import Dict
import pytest

from trading_agent_v2.agents.analyzer_llm import analyze_batch
from trading_agent_v2.agents.decision_llm import decide_batch
from trading_agent_v2.agents.single_agent_llm import decide_batch as single_agent_decide_batch
from trading_agent_v2.backtest.strategies.llm_decision import Strategy


def _mock_features() -> Dict:
    """创建一个简单的特征字典用于测试"""
    return {
        "symbol": "AAPL",
        "ts_utc": "2025-01-01T00:00:00Z",
        "schema_version": "v1",
        "tech": {
            "trend": "up",
            "mom": 0.7,
            "atr_pct": 0.02,
            "ret": {"1d": 0.01, "5d": 0.03, "20d": 0.05},
            "vwap_dev": 0.5
        },
        "news": {
            "sentiment": 0.2,
            "src_count": 10,
            "freshness_min": 60
        },
        "fund": {
            "fin_rev_yoy": 0.15,
            "fin_eps_yoy": 0.2,
            "fin_gross_margin": 0.4,
            "fin_net_margin": 0.2,
            "fin_debt_to_equity": 0.5
        },
        "market_ctx": {
            "last_price": 150.0
        },
        "position_state": {
            "current_position_pct": 0.0,
            "avg_price": None,
            "pnl_pct": 0.0,
            "holding_days": 0
        }
    }


def test_multi_agent_mode_degradation():
    """测试多智能体模式在缓存缺失时的降级行为"""
    # 1. 分析降级
    features_list = [_mock_features()]
    analysis_map = analyze_batch(features_list, cfg={}, enable_llm=False)
    
    # 验证分析结果包含默认值
    assert "AAPL" in analysis_map
    assert analysis_map["AAPL"]["tech_score"] == 0.5
    assert analysis_map["AAPL"]["sent_score"] == 0.0
    assert analysis_map["AAPL"]["event_risk"] == "normal"
    
    # 2. 决策降级
    decisions_input = [{"features": features_list[0], "analysis": analysis_map["AAPL"], "limits": {"allowed": ["increase", "hold"], "max_pos_pct": 0.1}}]
    decisions_map = decide_batch(decisions_input, cfg={}, enable_llm=False)
    
    # 验证决策结果
    assert "AAPL" in decisions_map
    assert decisions_map["AAPL"]["action"] == "hold"
    assert decisions_map["AAPL"]["target_pos_pct"] == 0.0


def test_single_agent_mode_degradation():
    """测试单智能体模式在缓存缺失时的降级行为"""
    features_list = [_mock_features()]
    decisions_map = single_agent_decide_batch(features_list, cfg={}, enable_llm=False)
    
    # 验证决策结果
    assert "AAPL" in decisions_map
    assert decisions_map["AAPL"]["action"] == "hold"
    assert decisions_map["AAPL"]["target_pos_pct"] == 0.0
    assert "analysis_excerpt" in decisions_map["AAPL"]
    assert "tech_score" in decisions_map["AAPL"]
    assert "sent_score" in decisions_map["AAPL"]
    assert "event_risk" in decisions_map["AAPL"]


def test_strategy_mode_selection():
    """测试策略类正确选择代理模式"""
    # 创建一个基本配置
    cfg = {
        "agents": {"mode": "single"},
        "llm": {"backtest_cache_only": True},
        "news": {"lookback_days": 7, "page_limit": 50},
        "backtest": {"warmup_days": 60}
    }
    
    # 创建策略实例
    strategy = Strategy(cfg)
    
    # 验证代理模式设置正确
    assert strategy.agent_mode == "single"
    
    # 修改为多体模式
    cfg["agents"]["mode"] = "multi"
    strategy = Strategy(cfg)
    assert strategy.agent_mode == "multi" 