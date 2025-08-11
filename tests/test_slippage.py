from trading_agent_v2.backtest.slippage import Slippage


def test_slippage_apply_buy_sell():
    s = Slippage(bps=20)
    assert abs(s.apply_buy(100.0) - 100.2) < 1e-6
    assert abs(s.apply_sell(100.0) - 99.8) < 1e-6 