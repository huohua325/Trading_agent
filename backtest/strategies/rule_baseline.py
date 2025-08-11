from __future__ import annotations

from typing import Dict, List
import pandas as pd


class Strategy:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg

    def _sma(self, series: pd.Series, win: int) -> float:
        if series is None or len(series) < win:
            return float(series.mean()) if len(series) > 0 else 0.0
        return float(series.iloc[-win:].mean())

    def on_bar(self, ctx) -> List[Dict]:
        # ctx: {date, symbols, open_map, portfolio, cfg, datasets}
        date = ctx["date"]
        symbols = ctx["symbols"]
        open_map = ctx["open_map"]
        pf = ctx["portfolio"]
        datasets = ctx["datasets"]
        cfg = ctx["cfg"]

        max_pos_pct = float(cfg.get("risk", {}).get("max_pos_pct", 0.1))
        warmup_days = int(cfg.get("backtest", {}).get("warmup_days", 60))

        orders: List[Dict] = []
        if not open_map:
            return orders

        start = (date - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        end = date.strftime("%Y-%m-%d")

        for s in symbols:
            open_px = open_map.get(s)
            if open_px is None or open_px <= 0:
                continue
            # 读取近 warmup 天的日线
            bars = datasets.get_day_bars(s, start, end)
            if bars is None or bars.empty:
                continue
            bars = bars.sort_values("date")
            closes = bars["close"].astype(float)
            sma5 = self._sma(closes, 5)
            sma20 = self._sma(closes, 20)
            up_trend = sma5 > sma20 and len(closes) >= 20
            if not up_trend:
                continue
            # 目标持仓按 max_pos_pct
            target_value = max_pos_pct * float(pf.equity)
            # 估算当前持仓市值（用开盘价）
            pos = pf.positions.get(s)
            current_value = (pos.shares * open_px) if pos else 0.0
            delta_value = max(0.0, target_value - current_value)
            qty = int(delta_value / open_px)
            if qty > 0:
                orders.append({"symbol": s, "side": "buy", "qty": qty})
        return orders 