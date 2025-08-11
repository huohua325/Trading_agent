from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

from trading_agent_v2.backtest.metrics import evaluate
from trading_agent_v2.backtest.slippage import Slippage


@dataclass
class Position:
    shares: int = 0
    avg_price: float = 0.0
    holding_days: int = 0
    last_trade_date: pd.Timestamp | None = None


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    @property
    def equity(self) -> float:
        return self.cash + sum(pos.shares * pos.avg_price for pos in self.positions.values())


class BacktestEngine:
    def __init__(self, cfg: Dict, datasets, slippage_model: Slippage) -> None:
        self.cfg = cfg
        self.datasets = datasets
        self.slippage = slippage_model
        self.commission_bps = float(cfg.get("backtest", {}).get("commission_bps", 0.0))
        self.fill_ratio = float(cfg.get("backtest", {}).get("fill_ratio", 1.0))
        # 风控参数
        rcfg = cfg.get("risk", {}) or {}
        self.max_positions = int(rcfg.get("max_positions", 999999))
        self.cooldown_days = int(rcfg.get("cooldown_days", 0))
        self.min_holding_days = int(rcfg.get("min_holding_days", 0))

    def _ensure_position(self, pf: Portfolio, symbol: str) -> Position:
        if symbol not in pf.positions:
            pf.positions[symbol] = Position()
        return pf.positions[symbol]

    def _apply_commission(self, notional: float) -> float:
        return notional * (self.commission_bps / 10_000.0)

    def _fill_at_open(self, symbol: str, trade_date: pd.Timestamp, qty: int, open_price: float) -> Tuple[int, float, float]:
        # 按开盘价撮合，考虑滑点与佣金与成交比例
        side = 1 if qty > 0 else -1
        px = self.slippage.apply_buy(open_price) if side > 0 else self.slippage.apply_sell(open_price)
        planned_qty = abs(qty)
        filled_qty = int(round(planned_qty * max(0.0, min(1.0, self.fill_ratio))))
        filled_qty = filled_qty * side
        gross = px * abs(filled_qty)
        commission = self._apply_commission(gross)
        net_cost = gross + commission if side > 0 else -(gross - commission)
        return filled_qty, px, net_cost

    def _apply_corporate_actions(self, pf: Portfolio, symbol: str, date: pd.Timestamp) -> None:
        # 占位：可在此读取 datasets 的 splits/dividends 并调整 shares/avg_price/cash。
        return

    def _enforce_max_positions(self, pf: Portfolio, orders: List[Dict]) -> List[Dict]:
        # 允许所有卖单；买单受最大持仓数限制
        if self.max_positions >= 999999:
            return orders
        current_positions = sum(1 for p in pf.positions.values() if p.shares > 0)
        allowed_new = max(0, self.max_positions - current_positions)
        if allowed_new <= 0:
            # 仅保留卖单
            return [od for od in orders if od.get("side", "buy") == "sell"]
        filtered: List[Dict] = []
        new_buys = 0
        for od in orders:
            if od.get("side", "buy") == "sell":
                filtered.append(od)
            else:
                # 仅限制从 0 -> 正仓的买入；加仓不占新增名额
                sym = od.get("symbol")
                pos = pf.positions.get(sym)
                if pos and pos.shares > 0:
                    filtered.append(od)
                elif new_buys < allowed_new:
                    filtered.append(od)
                    new_buys += 1
        return filtered

    def run(self, strategy, start: str, end: str, symbols: List[str], timespan: str = "day") -> Dict:
        dates = pd.date_range(start=start, end=end, freq="B")
        nav = []
        trade_rows = []
        pf = Portfolio(cash=float(self.cfg.get("backtest", {}).get("cash", 1_000_000)))

        for d in dates:
            # 公司行为占位处理（每个标的每日开盘前）
            for s in symbols:
                self._apply_corporate_actions(pf, s, d)

            # 计算每个 symbol 的当日撮合价与估值价
            open_map: Dict[str, float] = {}
            mark_map: Dict[str, float] = {}
            if (timespan or "day").lower() == "minute":
                for s in symbols:
                    mbars = self.datasets.get_min_bars(s, d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"))
                    if not mbars.empty:
                        mbars = mbars.sort_values("timestamp")
                        first_px = float(mbars.iloc[0]["vwap" if "vwap" in mbars.columns else "close"]) if len(mbars) > 0 else None
                        last_px = float(mbars.iloc[-1]["close"]) if len(mbars) > 0 else None
                        if first_px is not None:
                            open_map[s] = first_px
                        if last_px is not None:
                            mark_map[s] = last_px
            else:
                for s in symbols:
                    bars = self.datasets.get_day_bars(s, d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"))
                    if not bars.empty:
                        try:
                            row = bars.loc[bars["date"] == d.date()].iloc[0]
                            open_px = float(row["open"])
                            close_px = float(row["close"]) if "close" in row else open_px
                        except Exception:
                            open_px = None
                            close_px = None
                    else:
                        open_px = None
                        close_px = None
                    if open_px is not None:
                        open_map[s] = open_px
                    if close_px is not None:
                        mark_map[s] = close_px

            # 策略钩子：提供上下文
            ctx = {
                "date": d,
                "symbols": symbols,
                "open_map": open_map,
                "portfolio": pf,
                "cfg": self.cfg,
                "datasets": self.datasets,
            }
            try:
                orders = strategy.on_bar(ctx) or []
            except Exception:
                orders = []

            # 风控：最大持仓限制（优先级在撮合前）
            orders = self._enforce_max_positions(pf, orders)

            # 撮合订单
            if (timespan or "day").lower() == "minute":
                # 聚合为每标的净买卖量
                rem: Dict[str, int] = {}
                for od in orders:
                    s = od["symbol"]
                    side = od.get("side", "buy")
                    raw_qty = int(od.get("qty", 0))
                    qty = raw_qty if side == "buy" else -abs(raw_qty)
                    if qty == 0 or s not in open_map:
                        continue
                    # 风控：开仓冷却与最小持有天数（仅在当日开始检查一次）
                    pos = self._ensure_position(pf, s)
                    if qty > 0:
                        if pos.shares == 0 and self.cooldown_days > 0 and pos.last_trade_date is not None:
                            if (d - pos.last_trade_date).days < self.cooldown_days:
                                continue
                    else:
                        if self.min_holding_days > 0 and pos.shares > 0 and pos.holding_days < self.min_holding_days:
                            continue
                    rem[s] = rem.get(s, 0) + qty

                # 逐标的按分钟均匀切片撮合
                for s, q_total in rem.items():
                    # 取当日分钟序列
                    mbars = self.datasets.get_min_bars(s, d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"))
                    if mbars is None or mbars.empty:
                        continue
                    mbars = mbars.sort_values("timestamp").reset_index(drop=True)
                    remaining = int(q_total)
                    n = len(mbars)
                    if remaining == 0 or n == 0:
                        continue
                    sign = 1 if remaining > 0 else -1
                    # 均匀切片：保证在当日内尽量完成
                    for i in range(n):
                        if remaining == 0:
                            break
                        # 剩余分钟数
                        mins_left = n - i
                        # 目标切片（向上取整），再应用 fill_ratio
                        base = (abs(remaining) + mins_left - 1) // mins_left
                        step = int(round(base * self.fill_ratio)) * sign
                        # 避免过量成交
                        if abs(step) > abs(remaining):
                            step = remaining
                        if step == 0:
                            continue
                        # 执行价：本分钟 vwap 优先，否则 close
                        row = mbars.iloc[i]
                        ref_px = float(row["vwap"] if "vwap" in row and pd.notna(row["vwap"]) else row["close"])
                        # 复用开盘撮合函数计算滑点与佣金
                        filled, exec_px, cash_delta = self._fill_at_open(s, d, step, ref_px)
                        if filled == 0:
                            continue
                        pos = self._ensure_position(pf, s)
                        if filled > 0:
                            new_shares = pos.shares + filled
                            pos.avg_price = (pos.avg_price * pos.shares + exec_px * abs(filled)) / max(new_shares, 1)
                            pos.shares = new_shares
                            pf.cash -= abs(cash_delta)
                            pos.last_trade_date = d
                            if pos.shares > 0 and (pos.holding_days == 0 or pos.holding_days < 0):
                                pos.holding_days = 0
                        elif filled < 0:
                            sell_qty = min(abs(filled), pos.shares)
                            pos.shares -= sell_qty
                            pf.cash += abs(cash_delta)
                            pos.last_trade_date = d
                            if pos.shares == 0:
                                pos.holding_days = 0
                        remaining -= filled
                        trade_rows.append({
                            "ts": row["timestamp"],
                            "symbol": s,
                            "side": "buy" if filled > 0 else "sell",
                            "qty": abs(filled),
                            "exec_price": exec_px,
                            "open_price": ref_px,
                            "commission_bps": self.commission_bps,
                            "fill_ratio": self.fill_ratio,
                        })
            else:
                for od in orders:
                    s = od["symbol"]
                    side = od.get("side", "buy")
                    raw_qty = int(od.get("qty", 0))
                    qty = raw_qty if side == "buy" else -abs(raw_qty)
                    if qty == 0 or s not in open_map:
                        continue
                    exec_ref_px = open_map[s]
                    pos = self._ensure_position(pf, s)
                    # 风控：冷却与最小持有天数
                    if side == "buy":
                        if pos.shares == 0 and self.cooldown_days > 0 and pos.last_trade_date is not None:
                            if (d - pos.last_trade_date).days < self.cooldown_days:
                                continue
                    else:  # sell
                        if self.min_holding_days > 0 and pos.shares > 0 and pos.holding_days < self.min_holding_days:
                            continue

                    filled, exec_px, cash_delta = self._fill_at_open(s, d, qty, exec_ref_px)
                    if filled == 0:
                        continue

                    if filled > 0:
                        new_shares = pos.shares + filled
                        pos.avg_price = (pos.avg_price * pos.shares + exec_px * abs(filled)) / max(new_shares, 1)
                        pos.shares = new_shares
                        pf.cash -= abs(cash_delta)
                        pos.last_trade_date = d
                        if pos.shares > 0 and (pos.holding_days == 0 or pos.holding_days < 0):
                            pos.holding_days = 0
                    elif filled < 0:
                        sell_qty = min(abs(filled), pos.shares)
                        pos.shares -= sell_qty
                        pf.cash += abs(cash_delta)
                        pos.last_trade_date = d
                        if pos.shares == 0:
                            pos.holding_days = 0

                    trade_rows.append({
                        "ts": d,
                        "symbol": s,
                        "side": "buy" if filled > 0 else "sell",
                        "qty": abs(filled),
                        "exec_price": exec_px,
                        "open_price": exec_ref_px,
                        "commission_bps": self.commission_bps,
                        "fill_ratio": self.fill_ratio,
                    })

            # 估值：用日线收盘或当日最后一分钟收盘估值，并更新持有天数
            mark_to_market = 0.0
            for s, pos in pf.positions.items():
                px = mark_map.get(s, pos.avg_price)
                mark_to_market += pos.shares * px
                if pos.shares > 0:
                    pos.holding_days += 1
            nav.append({"date": d, "nav": (pf.cash + mark_to_market) / float(self.cfg.get("backtest", {}).get("cash", 1_000_000))})

        if not nav:
            nav_df = pd.Series(dtype=float, name="nav")
            trades = pd.DataFrame(trade_rows)
            metrics = evaluate(nav_df, trades)
            return {"nav": nav_df, "trades": trades, "metrics": metrics}

        nav_df = pd.DataFrame(nav).set_index("date")["nav"].astype(float)
        trades = pd.DataFrame(trade_rows)
        metrics = evaluate(nav_df, trades)
        return {"nav": nav_df, "trades": trades, "metrics": metrics} 