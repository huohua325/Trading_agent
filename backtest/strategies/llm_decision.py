from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import pandas as pd

from trading_agent_v2.core.features import build_features
from trading_agent_v2.agents.analyzer_llm import analyze_batch
from trading_agent_v2.agents.decision_llm import decide_batch
from trading_agent_v2.core import data_hub


class Strategy:
    def __init__(self, cfg: Dict, replay: bool = False, audit_dir: str | None = None) -> None:
        self.cfg = cfg
        self.replay = replay
        self.audit_dir = audit_dir
        self._cache: List[Dict] | None = None
        self.llm_cache_only = bool((cfg or {}).get("llm", {}).get("backtest_cache_only", True))
        self.news_lookback_days = int((cfg or {}).get("news", {}).get("lookback_days", 7))
        self.page_limit = int((cfg or {}).get("news", {}).get("page_limit", 50))
        self.warmup_days = int((cfg or {}).get("backtest", {}).get("warmup_days", 60))

    def _load_all_orders(self) -> List[Dict]:
        if self._cache is not None:
            return self._cache
        if not self.audit_dir:
            self._cache = []
            return self._cache
        out: List[Dict] = []
        p = Path(self.audit_dir)
        if not p.exists() or not p.is_dir():
            self._cache = []
            return self._cache
        for file in sorted(p.glob("*.jsonl")):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        symbol = rec.get("symbol")
                        ts = rec.get("ts_utc") or rec.get("ts")
                        trade_date = (ts or "")[:10] if isinstance(ts, str) else None
                        for od in rec.get("orders", []) or []:
                            qty = int(od.get("qty", 0))
                            side = od.get("side", "buy")
                            out.append({
                                "symbol": symbol,
                                "side": side,
                                "qty": qty if side == "buy" else -qty,
                                "date": trade_date,
                            })
                    except Exception:
                        continue
        self._cache = out
        return self._cache

    def _build_features_for_day(self, ctx) -> List[Dict]:
        date = ctx["date"]
        symbols = ctx["symbols"]
        open_map = ctx["open_map"]
        cfg = ctx["cfg"]
        # 构建所有标的特征（仅用日线 + 开盘价），分钟线留空
        start = (date - pd.Timedelta(days=self.warmup_days)).strftime("%Y-%m-%d")
        end = date.strftime("%Y-%m-%d")
        gte_news = (date - pd.Timedelta(days=self.news_lookback_days)).strftime("%Y-%m-%d")
        features_list: List[Dict] = []
        for s in symbols:
            open_px = open_map.get(s)
            bars_day = ctx["datasets"].get_day_bars(s, start, end)
            news_items, _ = data_hub.get_news(s, gte_news, end, limit=self.page_limit)
            dividends = data_hub.get_dividends(s)
            splits = data_hub.get_splits(s)
            financials = data_hub.get_financials(s, timeframe=None, limit=100)
            snapshot = {"symbol": s, "price": float(open_px) if open_px is not None else None, "ts_utc": f"{end}T00:00:00Z"}
            details = {"ticker": s, "news_agg": cfg.get("news", {}).get("agg", "mean"), "news_trim_alpha": cfg.get("news", {}).get("trim_alpha", 0.1)}
            position = ctx["portfolio"].positions.get(s)
            current_value = (position.shares * float(open_px)) if position and open_px else 0.0
            current_position_pct = current_value / float(ctx["portfolio"].equity) if float(ctx["portfolio"].equity) > 0 else 0.0
            position_state = {"current_position_pct": max(0.0, min(1.0, current_position_pct)), "avg_price": (position.avg_price if position else None)}
            fi = build_features(
                bars_min=pd.DataFrame([]),
                bars_day=bars_day,
                indicators={},
                snapshot=snapshot,
                news_items=news_items,
                dividends=dividends,
                splits=splits,
                financials=financials,
                details=details,
                position_state=position_state,
            )
            features_list.append(fi)
        return features_list

    def on_bar(self, ctx) -> List[Dict]:
        if self.replay:
            all_orders = self._load_all_orders()
            dstr = ctx["date"].strftime("%Y-%m-%d")
            return [
                {"symbol": od["symbol"], "side": od["side"], "qty": abs(int(od["qty"]))}
                for od in all_orders if od.get("date") == dstr
            ]
        # 非回放：调用 LLM（默认 cache_only）生成决策
        open_map = ctx["open_map"]
        if not open_map:
            return []
        features_list = self._build_features_for_day(ctx)
        analysis_map = analyze_batch(features_list, cfg=self.cfg, enable_llm=True, cache_only=self.llm_cache_only)
        decisions_input = []
        for fi in features_list:
            symbol = fi["symbol"]
            limits = {"allowed": ["increase", "hold", "decrease", "close"], "max_pos_pct": float(self.cfg.get("risk", {}).get("max_pos_pct", 0.1))}
            decisions_input.append({"features": fi, "analysis": analysis_map.get(symbol, {}), "limits": limits})
        decisions_map = decide_batch(decisions_input, cfg=self.cfg, enable_llm=True, cache_only=self.llm_cache_only)
        orders: List[Dict] = []
        pf = ctx["portfolio"]
        for fi in features_list:
            s = fi["symbol"]
            decision = decisions_map.get(s, {})
            action = decision.get("action", "hold")
            target = float(decision.get("target_pos_pct", 0.0))
            open_px = open_map.get(s)
            if open_px is None or open_px <= 0:
                continue
            pos = pf.positions.get(s)
            current_value = (pos.shares * float(open_px)) if pos else 0.0
            target_value = max(0.0, min(1.0, target)) * float(pf.equity)
            delta_value = target_value - current_value
            if action in ("increase", "hold") and delta_value > 0:
                qty = int(delta_value / float(open_px))
                if qty > 0:
                    orders.append({"symbol": s, "side": "buy", "qty": qty})
            elif action in ("decrease", "close") and delta_value < 0:
                qty = int(abs(delta_value) / float(open_px))
                if pos and pos.shares > 0:
                    qty = min(qty, pos.shares)
                if qty > 0:
                    orders.append({"symbol": s, "side": "sell", "qty": qty})
        return orders 