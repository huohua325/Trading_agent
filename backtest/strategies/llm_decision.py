"""
LLM 决策策略（回测阶段使用）

该模块提供一个基于 LLM 的交易策略 `Strategy`，在回测时按日（或分钟聚合）被回测引擎调用：
- 正常模式：构建因子/新闻/财务等特征，调用 LLM 分析与决策，生成买卖订单
- 回放模式（replay）：从审计目录（audit_dir）读取历史订单，按原始日期直接回放

设计目标：
- 与回测引擎通过统一的 `on_bar(ctx)` 接口交互，返回当日订单列表
- 通过配置控制 LLM 是否仅使用缓存（避免真实调用），以及新闻回看窗口、特征窗口等
- 回放模式便于复现实盘/历史的 LLM 决策并做成本/执行假设的敏感性分析
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import pandas as pd

from trading_agent_v2.core.features import build_features
from trading_agent_v2.agents.analyzer_llm import analyze_batch
from trading_agent_v2.agents.decision_llm import decide_batch
from trading_agent_v2.agents.single_agent_llm import decide_batch as single_agent_decide_batch
from trading_agent_v2.core import data_hub


class Strategy:
    """基于 LLM 的回测策略。

    使用方式：由回测引擎在每个回测日调用 `on_bar(ctx)` 获取订单列表；
    若 `replay=True`，则不调用 LLM，而是从审计目录回放订单。

    属性说明：
    - cfg: 策略与回测相关的配置字典
    - replay: 是否启用回放模式（使用审计订单而非实时 LLM 决策）
    - audit_dir: 回放模式下存放审计 `*.jsonl` 的目录路径
    - _cache: 缓存已加载的审计订单，避免重复 IO
    - llm_cache_only: 是否仅使用缓存生成 LLM 结果（避免真实调用）
    - news_lookback_days: 新闻回看窗口天数，用于特征构建
    - page_limit: 新闻条目抓取上限
    - warmup_days: 特征构建所需的历史回看天数（例如均线、财务等）
    - agent_mode: 代理模式，"multi"(多体) 或 "single"(单体)
    """
    def __init__(self, cfg: Dict, replay: bool = False, audit_dir: str | None = None) -> None:
        """初始化策略。

        参数：
        - cfg: 配置字典，包含 risk/news/backtest/llm 等子配置
        - replay: 是否启用回放模式
        - audit_dir: 回放模式下的审计目录（包含 *.jsonl 文件）
        """
        self.cfg = cfg
        self.replay = replay
        self.audit_dir = audit_dir
        self._cache: List[Dict] | None = None
        self.llm_cache_only = bool((cfg or {}).get("llm", {}).get("backtest_cache_only", True))
        self.news_lookback_days = int((cfg or {}).get("news", {}).get("lookback_days", 7))
        self.page_limit = int((cfg or {}).get("news", {}).get("page_limit", 50))
        self.warmup_days = int((cfg or {}).get("backtest", {}).get("warmup_days", 60))
        # 代理模式："multi"(多体) 或 "single"(单体)，默认多体
        self.agent_mode = str((cfg or {}).get("agents", {}).get("mode", "multi")).lower()

    def _load_all_orders(self) -> List[Dict]:
        """加载审计目录中所有订单记录（仅在回放模式下使用）。

        读取 `audit_dir/*.jsonl` 中的每行 JSON 记录，提取出订单并标准化为：
        `{"symbol": str, "side": "buy"|"sell", "qty": int(买为正、卖为负), "date": YYYY-MM-DD}`。

        返回：
        - List[Dict]：按日期聚合前的原始订单列表；若目录不存在或为空，返回空列表
        """
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
                        # 仅保留日期部分作为回测匹配键
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
                        # 忽略单行解析错误，尽量容错
                        continue
        self._cache = out
        return self._cache

    def _build_features_for_day(self, ctx) -> List[Dict]:
        """为当日构建每个标的的特征输入。

        ctx 结构（由回测引擎传入）：
        - date: pd.Timestamp，当日交易日
        - symbols: List[str]，回测标的集合
        - open_map: Dict[str, float]，当日开盘撮合价（或分钟模式下首分钟价）
        - cfg: Dict，配置对象
        - datasets: 数据读取器，提供 `get_day_bars/get_min_bars` 等
        - timespan: 回测粒度（"day" 或 "minute"）。当为 "minute" 时，本函数会加载当日分钟K并交给
          build_features 计算分钟级因子（如 VWAP 偏离等）；否则分钟数据传空，特征退化为日线/新闻/财务等。

        输出：
        - List[Dict]：每个标的一份特征字典，供 LLM 分析/决策使用
        """
        date = ctx["date"]
        symbols = ctx["symbols"]
        open_map = ctx["open_map"]
        cfg = ctx["cfg"]
        # 历史回看窗口（用于技术指标/财务/新闻等特征）
        start = (date - pd.Timedelta(days=self.warmup_days)).strftime("%Y-%m-%d")
        end = date.strftime("%Y-%m-%d")
        gte_news = (date - pd.Timedelta(days=self.news_lookback_days)).strftime("%Y-%m-%d")
        features_list: List[Dict] = []
        for s in symbols:
            # 参考价（与 sizing/risk 一致性）：优先使用引擎提供的统一参考价
            ref_price_map = ctx.get("ref_price_map", {}) or {}
            open_px = open_map.get(s)
            ref_px = ref_price_map.get(s, open_px)
            # K 线、新闻、分红、拆分、财务等原始数据
            bars_day = ctx["datasets"].get_day_bars(s, start, end)
            # 分钟模式下加载当日分钟K，否则传空DataFrame
            bars_min = pd.DataFrame([])
            if str(ctx.get("timespan", "day")).lower() == "minute":
                # 注意：此处按当日取分钟K（end~end）；build_features 将基于分钟末推导 ts_utc 与价格
                bars_min = ctx["datasets"].get_min_bars(s, end, end)
            news_items, _ = data_hub.get_news(s, gte_news, end, limit=self.page_limit)
            dividends = data_hub.get_dividends(s)
            splits = data_hub.get_splits(s)
            financials = data_hub.get_financials(s, timeframe=None, limit=100)
            # 当日快照与策略细节（如新闻聚合方式）
            # 快照策略：
            # - minute：不提供 price/ts（避免与分钟末冲突），由 build_features 用分钟数据推导最新价与时间
            # - day：提供开盘价与当日 00:00:00Z，缺失时 build_features 会回退到日线末收盘价
            if str(ctx.get("timespan", "day")).lower() == "minute":
                snapshot = {"symbol": s}
            else:
                snapshot = {"symbol": s, "price": float(ref_px) if ref_px is not None else None, "ts_utc": f"{end}T00:00:00Z"}
            details = {"ticker": s, "news_agg": cfg.get("news", {}).get("agg", "mean"), "news_trim_alpha": cfg.get("news", {}).get("trim_alpha", 0.1)}
            # 组合状态（当前头寸与仓位占比）
            position = ctx["portfolio"].positions.get(s)
            equity_for_risk = float(ctx.get("equity_for_risk") or 0.0)
            current_value = (position.shares * float(ref_px)) if position and ref_px else 0.0
            current_position_pct = (current_value / equity_for_risk) if equity_for_risk > 0 else 0.0
            # 估算 pnl_pct（基于参考价与均价）
            pnl_pct = 0.0
            try:
                if position and position.avg_price and position.avg_price > 0 and ref_px:
                    pnl_pct = float(ref_px / float(position.avg_price) - 1.0)
            except Exception:
                pnl_pct = 0.0
            holding_days = int(getattr(position, "holding_days", 0) or 0) if position else 0
            position_state = {
                "current_position_pct": max(0.0, min(1.0, current_position_pct)),
                "avg_price": (position.avg_price if position else None),
                "pnl_pct": float(pnl_pct),
                "holding_days": holding_days,
            }
            # 统一构建特征：
            # - minute：bars_min 非空，分钟级特征（如 VWAP 偏离）将被计算；
            # - day：bars_min 为空，分钟级特征退化为 0 或空，不影响日线/新闻/财务等特征。
            fi = build_features(
                bars_min=bars_min,
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
            # 注入市场上下文的风险字段（来自引擎）
            try:
                fi.setdefault("market_ctx", {})["daily_drawdown_pct"] = float(ctx.get("daily_drawdown_pct") or 0.0)
            except Exception:
                pass
            features_list.append(fi)
        return features_list

    def on_bar(self, ctx) -> List[Dict]:
        """策略在每个回测日被调用的主钩子。

        引擎会传入 `ctx`（见 `_build_features_for_day` 的说明）。
        该方法应返回订单列表：`[{"symbol": str, "side": "buy"|"sell", "qty": int}, ...]`。

        行为：
        - 回放模式：从审计目录加载订单，按当日日期筛选并直接返回
        - 正常模式：构建特征，调用 LLM 分析与决策，转化为买卖数量订单
        """
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
        
        # 1) 特征构建
        features_list = self._build_features_for_day(ctx)
        
        # 2) 根据代理模式选择不同的处理路径
        if self.agent_mode == "single":
            # 单体模式：直接调用单体代理
            decisions_map = single_agent_decide_batch(features_list, cfg=self.cfg, enable_llm=True, cache_only=self.llm_cache_only)
        else:
            # 多体模式（默认）：先分析再决策
            analysis_map = analyze_batch(features_list, cfg=self.cfg, enable_llm=True, cache_only=self.llm_cache_only)
            decisions_input = []
            for fi in features_list:
                symbol = fi["symbol"]
                # 使用统一风控：基于分析输出/仓位/风险配置/市场上下文（含 atr_pct/daily_drawdown_pct）
                try:
                    from trading_agent_v2.core.risk_guard import make_limits
                    limits = make_limits(
                        analysis=analysis_map.get(symbol, {}),
                        position_state=fi.get("position_state", {}),
                        risk_cfg=self.cfg.get("risk", {}),
                        market_ctx={**(fi.get("market_ctx", {}) or {}), "atr_pct": (fi.get("tech", {}) or {}).get("atr_pct", fi.get("market_ctx", {}).get("atr_pct", 0.0))},
                    )
                except Exception:
                    limits = {"allowed": ["increase", "hold", "decrease", "close"], "max_pos_pct": float(self.cfg.get("risk", {}).get("max_pos_pct", 0.1))}
                decisions_input.append({"features": fi, "analysis": analysis_map.get(symbol, {}), "limits": limits})
            decisions_map = decide_batch(decisions_input, cfg=self.cfg, enable_llm=True, cache_only=self.llm_cache_only)
        
        # 3) 生成订单
        orders: List[Dict] = []
        pf = ctx["portfolio"]
        ref_price_map = ctx.get("ref_price_map", {}) or {}
        equity_for_sizing = float(ctx.get("equity_for_sizing") or 0.0)
        for fi in features_list:
            s = fi["symbol"]
            decision = decisions_map.get(s, {})
            action = decision.get("action", "hold")
            target = float(decision.get("target_pos_pct", 0.0))
            ref_px = ref_price_map.get(s) or open_map.get(s)
            if ref_px is None or ref_px <= 0:
                continue
            pos = pf.positions.get(s)
            current_value = (pos.shares * float(ref_px)) if pos else 0.0
            # 将 LLM 输出的目标仓位（比例）转换成实际订单数量
            sizing_equity = equity_for_sizing if equity_for_sizing > 0 else float(pf.equity)
            target_value = max(0.0, min(1.0, target)) * sizing_equity
            delta_value = target_value - current_value
            if action in ("increase", "hold") and delta_value > 0:
                qty = int(delta_value / float(ref_px))
                if qty > 0:
                    orders.append({"symbol": s, "side": "buy", "qty": qty})
            elif action in ("decrease", "close") and delta_value < 0:
                qty = int(abs(delta_value) / float(ref_px))
                if pos and pos.shares > 0:
                    qty = min(qty, pos.shares)
                if qty > 0:
                    orders.append({"symbol": s, "side": "sell", "qty": qty})
        return orders 