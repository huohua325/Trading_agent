from __future__ import annotations

from typing import Dict, List

import math
import numpy as np
import pandas as pd
from trading_agent_v2.core.schemas import FeatureInput, TechSnapshot, NewsSnapshot, PositionState


def _safe_tail_float(series: pd.Series, default: float) -> float:
    try:
        if series is None or len(series) == 0:
            return default
        val = series.iloc[-1]
        return float(val) if pd.notna(val) else default
    except Exception:
        return default


def _compute_returns(close: pd.Series, windows: List[int]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if close is None or len(close) == 0:
        for w in windows:
            results[f"{w}d"] = 0.0
        return results
    try:
        for w in windows:
            if len(close) > w:
                prev = float(close.iloc[-(w + 1)])
                curr = float(close.iloc[-1])
                ret = (curr / prev - 1.0) if prev and not math.isclose(prev, 0.0) else 0.0
                # 夹逼到 [-1, 1]
                results[f"{w}d"] = float(max(-1.0, min(1.0, ret)))
            else:
                results[f"{w}d"] = 0.0
    except Exception:
        for w in windows:
            results[f"{w}d"] = 0.0
    return results


def _ema(series: pd.Series, span: int) -> pd.Series:
    try:
        return series.ewm(span=span, adjust=False).mean()
    except Exception:
        return pd.Series([], dtype=float)


def _compute_atr_pct(day_df: pd.DataFrame, window: int = 14) -> float:
    try:
        if day_df is None or day_df.empty:
            return 0.0
        df = day_df.copy()
        if not set(["high", "low", "close"]).issubset(df.columns):
            return 0.0
        df = df.sort_values("date").reset_index(drop=True)
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = _ema(tr, span=window)
        last_close = _safe_tail_float(df["close"], 0.0)
        if last_close <= 0:
            return 0.0
        atr_pct = float(atr.iloc[-1] / last_close) if len(atr) > 0 and pd.notna(atr.iloc[-1]) else 0.0
        # 限制到 [0,1]
        return max(0.0, min(1.0, atr_pct))
    except Exception:
        return 0.0


def _compute_trend(day_df: pd.DataFrame) -> str:
    try:
        if day_df is None or day_df.empty or "close" not in day_df.columns:
            return "sideways"
        df = day_df.sort_values("date").reset_index(drop=True)
        close = df["close"].astype(float)
        sma20 = close.rolling(window=20, min_periods=1).mean()
        sma50 = close.rolling(window=50, min_periods=1).mean()
        cond_up = len(df) >= 50 and sma20.iloc[-1] > sma50.iloc[-1] and (sma50.iloc[-1] - sma50.iloc[-2] if len(sma50) > 1 else 0.0) > 0
        cond_down = len(df) >= 50 and sma20.iloc[-1] < sma50.iloc[-1] and (sma50.iloc[-1] - sma50.iloc[-2] if len(sma50) > 1 else 0.0) < 0
        if cond_up:
            return "up"
        if cond_down:
            return "down"
        return "sideways"
    except Exception:
        return "sideways"


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _compute_momentum(day_df: pd.DataFrame) -> float:
    """以 20 日收益的 zscore 近似为动量打分，回映射到 [0,1]。
    数据不足时退化到 0.5。
    """
    try:
        if day_df is None or day_df.empty or "close" not in day_df.columns:
            return 0.5
        df = day_df.sort_values("date").reset_index(drop=True)
        close = df["close"].astype(float)
        # 计算 20 日滚动收益率
        if len(close) <= 21:
            ret_20 = 0.0
        else:
            ret_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        # 用最近 252 日的日收益估 zscore（近似），不足则用缩放 logistic
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) >= 60 and daily_ret.std() > 1e-9:
            z = (ret_20 - daily_ret.mean()) / (daily_ret.std() * math.sqrt(20))
            score = _sigmoid(z)  # 将 z 映射到 [0,1]
        else:
            score = _sigmoid(5.0 * ret_20)  # 简化：对 20 日收益做缩放 logistic
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.5


def _compute_vwap_dev(min_df: pd.DataFrame, last_price: float, window: int = 30) -> float:
    try:
        if min_df is None or min_df.empty or last_price is None:
            return 0.0
        df = min_df.sort_values("timestamp").reset_index(drop=True)
        tail = df.tail(window)
        if tail.empty:
            return 0.0
        # 以分钟收盘的波动作为标准化分母，优先使用 vwap 均值
        close_tail = tail["close"].astype(float) if "close" in tail.columns else pd.Series([], dtype=float)
        vwap_tail = tail["vwap"].astype(float) if "vwap" in tail.columns else None
        denom = float(close_tail.std()) if len(close_tail) > 1 else 0.0
        mean_vwap = float(vwap_tail.mean()) if vwap_tail is not None and len(vwap_tail) > 0 else float(close_tail.mean()) if len(close_tail) > 0 else 0.0
        if denom <= 1e-9:
            return 0.0
        dev = (float(last_price) - mean_vwap) / denom
        # 限幅，避免极端值
        return float(max(-5.0, min(5.0, dev)))
    except Exception:
        return 0.0


def _compute_news_snapshot(news_items: List[Dict], now_iso_utc: str, agg: str = "mean", trim_alpha: float = 0.1) -> NewsSnapshot:
    try:
        if not news_items:
            return NewsSnapshot(sentiment=0.0, top_k_events=[], src_count=0, freshness_min=0)
        sentiments: List[float] = []
        titles: List[str] = []
        latest_ts = None
        for it in news_items:
            s = it.get("sentiment")
            if isinstance(s, (int, float)) and not math.isnan(float(s)):
                sentiments.append(float(s))
            title = it.get("title")
            if isinstance(title, str) and title:
                titles.append(title)
            p = it.get("published_utc") or it.get("published_date")
            try:
                dt = pd.to_datetime(p, utc=True, errors="coerce")
                if pd.notna(dt):
                    latest_ts = dt if latest_ts is None or dt > latest_ts else latest_ts
            except Exception:
                pass
        # 聚合策略
        if sentiments:
            if agg == "median":
                import numpy as _np
                sent = float(_np.median(sentiments))
            elif agg == "trimmed_mean":
                import numpy as _np
                a = max(0.0, min(0.49, float(trim_alpha)))
                arr = sorted(sentiments)
                k = int(len(arr) * a)
                core = arr[k: len(arr) - k] if len(arr) - 2 * k > 0 else arr
                sent = float(_np.mean(core)) if core else 0.0
            else:
                import numpy as _np
                sent = float(_np.mean(sentiments))
            sent = float(max(-1.0, min(1.0, sent)))
        else:
            sent = 0.0
        top_k = titles[:5]
        # freshness_min：now - latest_news
        freshness_min = 0
        try:
            if latest_ts is not None:
                now_dt = pd.to_datetime(now_iso_utc, utc=True, errors="coerce")
                if pd.notna(now_dt):
                    freshness_min = int((now_dt - latest_ts).total_seconds() // 60)
        except Exception:
            freshness_min = 0
        return NewsSnapshot(sentiment=sent, top_k_events=top_k, src_count=len(news_items), freshness_min=freshness_min)
    except Exception:
        return NewsSnapshot()


def _compute_vol_zscore(day_df: pd.DataFrame, window: int = 20) -> float:
    try:
        if day_df is None or day_df.empty or "volume" not in day_df.columns:
            return 0.0
        vol = day_df.sort_values("date")["volume"].astype(float)
        if len(vol) < window + 1:
            return 0.0
        tail = vol.iloc[-window:]
        mu, sigma = float(tail.mean()), float(tail.std(ddof=0))
        if sigma <= 1e-9:
            return 0.0
        z = float((vol.iloc[-1] - mu) / sigma)
        return max(-5.0, min(5.0, z))
    except Exception:
        return 0.0


def _compute_realized_vol(day_df: pd.DataFrame, window: int = 20) -> float:
    try:
        if day_df is None or day_df.empty or "close" not in day_df.columns:
            return 0.0
        close = day_df.sort_values("date")["close"].astype(float)
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) < window:
            return float(daily_ret.std()) if len(daily_ret) > 1 else 0.0
        rv = float(daily_ret.iloc[-window:].std())
        return max(0.0, min(1.0, rv))  # 简单夹逼
    except Exception:
        return 0.0


def _compute_gap_1d(day_df: pd.DataFrame) -> float:
    try:
        if day_df is None or day_df.empty or not set(["open", "close"]).issubset(day_df.columns):
            return 0.0
        df = day_df.sort_values("date").reset_index(drop=True)
        if len(df) < 2:
            return 0.0
        prev_close = float(df.iloc[-2]["close"])
        today_open = float(df.iloc[-1]["open"])
        if prev_close == 0:
            return 0.0
        gap = (today_open / prev_close) - 1.0
        return float(max(-1.0, min(1.0, gap)))
    except Exception:
        return 0.0


# ========= 财报摘要 =========

def _get_nested(d: Dict, path: List[str]) -> float | None:
    try:
        cur = d
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        if isinstance(cur, (int, float)) and not math.isnan(float(cur)):
            return float(cur)
    except Exception:
        return None
    return None


def _safe_div(a: float | None, b: float | None) -> float:
    try:
        a = float(a) if a is not None else 0.0
        b = float(b) if b is not None else 0.0
        if math.isclose(b, 0.0):
            return 0.0
        return float(a / b)
    except Exception:
        return 0.0


def _compute_financials_summary(financials: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "fin_rev_yoy": 0.0,
        "fin_eps_yoy": 0.0,
        "fin_gross_margin": 0.0,
        "fin_op_margin": 0.0,
        "fin_net_margin": 0.0,
        "fin_fcf_margin": 0.0,
        "fin_debt_to_equity": 0.0,
        "fin_shares_dilution_1y": 0.0,
    }
    try:
        if not financials:
            return out
        # 排序：尽量按 end_date 或 filing_date 降序
        def _key(it: Dict) -> str:
            for k in ["end_date", "filing_date", "start_date"]:
                v = it.get(k)
                if v:
                    return str(v)
            return ""
        items = sorted(financials, key=_key, reverse=True)

        def _timeframe(it: Dict) -> str:
            tf = (it.get("timeframe") or "").lower()
            fp = str(it.get("fiscal_period") or it.get("period") or "").upper()
            if tf:
                return tf
            if fp.startswith("Q"):
                return "quarterly"
            if fp in ("FY", "ANNUAL"):
                return "annual"
            return ""

        # 选择当前记录：优先季度 → 年度 → 其它（如 TTM）
        cur_rec = next((it for it in items if _timeframe(it) == "quarterly"), None)
        if cur_rec is None:
            cur_rec = next((it for it in items if _timeframe(it) == "annual"), None)
        if cur_rec is None:
            cur_rec = items[0]

        cur_fin = cur_rec.get("financials") or {}
        cur_period = str(cur_rec.get("fiscal_period") or cur_rec.get("period") or "").upper()
        cur_year = None
        try:
            fy = cur_rec.get("fiscal_year") or cur_rec.get("year")
            cur_year = int(str(fy)) if fy not in (None, "") else None
        except Exception:
            cur_year = None

        # 找到去年同期（同比基准）
        prev_fin = None
        if cur_year is not None and cur_period:
            for it in items:
                try:
                    it_year_raw = it.get("fiscal_year") or it.get("year")
                    it_year = int(str(it_year_raw)) if it_year_raw not in (None, "") else None
                except Exception:
                    it_year = None
                it_period = str(it.get("fiscal_period") or it.get("period") or "").upper()
                if it_year is not None and it_year == cur_year - 1 and it_period == cur_period:
                    prev_fin = it.get("financials") or {}
                    break

        # 提取关键科目（多路径兼容）
        rev_cur = (
            _get_nested(cur_fin, ["income_statement", "revenues", "value"]) or
            _get_nested(cur_fin, ["income_statement", "revenue", "value"]) or
            _get_nested(cur_fin, ["income_statement", "total_revenue", "value"]) or 0.0
        )
        rev_prev = (
            _get_nested(prev_fin, ["income_statement", "revenues", "value"]) if isinstance(prev_fin, dict) else None
        ) or 0.0

        gross_profit = (
            _get_nested(cur_fin, ["income_statement", "gross_profit", "value"]) or 0.0
        )
        operating_income = (
            _get_nested(cur_fin, ["income_statement", "operating_income", "value"]) or
            _get_nested(cur_fin, ["income_statement", "operating_income_loss", "value"]) or 0.0
        )
        net_income = (
            _get_nested(cur_fin, ["income_statement", "net_income", "value"]) or
            _get_nested(cur_fin, ["income_statement", "net_income_loss", "value"]) or
            _get_nested(cur_fin, ["income_statement", "net_income_loss_attributable_to_parent", "value"]) or 0.0
        )

        basic_eps = (
            _get_nested(cur_fin, ["income_statement", "basic_earnings_per_share", "value"]) or
            _get_nested(cur_fin, ["income_statement", "basic_eps", "value"]) or 0.0
        )
        basic_eps_prev = (
            _get_nested(prev_fin, ["income_statement", "basic_earnings_per_share", "value"]) if isinstance(prev_fin, dict) else None
        ) or 0.0

        # 现金流/自由现金流（近似）
        cash_ops = (
            _get_nested(cur_fin, ["cash_flow_statement", "net_cash_flow_from_operating_activities", "value"]) or
            _get_nested(cur_fin, ["cash_flow_statement", "net_cash_flow_from_operating_activities_continuing", "value"]) or 0.0
        )
        capex = (
            _get_nested(cur_fin, ["cash_flow_statement", "net_cash_flow_from_investing_activities", "value"]) or
            _get_nested(cur_fin, ["cash_flow_statement", "capital_expenditures", "value"]) or 0.0
        )
        fcf = float(cash_ops - abs(capex))

        # 负债与权益
        total_debt = (
            _get_nested(cur_fin, ["balance_sheet", "total_debt", "value"]) or
            _get_nested(cur_fin, ["balance_sheet", "long_term_debt", "value"]) or 0.0
        )
        equity = (
            _get_nested(cur_fin, ["balance_sheet", "shareholders_equity", "value"]) or
            _get_nested(cur_fin, ["balance_sheet", "total_shareholders_equity", "value"]) or
            _get_nested(cur_fin, ["balance_sheet", "equity", "value"]) or
            _get_nested(cur_fin, ["balance_sheet", "equity_attributable_to_parent", "value"]) or 0.0
        )

        # 股本（同比）
        shares_basic = (
            _get_nested(cur_fin, ["income_statement", "basic_shares_outstanding", "value"]) or
            _get_nested(cur_fin, ["income_statement", "shares_basic", "value"]) or
            _get_nested(cur_fin, ["income_statement", "basic_average_shares", "value"]) or 0.0
        )
        shares_basic_prev = (
            _get_nested(prev_fin, ["income_statement", "basic_shares_outstanding", "value"]) if isinstance(prev_fin, dict) else None
        ) or (
            _get_nested(prev_fin, ["income_statement", "basic_average_shares", "value"]) if isinstance(prev_fin, dict) else None
        ) or 0.0

        # 计算比率
        out["fin_gross_margin"] = float(max(-1.0, min(1.0, _safe_div(gross_profit, rev_cur))))
        out["fin_op_margin"] = float(max(-1.0, min(1.0, _safe_div(operating_income, rev_cur))))
        out["fin_net_margin"] = float(max(-1.0, min(1.0, _safe_div(net_income, rev_cur))))
        out["fin_fcf_margin"] = float(max(-2.0, min(2.0, _safe_div(fcf, rev_cur))))
        out["fin_debt_to_equity"] = float(max(0.0, min(10.0, _safe_div(total_debt, equity))))

        # 同比
        if rev_prev and rev_prev != 0:
            out["fin_rev_yoy"] = float(max(-1.0, min(1.0, (rev_cur / rev_prev - 1.0))))
        if basic_eps_prev and basic_eps_prev != 0:
            out["fin_eps_yoy"] = float(max(-1.0, min(1.0, (basic_eps / basic_eps_prev - 1.0))))

        # 股本稀释（同比）
        if shares_basic_prev and shares_basic_prev != 0:
            out["fin_shares_dilution_1y"] = float(max(-1.0, min(1.0, (shares_basic / shares_basic_prev - 1.0))))
    except Exception:
        pass
    return out


def _compute_fund_snapshot(dividends: pd.DataFrame, splits: pd.DataFrame, last_price: float, now_iso_utc: str) -> Dict[str, float]:
    """最小可用的基本面快照：
    - has_dividend_30d: 过去30天是否有分红（0/1）
    - dividend_cash_last: 最近一次分红现金金额
    - dividend_yield_last: 最近一次分红金额/last_price（若可得）
    - has_split_180d: 过去180天是否有拆分（0/1）
    所有字段均返回 float，确保与 schema 兼容。
    """
    out: Dict[str, float] = {
        "has_dividend_30d": 0.0,
        "dividend_cash_last": 0.0,
        "dividend_yield_last": 0.0,
        "has_split_180d": 0.0,
    }
    try:
        now_dt = pd.to_datetime(now_iso_utc, utc=True, errors="coerce")
        if pd.isna(now_dt):
            now_dt = pd.Timestamp.utcnow().tz_localize("UTC")
    except Exception:
        now_dt = pd.Timestamp.utcnow().tz_localize("UTC")

    # 处理分红
    try:
        if isinstance(dividends, pd.DataFrame) and not dividends.empty:
            df = dividends.copy()
            # 兼容常见字段：ex_dividend_date 或 pay_date 或 declared_date
            for col in ["ex_dividend_date", "pay_date", "payment_date", "declaration_date", "declared_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            # 近30天有无分红
            date_cols = [c for c in ["ex_dividend_date", "pay_date", "payment_date", "declaration_date", "declared_date"] if c in df.columns]
            recent = False
            if date_cols:
                last_30 = now_dt - pd.Timedelta(days=30)
                for c in date_cols:
                    recent = recent or bool(((df[c] >= last_30) & (df[c] <= now_dt)).fillna(False).any())
            out["has_dividend_30d"] = 1.0 if recent else 0.0
            # 最近一次现金分红金额
            cash_col = None
            for c in ["cash_amount", "cash", "amount"]:
                if c in df.columns:
                    cash_col = c
                    break
            if cash_col:
                # 取日期最大的一行
                if date_cols:
                    df = df.sort_values(date_cols[0]).dropna(subset=[date_cols[0]])
                cash_last = float(df[cash_col].iloc[-1]) if len(df) > 0 and pd.notna(df[cash_col].iloc[-1]) else 0.0
                out["dividend_cash_last"] = cash_last
                if last_price and last_price > 0:
                    out["dividend_yield_last"] = float(max(0.0, min(1.0, cash_last / last_price)))
    except Exception:
        pass

    # 处理拆分
    try:
        if isinstance(splits, pd.DataFrame) and not splits.empty:
            df = splits.copy()
            for col in ["execution_date", "effective_date", "declared_date", "date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            date_cols = [c for c in ["execution_date", "effective_date", "declared_date", "date"] if c in df.columns]
            if date_cols:
                last_180 = now_iso_utc if isinstance(now_iso_utc, pd.Timestamp) else pd.Timestamp.utcnow().tz_localize("UTC")
                last_180 = pd.to_datetime(now_iso_utc, utc=True, errors="coerce") if not isinstance(now_iso_utc, pd.Timestamp) else now_iso_utc
                last_180 = last_180 - pd.Timedelta(days=180)
                recent_split = False
                for c in date_cols:
                    recent_split = recent_split or bool(((df[c] >= last_180) & (df[c] <= (pd.to_datetime(now_iso_utc, utc=True, errors="coerce") if not isinstance(now_iso_utc, pd.Timestamp) else now_iso_utc))).fillna(False).any())
                out["has_split_180d"] = 1.0 if recent_split else 0.0
    except Exception:
        pass

    return out


def build_features(
    bars_min: pd.DataFrame,
    bars_day: pd.DataFrame,
    indicators: Dict[str, pd.DataFrame],
    snapshot: Dict,
    news_items: List[Dict],
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    financials: List[Dict],
    details: Dict,
    position_state: Dict,
) -> Dict:
    # 统一与排序
    day_df = bars_day.copy() if isinstance(bars_day, pd.DataFrame) else pd.DataFrame([])
    min_df = bars_min.copy() if isinstance(bars_min, pd.DataFrame) else pd.DataFrame([])
    if not day_df.empty and "date" in day_df.columns:
        day_df = day_df.sort_values("date").reset_index(drop=True)
    if not min_df.empty and "timestamp" in min_df.columns:
        min_df = min_df.sort_values("timestamp").reset_index(drop=True)

    # 确定 symbol 与时间/价格
    symbol = (details.get("ticker") if isinstance(details, dict) else None) or (snapshot.get("symbol") if isinstance(snapshot, dict) else None) or "UNKNOWN"
    # ts_utc 优先：snapshot.ts_utc → 分钟末 → 日线末（当日 00:00:00Z）
    ts_utc = None
    if isinstance(snapshot, dict):
        ts_utc = snapshot.get("ts_utc")
    if not ts_utc or ts_utc.startswith("1970-"):
        if not min_df.empty and "timestamp" in min_df.columns:
            last_ts = min_df["timestamp"].iloc[-1]
            try:
                ts_utc = pd.Timestamp(last_ts).tz_localize('UTC').strftime("%Y-%m-%dT%H:%M:%SZ") if pd.Timestamp(last_ts).tzinfo else pd.Timestamp(last_ts).tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                ts_utc = pd.Timestamp(last_ts).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif not day_df.empty and "date" in day_df.columns:
            last_d = day_df["date"].iloc[-1]
            try:
                d = pd.to_datetime(str(last_d)).strftime("%Y-%m-%dT00:00:00Z")
                ts_utc = d
            except Exception:
                ts_utc = "1970-01-01T00:00:00Z"
    if not ts_utc:
        ts_utc = "1970-01-01T00:00:00Z"

    # price 优先：snapshot.price → 分钟末 close → 日线末 close
    price = None
    if isinstance(snapshot, dict):
        price = snapshot.get("price")
    if price is None or (isinstance(price, float) and math.isnan(price)):
        if not min_df.empty and "close" in min_df.columns:
            price = _safe_tail_float(min_df["close"], None)
        elif not day_df.empty and "close" in day_df.columns:
            price = _safe_tail_float(day_df["close"], None)

    # 技术指标
    close_series = day_df["close"] if "close" in day_df.columns else pd.Series([], dtype=float)
    rets = _compute_returns(close_series, [1, 5, 20])
    atr_pct = _compute_atr_pct(day_df, window=14)
    trend = _compute_trend(day_df)
    mom = _compute_momentum(day_df)
    vwap_dev = _compute_vwap_dev(min_df, price, window=30)

    tech = TechSnapshot(
        ret=rets,
        atr_pct=atr_pct,
        trend=trend,
        mom=mom,
        vwap_dev=vwap_dev,
    )

    # 补充特征（落入 market_ctx）
    vol_z = _compute_vol_zscore(day_df, window=20)
    rv_20 = _compute_realized_vol(day_df, window=20)
    gap_1d = _compute_gap_1d(day_df)

    # 新闻情绪快照
    news = _compute_news_snapshot(news_items, ts_utc, agg=(details.get("news_agg") if isinstance(details, dict) else "mean") or "mean", trim_alpha=float((details.get("news_trim_alpha") if isinstance(details, dict) else 0.1) or 0.1))

    # 基本面最小快照 + 财报摘要
    fund = _compute_fund_snapshot(dividends, splits, float(price) if price is not None else 0.0, ts_utc)
    try:
        fin_summary = _compute_financials_summary(financials)
        fund.update(fin_summary)
    except Exception:
        pass

    # 事件窗：使用 Polygon 事件日历判断是否处于财报窗（±3天）
    from trading_agent_v2.core import data_hub as _dh
    try:
        in_earnings_window = bool(_dh.is_in_earnings_window(symbol, ts_utc, window_days=3))
    except Exception:
        in_earnings_window = False

    # 仓位状态
    pos = PositionState(**position_state) if isinstance(position_state, dict) and position_state is not None else PositionState()

    fi = FeatureInput(
        symbol=symbol,
        ts_utc=ts_utc,
        tech=tech,
        news=news,
        fund=fund,
        market_ctx={
            "last_price": float(price) if price is not None else None,
            "vol_zscore_20d": float(vol_z),
            "realized_vol_20d": float(rv_20),
            "gap_1d": float(gap_1d),
            "atr_pct": float(atr_pct),
            "in_earnings_window": bool(in_earnings_window),
        },
        position_state=pos,
    )
    return fi.model_dump() 