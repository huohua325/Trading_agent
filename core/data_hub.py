from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from trading_agent_v2.adapters.polygon_client import PolygonClient
from trading_agent_v2.utils.io import (
    ensure_dir,
    write_parquet_idempotent,
    atomic_append_jsonl,
)
from trading_agent_v2.llm.llm_client import LLMClient, LLMConfig
from trading_agent_v2.agents.news_sentiment_llm import score_news_batch


# 计算项目根目录（绝对路径）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKTEST_DIR = os.path.join(_PROJECT_ROOT, "backtest_data")
_STORAGE_BASE = os.path.join(_PROJECT_ROOT, "trading_agent_v2", "storage")
_PARQUET_BASE = os.path.join(_STORAGE_BASE, "parquet")
_CACHE_BASE = os.path.join(_STORAGE_BASE, "cache")
_REPORT_BASE = os.path.join(_STORAGE_BASE, "reports")

_polygon_client = PolygonClient(os.getenv("POLYGON_API_KEY", ""))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _empty_bars_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "open", "high", "low", "close", "volume", "vwap"
    ])


def _normalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # 确保 timestamp 无时区，按时间排序
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _normalize_day_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _log_quality_issue(kind: str, symbol: str, payload: Dict[str, object]) -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = os.path.join(_REPORT_BASE, "quality", date_str)
    ensure_dir(out_dir)
    rec = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": kind,
        "symbol": symbol,
        **payload,
    }
    atomic_append_jsonl(os.path.join(out_dir, f"{symbol}.jsonl"), rec)


def _detect_duplicates(df: pd.DataFrame, key: str) -> int:
    if df.empty or key not in df.columns:
        return 0
    dup = int(df.duplicated(subset=[key]).sum())
    return dup


def _detect_minute_sparsity(df: pd.DataFrame) -> bool:
    # 经验阈值：若单日分钟数 < 100，认为稀疏（可能缺页或非交易日）
    return len(df) > 0 and len(df) < 100


def _read_local_day_csv(symbol: str) -> pd.DataFrame:
    processed = os.path.join(_BACKTEST_DIR, f"{symbol}_prices_processed.csv")
    raw = os.path.join(_BACKTEST_DIR, f"{symbol}_prices.csv")
    path = processed if os.path.exists(processed) else raw
    if not os.path.exists(path):
        return _empty_bars_df()

    df = pd.read_csv(path)
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    # 处理日期时区：去时区，转为日期列
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["date"] = df["date"].dt.date
    if "vwap" not in df.columns:
        df["vwap"] = df["close"]
    df = df[["date", "open", "high", "low", "close", "volume", "vwap"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _write_partitioned_parquet(df: pd.DataFrame, symbol: str, granularity: str) -> None:
    if df.empty:
        return
    base_dir = os.path.join(_PARQUET_BASE, symbol, granularity)
    ensure_dir(base_dir)
    # 日线与分钟都按“日期”分片写入（分钟分片文件名不包含时间，避免产生海量小文件与非法字符）
    key_col = "date"
    if key_col not in df.columns:
        return
    # 规范化与质量检查
    df = _normalize_day_df(df) if granularity == "day" else _normalize_minute_df(df)
    for key, g in df.groupby(key_col):
        fname = os.path.join(base_dir, f"{str(key)}.parquet")
        # 质量检测：重复/稀疏
        dups = _detect_duplicates(g, key_col)
        if dups > 0:
            _log_quality_issue("duplicate_rows", symbol, {"granularity": granularity, "key": str(key), "duplicates": dups})
        if granularity == "minute" and _detect_minute_sparsity(g):
            _log_quality_issue("sparse_minute_bars", symbol, {"key": str(key), "num_rows": int(len(g))})
        try:
            # 按内容哈希进行幂等写（atomic replace）
            write_parquet_idempotent(g, fname, sort_by=[c for c in g.columns if c != key_col])
        except Exception as exc:  # pragma: no cover
            logger.warning(f"write parquet failed: {fname}: {exc}")


def _filter_day_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    s = pd.to_datetime(start).date() if start else df["date"].min()
    e = pd.to_datetime(end).date() if end else df["date"].max()
    m = (df["date"] >= s) & (df["date"] <= e)
    out = df.loc[m].reset_index(drop=True)
    # 粗略缺口检测：相邻日期跨度>5天时记录（不精确剔除周末/节假日，仅提示）
    if len(out) >= 2:
        diffs = pd.Series(out["date"]).sort_values().diff().dt.days.fillna(0)
        if (diffs > 5).any():
            _log_quality_issue("large_day_gap", out.iloc[0 if len(out)==0 else 0]["symbol"] if "symbol" in out.columns else "UNKNOWN", {"max_gap_days": int(diffs.max())})
    return out


def _read_parquet_range(symbol: str, granularity: str, start: str, end: str) -> pd.DataFrame:
    try:
        base_dir = os.path.join(_PARQUET_BASE, symbol, granularity)
        if not os.path.isdir(base_dir):
            return pd.DataFrame([])
        files = [f for f in os.listdir(base_dir) if f.endswith('.parquet')]
        if not files:
            return pd.DataFrame([])
        s = pd.to_datetime(start).date() if start else None
        e = pd.to_datetime(end).date() if end else None
        selected: List[str] = []
        for f in files:
            try:
                d = pd.to_datetime(f.replace('.parquet', '')).date()
            except Exception:
                continue
            if (s is None or d >= s) and (e is None or d <= e):
                selected.append(os.path.join(base_dir, f))
        if not selected:
            return pd.DataFrame([])
        dfs = [pd.read_parquet(p) for p in sorted(selected)]
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame([])
        if granularity == "day":
            df = _normalize_day_df(df)
        else:
            df = _normalize_minute_df(df)
        return df
    except Exception as exc:
        logger.warning(f"read parquet range failed: {symbol} {granularity} {start}-{end}: {exc}")
        return pd.DataFrame([])


def _load_cached_news(ticker: str) -> Optional[List[Dict]]:
    cache_dir = os.path.join(_CACHE_BASE, "news")
    _ensure_dir(cache_dir)
    path = os.path.join(cache_dir, f"{ticker}.json")
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    # 24h 以内有效
    if (datetime.now().timestamp() - st.st_mtime) > 24 * 3600:
        # 过期即删除
        try:
            os.remove(path)
        except Exception:
            pass
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 损坏或读取失败：删除以释放空间
        try:
            os.remove(path)
        except Exception:
            pass
        return None


def _dedup_news(items: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    result: List[Dict] = []
    for it in items:
        key = str(it.get("id") or (it.get("title"), it.get("published_utc") or it.get("published_date")))
        if key in seen:
            continue
        seen.add(key)
        result.append(it)
    return result


def _save_cached_news(ticker: str, items: List[Dict]) -> None:
    cache_dir = os.path.join(_CACHE_BASE, "news")
    _ensure_dir(cache_dir)
    path = os.path.join(cache_dir, f"{ticker}.json")
    try:
        deduped = _dedup_news(items)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, ensure_ascii=False)
    except Exception:
        pass


def enrich_news_with_llm_sentiment(items: List[Dict], cfg: Dict | None = None, cache_only: bool = False, batch_size: int = 32, agg: str = "mean", trim_alpha: float = 0.1) -> List[Dict]:
    """使用 LLM 对新闻标题/摘要进行情绪打分，范围 [-1,1]，并写回 items[i]['sentiment']。
    聚合策略（用于上层汇总到 features.news.sentiment 时参考）：
    - mean: 简单均值
    - median: 中位数
    - trimmed_mean: 截尾均值（两侧各剔除 alpha 比例）
    注意：此函数只在 item 级别填充 sentiment；聚合由 features 构建时进行（此处返回 items 以保持兼容）。
    - 若已有数值 sentiment 则保留不改（避免覆盖外部来源）。
    - 失败或不可用时原样返回。
    """
    try:
        if not items:
            return items
        # 仅对缺失 sentiment 的项打分
        need_score_idx: List[int] = []
        batch_items: List[Dict] = []
        for idx, it in enumerate(items):
            s = it.get("sentiment")
            if isinstance(s, (int, float)):
                continue
            title = (it.get("title") or "").strip()
            summary = (it.get("description") or it.get("summary") or it.get("article_body") or "").strip()
            if not title and not summary:
                continue
            need_score_idx.append(idx)
            batch_items.append({"title": title, "summary": summary})
        if not batch_items:
            return items
        scores = score_news_batch(batch_items, cfg=cfg, cache_only=cache_only, batch_size=batch_size)
        for k, v in enumerate(scores):
            try:
                items[need_score_idx[k]]["sentiment"] = float(v)
            except Exception:
                continue
        return items
    except Exception:
        return items

# 获取日线或分钟线数据
def get_bars(ticker: str, start: str, end: str, multiplier: int, timespan: str, adjusted: bool) -> pd.DataFrame:
    try:
        if timespan == "day":
            # 1) 优先读取本地 Parquet 分区
            local = _read_parquet_range(ticker, "day", start, end)
            if not local.empty:
                return _filter_day_by_date(local, start, end)
            # 2) 本地 CSV → 写 Parquet → 过滤
            df = _read_local_day_csv(ticker)
            df = _normalize_day_df(df)
            _write_partitioned_parquet(df, ticker, granularity="day")
            if df.empty:
                # 3) 回退 API
                aggs = _polygon_client.list_aggs(ticker, start, end, multiplier=1, timespan="day", adjusted=adjusted)
                if not aggs:
                    return _empty_bars_df()
                df = pd.DataFrame([{
                    "date": pd.to_datetime(x.get("t"), unit="ms", utc=True).tz_localize(None).date(),
                    "open": x.get("o"),
                    "high": x.get("h"),
                    "low": x.get("l"),
                    "close": x.get("c"),
                    "volume": x.get("v"),
                    "vwap": x.get("vw", x.get("c")),
                } for x in aggs])
                df = _normalize_day_df(df)
                _write_partitioned_parquet(df, ticker, granularity="day")
            out = _filter_day_by_date(df, start, end)
            return out
        elif timespan == "minute":
            # 1) 优先读取本地 Parquet 分区（按日分片）
            local = _read_parquet_range(ticker, "minute", start, end)
            if not local.empty:
                return local
            # 2) 按天切片流式拉取，边拉边落
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"]).astype({"timestamp": "datetime64[ns]"})
            all_rows: List[pd.DataFrame] = []
            days = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="D")
            for d in days:
                d_start = d.strftime("%Y-%m-%d")
                d_end = d.strftime("%Y-%m-%d")
                aggs = _polygon_client.list_aggs(ticker, d_start, d_end, multiplier=multiplier, timespan="minute", adjusted=adjusted)
                if not aggs:
                    continue
                mdf = pd.DataFrame([{
                    "timestamp": pd.to_datetime(x.get("t"), unit="ms", utc=True).tz_localize(None),
                    "open": x.get("o"),
                    "high": x.get("h"),
                    "low": x.get("l"),
                    "close": x.get("c"),
                    "volume": x.get("v"),
                    "vwap": x.get("vw", x.get("c")),
                    "date": pd.to_datetime(x.get("t"), unit="ms", utc=True).tz_localize(None).date(),
                } for x in aggs])
                mdf = _normalize_minute_df(mdf)
                _write_partitioned_parquet(mdf, ticker, granularity="minute")
                all_rows.append(mdf)
            if not all_rows:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"]).astype({"timestamp": "datetime64[ns]"})
            mdf_all = pd.concat(all_rows, ignore_index=True)
            # 写入后再做一次重复行检查（跨分片）
            if _detect_duplicates(mdf_all, "timestamp") > 0:
                _log_quality_issue("duplicate_rows", ticker, {"granularity": "minute", "scope": "batch", "duplicates": int(_detect_duplicates(mdf_all, "timestamp"))})
            return mdf_all
        else:
            return _empty_bars_df()
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_bars failed: {ticker}: {exc}")
        return _empty_bars_df()

# 一次性获取指定日期全市场的日线数据
def get_grouped_daily(date: str, adjusted: bool) -> pd.DataFrame:
    try:
        res = _polygon_client.get_grouped_daily_aggs(date, adjusted)
        if not res:
            return pd.DataFrame([])
        df = pd.DataFrame(res)
        return df
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_grouped_daily failed: {date}: {exc}")
        return pd.DataFrame([])


def get_indicators(ticker: str, timespan: str, windows: List[int]) -> Dict[str, pd.DataFrame]:
    try:
        # 建议本地计算，这里保留空实现
        return {}
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_indicators failed: {ticker}: {exc}")
        return {}

# 获取指定股票的快照数据
def get_universal_snapshots(symbols: List[str]) -> Dict[str, Dict]:
    try:
        return _polygon_client.get_universal_snapshots(symbols)
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_universal_snapshots failed: {exc}")
        return {s: {} for s in symbols}

# 获取指定股票的涨跌榜
def get_gainers_losers(top_n: int) -> Dict[str, List[str]]:
    try:
        return _polygon_client.get_gainers_losers(top_n)
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_gainers_losers failed: {exc}")
        return {"gainers": [], "losers": []}


def get_news(ticker: str, gte: str, lte: str, limit: int = 100, page_token: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
    # 先读缓存
    cached = _load_cached_news(ticker)
    if cached is not None:
        deduped = _dedup_news(cached)
        return deduped[:limit], None
    # 回退到本地 backtest_data
    path = os.path.join(_BACKTEST_DIR, "news_data.json")
    if os.path.exists(path):
        try:
            data = pd.read_json(path)
            data = data[data["tickers"].apply(lambda arr: isinstance(arr, list) and ticker in arr)]
            gte_dt = pd.to_datetime(gte) if gte else None
            lte_dt = pd.to_datetime(lte) if lte else None
            data["published_date"] = pd.to_datetime(data["published_date"], errors="coerce")
            if gte_dt is not None:
                data = data[data["published_date"] >= gte_dt]
            if lte_dt is not None:
                data = data[data["published_date"] <= lte_dt]
            data = data.sort_values("published_date").head(limit)
            items: List[Dict] = data.to_dict(orient="records")
            _save_cached_news(ticker, items)
            return items, None
        except Exception as exc:  # pragma: no cover
            logger.warning(f"local news read failed, fallback to API: {exc}")
    # API 拉取
    try:
        items_all: List[Dict] = []
        cursor: Optional[str] = None
        while True and len(items_all) < limit:
            items, cursor = _polygon_client.list_ticker_news(ticker, gte, lte, limit=limit, page_token=cursor)
            if not items:
                break
            items_all.extend(items)
            if not cursor:
                break
        _save_cached_news(ticker, items_all)
        deduped = _dedup_news(items_all)
        return deduped[:limit], None
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_news failed: {ticker}: {exc}")
        return [], None


def get_dividends(ticker: str) -> pd.DataFrame:
    """优先 Polygon；无结果时回退读取 backtest_data/dividends.json（若存在）。"""
    # 1) API 优先
    try:
        items = _polygon_client.list_dividends(ticker)
        if items:
            return pd.DataFrame(items)
    except Exception:
        pass
    # 2) 本地回退
    try:
        path = os.path.join(_BACKTEST_DIR, "dividends.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 期望结构：list[dict]，包含 ticker/现金金额/日期字段
            rows = [it for it in data if (it.get("ticker") == ticker)]
            return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning(f"local dividends read failed: {exc}")
    return pd.DataFrame([])


def get_splits(ticker: str) -> pd.DataFrame:
    """优先 Polygon；无结果时回退读取 backtest_data/splits.json（若存在）。"""
    # 1) API 优先
    try:
        items = _polygon_client.list_splits(ticker)
        if items:
            return pd.DataFrame(items)
    except Exception:
        pass
    # 2) 本地回退
    try:
        path = os.path.join(_BACKTEST_DIR, "splits.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = [it for it in data if (it.get("ticker") == ticker)]
            return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning(f"local splits read failed: {exc}")
    return pd.DataFrame([])

# 获取指定股票的详情数据
def get_ticker_details(ticker: str, date: Optional[str] = None) -> Dict:
    try:
        return _polygon_client.get_ticker_details(ticker, date)
    except Exception:
        return {}

# 获取市场状态
def get_market_status() -> Dict:
    try:
        return _polygon_client.get_market_status()
    except Exception:
        return {"market": "unknown"}


def get_earnings_events(ticker: str, limit: int = 1000) -> List[Dict]:
    """获取标的的事件日历中“财报相关”的事件（本地过滤）。

    注意：Polygon 的 events 接口对 types 参数的可用值有限（如 ticker_change）。
    实测传 "earnings" 可能 400。为稳健起见，这里不传 types，
    本地依据 event["type"] 与字段名包含 "earn" 的子对象进行过滤。
    """
    try:
        items: List[Dict] = _polygon_client.list_ticker_events(ticker, types=None, limit=limit) or []
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_earnings_events failed: {ticker}: {exc}")
        items = []
    # 本地过滤 earnings 相关
    results: List[Dict] = []
    for it in items:
        try:
            t = str(it.get("type") or "").lower()
            if any(k in t for k in ["earn", "eps"]):
                results.append(it)
                continue
            # 兼容形如 {"earnings_announcement":{...},"type":"earnings_announcement"}
            keys = " ".join([str(k).lower() for k in it.keys()])
            if any("earn" in k for k in keys.split()):
                results.append(it)
        except Exception:
            continue
    return results


def is_in_earnings_window(ticker: str, as_of_iso_utc: str, window_days: int = 3) -> bool:
    """判断 as_of 是否处于财报事件窗（±window_days）。
    事件窗依据 Polygon events 的 earnings 类型事件（如 earnings_announcement）。
    """
    try:
        now_dt = pd.to_datetime(as_of_iso_utc, utc=True, errors="coerce")
        if pd.isna(now_dt):
            now_dt = pd.Timestamp.utcnow().tz_localize("UTC")
        items = get_earnings_events(ticker, limit=200)
        for it in items:
            # 兼容常见字段：start_date、timestamp、event_time
            ts = it.get("start_date") or it.get("event_time") or it.get("timestamp") or it.get("date")
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            delta = abs((now_dt - dt).days)
            if delta <= int(window_days):
                return True
        return False
    except Exception as exc:  # pragma: no cover
        logger.exception(f"is_in_earnings_window failed: {ticker}: {exc}")
        return False


def get_nearest_earnings_event(
    ticker: str,
    as_of_iso_utc: str,
    lookback_days: int = 120,
    lookahead_days: int = 120,
) -> Optional[Dict]:
    """查找距 as_of 最近的“财报相关”事件。

    返回：
    - {"date": "YYYY-MM-DD", "delta_days": int, "type": str, "raw": dict}
      若找不到，返回 None。
    逻辑：
    1) 读取 events 全量并本地过滤 earnings 相关；选取与 as_of 最近的一条（优先落在 [as_of-回看, as_of+前瞻] 内）。
    2) 若 events 空，回退：用新闻关键词法近似（取最近一条含 earn/eps 关键词的新闻日期）。
    """
    try:
        as_of = pd.to_datetime(as_of_iso_utc, utc=True, errors="coerce")
        if pd.isna(as_of):
            as_of = pd.Timestamp.utcnow().tz_localize("UTC")
        # 1) events 源
        items = get_earnings_events(ticker, limit=500) or []
        candidates: List[Tuple[pd.Timestamp, Dict]] = []
        for it in items:
            ts = it.get("start_date") or it.get("event_time") or it.get("timestamp") or it.get("date")
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            candidates.append((dt, it))
        if candidates:
            # 先过滤窗口，再就近
            lb = as_of - pd.Timedelta(days=int(lookback_days))
            ub = as_of + pd.Timedelta(days=int(lookahead_days))
            in_win = [(dt, it) for dt, it in candidates if lb <= dt <= ub]
            pool = in_win if in_win else candidates
            dt_best, it_best = sorted(pool, key=lambda x: abs((x[0] - as_of).days))[0]
            return {
                "date": dt_best.strftime("%Y-%m-%d"),
                "delta_days": int(abs((dt_best - as_of).days)),
                "type": str(it_best.get("type") or ""),
                "raw": it_best,
            }
        # 2) 回退：新闻关键词近似
        # 取 as_of 前后 90 天新闻，找含 earn/eps 关键词的最近一条
        gte = (as_of - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        lte = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        news, _ = get_news(ticker, gte, lte, limit=1000)
        nearest_dt = None
        nearest_it: Optional[Dict] = None
        for it in news or []:
            title = str(it.get("title") or "").lower()
            desc = str(it.get("description") or it.get("summary") or it.get("article_body") or "").lower()
            text = title + " " + desc
            if ("earn" in text) or ("eps" in text):
                p = it.get("published_utc") or it.get("published_date")
                dt = pd.to_datetime(p, utc=True, errors="coerce")
                if pd.isna(dt):
                    continue
                if (nearest_dt is None) or (abs((dt - as_of).days) < abs((nearest_dt - as_of).days)):
                    nearest_dt = dt
                    nearest_it = it
        if nearest_dt is not None and nearest_it is not None:
            return {
                "date": nearest_dt.strftime("%Y-%m-%d"),
                "delta_days": int(abs((nearest_dt - as_of).days)),
                "type": "news_keyword",
                "raw": nearest_it,
            }
    except Exception as exc:  # pragma: no cover
        logger.exception(f"get_nearest_earnings_event failed: {ticker}: {exc}")
    return None


def compare_with_legacy_day(symbol: str, start: str, end: str, tolerance: float = 1e-6) -> Dict[str, object]:
    """对齐本地 CSV（legacy）与当前 Parquet（日线）口径，输出差异报告并落盘到 reports/alignments。
    返回简要统计字典。
    """
    try:
        legacy = _read_local_day_csv(symbol)
        legacy = _normalize_day_df(legacy)
        # 读取本地已落盘的日线 Parquet 分片
        base_dir = os.path.join(_PARQUET_BASE, symbol, "day")
        rows: List[pd.DataFrame] = []
        if os.path.isdir(base_dir):
            for fname in os.listdir(base_dir):
                if not fname.endswith(".parquet"):
                    continue
                d = fname.replace(".parquet", "")
                rows.append(pd.read_parquet(os.path.join(base_dir, fname)))
        current = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=legacy.columns)
        current = _normalize_day_df(current)
        # 过滤区间
        s = pd.to_datetime(start).date() if start else None
        e = pd.to_datetime(end).date() if end else None
        if s:
            legacy = legacy[legacy["date"] >= s]
            current = current[current["date"] >= s]
        if e:
            legacy = legacy[legacy["date"] <= e]
            current = current[current["date"] <= e]
        # 合并并比较 close 差异
        merged = legacy.merge(current, on="date", how="inner", suffixes=("_legacy", "_cur"))
        merged["close_diff"] = (merged["close_legacy"] - merged["close_cur"]).abs()
        mismatch = int((merged["close_diff"] > tolerance).sum())
        report = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "rows_compared": int(len(merged)),
            "mismatch_close_gt_tol": mismatch,
            "tolerance": tolerance,
        }
        out_dir = os.path.join(_REPORT_BASE, "alignments")
        ensure_dir(out_dir)
        atomic_append_jsonl(os.path.join(out_dir, f"{symbol}.jsonl"), report)
        return report
    except Exception as exc:  # pragma: no cover
        logger.warning(f"compare_with_legacy_day failed for {symbol}: {exc}")
        return {"symbol": symbol, "error": str(exc)}


def get_financials(ticker: str, timeframe: Optional[str] = None, limit: int = 50, use_cache: bool = True) -> List[Dict]:
    """获取财务报表（annual/quarterly）。默认开启 24h 文件缓存。
    缓存键：storage/cache/financials/{ticker}.{timeframe or all}.json
    """
    try:
        cache_dir = os.path.join(_CACHE_BASE, "financials")
        _ensure_dir(cache_dir)
        cache_name = f"{ticker}.{timeframe or 'all'}.json"
        cache_path = os.path.join(cache_dir, cache_name)
        if use_cache and os.path.exists(cache_path):
            st = os.stat(cache_path)
            if (datetime.now().timestamp() - st.st_mtime) <= 24 * 3600:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return data[:limit]
                except Exception:
                    # 损坏：删除后视为未命中
                    try:
                        os.remove(cache_path)
                    except Exception:
                        pass
            else:
                # 过期：删除后视为未命中
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
        items = _polygon_client.list_financials(ticker, timeframe=timeframe, limit=limit)
        # 简单去重（按报告期+文件类型或 id）
        seen: set[str] = set()
        deduped: List[Dict] = []
        for it in items or []:
            key = str(it.get("id") or (it.get("fiscal_period"), it.get("fiscal_year"), it.get("start_date"), it.get("end_date")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(deduped, f, ensure_ascii=False)
        except Exception:
            pass
        return deduped[:limit]
    except Exception as exc:  # pragma: no cover
        logger.warning(f"get_financials failed: {ticker}: {exc}")
        return []