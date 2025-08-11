from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta, timezone

import typer
import yaml

from trading_agent_v2.core import data_hub
from trading_agent_v2.core.features import build_features
from trading_agent_v2.agents.analyzer_llm import analyze_batch
from trading_agent_v2.core.risk_guard import make_limits
from trading_agent_v2.agents.decision_llm import decide_batch
from trading_agent_v2.core.executor import plan_orders, record_orders
from trading_agent_v2.utils.logging_setup import setup_json_logging, Metrics
from trading_agent_v2.llm.llm_client import LLMClient
from trading_agent_v2.core.data_hub import enrich_news_with_llm_sentiment
from trading_agent_v2.agents.report_llm import generate_report, write_report

app = typer.Typer(add_completion=False)

# 初始化结构化日志
try:
    setup_json_logging()
except Exception:
    pass


def _load_cfg(cfg_path: Path) -> Dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.command()
def main(
    cfg: Path = typer.Option(..., exists=True, readable=True, help="配置文件路径"),
    since: str = typer.Option("2024-01-01", help="数据起始日期（bars/news）"),
    until: str = typer.Option("2024-01-02", help="数据结束日期（bars/news）"),
    max_symbols: int = typer.Option(0, min=0, help="最多处理标的数，0 表示不限制"),
    dry_run: bool = typer.Option(False, help="干跑：不写盘（不落审计、不落 Parquet）"),
    gray_percent: int = typer.Option(100, min=0, max=100, help="灰度比例：处理 symbols 的百分比（随机截断方式示意）"),
    enable_llm: bool = typer.Option(False, help="启用 LLM 调用（默认关闭）"),
    llm_cache_only: bool = typer.Option(False, help="仅使用缓存（不真实调用）"),
    llm_news_sent: bool = typer.Option(False, help="使用 LLM 为新闻打分并汇总到 features.news.sentiment"),
    print_features: bool = typer.Option(False, help="打印构建的 features_list 以便核验"),
    print_analysis: bool = typer.Option(False, help="打印 analyzer 输出以便核验"),
    print_llm_prompts: bool = typer.Option(False, help="打印两次 LLM 的 prompt 与回答（从缓存中读取）"),
) -> None:
    """盘中任务最小闭环：features→analyzer→risk→decision→executor→审计。
    新增：LLM 开关与缓存开关。
    """
    setup_json_logging()
    metrics = Metrics()
    metrics.incr("run_intraday.start", 1)

    config = _load_cfg(cfg)

    symbols: List[str] = list(config.get("symbols_universe", []))
    if not symbols:
        typer.echo("[run_intraday] 警告：symbols_universe 为空")
        return

    # 灰度与限量
    if 0 <= gray_percent < 100:
        cut = max(1, int(len(symbols) * (gray_percent / 100.0)))
        symbols = symbols[:cut]
    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]

    # News 拉取窗口（统一 UTC）
    news_cfg = config.get("news", {})
    lookback_days = int(news_cfg.get("lookback_days", 7))
    page_limit = int(news_cfg.get("page_limit", 100))
    try:
        until_dt = datetime.fromisoformat(until)
        # 若无时区信息，强制视为 UTC
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)
        else:
            until_dt = until_dt.astimezone(timezone.utc)
    except Exception:
        until_dt = datetime.now(timezone.utc)
    gte_dt = until_dt - timedelta(days=lookback_days)
    gte_str = gte_dt.strftime("%Y-%m-%d")

    # 批量获取实时快照（作为真实最新价）
    snapshots = data_hub.get_universal_snapshots(symbols)

    features_list: List[Dict] = []
    symbol_to_financials: Dict[str, List[Dict]] = {}
    for s in symbols:
        bars_min = data_hub.get_bars(s, since, until, 1, "minute", True)
        bars_day = data_hub.get_bars(s, "2023-12-01", until, 1, "day", True)
        indicators = {}
        snap = snapshots.get(s, {}) or {}
        # Polygon snapshot 字段名对齐：优先最新成交价和时间
        price = None
        ts_utc = "1970-01-01T00:00:00Z"
        try:
            # 兼容 Polygon 通用快照结构：最新价可能在 lastTrade.p 或 minute.o/c 等
            last_trade = (snap.get("lastTrade") or {}) if isinstance(snap, dict) else {}
            prev_day = (snap.get("prevDay") or {}) if isinstance(snap, dict) else {}
            minute = (snap.get("minute") or {}) if isinstance(snap, dict) else {}
            price_candidates = [
                last_trade.get("p"),
                minute.get("c"),
                minute.get("o"),
                (snap.get("ticker", {}) if isinstance(snap.get("ticker"), dict) else {}).get("lastPrice"),
            ]
            for cand in price_candidates:
                if isinstance(cand, (int, float)):
                    price = float(cand)
                    break
            # 时间：优先 lastTrade.t（纳秒/毫秒时间戳），回退到 until_dt
            t_candidates = [last_trade.get("t"), minute.get("t"), prev_day.get("t")]
            for t in t_candidates:
                if t is None:
                    continue
                try:
                    # Polygon t 通常为纳秒或毫秒，这里尝试按 ns→ms→秒解读
                    # 先按纳秒
                    dt = datetime.fromtimestamp(int(t) / 1e9, tz=timezone.utc)
                    ts_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    break
                except Exception:
                    try:
                        dt = datetime.fromtimestamp(int(t) / 1e3, tz=timezone.utc)
                        ts_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        break
                    except Exception:
                        try:
                            dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
                            ts_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                            break
                        except Exception:
                            pass
            if ts_utc == "1970-01-01T00:00:00Z":
                ts_utc = until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            price = None
            ts_utc = until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        snapshot = {"symbol": s, "price": price, "ts_utc": ts_utc}

        # 新闻、分红、拆股、财务报表
        news_items, _ = data_hub.get_news(s, gte_str, until, limit=page_limit)
        if llm_news_sent and enable_llm:
            try:
                news_items = enrich_news_with_llm_sentiment(news_items, cfg=config, cache_only=llm_cache_only)
            except Exception:
                pass
        dividends = data_hub.get_dividends(s)
        splits = data_hub.get_splits(s)
        financials = data_hub.get_financials(s, timeframe=None, limit=100)
        symbol_to_financials[s] = financials
        details = data_hub.get_ticker_details(s) or {"ticker": s}
        position_state = {"current_position_pct": 0.0}

        fi = build_features(bars_min, bars_day, indicators, snapshot, news_items, dividends, splits, financials, details, position_state)
        features_list.append(fi)

    # 按需打印 features_list 以便核验
    if print_features:
        try:
            typer.echo("==== FEATURES (FeatureInput list) ====")
            typer.echo(json.dumps(features_list, ensure_ascii=False, indent=2))
        except Exception:
            for idx, fi in enumerate(features_list):
                try:
                    typer.echo(f"[features {idx}] " + json.dumps(fi, ensure_ascii=False))
                except Exception:
                    typer.echo(f"[features {idx}] {fi}")

    # Analyzer（带 LLM 开关）
    analysis_map = analyze_batch(features_list, cfg=config, enable_llm=enable_llm, cache_only=llm_cache_only)

    if print_analysis:
        try:
            typer.echo("==== ANALYZER OUTPUT ====")
            typer.echo(json.dumps(analysis_map, ensure_ascii=False, indent=2))
        except Exception:
            for k, v in analysis_map.items():
                try:
                    typer.echo(f"[analysis {k}] " + json.dumps(v, ensure_ascii=False))
                except Exception:
                    typer.echo(f"[analysis {k}] {v}")

    # 若需要打印两次 LLM 的 prompt & 回答（从缓存读取）
    if print_llm_prompts and enable_llm:
        try:
            client = LLMClient()
            import os
            cache_dir = client.cache_dir
            files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
            files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)
            # 仅取最近的 analyzer 与 decision 记录各一条
            payloads = {"analyzer": None, "decision": None, "news_sentiment": None}
            for fname in files:
                key = fname.replace('.json', '')
                obj = client.get_cached_payload(key)
                if not obj or not isinstance(obj, dict):
                    continue
                role = obj.get("role") or "unknown"
                if role in payloads and payloads[role] is None:
                    payloads[role] = {"cache_key": key, **obj}
                # 收齐后即可停止
                if all(payloads[k] is not None for k in payloads):
                    break
            # 分段打印
            typer.echo("==== LLM PROMPTS (analyzer) ====")
            typer.echo(json.dumps(payloads.get("analyzer"), ensure_ascii=False, indent=2))
            typer.echo("==== LLM PROMPTS (decision) ====")
            typer.echo(json.dumps(payloads.get("decision"), ensure_ascii=False, indent=2))
            typer.echo("==== LLM PROMPTS (news_sentiment) ====")
            typer.echo(json.dumps(payloads.get("news_sentiment"), ensure_ascii=False, indent=2))
        except Exception:
            pass

    # Risk
    decisions_input = []
    for fi in features_list:
        symbol = fi["symbol"]
        limits = make_limits(analysis_map[symbol], fi.get("position_state", {}), config.get("risk", {}), {})
        decisions_input.append({"features": fi, "analysis": analysis_map[symbol], "limits": limits})

    # Decision（带 LLM 开关）
    decisions_map = decide_batch(decisions_input, cfg=config, enable_llm=enable_llm, cache_only=llm_cache_only)

    # Executor + 审计（逐票）
    for fi in features_list:
        symbol = fi["symbol"]
        decision = dict(decisions_map[symbol])
        decision["symbol"] = symbol
        # 使用真实快照价格
        snap = snapshots.get(symbol, {}) or {}
        snapshot_price = None
        try:
            last_trade = (snap.get("lastTrade") or {}) if isinstance(snap, dict) else {}
            minute = (snap.get("minute") or {}) if isinstance(snap, dict) else {}
            for cand in [last_trade.get("p"), minute.get("c"), minute.get("o")]:
                if isinstance(cand, (int, float)):
                    snapshot_price = float(cand)
                    break
        except Exception:
            snapshot_price = None
        # 生成报告（覆盖写入）
        try:
            rpt = generate_report(fi, analysis_map[symbol], next(di["limits"] for di in decisions_input if di["features"]["symbol"] == symbol), decision, cfg=config, cache_only=False)
            rpt_path = write_report(symbol, rpt)
            typer.echo(f"[report] {symbol} -> {rpt_path}")
        except Exception:
            pass
        # 审计 + 最终输出
        orders = plan_orders(decision, snapshot_price if snapshot_price is not None else 0.0, cfg=config, portfolio={"equity": config.get("backtest", {}).get("cash", 1_000_000), "positions": {}})
        audit_payload = {
            "symbol": symbol,
            "features": fi,
            "analyzer": analysis_map[symbol],
            "limits": next(di["limits"] for di in decisions_input if di["features"]["symbol"] == symbol),
            "decision": decision,
            "snapshot": {"price": snapshot_price, "source": "polygon_snapshot"},
            "config": {"prompt_version": "p1", "risk_version": "r1", "execution": config.get("execution", {}), "llm": config.get("llm", {})},
            "meta": {"api_calls": 0, "llm_tokens": 0},
        }
        if not dry_run:
            record_orders(orders, audit_payload)

    # 打印最终输出（决策映射）
    typer.echo("==== DECISIONS (per symbol) ====")
    typer.echo(json.dumps({k: v for k, v in decisions_map.items() if k != "__meta__"}, ensure_ascii=False, indent=2))

    typer.echo(f"[run_intraday] 完成最小闭环，已处理 {len(symbols)} 支标的，写入审计记录，dry_run={dry_run}, gray={gray_percent}%, enable_llm={enable_llm}, cache_only={llm_cache_only}")
    metrics.incr("run_intraday.finished", 1)
    metrics.flush({"component": "run_intraday", "dry_run": dry_run, "gray_percent": gray_percent, "num_symbols": len(symbols), "enable_llm": enable_llm, "llm_cache_only": llm_cache_only})


if __name__ == "__main__":  # pragma: no cover
    app() 