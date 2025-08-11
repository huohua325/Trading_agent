from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any
import sys

import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_outputs(result: Dict, run_id: str | None = None, cfg: Dict[str, Any] | None = None) -> str:
    run_id = run_id or _default_run_id()
    base = os.path.join(os.getcwd(), "trading_agent_v2", "storage", "reports", "backtest", run_id)
    _ensure_dir(base)

    # trades
    trades: pd.DataFrame = result.get("trades") if isinstance(result.get("trades"), pd.DataFrame) else pd.DataFrame()
    trades_path = os.path.join(base, "trades.parquet")
    if not trades.empty:
        trades.to_parquet(trades_path, index=False)
    else:
        pd.DataFrame([]).to_parquet(trades_path)

    # daily nav
    nav = result.get("nav")
    nav_path = os.path.join(base, "daily_nav.parquet")
    if isinstance(nav, pd.Series):
        nav.to_frame(name="nav").to_parquet(nav_path)
    elif isinstance(nav, pd.DataFrame):
        nav.to_parquet(nav_path)
    else:
        pd.DataFrame([]).to_parquet(nav_path)

    # metrics
    metrics = result.get("metrics") or {}
    with open(os.path.join(base, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # config snapshot
    if isinstance(cfg, dict) and cfg:
        with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # meta snapshot（环境与库版本）
    meta = {
        "python": sys.version,
    }
    try:
        import pandas as _pd  # type: ignore
        meta["pandas"] = getattr(_pd, "__version__", "unknown")
    except Exception:
        meta["pandas"] = "unknown"
    try:
        import pyarrow as _pa  # type: ignore
        meta["pyarrow"] = getattr(_pa, "__version__", "unknown")
    except Exception:
        meta["pyarrow"] = "unknown"
    try:
        import httpx as _httpx  # type: ignore
        meta["httpx"] = getattr(_httpx, "__version__", "unknown")
    except Exception:
        meta["httpx"] = "unknown"
    with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # summary
    summary_lines = []
    summary_lines.append(f"run_id: {run_id}")
    try:
        start = str(trades["ts"].min()) if not trades.empty else "NA"
        end = str(trades["ts"].max()) if not trades.empty else "NA"
    except Exception:
        start = end = "NA"
    summary_lines.append(f"period: {start} ~ {end}")
    # params snapshot（关键）
    try:
        bt = (cfg or {}).get("backtest", {}) if isinstance(cfg, dict) else {}
        news_cfg = (cfg or {}).get("news", {}) if isinstance(cfg, dict) else {}
        strat = (cfg or {}).get("strategy", "") if isinstance(cfg, dict) else ""
        timespan = (cfg or {}).get("backtest", {}).get("timespan") if isinstance(cfg, dict) else None
        summary_lines.append(
            "params: "
            f"commission_bps={bt.get('commission_bps')}, slippage_bps={bt.get('slippage_bps')}, fill_ratio={bt.get('fill_ratio')}, "
            f"news_agg={news_cfg.get('agg')}, trim_alpha={news_cfg.get('trim_alpha')}, strategy={strat or 'N/A'}, timespan={timespan or 'N/A'}"
        )
    except Exception:
        pass
    summary_lines.append(
        "metrics: "
        f"cum_return={metrics.get('cum_return', 0):.4f}, "
        f"max_drawdown={metrics.get('max_drawdown', 0):.4f}, "
        f"volatility={metrics.get('volatility', 0):.4f}, "
        f"sharpe={metrics.get('sharpe', 0):.4f}"
    )
    summary_lines.append(f"trades: {0 if trades is None else len(trades)} rows")
    with open(os.path.join(base, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 固化基线（可选）：cfg.backtest.baseline_name 存在时写入 baselines 目录
    try:
        baseline_name = None
        if isinstance(cfg, dict):
            baseline_name = (cfg.get("backtest", {}) or {}).get("baseline_name")
        if baseline_name:
            bdir = os.path.join(os.getcwd(), "trading_agent_v2", "storage", "reports", "backtest", "baselines")
            _ensure_dir(bdir)
            payload = {
                "run_id": run_id,
                "metrics": metrics,
                "period": {"start": start, "end": end},
                "params": {
                    "commission_bps": (cfg or {}).get("backtest", {}).get("commission_bps") if isinstance(cfg, dict) else None,
                    "slippage_bps": (cfg or {}).get("backtest", {}).get("slippage_bps") if isinstance(cfg, dict) else None,
                    "fill_ratio": (cfg or {}).get("backtest", {}).get("fill_ratio") if isinstance(cfg, dict) else None,
                    "news_agg": (cfg or {}).get("news", {}).get("agg") if isinstance(cfg, dict) else None,
                    "trim_alpha": (cfg or {}).get("news", {}).get("trim_alpha") if isinstance(cfg, dict) else None,
                },
            }
            with open(os.path.join(bdir, f"{baseline_name}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return base 