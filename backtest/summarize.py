from __future__ import annotations

from pathlib import Path
import json
import os

import pandas as pd
import typer
import yaml

from trading_agent_v2.agents.backtest_report_llm import generate_backtest_report

app = typer.Typer(add_completion=False)


def _load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _collect_stats(run_dir: Path) -> dict:
    stats = {}
    # trades count
    trades_path = run_dir / "trades.parquet"
    try:
        if trades_path.exists():
            df = pd.read_parquet(trades_path)
            stats["trades_count"] = int(len(df))
            if "symbol" in df.columns:
                stats["symbols"] = sorted(list(map(str, set(df["symbol"].dropna().unique().tolist()))))
            if "ts" in df.columns and len(df) > 0:
                stats["period"] = {
                    "start": str(df["ts"].min()),
                    "end": str(df["ts"].max()),
                }
    except Exception:
        pass
    # params snapshot
    cfg = _load_config(run_dir)
    bt = (cfg.get("backtest") or {}) if isinstance(cfg, dict) else {}
    news = (cfg.get("news") or {}) if isinstance(cfg, dict) else {}
    stats["timespan"] = bt.get("timespan")
    stats["costs"] = {
        "commission_bps": bt.get("commission_bps"),
        "slippage_bps": bt.get("slippage_bps"),
        "fill_ratio": bt.get("fill_ratio"),
    }
    stats["news_agg"] = news.get("agg")
    stats["trim_alpha"] = news.get("trim_alpha")
    return stats


@app.command()
def main(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True, help="回测输出目录，如 trading_agent_v2/storage/reports/backtest/llm_single_day"),
    cfg: Path = typer.Option(None, exists=True, readable=True, help="可选：加载 YAML 配置以传入 LLM 客户端设置"),
    llm: bool = typer.Option(False, help="是否调用真实 LLM 生成总结（默认否）"),
    out: Path = typer.Option(None, help="输出文本路径（默认写在 run_dir/nl_summary.txt）"),
):
    # 配置
    cfg_dict = {}
    if cfg is not None:
        with cfg.open("r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

    metrics = _load_metrics(run_dir)
    stats = _collect_stats(run_dir)
    payload = {"metrics": metrics, "stats": stats}

    text = generate_backtest_report(payload, cfg=cfg_dict, cache_only=(not llm))

    out_path = out or (run_dir / "nl_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    typer.echo(f"已生成总结：{out_path}")


if __name__ == "__main__":  # pragma: no cover
    app() 