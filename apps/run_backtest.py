from __future__ import annotations

from pathlib import Path
from typing import List
import itertools
import copy
import re

import typer
import yaml

from trading_agent_v2.backtest.pipeline import run_backtest
from trading_agent_v2.backtest.strategies.rule_baseline import Strategy as RuleBaseline
from trading_agent_v2.backtest.strategies.llm_decision import Strategy as LlmDecision
from trading_agent_v2.utils.logging_setup import setup_json_logging, Metrics

app = typer.Typer(add_completion=False)


@app.command()
def main(
    cfg: Path = typer.Option(..., exists=True, readable=True, help="配置文件路径"),
    start: str = typer.Option("2025-03-01", help="开始日期"),
    end: str = typer.Option("2025-07-31", help="结束日期"),
    symbols: str = typer.Option("", "--symbols", "-s", help="回测标的（逗号或空格分隔）"),
    strategy: str = typer.Option("rule_baseline", help="策略：rule_baseline|llm_decision"),
    replay: bool = typer.Option(False, help="回放模式：读取审计生成订单"),
    audit_dir: Path = typer.Option(None, help="回放模式下的审计目录（如 trading_agent_v2/storage/audit/2025-08-10)"),
    run_id: str = typer.Option(None, help="输出 run_id（默认自动生成）"),
    # 风控与 LLM 控制
    max_positions: int = typer.Option(None, help="最大持仓数上限（为空则使用配置）"),
    cooldown_days: int = typer.Option(None, help="清仓后冷却天数（为空则使用配置）"),
    min_holding_days: int = typer.Option(None, help="最小持有天数（为空则使用配置）"),
    llm_cache_only: bool = typer.Option(None, help="回测中 LLM 调用是否仅用缓存（为空则使用配置）"),
    # 执行与成本
    timespan: str = typer.Option("day", help="撮合时间粒度：day|minute（minute 为预留/近似）"),
    cost_tier: str = typer.Option(None, help="成本与执行假设分层：low|med|high（为空则使用配置）"),
    # 新闻情绪聚合（稳健性）
    news_agg: str = typer.Option(None, help="新闻情绪聚合：mean|median|trimmed_mean（为空则使用配置）"),
    news_trim_alpha: float = typer.Option(None, help="新闻情绪截尾均值参数 alpha（0.0~0.49，为空则使用配置）"),
    # 敏感性扫描（逗号分隔）
    sweep_news_agg: str = typer.Option("", help="敏感性扫描：news_agg 候选，逗号分隔，如 mean,median,trimmed_mean"),
    sweep_news_trim_alpha: str = typer.Option("", help="敏感性扫描：trim_alpha 候选，逗号分隔，如 0.0,0.1,0.2"),
    sweep_cost_tier: str = typer.Option("", help="敏感性扫描：cost_tier 候选，逗号分隔，如 low,med,high"),
    # 回测总结（自然语言）
    summary_llm: bool = typer.Option(False, help="回测结束后是否调用真实 LLM 生成自然语言总结（默认否；关闭时本地生成简要文本）"),
):
    setup_json_logging()
    m = Metrics()
    m.incr("run_backtest.start", 1)

    with cfg.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 解析 symbols（支持逗号或空格分隔）；为空则回退配置中的 universe
    sym_list: List[str] = []
    if symbols:
        sym_list = [s for s in re.split(r"[\s,]+", symbols.strip()) if s]
    if not sym_list:
        sym_list = list(config.get("symbols_universe", []))

    # 覆盖配置（风控与 LLM）
    if max_positions is not None:
        config.setdefault("risk", {})["max_positions"] = int(max_positions)
    if cooldown_days is not None:
        config.setdefault("risk", {})["cooldown_days"] = int(cooldown_days)
    if min_holding_days is not None:
        config.setdefault("risk", {})["min_holding_days"] = int(min_holding_days)
    if llm_cache_only is not None:
        config.setdefault("llm", {})["backtest_cache_only"] = bool(llm_cache_only)
    # 回测总结：是否启用真实 LLM
    config.setdefault("backtest", {})["summary_llm"] = bool(summary_llm)

    # 成本与执行分层覆盖（单值）
    tier = (cost_tier or "").lower() if cost_tier else None
    if tier in {"low", "med", "high"}:
        bt = config.setdefault("backtest", {})
        if tier == "low":
            bt["commission_bps"] = 0.0
            bt["slippage_bps"] = 5.0
            bt["fill_ratio"] = 1.0
        elif tier == "med":
            bt["commission_bps"] = 2.0
            bt["slippage_bps"] = 10.0
            bt["fill_ratio"] = 0.8
        else:  # high
            bt["commission_bps"] = 5.0
            bt["slippage_bps"] = 20.0
            bt["fill_ratio"] = 0.6

    # 覆盖配置（新闻聚合：单值）
    if news_agg is not None:
        config.setdefault("news", {})["agg"] = str(news_agg)
    if news_trim_alpha is not None:
        config.setdefault("news", {})["trim_alpha"] = float(news_trim_alpha)

    # 敏感性扫描：解析候选集合（为空则降级为单值）
    agg_list = [a for a in (sweep_news_agg.split(",") if sweep_news_agg else []) if a]
    trim_list_raw = [t for t in (sweep_news_trim_alpha.split(",") if sweep_news_trim_alpha else []) if t]
    cost_list = [c for c in (sweep_cost_tier.split(",") if sweep_cost_tier else []) if c]
    # 若未提供扫描项，则使用当前配置值单元素
    if not agg_list:
        agg_list = [config.get("news", {}).get("agg", "mean")]
    if not trim_list_raw:
        trim_list_raw = [str(config.get("news", {}).get("trim_alpha", 0.1))]
    if not cost_list:
        cost_list = [tier or (config.get("backtest", {}).get("cost_tier") or "")]  # 可能为空串

    # 规范化 trim_alpha 列表为 float
    trim_list: List[float] = []
    for t in trim_list_raw:
        try:
            trim_list.append(float(t))
        except Exception:
            continue
    if not trim_list:
        trim_list = [0.1]

    # 运行一个或多个组合
    any_multi = (len(agg_list) > 1) or (len(trim_list) > 1) or (len(cost_list) > 1)
    results = []
    for agg_val, trim_val, tier_val in itertools.product(agg_list, trim_list, cost_list):
        cfg_i = copy.deepcopy(config)
        cfg_i.setdefault("news", {})["agg"] = agg_val
        cfg_i.setdefault("news", {})["trim_alpha"] = float(trim_val)
        # 按 cost_tier 覆盖三项
        tv = (tier_val or "").lower()
        if tv in {"low", "med", "high"}:
            bt = cfg_i.setdefault("backtest", {})
            if tv == "low":
                bt["commission_bps"] = 0.0
                bt["slippage_bps"] = 5.0
                bt["fill_ratio"] = 1.0
            elif tv == "med":
                bt["commission_bps"] = 2.0
                bt["slippage_bps"] = 10.0
                bt["fill_ratio"] = 0.8
            else:
                bt["commission_bps"] = 5.0
                bt["slippage_bps"] = 20.0
                bt["fill_ratio"] = 0.6
        # 针对多组合，自动拼接 run_id 后缀
        rid = run_id
        if any_multi:
            base = run_id or "auto"
            suffix = f"agg={agg_val}|alpha={trim_val}|tier={tv or 'cfg'}"
            rid = f"{base}_{suffix}"

        if strategy == "llm_decision":
            strat = LlmDecision(cfg_i, replay=replay, audit_dir=str(audit_dir) if audit_dir else None)
        else:
            strat = RuleBaseline(cfg_i)

        result = run_backtest(cfg=cfg_i, strategy=strat, start=start, end=end, symbols=sym_list, replay=replay, run_id=rid, timespan=timespan)
        metrics = result["metrics"]
        typer.echo(f"回测完成：{start}~{end} 标的数={len(sym_list)} replay={replay} news_agg={agg_val} trim_alpha={trim_val} cost_tier={tv or tier}")
        typer.echo(f"参数：max_positions={cfg_i.get('risk',{}).get('max_positions')}, cooldown_days={cfg_i.get('risk',{}).get('cooldown_days')}, min_holding_days={cfg_i.get('risk',{}).get('min_holding_days')}, llm_cache_only={cfg_i.get('llm',{}).get('backtest_cache_only')}, timespan={timespan}")
        typer.echo(f"指标：cum_return={metrics['cum_return']:.4f}, max_dd={metrics['max_drawdown']:.4f}, sharpe={metrics['sharpe']:.4f}")
        typer.echo(f"输出目录：{result.get('output_dir')}")
        results.append({"agg": agg_val, "alpha": float(trim_val), "tier": tv, "metrics": metrics, "output_dir": result.get("output_dir")})

        m.gauge("bt.cum_return", float(metrics.get("cum_return", 0)))
        m.gauge("bt.max_drawdown", float(metrics.get("max_drawdown", 0)))
        m.gauge("bt.sharpe", float(metrics.get("sharpe", 0)))
        m.gauge("bt.symbols", len(sym_list))
        m.incr("run_backtest.finished", 1)

    m.flush({"component": "run_backtest", "replay": replay, "start": start, "end": end, "strategy": strategy, "multi": any_multi})


if __name__ == "__main__":  # pragma: no cover
    app() 