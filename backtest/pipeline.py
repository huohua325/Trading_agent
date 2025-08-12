from __future__ import annotations

from typing import Dict, List, Optional
import os

from trading_agent_v2.backtest.datasets import Datasets
from trading_agent_v2.backtest.engine import BacktestEngine
from trading_agent_v2.backtest.slippage import Slippage
from trading_agent_v2.backtest.reports import write_outputs
from trading_agent_v2.agents.backtest_report_llm import generate_backtest_report


def run_backtest(cfg: Dict, strategy, start: str, end: str, symbols: List[str], replay: bool = False, run_id: str | None = None, timespan: Optional[str] = None) -> Dict:
    datasets = Datasets()
    slippage = Slippage.from_cfg(cfg)
    engine = BacktestEngine(cfg, datasets, slippage)
    # 选择 timespan：优先 CLI 传入；否则读配置；最终回退 "day"
    effective_timespan = (timespan or (cfg.get("backtest", {}) or {}).get("timespan") or "day")
    # 运行
    result = engine.run(strategy=strategy, start=start, end=end, symbols=symbols, timespan=effective_timespan)
    # 将 timespan 写回 cfg 以便报告显示
    try:
        cfg.setdefault("backtest", {})["timespan"] = effective_timespan
    except Exception:
        pass
    out_dir = write_outputs(result, run_id=run_id, cfg=cfg)
    result["output_dir"] = out_dir
    # 回测自然语言总结（作为回测一环）
    try:
        enable_llm = bool((cfg or {}).get("backtest", {}).get("summary_llm", False))
        payload = {
            "metrics": result.get("metrics") or {},
            "stats": {
                # trades_count/dates 将在 summarize 中更完善；此处简要写入
            },
        }
        text = generate_backtest_report(payload, cfg=cfg, cache_only=(not enable_llm))
        summary_path = os.path.join(out_dir, "nl_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(text)
        result["nl_summary"] = summary_path
    except Exception:
        pass
    return result 