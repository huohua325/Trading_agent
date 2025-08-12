from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from trading_agent_v2.core.data_hub import get_nearest_earnings_event
from trading_agent_v2.utils.logging_setup import setup_json_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    symbol: str = typer.Option(..., "--symbol", "-s", help="标的"),
    asof: str = typer.Option(None, help="基准日期 ISO，如 2025-01-15T00:00:00Z；为空则使用当前 UTC"),
    lookback: int = typer.Option(120, help="回看天数"),
    lookahead: int = typer.Option(120, help="前瞻天数"),
) -> None:
    setup_json_logging()
    if asof is None:
        import datetime as _dt
        asof = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    res = get_nearest_earnings_event(symbol, as_of_iso_utc=asof, lookback_days=lookback, lookahead_days=lookahead)
    if res is None:
        typer.echo("未找到财报相关事件")
        raise typer.Exit(code=1)
    typer.echo(f"symbol={symbol} asof={asof} nearest_earnings_date={res['date']} delta_days={res['delta_days']} type={res['type']}")


if __name__ == "__main__":  # pragma: no cover
    app() 