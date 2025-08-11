from __future__ import annotations

from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False)


@app.command()
def main(cfg: Path = typer.Option(..., exists=True, readable=True, help="配置文件路径")) -> None:
    with cfg.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    typer.echo("[run_eod] 占位：日终流程完成")


if __name__ == "__main__":  # pragma: no cover
    app() 