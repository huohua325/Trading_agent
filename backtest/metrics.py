from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    roll_max = nav.cummax()
    dd = (nav / roll_max) - 1.0
    return float(dd.min())


def evaluate(nav_series: pd.Series, trades: pd.DataFrame) -> Dict:
    if nav_series.empty:
        return {"cum_return": 0.0, "max_drawdown": 0.0, "volatility": 0.0, "sharpe": 0.0}
    ret = nav_series.pct_change().fillna(0.0)
    cum_return = float(nav_series.iloc[-1] / max(nav_series.iloc[0], 1e-9) - 1.0)
    vol = float(ret.std() * (252 ** 0.5))
    sharpe = float((ret.mean() * 252) / vol) if vol > 0 else 0.0
    mdd = _max_drawdown(nav_series)
    return {"cum_return": cum_return, "max_drawdown": mdd, "volatility": vol, "sharpe": sharpe} 