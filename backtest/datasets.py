from __future__ import annotations

from typing import Dict, List

import pandas as pd

from trading_agent_v2.core import data_hub


class Datasets:
    def get_day_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return data_hub.get_bars(symbol, start, end, multiplier=1, timespan="day", adjusted=True)

    def get_min_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return data_hub.get_bars(symbol, start, end, multiplier=1, timespan="minute", adjusted=True)

    def get_news(self, symbol: str, gte: str, lte: str) -> List[Dict]:
        data, _ = data_hub.get_news(symbol, gte, lte, limit=100)
        return data 