from __future__ import annotations

import types
import pandas as pd

from trading_agent_v2.core import data_hub


def test_minute_chunking_merges_days(monkeypatch):
    calls = []

    def fake_list_aggs(ticker, start, end, multiplier, timespan, adjusted):
        calls.append((start, end))
        # 生成两条分钟数据
        base_ts = pd.Timestamp(f"{start} 14:30:00")
        rows = [
            {"t": int(base_ts.value // 1_000_000), "o": 1, "h": 1, "l": 1, "c": 1, "v": 100, "vw": 1},
            {"t": int((base_ts + pd.Timedelta(minutes=1)).value // 1_000_000), "o": 1, "h": 1, "l": 1, "c": 1, "v": 100, "vw": 1},
        ]
        return rows

    monkeypatch.setattr(data_hub._polygon_client, "list_aggs", fake_list_aggs)

    df = data_hub.get_bars("AAPL", "2024-01-01", "2024-01-02", 1, "minute", True)
    # 预期调用两天各一次
    assert ("2024-01-01", "2024-01-01") in calls
    assert ("2024-01-02", "2024-01-02") in calls
    # 合并后应有 4 行
    assert len(df) == 4 