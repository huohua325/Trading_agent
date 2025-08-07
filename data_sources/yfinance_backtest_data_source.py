import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import yfinance as yf

from .base_data_source import BaseDataSource
from .yfinance_data_source import YFinanceDataSource


class YFinanceBacktestDataSource(YFinanceDataSource):
    """使用 yfinance 直接在线拉取历史行情的回测数据源。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.start_date: str = config.get("start_date")
        self.end_date: str = config.get("end_date")
        self.current_date: Optional[datetime] = None
        self.price_data: Dict[str, pd.DataFrame] = {}

    # --------------------------------------------------
    # 通用
    # --------------------------------------------------
    def set_current_date(self, date: datetime):
        """回测循环中由外部调用，设置当前模拟日期"""
        self.current_date = pd.Timestamp(date).tz_localize(None).to_pydatetime()

    async def load_data(self, symbols: List[str]):
        """一次性拉取所有股票的历史日线"""
        tasks = [self._fetch_price(s) for s in symbols]
        await asyncio.gather(*tasks)

    async def _fetch_price(self, symbol: str):
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.Ticker(symbol).history(start=self.start_date, end=self.end_date, interval="1d")
        )
        if not df.empty:
            df.index = df.index.tz_localize(None)
            self.price_data[symbol] = df
        else:
            print(f"⚠️ yfinance 无 {symbol} 行情数据")

    # --------------------------------------------------
    # 行情接口
    # --------------------------------------------------
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.price_data:
            await self._fetch_price(symbol)
            if symbol not in self.price_data:
                raise Exception(f"No historical data for {symbol}")

        if self.current_date is None:
            raise ValueError("current_date 未设置")

        df = self.price_data[symbol]
        available = df[df.index <= self.current_date]
        if available.empty:
            raise Exception(f"{symbol} 在 {self.current_date.date()} 之前没有价格数据")

        row = available.iloc[-1]
        prev = available.iloc[-2] if len(available) > 1 else row
        change = row["Close"] - prev["Close"]
        change_pct = (change / prev["Close"]) * 100 if prev["Close"] else 0

        return {
            "symbol": symbol,
            "price": row["Close"],
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "volume": row["Volume"],
            "date": row.name.strftime("%Y-%m-%d"),
            "change": change,
            "change_percent": change_pct,
        }

    async def get_market_data(self) -> Dict[str, Any]:
        data = {}
        for s in self.trading_symbols:
            try:
                data[s] = await self.get_real_time_price(s)
            except Exception as e:
                print(f"获取 {s} 行情失败: {e}")
        return data 

    # --------------------------------------------------
    # 财务数据过滤，避免未来信息泄漏
    # --------------------------------------------------
    async def get_company_financials(self, symbol: str, quarters: int = 4, earnings_limit: int = 4) -> Dict[str, Any]:
        data = await super().get_company_financials(symbol, quarters, earnings_limit)
        if self.current_date is None:
            return data
        # 过滤 reported_financials 时间
        date_cut = self.current_date.strftime('%Y-%m-%d')
        rep = data.get("reported_financials", {}).get("data", {})
        if isinstance(rep, dict):
            for stmt in rep.values():
                for k in list(stmt.keys()):
                    if k > date_cut:
                        stmt.pop(k, None)
        # 过滤 earnings_surprises
        es = data.get("earnings_surprises", [])
        if isinstance(es, list):
            data["earnings_surprises"] = [e for e in es if e.get("period", "9999-99-99") <= date_cut]
        # 过滤 recommendation_trends
        rt = data.get("recommendation_trends", [])
        if isinstance(rt, list):
            data["recommendation_trends"] = [r for r in rt if r.get("period", "9999-99-99") <= date_cut]
        return data 