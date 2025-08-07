import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd

from .finnhub_data_source import FinnhubDataSource


class FinnhubBacktestDataSource(FinnhubDataSource):
    """基于 Finnhub API 的回测数据源。

    设计思路：
    1. 在回测开始前一次性拉取 start_date~end_date 的历史价格数据并缓存。
    2. 回测循环中通过 ``set_current_date`` 设定当前回测日期，随后
       ``get_real_time_price`` / ``get_market_data`` 等接口均返回该日期
       (或之前最近一个交易日) 的行情，行为与 ``BacktestDataSource`` 保持一致。
    3. 其他如 ``get_company_financials``、``get_news`` 等仍沿用 Finnhub 实时
       API，但在内部根据 ``current_date`` 做日期过滤，保证只使用“当前日期
       之前”的信息，避免未来数据泄露。
    """

    def __init__(self, config: Dict[str, Any]):
        # 继承 FinnhubDataSource 初始化（含 API key 校验）
        super().__init__(config)

        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        self.current_date: Optional[datetime] = None

        # 缓存
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.market_info_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 通用工具
    # ------------------------------------------------------------------
    def set_current_date(self, date: datetime):
        """在回测过程中由外部设置当前模拟日期"""
        # 去掉时区保持一致
        self.current_date = pd.Timestamp(date).tz_localize(None).to_pydatetime()

    async def load_data(self, symbols: List[str]):
        """一次性拉取所有股票的历史日线数据"""
        if not self.start_date or not self.end_date:
            raise ValueError("start_date / end_date 必须在 config 中提供以便回测")

        start_dt = pd.Timestamp(self.start_date)
        end_dt = pd.Timestamp(self.end_date)

        tasks = [self._fetch_and_cache_price(s, start_dt, end_dt) for s in symbols]
        await asyncio.gather(*tasks)

    async def _fetch_and_cache_price(self, symbol: str, start_dt: datetime, end_dt: datetime):
        df = await self.get_historical_data(symbol, start_dt, end_dt, interval="D")
        if not df.empty:
            # 移除时区信息
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            self.price_data[symbol] = df

    # ------------------------------------------------------------------
    # 行情接口覆写
    # ------------------------------------------------------------------
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """回测环境下返回当前日期对应的收盘价，而非实时价"""
        if symbol not in self.price_data:
            # 若未提前加载则动态拉取
            await self._fetch_and_cache_price(symbol, pd.Timestamp(self.start_date), pd.Timestamp(self.end_date))
            if symbol not in self.price_data:
                raise Exception(f"No historical data for {symbol}")

        if self.current_date is None:
            raise ValueError("current_date 尚未设置")

        df = self.price_data[symbol]
        # 确保索引为 datetime 且无 tz
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        available = df[df.index <= self.current_date]
        if available.empty:
            raise Exception(f"{symbol} 在 {self.current_date.date()} 之前没有价格数据")

        row = available.iloc[-1]
        prev_row = available.iloc[-2] if len(available) > 1 else row
        change = row['Close'] - prev_row['Close']
        change_pct = (change / prev_row['Close']) * 100 if prev_row['Close'] else 0

        return {
            "symbol": symbol,
            "price": row['Close'],
            "open": row['Open'],
            "high": row['High'],
            "low": row['Low'],
            "volume": row['Volume'],
            "date": available.index[-1].strftime("%Y-%m-%d"),
            "change": change,
            "change_percent": change_pct,
        }

    async def get_market_data(self) -> Dict[str, Any]:
        data = {}
        for s in self.trading_symbols:
            try:
                data[s] = await self.get_real_time_price(s)
            except Exception as e:
                print(f"获取 {s} 数据失败: {e}")
        return data

    # ------------------------------------------------------------------
    # 市场信息缓存（避免重复调用）
    # ------------------------------------------------------------------
    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        if symbol in self.market_info_cache:
            return self.market_info_cache[symbol]
        info = await super().get_market_info(symbol)
        self.market_info_cache[symbol] = info
        return info

    # ------------------------------------------------------------------
    # 新闻过滤（不泄漏未来信息）
    # ------------------------------------------------------------------
    async def get_news(self, symbol: Optional[str] = None, limit: int = 10, days_back: int = 7):
        raw_news = await super().get_news(symbol=symbol, limit=limit * 3, days_back=days_back)
        if self.current_date is None:
            return raw_news[:limit]
        filtered = [n for n in raw_news if n.get('published_date', '')[:10] <= self.current_date.strftime('%Y-%m-%d')]
        return filtered[:limit] 

    # --------------------------------------------------
    # 财务数据过滤，避免未来信息泄漏
    # --------------------------------------------------
    async def get_company_financials(self, symbol: str, quarters: int = 4, earnings_limit: int = 4) -> Dict[str, Any]:
        data = await super().get_company_financials(symbol, quarters, earnings_limit)
        if self.current_date is None:
            return data
        date_cut = self.current_date.strftime('%Y-%m-%d')
        # reported_financials
        rep = data.get("reported_financials", {}).get("data", [])
        if isinstance(rep, list):
            data["reported_financials"]["data"] = [d for d in rep if d.get("report", {}).get("period", "9999-99-99") <= date_cut]
        # earnings_surprises
        es = data.get("earnings_surprises", [])
        if isinstance(es, list):
            data["earnings_surprises"] = [e for e in es if e.get("period", "9999-99-99") <= date_cut]
        # recommendation_trends
        rt = data.get("recommendation_trends", [])
        if isinstance(rt, list):
            data["recommendation_trends"] = [r for r in rt if r.get("period", "9999-99-99") <= date_cut]
        return data 