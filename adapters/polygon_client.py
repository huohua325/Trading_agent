from __future__ import annotations

import os
import time
import random
from typing import Dict, List, Optional, Tuple

import httpx


class PolygonError(Exception):
    def __init__(self, status_code: int, message: str = "", payload: Optional[dict] = None) -> None:
        super().__init__(f"PolygonError {status_code}: {message}")
        self.status_code = status_code
        self.payload = payload or {}


class PolygonClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.polygon.io") -> None:
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=30.0)
        return self._client

    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        params = dict(params or {})
        if "apiKey" not in params:
            params["apiKey"] = self.api_key
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        client = self._get_client()

        # 指数退避 + 抖动
        backoff = 0.5
        for attempt in range(6):
            try:
                resp = client.request(method, url, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    # 可重试
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        sleep_s = float(retry_after) if retry_after else backoff * (2 ** attempt)
                    except Exception:
                        sleep_s = backoff * (2 ** attempt)
                    sleep_s = min(sleep_s, 30.0)
                    # 抖动 ±20%
                    sleep_s *= (0.8 + random.random() * 0.4)
                    time.sleep(sleep_s)
                    continue
                if 400 <= resp.status_code < 500:
                    # 非429的客户端错误直接抛出统一异常
                    raise PolygonError(resp.status_code, message=resp.text)
                resp.raise_for_status()
                return resp.json()
            except httpx.RequestError:
                sleep_s = min(backoff * (2 ** attempt), 30.0)
                sleep_s *= (0.8 + random.random() * 0.4)
                time.sleep(sleep_s)
                continue
            except httpx.HTTPStatusError as e:
                # 兼容某些情况下由 raise_for_status 抛错
                sc = e.response.status_code if e.response is not None else 0
                if sc != 429 and 400 <= sc < 500:
                    raise PolygonError(sc, message=str(e))
                # 其它情况（如5xx）走重试
        # 超过重试返回空
        return {}

    # Aggregates（分钟/日线）
    def list_aggs(self, ticker: str, start: str, end: str, multiplier: int, timespan: str, adjusted: bool) -> List[Dict]:
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}
        results: List[Dict] = []
        url = path
        cursor = None
        while True:
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", url, params)
            if not data:
                break
            items = data.get("results") or []
            results.extend(items)
            cursor = data.get("next_url") or data.get("next_page_token") or data.get("nextCursor") or data.get("next_cursor") or data.get("next") or data.get("cursor")
            if not cursor:
                break
            # 有的返回 next_url 完整 URL
            if str(cursor).startswith("http"):
                url = cursor
                params = {"apiKey": self.api_key}
            else:
                url = path
        return results

    # Grouped daily（全市场日线）
    def get_grouped_daily_aggs(self, date: str, adjusted: bool) -> List[Dict]:
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"
        params = {"adjusted": str(adjusted).lower()}
        data = self._request("GET", path, params)
        return data.get("results") or []

    # 技术指标（若使用 API）
    def list_indicators(self, ticker: str, timespan: str, windows: List[int]) -> Dict[str, List[Dict]]:
        # 建议本地计算，这里保留占位
        return {}

    # Snapshots（通用多票）
    def get_universal_snapshots(self, symbols: List[str]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        # Polygon 对 snapshot 多票接口可能限制批量数，分批请求
        CHUNK = 50
        for i in range(0, len(symbols), CHUNK):
            batch = symbols[i:i+CHUNK]
            tickers = ",".join(batch)
            path = "/v2/snapshot/locale/us/markets/stocks/tickers"
            params = {"tickers": tickers}
            data = self._request("GET", path, params)
            for item in data.get("tickers", []) or []:
                sym = item.get("ticker")
                if sym:
                    out[sym] = item
        # 对缺失补空
        for s in symbols:
            out.setdefault(s, {})
        return out

    def get_gainers_losers(self, top_n: int) -> Dict[str, List[str]]:
        top_n = max(0, int(top_n))
        gainers = self._request("GET", "/v2/snapshot/locale/us/markets/stocks/gainers", {})
        losers = self._request("GET", "/v2/snapshot/locale/us/markets/stocks/losers", {})
        g = [x.get("ticker") for x in (gainers.get("tickers") or [])][:top_n]
        l = [x.get("ticker") for x in (losers.get("tickers") or [])][:top_n]
        return {"gainers": [t for t in g if t], "losers": [t for t in l if t]}

    # News v2（需要循环调用，通过next_cursor来判断是否读取完）
    def list_ticker_news(self, ticker: str, gte: str, lte: str, limit: int = 100, page_token: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        path = "/v2/reference/news"
        params: Dict[str, object] = {
            "ticker": ticker,
            "limit": min(limit, 1000),
            "order": "asc",
        }
        if gte:
            params["published_utc.gte"] = gte
        if lte:
            params["published_utc.lte"] = lte
        if page_token:
            params["cursor"] = page_token
        data = self._request("GET", path, params)
        items: List[Dict] = data.get("results") or []
        next_cursor = data.get("next_url") or data.get("next_page_token") or data.get("next_cursor")
        return items, (next_cursor if next_cursor else None)

    # Ticker events（日历/事件：可筛选 earnings 等）
    def list_ticker_events(self, ticker: str, types: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        path = f"/vX/reference/tickers/{ticker}/events"
        params: Dict[str, object] = {"limit": max(1, min(int(limit), 1000))}
        if types:
            params["types"] = types
        results: List[Dict] = []
        url = path
        cursor: Optional[str] = None
        while True:
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", url, params)
            if not data:
                break
            items = data.get("results") or []
            results.extend(items)
            cursor = data.get("next_url") or data.get("next_page_token") or data.get("next_cursor")
            if not cursor:
                break
            if str(cursor).startswith("http"):
                url = cursor
                params = {"apiKey": self.api_key}
            else:
                url = path
        return results

    # Corporate actions（分红）
    def list_dividends(self, ticker: str) -> List[Dict]:
        path = "/v3/reference/dividends"
        params = {"ticker": ticker, "limit": 1000}
        data = self._request("GET", path, params)
        return data.get("results") or []

    # Corporate actions（拆股，获取股票的拆分记录）
    def list_splits(self, ticker: str) -> List[Dict]:
        path = "/v3/reference/splits"
        params = {"ticker": ticker, "limit": 1000}
        data = self._request("GET", path, params)
        return data.get("results") or []

    # Corporate actions（股票详情，获取股票的详情信息）
    def get_ticker_details(self, ticker: str, date: Optional[str] = None) -> Dict:
        path = f"/v3/reference/tickers/{ticker}"
        params: Dict[str, object] = {}
        if date:
            params["date"] = date
        data = self._request("GET", path, params)
        return data.get("results") or {}

    # Market status（获取市场状态）
    def get_market_status(self) -> Dict:
        data = self._request("GET", "/v1/marketstatus/now", {})
        return data or {"market": "unknown"}

    # Financials（财务报表，vX）
    def list_financials(self, ticker: str, timeframe: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """获取财务报表列表。timeframe 可为 'annual' 或 'quarterly'（Polygon 接口支持的口径）。
        返回原始 results 列表，调用方自行裁剪/缓存。
        """
        path = "/vX/reference/financials"
        params: Dict[str, object] = {"ticker": ticker, "limit": max(1, min(int(limit), 1000))}
        if timeframe:
            params["timeframe"] = timeframe
        data = self._request("GET", path, params)
        return data.get("results") or [] 