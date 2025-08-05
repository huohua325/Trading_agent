import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .base_data_source import BaseDataSource


class TiingoDataSource(BaseDataSource):
    """Tiingo数据源实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("tiingo_api_key")
        self.base_url = config.get("tiingo_base_url", "https://api.tiingo.com/tiingo")
        
        if not self.api_key:
            raise ValueError("Tiingo API key is required")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "daily"
    ) -> pd.DataFrame:
        """获取历史价格数据"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        url = f"{self.base_url}/daily/{symbol}/prices"
        params = {
            "token": self.api_key,
            "startDate": self.format_datetime(start_date),
            "endDate": self.format_datetime(end_date),
            "format": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get historical data: {response.status}")
                
                data = await response.json()
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # 重命名列以匹配标准格式
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adjClose': 'Adj Close'
                }, inplace=True)
                
                return df
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """获取实时价格数据"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        url = f"{self.base_url}/daily/{symbol}/prices"
        params = {
            "token": self.api_key,
            "format": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get real-time price: {response.status}")
                
                data = await response.json()
                
                if not data:
                    raise Exception("No price data available")
                
                # 获取最新的价格数据
                latest = data[-1] if isinstance(data, list) else data
                
                return {
                    "symbol": symbol,
                    "price": latest.get("close"),
                    "open": latest.get("open"),
                    "high": latest.get("high"),
                    "low": latest.get("low"),
                    "volume": latest.get("volume"),
                    "date": latest.get("date"),
                    "change": latest.get("close", 0) - latest.get("open", 0),
                    "change_percent": ((latest.get("close", 0) - latest.get("open", 0)) / 
                                     latest.get("open", 1)) * 100 if latest.get("open") else 0
                }
    
    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取市场信息"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Tiingo的meta信息API
        url = f"{self.base_url}/daily/{symbol}"
        params = {
            "token": self.api_key,
            "format": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get market info: {response.status}")
                
                data = await response.json()
                
                return {
                    "symbol": symbol,
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "exchange": data.get("exchangeCode", ""),
                    "start_date": data.get("startDate", ""),
                    "end_date": data.get("endDate", ""),
                    "is_active": data.get("isActive", False)
                }
    
    async def get_news(
        self, 
        symbol: Optional[str] = None,
        limit: int = 10,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """获取新闻数据"""
        url = f"{self.base_url}/news"
        
        params = {
            "token": self.api_key,
            "limit": limit,
            "format": "json"
        }
        
        if symbol:
            params["tickers"] = symbol
        
        # 添加日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        params["startDate"] = self.format_datetime(start_date)
        params["endDate"] = self.format_datetime(end_date)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get news: {response.status}")
                
                data = await response.json()
                
                news_list = []
                for article in data:
                    news_list.append({
                        "id": article.get("id"),
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "published_date": article.get("publishedDate", ""),
                        "source": article.get("source", ""),
                        "tags": article.get("tags", []),
                        "tickers": article.get("tickers", [])
                    })
                
                return news_list
    
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票代码"""
        # Tiingo不直接支持搜索，这里提供一个基础实现
        # 在实际使用中，可能需要使用其他服务或预定义的股票列表
        
        # 简单的验证，检查是否是有效的股票代码格式
        if len(query) <= 5 and query.isalpha():
            try:
                # 尝试获取该股票的信息来验证是否存在
                info = await self.get_market_info(query.upper())
                return [{
                    "symbol": query.upper(),
                    "name": info.get("name", ""),
                    "exchange": info.get("exchange", "")
                }]
            except:
                return []
        
        return []
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            url = f"{self.base_url}/daily/AAPL"
            params = {
                "token": self.api_key,
                "format": "json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    return response.status == 200
        except:
            return False 