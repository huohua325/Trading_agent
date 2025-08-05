import finnhub
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .base_data_source import BaseDataSource


class FinnhubDataSource(BaseDataSource):
    """Finnhub数据源实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("finnhub_api_key")
        
        if not self.api_key:
            raise ValueError("Finnhub API key is required")
        
        # 初始化Finnhub客户端
        self.client = finnhub.Client(api_key=self.api_key)
    
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
        
        # 转换为Unix时间戳
        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())
        
        # 转换interval到Finnhub格式
        resolution = 'D'  # 默认为日线
        if interval == 'hourly':
            resolution = '60'
        elif interval == 'minute':
            resolution = '1'
        
        try:
            # Finnhub API调用是同步的，使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            candles = await loop.run_in_executor(
                None, 
                lambda: self.client.stock_candles(symbol, resolution, from_timestamp, to_timestamp)
            )
            
            if candles['s'] == 'no_data':
                return pd.DataFrame()
            
            # 创建DataFrame
            df = pd.DataFrame({
                'Open': candles['o'],
                'High': candles['h'],
                'Low': candles['l'],
                'Close': candles['c'],
                'Volume': candles['v']
            })
            
            # 添加日期索引
            df.index = pd.to_datetime(candles['t'], unit='s')
            df.index.name = 'date'
            
            return df
            
        except Exception as e:
            print(f"获取 {symbol} 历史数据失败: {e}")
            return pd.DataFrame()
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """获取实时价格数据"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            # Finnhub API调用是同步的，使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(
                None, 
                lambda: self.client.quote(symbol)
            )
            
            # 计算涨跌幅
            change = quote.get('c', 0) - quote.get('pc', 0)
            change_percent = (change / quote.get('pc', 1)) * 100 if quote.get('pc') else 0
            
            return {
                "symbol": symbol,
                "price": quote.get('c'),  # 当前价格
                "open": quote.get('o'),   # 开盘价
                "high": quote.get('h'),   # 最高价
                "low": quote.get('l'),    # 最低价
                "volume": 0,              # Finnhub quote API不提供成交量
                "date": datetime.now().strftime("%Y-%m-%d"),
                "change": change,
                "change_percent": change_percent
            }
            
        except Exception as e:
            print(f"获取 {symbol} 实时价格失败: {e}")
            raise Exception(f"Failed to get real-time price: {e}")
    
    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取市场信息"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            # Finnhub API调用是同步的，使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            profile = await loop.run_in_executor(
                None, 
                lambda: self.client.company_profile2(symbol=symbol)
            )
            
            if not profile:
                raise Exception(f"No market info available for {symbol}")
            
            return {
                "symbol": symbol,
                "name": profile.get("name", ""),
                "description": profile.get("finnhubIndustry", ""),
                "exchange": profile.get("exchange", ""),
                "start_date": "",  # Finnhub不提供此信息
                "end_date": "",    # Finnhub不提供此信息
                "is_active": True  # 假设是活跃的
            }
            
        except Exception as e:
            print(f"获取 {symbol} 市场信息失败: {e}")
            raise Exception(f"Failed to get market info: {e}")
    
    async def get_news(
        self, 
        symbol: Optional[str] = None,
        limit: int = 10,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """获取新闻数据"""
        try:
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            
            # Finnhub API调用是同步的，使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            
            if symbol:
                # 获取特定公司的新闻
                news_data = await loop.run_in_executor(
                    None, 
                    lambda: self.client.company_news(symbol, _from=from_date, to=to_date)
                )
            else:
                # 获取一般市场新闻
                news_data = await loop.run_in_executor(
                    None, 
                    lambda: self.client.general_news('general', min_id=0)
                )
            
            # 限制返回的新闻数量
            news_data = news_data[:limit]
            
            # 转换为标准格式
            news_list = []
            for article in news_data:
                # 转换Unix时间戳为ISO格式日期
                published_date = datetime.fromtimestamp(article.get("datetime", 0)).isoformat()
                
                news_list.append({
                    "id": str(article.get("id", "")),
                    "title": article.get("headline", ""),
                    "description": article.get("summary", ""),
                    "url": article.get("url", ""),
                    "published_date": published_date,
                    "source": article.get("source", ""),
                    "tags": article.get("category", "").split(",") if article.get("category") else [],
                    "tickers": article.get("related", "").split(",") if article.get("related") else []
                })
            
            return news_list
            
        except Exception as e:
            print(f"获取新闻数据时出错: {e}")
            # 返回模拟数据作为降级方案
            return self._generate_mock_news(symbol)
    
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票代码"""
        try:
            # Finnhub API调用是同步的，使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None, 
                lambda: self.client.symbol_lookup(query)
            )
            
            results = []
            for item in search_results.get('result', [])[:10]:  # 限制结果数量
                results.append({
                    "symbol": item.get("symbol", ""),
                    "name": item.get("description", ""),
                    "exchange": item.get("displaySymbol", "")
                })
            
            return results
            
        except Exception as e:
            print(f"搜索股票代码失败: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            # 尝试获取AAPL的报价作为连接测试
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(
                None, 
                lambda: self.client.quote("AAPL")
            )
            
            # 如果能获取到价格，说明连接正常
            return quote.get('c', 0) > 0
        except:
            return False
    
    def _generate_mock_news(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """生成模拟新闻数据（当API不可用时）"""
        current_date = datetime.now().isoformat()
        
        mock_news = [
            {
                "id": "mock_1",
                "title": "市场概览：今日股市表现平稳",
                "description": "主要指数小幅波动，投资者保持谨慎态度。",
                "url": "https://example.com/market-overview",
                "published_date": current_date,
                "source": "模拟新闻源",
                "tags": ["市场", "概览"],
                "tickers": []
            },
            {
                "id": "mock_2",
                "title": "经济数据：通胀率保持稳定",
                "description": "最新经济数据显示通胀率与上月持平，经济增长缓慢但稳定。",
                "url": "https://example.com/economic-data",
                "published_date": current_date,
                "source": "模拟新闻源",
                "tags": ["经济", "通胀"],
                "tickers": []
            },
            {
                "id": "mock_3",
                "title": "科技股今日领涨大盘",
                "description": "大型科技公司股价上涨，带动市场整体走高。",
                "url": "https://example.com/tech-stocks",
                "published_date": current_date,
                "source": "模拟新闻源",
                "tags": ["科技", "股票"],
                "tickers": ["AAPL", "MSFT", "GOOGL"]
            }
        ]
        
        # 如果指定了股票代码，添加相关的模拟新闻
        if symbol:
            mock_news.append({
                "id": f"mock_{symbol}",
                "title": f"{symbol}公司发布季度财报",
                "description": f"{symbol}公司最新财报显示业绩符合预期，收入稳定增长。",
                "url": f"https://example.com/{symbol}-earnings",
                "published_date": current_date,
                "source": "模拟新闻源",
                "tags": ["财报", "收益"],
                "tickers": [symbol]
            })
        
        return mock_news 