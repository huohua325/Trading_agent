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
        elif interval in ['1', '5', '15', '30', '60', 'D', 'W', 'M']:
            resolution = interval
        
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
            return []
    
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

    async def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """获取基本财务数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含基本财务指标的字典
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            loop = asyncio.get_event_loop()
            financials = await loop.run_in_executor(
                None, 
                lambda: self.client.company_basic_financials(symbol, 'all')
            )
            
            return financials
        except Exception as e:
            print(f"获取 {symbol} 基本财务数据失败: {e}")
            return {"error": str(e)}
    
    async def get_reported_financials(self, symbol: str, freq: str = 'annual') -> Dict[str, Any]:
        """获取已报告的财务报表
        
        Args:
            symbol: 股票代码
            freq: 频率，'annual'或'quarterly'
            
        Returns:
            包含财务报表的字典
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            loop = asyncio.get_event_loop()
            financials = await loop.run_in_executor(
                None, 
                lambda: self.client.financials_reported(symbol=symbol, freq=freq)
            )
            
            return financials
        except Exception as e:
            print(f"获取 {symbol} 已报告的财务报表失败: {e}")
            return {"error": str(e)}
    
    async def get_earnings_surprises(self, symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
        """获取盈利惊喜数据
        
        Args:
            symbol: 股票代码
            limit: 返回的数据点数量
            
        Returns:
            盈利惊喜数据列表
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            loop = asyncio.get_event_loop()
            earnings = await loop.run_in_executor(
                None, 
                lambda: self.client.company_earnings(symbol, limit=limit)
            )
            
            return earnings
        except Exception as e:
            print(f"获取 {symbol} 盈利惊喜数据失败: {e}")
            return []
    
    async def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """获取分析师推荐趋势
        
        Args:
            symbol: 股票代码
            
        Returns:
            分析师推荐趋势数据列表
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            loop = asyncio.get_event_loop()
            trends = await loop.run_in_executor(
                None, 
                lambda: self.client.recommendation_trends(symbol)
            )
            
            return trends
        except Exception as e:
            print(f"获取 {symbol} 推荐趋势失败: {e}")
            return []
    
    async def get_company_financials(self, symbol: str, quarters: int = 4, earnings_limit: int = 4) -> Dict[str, Any]:
        """获取公司综合财务数据（整合多个API）
        
        Args:
            symbol: 股票代码
            quarters: 获取的季度财务数据数量
            earnings_limit: 获取的盈利惊喜数据点数量
            
        Returns:
            包含多种财务数据的字典
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # 并行获取所有财务数据
        basic_financials_task = self.get_basic_financials(symbol)
        reported_financials_task = self.get_reported_financials(symbol, freq='quarterly' if quarters > 0 else 'annual')
        earnings_surprises_task = self.get_earnings_surprises(symbol, limit=earnings_limit)
        recommendation_trends_task = self.get_recommendation_trends(symbol)
        
        # 等待所有任务完成
        results = await asyncio.gather(
            basic_financials_task,
            reported_financials_task,
            earnings_surprises_task,
            recommendation_trends_task,
            return_exceptions=True
        )
        
        # 处理结果
        financials = {
            "basic_financials": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "reported_financials": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "earnings_surprises": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "recommendation_trends": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])}
        }
        
        return financials
    
    async def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """提取关键财务指标
        
        Args:
            symbol: 股票代码
            
        Returns:
            关键财务指标字典
        """
        try:
            financials = await self.get_basic_financials(symbol)
            
            if "error" in financials:
                return {"error": financials["error"]}
            
            metrics = financials.get("metric", {})
            
            # 提取关键指标
            key_metrics = {
                "pe_ratio": metrics.get("peBasicExclExtraTTM", None),
                "eps_ttm": metrics.get("epsBasicExclExtraItemsTTM", None),
                "dividend_yield": metrics.get("dividendYieldIndicatedAnnual", None),
                "market_cap": metrics.get("marketCapitalization", None),
                "52w_high": metrics.get("52WeekHigh", None),
                "52w_low": metrics.get("52WeekLow", None),
                "52w_change": metrics.get("52WeekPriceReturnDaily", None),
                "beta": metrics.get("beta", None),
                "avg_volume": metrics.get("10DayAverageTradingVolume", None)
            }
            
            return key_metrics
        except Exception as e:
            print(f"获取 {symbol} 财务指标失败: {e}")
            return {"error": str(e)} 