import yfinance as yf
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .base_data_source import BaseDataSource


class YFinanceDataSource(BaseDataSource):
    """YFinance数据源实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # YFinance不需要API密钥，但我们保留config参数以保持接口一致性
        self.cache = {}  # 简单的内存缓存
    
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
        
        # 转换interval到YFinance格式
        yf_interval = '1d'  # 默认为日线
        if interval == 'hourly':
            yf_interval = '1h'
        elif interval == 'minute':
            yf_interval = '1m'
        elif interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']:
            yf_interval = interval
        
        try:
            # 使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            df = await loop.run_in_executor(
                None, 
                lambda: ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=yf_interval
                )
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列以匹配Finnhub格式
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return df
            
        except Exception as e:
            print(f"获取 {symbol} 历史数据失败: {e}")
            return pd.DataFrame()
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """获取实时价格数据"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            # 使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            
            # 获取最新的价格信息
            data = await loop.run_in_executor(
                None, 
                lambda: ticker.history(period='1d')
            )
            
            if data.empty:
                raise Exception(f"No price data available for {symbol}")
            
            # 获取最新行
            latest = data.iloc[-1]
            
            # 计算涨跌幅
            if len(data) > 1:
                prev_close = data.iloc[-2]['Close']
                change = latest['Close'] - prev_close
                change_percent = (change / prev_close) * 100 if prev_close else 0
            else:
                change = latest['Close'] - latest['Open']
                change_percent = (change / latest['Open']) * 100 if latest['Open'] else 0
            
            return {
                "symbol": symbol,
                "price": latest['Close'],
                "open": latest['Open'],
                "high": latest['High'],
                "low": latest['Low'],
                "volume": latest['Volume'],
                "date": latest.name.strftime("%Y-%m-%d"),
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
            # 使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            info = await loop.run_in_executor(
                None, 
                lambda: ticker.info
            )
            
            if not info:
                raise Exception(f"No market info available for {symbol}")
            
            return {
                "symbol": symbol,
                "name": info.get("shortName", ""),
                "description": info.get("longBusinessSummary", ""),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "start_date": "",  # YFinance不直接提供此信息
                "end_date": "",    # YFinance不直接提供此信息
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
            # 使用run_in_executor在异步环境中运行
            loop = asyncio.get_event_loop()
            
            if symbol:
                # 获取特定公司的新闻
                ticker = yf.Ticker(symbol)
                news_data = await loop.run_in_executor(
                    None, 
                    lambda: ticker.news
                )
                
                # 限制返回的新闻数量
                news_data = news_data[:limit]
                
                # 转换为标准格式
                news_list = []
                for article in news_data:
                    # 转换时间戳为ISO格式日期
                    if 'providerPublishTime' in article:
                        published_date = datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat()
                    else:
                        published_date = datetime.now().isoformat()
                    
                    news_list.append({
                        "id": str(article.get("uuid", "")),
                        "title": article.get("title", ""),
                        "description": article.get("summary", ""),
                        "url": article.get("link", ""),
                        "published_date": published_date,
                        "source": article.get("publisher", ""),
                        "tags": [],  # YFinance不提供标签
                        "tickers": article.get("relatedTickers", []) if article.get("relatedTickers") else []
                    })
                
                return news_list
            else:
                # YFinance不提供一般市场新闻，返回空列表
                return []
            
        except Exception as e:
            print(f"获取新闻数据时出错: {e}")
            return []
    
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票代码"""
        try:
            # YFinance没有直接的符号搜索API，这里提供一个简单的实现
            # 实际应用中可能需要更复杂的解决方案
            common_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
            results = []
            
            query = query.upper()
            for symbol in common_symbols:
                if query in symbol:
                    # 使用run_in_executor在异步环境中运行
                    loop = asyncio.get_event_loop()
                    ticker = yf.Ticker(symbol)
                    info = await loop.run_in_executor(
                        None, 
                        lambda: ticker.info
                    )
                    
                    results.append({
                        "symbol": symbol,
                        "name": info.get("shortName", ""),
                        "exchange": info.get("exchange", "")
                    })
            
            return results
            
        except Exception as e:
            print(f"搜索股票代码失败: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            # 尝试获取AAPL的信息作为连接测试
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker("AAPL")
            info = await loop.run_in_executor(
                None, 
                lambda: ticker.info
            )
            
            # 如果能获取到信息，说明连接正常
            return "shortName" in info
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
            ticker = yf.Ticker(symbol)
            info = await loop.run_in_executor(
                None, 
                lambda: ticker.info
            )
            
            # 提取关键财务指标
            financials = {
                "symbol": symbol,
                "metric": {
                    "peRatio": info.get("trailingPE"),
                    "peForward": info.get("forwardPE"),
                    "epsTTM": info.get("trailingEps"),
                    "epsForward": info.get("forwardEps"),
                    "dividendYield": info.get("dividendYield"),
                    "marketCap": info.get("marketCap"),
                    "52WeekHigh": info.get("fiftyTwoWeekHigh"),
                    "52WeekLow": info.get("fiftyTwoWeekLow"),
                    "beta": info.get("beta"),
                    "averageVolume": info.get("averageVolume")
                }
            }
            
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
            ticker = yf.Ticker(symbol)
            
            if freq == 'annual':
                income_stmt = await loop.run_in_executor(None, lambda: ticker.income_stmt)
                balance_sheet = await loop.run_in_executor(None, lambda: ticker.balance_sheet)
                cash_flow = await loop.run_in_executor(None, lambda: ticker.cashflow)
            else:  # quarterly
                income_stmt = await loop.run_in_executor(None, lambda: ticker.quarterly_income_stmt)
                balance_sheet = await loop.run_in_executor(None, lambda: ticker.quarterly_balance_sheet)
                cash_flow = await loop.run_in_executor(None, lambda: ticker.quarterly_cashflow)
            
            # 转换为可序列化的格式
            financials = {
                "symbol": symbol,
                "freq": freq,
                "data": {
                    "income_statement": self._convert_dataframe_to_dict(income_stmt),
                    "balance_sheet": self._convert_dataframe_to_dict(balance_sheet),
                    "cash_flow": self._convert_dataframe_to_dict(cash_flow)
                }
            }
            
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
            ticker = yf.Ticker(symbol)
            earnings_data = await loop.run_in_executor(
                None, 
                lambda: ticker.earnings_dates
            )
            
            if earnings_data is None or earnings_data.empty:
                return []
            
            # 限制返回的数据量
            earnings_data = earnings_data.head(limit)
            
            # 转换为列表格式
            earnings_list = []
            for date, row in earnings_data.iterrows():
                earnings_list.append({
                    "period": date.strftime("%Y-%m-%d"),
                    "epsActual": row.get("Reported EPS", None),
                    "epsEstimate": row.get("EPS Estimate", None),
                    "epsSurprise": None,  # YFinance不直接提供惊喜值
                    "epsSurprisePercent": row.get("Surprise(%)", None)
                })
            
            return earnings_list
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
            ticker = yf.Ticker(symbol)
            recommendations = await loop.run_in_executor(
                None, 
                lambda: ticker.recommendations
            )
            
            if recommendations is None or recommendations.empty:
                return []
            
            # 转换为列表格式
            trends = []
            for date, row in recommendations.iterrows():
                # 兼容 int / Timestamp / str 索引
                period = date
                if isinstance(period, (datetime, pd.Timestamp)):
                    period = period.strftime("%Y-%m-%d")
                else:
                    period = str(period)
                trends.append({
                    "period": period,
                    "strongBuy": row.get("Strong Buy", 0) if "Strong Buy" in row else 0,
                    "buy": row.get("Buy", 0) if "Buy" in row else 0,
                    "hold": row.get("Hold", 0) if "Hold" in row else 0,
                    "sell": row.get("Sell", 0) if "Sell" in row else 0,
                    "strongSell": row.get("Strong Sell", 0) if "Strong Sell" in row else 0,
                    "grade": row.get("To Grade", ""),
                    "action": row.get("Action", ""),
                    "firm": row.get("Firm", "")
                })
            
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
                "pe_ratio": metrics.get("peRatio", None),
                "eps_ttm": metrics.get("epsTTM", None),
                "dividend_yield": metrics.get("dividendYield", None),
                "market_cap": metrics.get("marketCap", None),
                "52w_high": metrics.get("52WeekHigh", None),
                "52w_low": metrics.get("52WeekLow", None),
                "52w_change": None,  # YFinance不直接提供
                "beta": metrics.get("beta", None),
                "avg_volume": metrics.get("averageVolume", None)
            }
            
            return key_metrics
        except Exception as e:
            print(f"获取 {symbol} 财务指标失败: {e}")
            return {"error": str(e)}
    
    def _convert_dataframe_to_dict(self, df):
        """将DataFrame转换为可序列化的字典格式"""
        if df is None or df.empty:
            return {}
            
        # 将索引转换为字符串
        df_dict = {}
        for col in df.columns:
            df_dict[str(col)] = {}
            for idx in df.index:
                # 将索引转换为ISO格式字符串
                idx_str = str(idx) if not isinstance(idx, pd.Timestamp) else idx.isoformat()
                df_dict[str(col)][idx_str] = df.loc[idx, col]
                
        return df_dict 