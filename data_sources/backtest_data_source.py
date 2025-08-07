import pandas as pd
import numpy as np
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from .base_data_source import BaseDataSource


class BacktestDataSource(BaseDataSource):
    """回测数据源实现，用于加载历史数据并按照时间点提供数据"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 回测配置
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        self.current_date = None
        self.data_dir = config.get("data_dir", "backtest_data")
        
        # 数据缓存
        self.price_data = {}  # 价格数据缓存 {symbol: DataFrame}
        self.news_data = []   # 新闻数据缓存 [news_items]
        self.financial_data = {}  # 财务数据缓存 {symbol: {data}}
        self.market_info = {}  # 市场信息缓存 {symbol: {info}}
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化当前日期为开始日期
        if self.start_date:
            if isinstance(self.start_date, str):
                self.current_date = datetime.strptime(self.start_date, "%Y-%m-%d")
            else:
                self.current_date = self.start_date
    
    def set_current_date(self, date: Union[str, datetime]):
        """设置当前回测日期"""
        if isinstance(date, str):
            self.current_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            # 确保日期对象没有时区信息
            if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                # 创建一个没有时区信息的新日期对象
                self.current_date = datetime(date.year, date.month, date.day)
            else:
                self.current_date = date
    
    async def load_data(self, symbols: List[str]):
        """加载回测所需的所有数据"""
        print(f"正在加载回测数据，时间范围: {self.start_date} 到 {self.end_date}")
        
        # 并行加载所有数据
        tasks = []
        for symbol in symbols:
            tasks.append(self.load_price_data(symbol))
            tasks.append(self.load_financial_data(symbol))
            tasks.append(self.load_market_info(symbol))
        
        # 加载新闻数据
        tasks.append(self.load_news_data())
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        print(f"数据加载完成: {len(symbols)} 个股票, {len(self.news_data)} 条新闻")
    
    async def load_price_data(self, symbol: str):
        """加载股票价格数据"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_prices_processed.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # 确保索引是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                
                # 安全地检查时区属性
                try:
                    if hasattr(df.index, 'tz') and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                except AttributeError:
                    # 如果索引没有tz属性，跳过这一步
                    pass
                
                self.price_data[symbol] = df
                print(f"加载 {symbol} 价格数据: {len(df)} 行")
            else:
                print(f"警告: {symbol} 价格数据文件不存在")
        except Exception as e:
            print(f"加载 {symbol} 价格数据失败: {e}")
    
    async def load_news_data(self):
        """加载新闻数据"""
        try:
            file_path = os.path.join(self.data_dir, "news_data.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.news_data = json.load(f)
                print(f"加载新闻数据: {len(self.news_data)} 条")
            else:
                print("警告: 新闻数据文件不存在")
        except Exception as e:
            print(f"加载新闻数据失败: {e}")
    
    async def load_financial_data(self, symbol: str):
        """加载财务数据"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_financials.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.financial_data[symbol] = json.load(f)
                print(f"加载 {symbol} 财务数据成功")
            else:
                print(f"警告: {symbol} 财务数据文件不存在")
        except Exception as e:
            print(f"加载 {symbol} 财务数据失败: {e}")
    
    async def load_market_info(self, symbol: str):
        """加载市场信息"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_info.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.market_info[symbol] = json.load(f)
                print(f"加载 {symbol} 市场信息成功")
            else:
                print(f"警告: {symbol} 市场信息文件不存在")
        except Exception as e:
            print(f"加载 {symbol} 市场信息失败: {e}")
    
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
        
        if symbol not in self.price_data:
            await self.load_price_data(symbol)
            
        if symbol not in self.price_data:
            return pd.DataFrame()
        
        # 获取数据
        df = self.price_data[symbol]
        
        # 过滤日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return df
    
    async def get_market_data(self) -> Dict[str, Dict[str, Any]]:
        """获取市场数据（在回测中，返回当前日期的价格数据）"""
        market_data = {}
        
        try:
            for symbol in self.trading_symbols:
                try:
                    price_data = await self.get_real_time_price(symbol)
                    market_data[symbol] = price_data
                except Exception as e:
                    print(f"获取 {symbol} 数据失败: {e}")
            
            return market_data
        except Exception as e:
            print(f"获取市场数据失败: {e}")
            return {}
            
    async def _get_current_price(self, symbol: str) -> float:
        """获取当前价格（异步）"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if not self.current_date:
            raise ValueError("Current date is not set")
        
        if symbol not in self.price_data:
            await self.load_price_data(symbol)
            
        if symbol not in self.price_data:
            raise Exception(f"No price data available for {symbol}")
        
        # 获取数据
        df = self.price_data[symbol]
        
        # 确保current_date是datetime类型，并处理时区问题
        if isinstance(self.current_date, str):
            current_date = pd.Timestamp(self.current_date).tz_localize(None)
        else:
            # 移除时区信息
            current_date = pd.Timestamp(self.current_date).tz_localize(None)
            
        # 确保索引是datetime类型，并且没有时区信息
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        # 安全地检查和移除索引的时区信息
        try:
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        except AttributeError:
            # 如果索引没有tz属性，跳过这一步
            pass
            
        # 获取当前日期或之前的数据
        available_dates = df.index[df.index <= current_date]
        
        if len(available_dates) == 0:
            raise Exception(f"No price data available for {symbol} on or before {self.current_date}")
        
        latest_date = available_dates[-1]
        latest = df.loc[latest_date]
        
        return latest['Close']
    
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """获取实时价格数据（在回测中，返回当前日期的价格）"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if not self.current_date:
            raise ValueError("Current date is not set")
        
        if symbol not in self.price_data:
            await self.load_price_data(symbol)
            
        if symbol not in self.price_data:
            raise Exception(f"No price data available for {symbol}")
        
        # 获取数据
        df = self.price_data[symbol]
        
        # 确保current_date是datetime类型，并处理时区问题
        if isinstance(self.current_date, str):
            current_date = pd.Timestamp(self.current_date).tz_localize(None)
        else:
            # 移除时区信息
            current_date = pd.Timestamp(self.current_date).tz_localize(None)
            
        # 确保索引是datetime类型，并且没有时区信息
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        # 安全地检查和移除索引的时区信息
        try:
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        except AttributeError:
            # 如果索引没有tz属性，跳过这一步
            pass
        
        # 获取当前日期或之前最近的数据
        available_dates = df.index[df.index <= current_date]
        
        if len(available_dates) == 0:
            raise Exception(f"No price data available for {symbol} on or before {self.current_date}")
        
        latest_date = available_dates[-1]
        latest = df.loc[latest_date]
        
        # 计算涨跌幅
        if len(available_dates) > 1:
            prev_date = available_dates[-2]
            prev_close = df.loc[prev_date]['Close']
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
            "date": latest_date.strftime("%Y-%m-%d"),
            "change": change,
            "change_percent": change_percent
        }
    
    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取市场信息"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if symbol not in self.market_info:
            await self.load_market_info(symbol)
            
        if symbol not in self.market_info:
            raise Exception(f"No market info available for {symbol}")
        
        return self.market_info[symbol]
    
    async def get_news(
        self, 
        symbol: Optional[str] = None,
        limit: int = 10,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """获取新闻数据（在回测中，返回当前日期之前的新闻）"""
        if not self.current_date:
            raise ValueError("Current date is not set")
        
        # 确保新闻数据已加载
        if not self.news_data:
            await self.load_news_data()
        
        # 确保current_date没有时区信息
        if hasattr(self.current_date, 'tzinfo') and self.current_date.tzinfo is not None:
            current_date = datetime(self.current_date.year, self.current_date.month, self.current_date.day)
        else:
            current_date = self.current_date
            
        # 获取当前日期的字符串表示
        current_date_str = current_date.strftime('%Y-%m-%d')
        filtered_news = []
        
        for news in self.news_data:
            news_date = news.get('published_date', '').split('T')[0]
            
            # 检查新闻日期是否在当前日期之前且在days_back范围内
            if news_date <= current_date_str:
                try:
                    # 将新闻日期转换为datetime对象（无时区）
                    news_datetime = datetime.strptime(news_date, '%Y-%m-%d')
                    
                    # 计算新闻日期与当前日期的差距
                    days_diff = (current_date - news_datetime).days
                    
                    # 检查是否在days_back范围内
                    if days_diff <= days_back:
                        # 如果指定了股票代码，检查新闻是否与该股票相关
                        if symbol:
                            tickers = news.get('tickers', [])
                            if symbol in tickers:
                                filtered_news.append(news)
                        else:
                            filtered_news.append(news)
                except Exception as e:
                    print(f"处理新闻日期时出错: {e}, 新闻日期: {news_date}")
                    continue
        
        # 限制返回的新闻数量
        return filtered_news[:limit]
    
    async def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """获取财务指标"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if symbol not in self.financial_data:
            await self.load_financial_data(symbol)
            
        if symbol not in self.financial_data:
            return {"error": f"No financial data available for {symbol}"}
        
        financials = self.financial_data[symbol]
        
        # 提取关键指标
        if "key_metrics" in financials:
            return financials["key_metrics"]
        else:
            return {"error": "Key metrics not available"}
    
    async def get_earnings_surprises(self, symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
        """获取盈利惊喜数据"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if symbol not in self.financial_data:
            await self.load_financial_data(symbol)
            
        if symbol not in self.financial_data:
            return []
        
        financials = self.financial_data[symbol]
        
        # 提取盈利惊喜数据
        if "earnings_surprises" in financials:
            surprises = financials["earnings_surprises"]
            
            # 过滤当前日期之前的数据
            if self.current_date:
                # 确保current_date没有时区信息
                if hasattr(self.current_date, 'tzinfo') and self.current_date.tzinfo is not None:
                    current_date = datetime(self.current_date.year, self.current_date.month, self.current_date.day)
                else:
                    current_date = self.current_date
                
                current_date_str = current_date.strftime('%Y-%m-%d')
                filtered_surprises = []
                
                for surprise in surprises:
                    period = surprise.get('period', '')
                    if isinstance(period, datetime):
                        period = period.strftime('%Y-%m-%d')
                    elif not isinstance(period, str):
                        period = str(period)
                        
                    if period <= current_date_str:
                        filtered_surprises.append(surprise)
                
                return filtered_surprises[:limit]
            else:
                return surprises[:limit]
        else:
            return []
    
    async def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """获取分析师推荐趋势"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if symbol not in self.financial_data:
            await self.load_financial_data(symbol)
            
        if symbol not in self.financial_data:
            return []
        
        financials = self.financial_data[symbol]
        
        # 提取推荐趋势数据
        if "recommendation_trends" in financials:
            trends = financials["recommendation_trends"]
            
            # 过滤当前日期之前的数据
            if self.current_date:
                # 确保current_date没有时区信息
                if hasattr(self.current_date, 'tzinfo') and self.current_date.tzinfo is not None:
                    current_date = datetime(self.current_date.year, self.current_date.month, self.current_date.day)
                else:
                    current_date = self.current_date
                
                current_date_str = current_date.strftime('%Y-%m-%d')
                filtered_trends = []
                
                for trend in trends:
                    period = trend.get('period', '')
                    if isinstance(period, datetime):
                        period = period.strftime('%Y-%m-%d')
                    elif not isinstance(period, str):
                        period = str(period)
                        
                    if period <= current_date_str:
                        filtered_trends.append(trend)
                
                return filtered_trends
            else:
                return trends
        else:
            return []
    
    async def get_company_financials(self, symbol: str, quarters: int = 4, earnings_limit: int = 4) -> Dict[str, Any]:
        """获取公司综合财务数据"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if symbol not in self.financial_data:
            await self.load_financial_data(symbol)
            
        if symbol not in self.financial_data:
            return {"error": f"No financial data available for {symbol}"}
        
        # 直接返回缓存的财务数据
        return self.financial_data[symbol]
    
    async def test_connection(self) -> bool:
        """测试连接（在回测中总是返回True）"""
        return True 

    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票代码（回测中使用预定义的符号列表）"""
        results = []
        
        # 在已加载的市场信息中搜索
        for symbol, info in self.market_info.items():
            if query.upper() in symbol or query.lower() in info.get("name", "").lower():
                results.append({
                    "symbol": symbol,
                    "name": info.get("name", ""),
                    "exchange": info.get("exchange", "")
                })
        
        # 如果没有找到匹配项，返回一些常见的股票
        if not results:
            common_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
            for symbol in common_symbols:
                if query.upper() in symbol:
                    # 尝试获取市场信息
                    if symbol in self.market_info:
                        info = self.market_info[symbol]
                        results.append({
                            "symbol": symbol,
                            "name": info.get("name", ""),
                            "exchange": info.get("exchange", "")
                        })
                    else:
                        # 如果没有市场信息，添加基本信息
                        results.append({
                            "symbol": symbol,
                            "name": "",
                            "exchange": ""
                        })
        
        return results 