import os
import json
import pandas as pd
import asyncio
import aiohttp
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """多数据源数据下载工具，支持从多个API源下载历史数据并保存为回测所需的格式"""
    
    def __init__(self, output_dir: str = "backtest_data"):
        """初始化数据下载器
        
        Args:
            output_dir: 数据保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # API密钥配置（从环境变量读取）
        self.api_keys = {
            "finnhub": os.getenv("FINNHUB_API_KEY"),
            "polygon": os.getenv("POLYGON_API_KEY"),
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "tiingo": os.getenv("TIINGO_API_KEY"),
            "quandl": os.getenv("QUANDL_API_KEY"),
            "newsapi": os.getenv("NEWS_API_KEY")
        }
        
        # 数据源优先级配置
        self.data_sources = {
            "price": ["yfinance", "finnhub", "polygon", "alpha_vantage", "tiingo", "quandl"],
            "news": ["finnhub", "newsapi", "yfinance"],
            "financials": ["yfinance", "finnhub", "alpha_vantage", "tiingo"],
            "market_info": ["yfinance", "finnhub", "polygon", "alpha_vantage", "tiingo"]
        }
        
        # API速率限制配置
        self.rate_limits = {
            "finnhub": {"calls_per_minute": 60, "last_call": 0},
            "polygon": {"calls_per_minute": 5, "last_call": 0},
            "alpha_vantage": {"calls_per_minute": 5, "last_call": 0},
            "tiingo": {"calls_per_minute": 10, "last_call": 0},
            "quandl": {"calls_per_minute": 10, "last_call": 0},
            "newsapi": {"calls_per_minute": 10, "last_call": 0}
        }
    
    async def _rate_limit(self, source: str):
        """实现API速率限制"""
        if source in self.rate_limits:
            limit = self.rate_limits[source]
            current_time = time.time()
            time_since_last = current_time - limit["last_call"]
            min_interval = 60.0 / limit["calls_per_minute"]
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
            
            self.rate_limits[source]["last_call"] = time.time()
    
    async def _make_api_request(self, session: aiohttp.ClientSession, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """通用API请求方法"""
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API请求失败: {response.status} - {url}")
                    return None
        except Exception as e:
            logger.error(f"API请求异常: {e}")
            return None
    
    async def download_all_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        include_news: bool = True,
        include_financials: bool = True,
        force_download: bool = False,
        test_all_apis: bool = False
    ):
        """下载所有需要的数据，使用多数据源确保完整性
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            include_news: 是否包含新闻数据
            include_financials: 是否包含财务数据
            force_download: 是否强制重新下载数据（即使文件已存在）
            test_all_apis: 是否测试所有API（会为每个API创建单独的目录）
        """
        if test_all_apis:
            await self._download_with_all_apis(symbols, start_date, end_date, include_news, include_financials, force_download)
            return
            
        print(f"开始多数据源下载: {len(symbols)} 个股票, 时间范围: {start_date} 到 {end_date}")
        
        # 创建任务列表
        tasks = []
        
        # 下载价格数据
        for symbol in symbols:
            price_file = os.path.join(self.output_dir, f"{symbol}_prices.csv")
            if force_download or not os.path.exists(price_file):
                tasks.append(self.download_price_data_multi_source(symbol, start_date, end_date))
            else:
                print(f"✅ {symbol} 价格数据文件已存在，跳过下载")
            
            # 下载市场信息
            info_file = os.path.join(self.output_dir, f"{symbol}_info.json")
            if force_download or not os.path.exists(info_file):
                tasks.append(self.download_market_info_multi_source(symbol))
            else:
                print(f"✅ {symbol} 市场信息文件已存在，跳过下载")
            
            # 下载财务数据
            if include_financials:
                financial_file = os.path.join(self.output_dir, f"{symbol}_financials.json")
                if force_download or not os.path.exists(financial_file):
                    tasks.append(self.download_financial_data_multi_source(symbol))
                else:
                    print(f"✅ {symbol} 财务数据文件已存在，跳过下载")
        
        # 下载新闻数据
        if include_news:
            news_file = os.path.join(self.output_dir, "news_data.json")
            if force_download or not os.path.exists(news_file):
                tasks.append(self.download_news_data_multi_source(symbols, start_date, end_date))
            else:
                print(f"✅ 新闻数据文件已存在，跳过下载")
        
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks)
            print("所有数据下载完成!")
        else:
            print("所有数据文件已存在，无需下载")
    
    async def _download_with_all_apis(self, symbols: List[str], start_date: str, end_date: str, include_news: bool, include_financials: bool, force_download: bool):
        """测试所有API并下载数据到不同目录"""
        print("🧪 开始测试所有API模式...")
        print(f"📊 测试股票: {', '.join(symbols)}")
        print(f"📅 时间范围: {start_date} 到 {end_date}")
        print("=" * 60)
        
        # 定义要测试的API列表
        apis_to_test = {
            "yfinance": {"enabled": True, "name": "YFinance"},
            "finnhub": {"enabled": bool(self.api_keys["finnhub"]), "name": "Finnhub"},
            "polygon": {"enabled": bool(self.api_keys["polygon"]), "name": "Polygon.io"},
            "alpha_vantage": {"enabled": bool(self.api_keys["alpha_vantage"]), "name": "Alpha Vantage"},
            "tiingo": {"enabled": bool(self.api_keys["tiingo"]), "name": "Tiingo"},
            "quandl": {"enabled": bool(self.api_keys["quandl"]), "name": "Quandl"},
            "newsapi": {"enabled": bool(self.api_keys["newsapi"]), "name": "NewsAPI"}
        }
        
        # 显示可用的API
        print("🔑 可用的API:")
        for api, config in apis_to_test.items():
            status = "✅ 可用" if config["enabled"] else "❌ 未配置"
            print(f"  {config['name']}: {status}")
        
        print("\n" + "=" * 60)
        
        # 为每个API创建单独的下载器并测试
        for api, config in apis_to_test.items():
            if not config["enabled"]:
                continue
                
            print(f"\n🚀 测试 {config['name']} API...")
            
            # 创建该API专用的输出目录
            api_output_dir = os.path.join(self.output_dir, f"test_{api}")
            os.makedirs(api_output_dir, exist_ok=True)
            
            # 创建该API专用的下载器
            api_downloader = DataDownloader(output_dir=api_output_dir)
            api_downloader.api_keys = self.api_keys
            api_downloader.rate_limits = self.rate_limits
            
            # 设置该API为最高优先级
            api_downloader.data_sources = {
                "price": [api] + [s for s in self.data_sources["price"] if s != api],
                "news": [api] + [s for s in self.data_sources["news"] if s != api],
                "financials": [api] + [s for s in self.data_sources["financials"] if s != api],
                "market_info": [api] + [s for s in self.data_sources["market_info"] if s != api]
            }
            
            try:
                # 测试该API的数据下载能力
                await self._test_single_api(api_downloader, symbols, start_date, end_date, include_news, include_financials, api, config["name"])
                
            except Exception as e:
                print(f"❌ {config['name']} API测试失败: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("🎉 所有API测试完成!")
        print(f"📁 数据保存在: {self.output_dir}/test_* 目录中")
        print("📊 你可以对比不同API的数据质量和完整性")
    
    async def _test_single_api(self, api_downloader, symbols: List[str], start_date: str, end_date: str, include_news: bool, include_financials: bool, api: str, api_name: str):
        """测试单个API的数据下载能力"""
        
        results = {
            "price_data": {},
            "market_info": {},
            "financial_data": {},
            "news_data": False
        }
        
        # 测试价格数据
        print(f"  📊 测试价格数据...")
        for symbol in symbols[:2]:  # 只测试前2个股票以节省时间
            try:
                price_results = await api_downloader.download_price_data_multi_source(symbol, start_date, end_date, test_mode=True)
                # 检查是否有成功的数据源
                success = any(result.get("success", False) for result in price_results.values()) if price_results else False
                results["price_data"][symbol] = success
                if success:
                    print(f"    ✅ {symbol}: 成功")
                else:
                    print(f"    ❌ {symbol}: 失败")
            except Exception as e:
                results["price_data"][symbol] = False
                print(f"    ❌ {symbol}: 失败 - {str(e)[:50]}...")
        
        # 测试市场信息
        print(f"  📋 测试市场信息...")
        for symbol in symbols[:2]:
            try:
                info_results = await api_downloader.download_market_info_multi_source(symbol, test_mode=True)
                # 检查是否有成功的数据源
                success = any(result.get("success", False) for result in info_results.values()) if info_results else False
                results["market_info"][symbol] = success
                if success:
                    print(f"    ✅ {symbol}: 成功")
                else:
                    print(f"    ❌ {symbol}: 失败")
            except Exception as e:
                results["market_info"][symbol] = False
                print(f"    ❌ {symbol}: 失败 - {str(e)[:50]}...")
        
        # 测试财务数据
        if include_financials:
            print(f"  💰 测试财务数据...")
            for symbol in symbols[:2]:
                try:
                    financial_results = await api_downloader.download_financial_data_multi_source(symbol, test_mode=True)
                    # 检查是否有成功的数据源
                    success = any(result.get("success", False) for result in financial_results.values()) if financial_results else False
                    results["financial_data"][symbol] = success
                    if success:
                        print(f"    ✅ {symbol}: 成功")
                    else:
                        print(f"    ❌ {symbol}: 失败")
                except Exception as e:
                    results["financial_data"][symbol] = False
                    print(f"    ❌ {symbol}: 失败 - {str(e)[:50]}...")
        
        # 测试新闻数据
        if include_news and api in ["finnhub", "newsapi", "yfinance"]:
            print(f"  📰 测试新闻数据...")
            try:
                news_results = await api_downloader.download_news_data_multi_source(symbols[:2], start_date, end_date, limit=20, test_mode=True)
                # 检查是否有成功的数据源
                success = any(result.get("success", False) for result in news_results.values()) if news_results else False
                results["news_data"] = success
                if success:
                    print(f"    ✅ 成功")
                else:
                    print(f"    ❌ 失败")
            except Exception as e:
                results["news_data"] = False
                print(f"    ❌ 失败 - {str(e)[:50]}...")
        
        # 统计结果
        total_tests = 0
        successful_tests = 0
        
        for data_type, status in results.items():
            if isinstance(status, dict):
                for symbol, success in status.items():
                    total_tests += 1
                    if success:
                        successful_tests += 1
            else:
                total_tests += 1
                if status:
                    successful_tests += 1
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"  📈 {api_name} 测试结果: {successful_tests}/{total_tests} 成功 ({success_rate:.1f}%)")
        
        # 保存测试结果
        result_file = os.path.join(api_downloader.output_dir, "test_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "api_name": api_name,
                "api_type": api,
                "test_date": datetime.now().isoformat(),
                "results": results,
                "success_rate": success_rate,
                "total_tests": total_tests,
                "successful_tests": successful_tests
            }, f, ensure_ascii=False, indent=2)
        
        print(f"  📄 详细结果已保存到: {result_file}")
    
    async def download_price_data_multi_source(self, symbol: str, start_date: str, end_date: str, test_mode: bool = False):
        """多数据源下载价格数据"""
        print(f"多数据源下载 {symbol} 价格数据...")
        
        results = {}
        successful_source = None
        
        for source in self.data_sources["price"]:
            try:
                df = None
                if source == "yfinance":
                    df = await self._download_price_yfinance(symbol, start_date, end_date)
                elif source == "finnhub" and self.api_keys["finnhub"]:
                    df = await self._download_price_finnhub(symbol, start_date, end_date)
                elif source == "polygon" and self.api_keys["polygon"]:
                    df = await self._download_price_polygon(symbol, start_date, end_date)
                elif source == "alpha_vantage" and self.api_keys["alpha_vantage"]:
                    df = await self._download_price_alpha_vantage(symbol, start_date, end_date)
                elif source == "tiingo" and self.api_keys["tiingo"]:
                    df = await self._download_price_tiingo(symbol, start_date, end_date)
                elif source == "quandl" and self.api_keys["quandl"]:
                    df = await self._download_price_quandl(symbol, start_date, end_date)
                
                # 记录每个数据源的结果
                if df is not None and not df.empty:
                    results[source] = {"success": True, "data": df, "rows": len(df)}
                    if successful_source is None:
                        successful_source = source
                        if not test_mode:
                            # 非测试模式下，找到第一个成功的数据源就保存并返回
                            output_path = os.path.join(self.output_dir, f"{symbol}_prices.csv")
                            df.to_csv(output_path)
                            print(f"✅ {symbol} 价格数据已保存 ({source}): {len(df)} 行")
                            return
                else:
                    results[source] = {"success": False, "data": None, "rows": 0}
                    
            except Exception as e:
                print(f"❌ {source} 下载 {symbol} 价格数据失败: {e}")
                results[source] = {"success": False, "data": None, "rows": 0, "error": str(e)}
                continue
        
        # 测试模式下，显示所有数据源的结果
        if test_mode:
            print(f"📊 {symbol} 价格数据测试结果:")
            for source, result in results.items():
                status = "✅ 成功" if result["success"] else "❌ 失败"
                rows = result.get("rows", 0)
                error = result.get("error", "")
                print(f"  {source}: {status} ({rows} 行)" + (f" - {error}" if error else ""))
            
            # 如果有成功的数据源，保存第一个成功的数据
            if successful_source:
                df = results[successful_source]["data"]
                output_path = os.path.join(self.output_dir, f"{symbol}_prices.csv")
                df.to_csv(output_path)
                print(f"💾 保存 {successful_source} 的数据作为最终结果")
                return results
        else:
            if successful_source:
                print(f"✅ {symbol} 价格数据已保存 ({successful_source}): {results[successful_source]['rows']} 行")
            else:
                print(f"❌ 所有数据源都无法获取 {symbol} 的价格数据")
        
        return results
    
    async def _download_price_yfinance(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """使用yfinance下载价格数据"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                return None
            
            # 标准化列名
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return df
            
        except Exception as e:
            logger.error(f"yfinance价格数据下载失败: {e}")
            return None
    
    async def _download_price_finnhub(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """使用Finnhub下载价格数据"""
        await self._rate_limit("finnhub")
        
        try:
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                "symbol": symbol,
                "resolution": "D",
                "from": int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()),
                "to": int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()),
                "token": self.api_keys["finnhub"]
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data and data.get("s") == "ok":
                    df_data = []
                    timestamps = data.get("t", [])
                    opens = data.get("o", [])
                    highs = data.get("h", [])
                    lows = data.get("l", [])
                    closes = data.get("c", [])
                    volumes = data.get("v", [])
                    
                    for i in range(len(timestamps)):
                        df_data.append({
                            "Date": datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d"),
                            "Open": opens[i],
                            "High": highs[i],
                            "Low": lows[i],
                            "Close": closes[i],
                            "Volume": volumes[i]
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df["Date"] = pd.to_datetime(df["Date"])
                        df.set_index("Date", inplace=True)
                        return df
                
                return None
                
        except Exception as e:
            logger.error(f"Finnhub价格数据下载失败: {e}")
            return None
    
    async def _download_price_polygon(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """使用Polygon.io下载价格数据"""
        await self._rate_limit("polygon")
        
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {
                "apiKey": self.api_keys["polygon"],
                "adjusted": "true",
                "sort": "asc"
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data and data.get("results"):
                    df_data = []
                    for result in data["results"]:
                        df_data.append({
                            "Date": datetime.fromtimestamp(result["t"] / 1000).strftime("%Y-%m-%d"),
                            "Open": result["o"],
                            "High": result["h"],
                            "Low": result["l"],
                            "Close": result["c"],
                            "Volume": result["v"]
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df["Date"] = pd.to_datetime(df["Date"])
                        df.set_index("Date", inplace=True)
                        return df
                
                return None
                
        except Exception as e:
            logger.error(f"Polygon价格数据下载失败: {e}")
            return None
    
    async def _download_price_alpha_vantage(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """使用Alpha Vantage下载价格数据"""
        await self._rate_limit("alpha_vantage")
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_keys["alpha_vantage"],
                "datatype": "json"
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data and "Time Series (Daily)" in data:
                    df_data = []
                    time_series = data["Time Series (Daily)"]
                    
                    for date, values in time_series.items():
                        if start_date <= date <= end_date:
                            df_data.append({
                                "Date": date,
                                "Open": float(values["1. open"]),
                                "High": float(values["2. high"]),
                                "Low": float(values["3. low"]),
                                "Close": float(values["4. close"]),
                                "Volume": int(values["5. volume"])
                            })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df["Date"] = pd.to_datetime(df["Date"])
                        df.set_index("Date", inplace=True)
                        df.sort_index(inplace=True)
                        return df
                
                return None
                
        except Exception as e:
            logger.error(f"Alpha Vantage价格数据下载失败: {e}")
            return None
    
    async def _download_price_tiingo(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """使用Tiingo下载价格数据"""
        await self._rate_limit("tiingo")
        
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "format": "json"
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_keys['tiingo']}"
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params, headers)
                
                if data:
                    df_data = []
                    for item in data:
                        df_data.append({
                            "Date": item["date"][:10],
                            "Open": item["open"],
                            "High": item["high"],
                            "Low": item["low"],
                            "Close": item["close"],
                            "Volume": item["volume"]
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df["Date"] = pd.to_datetime(df["Date"])
                        df.set_index("Date", inplace=True)
                        return df
                
                return None
                
        except Exception as e:
            logger.error(f"Tiingo价格数据下载失败: {e}")
            return None
    
    async def _download_price_quandl(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """使用Quandl下载价格数据"""
        await self._rate_limit("quandl")
        
        try:
            # Quandl免费API限制非常严格，大部分数据集都需要付费
            # 我们尝试一个简单的测试数据集
            url = "https://www.quandl.com/api/v3/datasets/ODA/POILWTI.json"  # 原油价格数据
            params = {
                "api_key": self.api_keys["quandl"],
                "start_date": start_date,
                "end_date": end_date,
                "order": "asc"
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data and data.get("dataset_data"):
                    dataset_data = data["dataset_data"]
                    data_points = dataset_data.get("data", [])
                    
                    if data_points:
                        df_data = []
                        for point in data_points:
                            if len(point) >= 2:
                                df_data.append({
                                    "Date": point[0],
                                    "Value": point[1]
                                })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df["Date"] = pd.to_datetime(df["Date"])
                            df.set_index("Date", inplace=True)
                            # 重命名列以匹配标准格式
                            df = df.rename(columns={"Value": "Close"})
                            # 添加其他必需的列
                            df["Open"] = df["Close"]
                            df["High"] = df["Close"]
                            df["Low"] = df["Close"]
                            df["Volume"] = 0
                            return df
                
                # 如果API调用失败，返回None
                logger.warning("Quandl API免费版限制严格，建议升级或使用其他数据源")
                return None
                
        except Exception as e:
            logger.error(f"Quandl价格数据下载失败: {e}")
            return None
    
    async def download_market_info_multi_source(self, symbol: str, test_mode: bool = False):
        """多数据源下载市场信息"""
        print(f"多数据源下载 {symbol} 市场信息...")
        
        results = {}
        successful_source = None
        
        for source in self.data_sources["market_info"]:
            try:
                info = None
                if source == "yfinance":
                    info = await self._download_market_info_yfinance(symbol)
                elif source == "finnhub" and self.api_keys["finnhub"]:
                    info = await self._download_market_info_finnhub(symbol)
                elif source == "polygon" and self.api_keys["polygon"]:
                    info = await self._download_market_info_polygon(symbol)
                elif source == "alpha_vantage" and self.api_keys["alpha_vantage"]:
                    info = await self._download_market_info_alpha_vantage(symbol)
                elif source == "tiingo" and self.api_keys["tiingo"]:
                    info = await self._download_market_info_tiingo(symbol)
                
                # 记录每个数据源的结果
                if info:
                    results[source] = {"success": True, "data": info}
                    if successful_source is None:
                        successful_source = source
                        if not test_mode:
                            # 非测试模式下，找到第一个成功的数据源就保存并返回
                            output_path = os.path.join(self.output_dir, f"{symbol}_info.json")
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(info, f, ensure_ascii=False, indent=2)
                            print(f"✅ {symbol} 市场信息已保存 ({source})")
                            return
                else:
                    results[source] = {"success": False, "data": None}
                    
            except Exception as e:
                print(f"❌ {source} 下载 {symbol} 市场信息失败: {e}")
                results[source] = {"success": False, "data": None, "error": str(e)}
                continue
        
        # 测试模式下，显示所有数据源的结果
        if test_mode:
            print(f"📊 {symbol} 市场信息测试结果:")
            for source, result in results.items():
                status = "✅ 成功" if result["success"] else "❌ 失败"
                error = result.get("error", "")
                print(f"  {source}: {status}" + (f" - {error}" if error else ""))
            
            # 如果有成功的数据源，保存第一个成功的数据
            if successful_source:
                info = results[successful_source]["data"]
                output_path = os.path.join(self.output_dir, f"{symbol}_info.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
                print(f"💾 保存 {successful_source} 的数据作为最终结果")
                return results
        else:
            if successful_source:
                print(f"✅ {symbol} 市场信息已保存 ({successful_source})")
            else:
                print(f"❌ 所有数据源都无法获取 {symbol} 的市场信息")
        
        return results
    
    async def _download_market_info_yfinance(self, symbol: str) -> Optional[Dict]:
        """使用yfinance下载市场信息"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            return {
                "symbol": symbol,
                "name": info.get("shortName", ""),
                "description": info.get("longBusinessSummary", ""),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "website": info.get("website", ""),
                "source": "yfinance"
            }
            
        except Exception as e:
            logger.error(f"yfinance市场信息下载失败: {e}")
            return None
    
    async def _download_market_info_finnhub(self, symbol: str) -> Optional[Dict]:
        """使用Finnhub下载市场信息"""
        await self._rate_limit("finnhub")
        
        try:
            url = "https://finnhub.io/api/v1/stock/profile2"
            params = {
                "symbol": symbol,
                "token": self.api_keys["finnhub"]
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data:
                    return {
                        "symbol": symbol,
                        "name": data.get("name", ""),
                        "description": data.get("finnhubIndustry", ""),
                        "exchange": data.get("exchange", ""),
                        "sector": data.get("finnhubIndustry", ""),
                        "industry": data.get("finnhubIndustry", ""),
                        "country": data.get("country", ""),
                        "employees": data.get("employeeTotal", 0),
                        "website": data.get("weburl", ""),
                        "source": "finnhub"
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Finnhub市场信息下载失败: {e}")
            return None
    
    async def _download_market_info_polygon(self, symbol: str) -> Optional[Dict]:
        """使用Polygon.io下载市场信息"""
        await self._rate_limit("polygon")
        
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {
                "apiKey": self.api_keys["polygon"]
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data and data.get("results"):
                    result = data["results"]
                    return {
                        "symbol": symbol,
                        "name": result.get("name", ""),
                        "description": result.get("description", ""),
                        "exchange": result.get("primary_exchange", ""),
                        "sector": result.get("sic_description", ""),
                        "industry": result.get("sic_description", ""),
                        "country": result.get("locale", ""),
                        "employees": 0,
                        "website": result.get("homepage_url", ""),
                        "source": "polygon"
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Polygon市场信息下载失败: {e}")
            return None
    
    async def _download_market_info_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """使用Alpha Vantage下载市场信息"""
        await self._rate_limit("alpha_vantage")
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_keys["alpha_vantage"]
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data and not data.get("Error Message"):
                    return {
                        "symbol": symbol,
                        "name": data.get("Name", ""),
                        "description": data.get("Description", ""),
                        "exchange": data.get("Exchange", ""),
                        "sector": data.get("Sector", ""),
                        "industry": data.get("Industry", ""),
                        "country": data.get("Country", ""),
                        "employees": int(data.get("FullTimeEmployees", 0)) if data.get("FullTimeEmployees") else 0,
                        "website": data.get("Website", ""),
                        "source": "alpha_vantage"
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Alpha Vantage市场信息下载失败: {e}")
            return None
    
    async def _download_market_info_tiingo(self, symbol: str) -> Optional[Dict]:
        """使用Tiingo下载市场信息"""
        await self._rate_limit("tiingo")
        
        try:
            # Tiingo没有专门的市场信息API，我们使用公司信息API
            url = f"https://api.tiingo.com/tiingo/utilities/search/{symbol}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_keys['tiingo']}"
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, headers=headers)
                
                if data and len(data) > 0:
                    # 使用第一个匹配的结果
                    result = data[0]
                    return {
                        "symbol": symbol,
                        "name": result.get("name", ""),
                        "description": result.get("description", ""),
                        "exchange": result.get("exchange", ""),
                        "sector": "",
                        "industry": "",
                        "country": "",
                        "employees": 0,
                        "website": "",
                        "source": "tiingo"
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Tiingo市场信息下载失败: {e}")
            return None
    
    async def download_financial_data_multi_source(self, symbol: str, test_mode: bool = False):
        """多数据源下载财务数据"""
        print(f"多数据源下载 {symbol} 财务数据...")
        
        results = {}
        successful_source = None
        
        for source in self.data_sources["financials"]:
            try:
                data = None
                if source == "yfinance":
                    data = await self._download_financial_data_yfinance(symbol)
                elif source == "finnhub" and self.api_keys["finnhub"]:
                    data = await self._download_financial_data_finnhub(symbol)
                elif source == "alpha_vantage" and self.api_keys["alpha_vantage"]:
                    data = await self._download_financial_data_alpha_vantage(symbol)
                elif source == "tiingo" and self.api_keys["tiingo"]:
                    data = await self._download_financial_data_tiingo(symbol)
                
                # 记录每个数据源的结果
                if data:
                    results[source] = {"success": True, "data": data}
                    if successful_source is None:
                        successful_source = source
                        if not test_mode:
                            # 非测试模式下，找到第一个成功的数据源就保存并返回
                            output_path = os.path.join(self.output_dir, f"{symbol}_financials.json")
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                            print(f"✅ {symbol} 财务数据已保存 ({source})")
                            return
                else:
                    results[source] = {"success": False, "data": None}
                    
            except Exception as e:
                print(f"❌ {source} 下载 {symbol} 财务数据失败: {e}")
                results[source] = {"success": False, "data": None, "error": str(e)}
                continue
        
        # 测试模式下，显示所有数据源的结果
        if test_mode:
            print(f"📊 {symbol} 财务数据测试结果:")
            for source, result in results.items():
                status = "✅ 成功" if result["success"] else "❌ 失败"
                error = result.get("error", "")
                print(f"  {source}: {status}" + (f" - {error}" if error else ""))
            
            # 如果有成功的数据源，保存第一个成功的数据
            if successful_source:
                data = results[successful_source]["data"]
                output_path = os.path.join(self.output_dir, f"{symbol}_financials.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                print(f"💾 保存 {successful_source} 的数据作为最终结果")
                return results
        else:
            if successful_source:
                print(f"✅ {symbol} 财务数据已保存 ({successful_source})")
            else:
                print(f"❌ 所有数据源都无法获取 {symbol} 的财务数据")
        
        return results
    
    async def _download_financial_data_yfinance(self, symbol: str) -> Optional[Dict]:
        """使用yfinance下载财务数据"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 获取盈利数据
            earnings_data = ticker.earnings_dates
            earnings_surprises = []
            
            if earnings_data is not None and not earnings_data.empty:
                for date, row in earnings_data.iterrows():
                    period = date
                    if isinstance(period, (datetime, pd.Timestamp)):
                        period = period.strftime("%Y-%m-%d")
                    elif not isinstance(period, str):
                        period = str(period)
                        
                    earnings_surprises.append({
                        "period": period,
                        "epsActual": row.get("Reported EPS", None),
                        "epsEstimate": row.get("EPS Estimate", None),
                        "epsSurprise": None,
                        "epsSurprisePercent": row.get("Surprise(%)", None)
                    })
            
            # 获取分析师推荐
            recommendations = ticker.recommendations
            recommendation_trends = []
            
            if recommendations is not None and not recommendations.empty:
                for date, row in recommendations.iterrows():
                    period = date
                    if isinstance(period, (datetime, pd.Timestamp)):
                        period = period.strftime("%Y-%m-%d")
                    elif not isinstance(period, str):
                        period = str(period)
                        
                    recommendation_trends.append({
                        "period": period,
                        "strongBuy": 0,
                        "buy": 0,
                        "hold": 0,
                        "sell": 0,
                        "strongSell": 0,
                        "grade": row.get("To Grade", ""),
                        "action": row.get("Action", ""),
                        "firm": row.get("Firm", "")
                    })
            
            return {
                "key_metrics": {
                    "pe_ratio": info.get("trailingPE"),
                    "eps_ttm": info.get("trailingEps"),
                    "dividend_yield": info.get("dividendYield"),
                    "market_cap": info.get("marketCap"),
                    "52w_high": info.get("fiftyTwoWeekHigh"),
                    "52w_low": info.get("fiftyTwoWeekLow"),
                    "beta": info.get("beta"),
                    "avg_volume": info.get("averageVolume")
                },
                "earnings_surprises": earnings_surprises,
                "recommendation_trends": recommendation_trends,
                "source": "yfinance"
            }
            
        except Exception as e:
            logger.error(f"yfinance财务数据下载失败: {e}")
            return None
    
    async def _download_financial_data_finnhub(self, symbol: str) -> Optional[Dict]:
        """使用Finnhub下载财务数据"""
        await self._rate_limit("finnhub")
        
        try:
            # 获取财务指标
            url = "https://finnhub.io/api/v1/quote"
            params = {
                "symbol": symbol,
                "token": self.api_keys["finnhub"]
            }
            
            async with aiohttp.ClientSession() as session:
                quote_data = await self._make_api_request(session, url, params)
                
                # 获取盈利数据
                earnings_url = "https://finnhub.io/api/v1/stock/earnings"
                earnings_params = {
                    "symbol": symbol,
                    "token": self.api_keys["finnhub"]
                }
                earnings_data = await self._make_api_request(session, earnings_url, earnings_params)
                
                # 获取分析师推荐
                recommendations_url = "https://finnhub.io/api/v1/stock/recommendation"
                recommendations_params = {
                    "symbol": symbol,
                    "token": self.api_keys["finnhub"]
                }
                recommendations_data = await self._make_api_request(session, recommendations_url, recommendations_params)
                
                return {
                    "key_metrics": {
                        "pe_ratio": quote_data.get("pe") if quote_data else None,
                        "eps_ttm": quote_data.get("eps") if quote_data else None,
                        "dividend_yield": None,
                        "market_cap": None,
                        "52w_high": quote_data.get("h") if quote_data else None,
                        "52w_low": quote_data.get("l") if quote_data else None,
                        "beta": None,
                        "avg_volume": quote_data.get("volume") if quote_data else None
                    },
                    "earnings_surprises": earnings_data if earnings_data else [],
                    "recommendation_trends": recommendations_data if recommendations_data else [],
                    "source": "finnhub"
                }
                
        except Exception as e:
            logger.error(f"Finnhub财务数据下载失败: {e}")
            return None
    
    async def _download_financial_data_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """使用Alpha Vantage下载财务数据"""
        await self._rate_limit("alpha_vantage")
        
        try:
            # 获取财务指标
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_keys["alpha_vantage"]
            }
            
            async with aiohttp.ClientSession() as session:
                overview_data = await self._make_api_request(session, url, params)
                
                # 获取盈利数据
                earnings_params = {
                    "function": "EARNINGS",
                    "symbol": symbol,
                    "apikey": self.api_keys["alpha_vantage"]
                }
                earnings_data = await self._make_api_request(session, url, earnings_params)
                
                return {
                    "key_metrics": {
                        "pe_ratio": float(overview_data.get("PERatio", 0)) if overview_data and overview_data.get("PERatio") else None,
                        "eps_ttm": float(overview_data.get("EPS", 0)) if overview_data and overview_data.get("EPS") else None,
                        "dividend_yield": float(overview_data.get("DividendYield", 0)) if overview_data and overview_data.get("DividendYield") else None,
                        "market_cap": overview_data.get("MarketCapitalization", None),
                        "52w_high": None,
                        "52w_low": None,
                        "beta": float(overview_data.get("Beta", 0)) if overview_data and overview_data.get("Beta") else None,
                        "avg_volume": overview_data.get("Volume", None)
                    },
                    "earnings_surprises": earnings_data.get("quarterlyEarnings", []) if earnings_data else [],
                    "recommendation_trends": [],
                    "source": "alpha_vantage"
                }
                
        except Exception as e:
            logger.error(f"Alpha Vantage财务数据下载失败: {e}")
            return None
    
    async def _download_financial_data_tiingo(self, symbol: str) -> Optional[Dict]:
        """使用Tiingo下载财务数据"""
        await self._rate_limit("tiingo")
        
        try:
            # Tiingo的财务数据API
            url = f"https://api.tiingo.com/tiingo/fundamentals/{symbol}/statements"
            params = {
                "format": "json"
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_keys['tiingo']}"
            }
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params, headers)
                
                if data:
                    return {
                        "key_metrics": {
                            "pe_ratio": None,
                            "eps_ttm": None,
                            "dividend_yield": None,
                            "market_cap": None,
                            "52w_high": None,
                            "52w_low": None,
                            "beta": None,
                            "avg_volume": None
                        },
                        "earnings_surprises": [],
                        "recommendation_trends": [],
                        "source": "tiingo"
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Tiingo财务数据下载失败: {e}")
            return None
    
    async def download_news_data_multi_source(self, symbols: List[str], start_date: str, end_date: str, limit: int = 1000, test_mode: bool = False):
        """多数据源下载新闻数据"""
        print(f"多数据源下载新闻数据...")
        
        results = {}
        all_news = []
        successful_source = None
        
        for source in self.data_sources["news"]:
            try:
                news = None
                if source == "finnhub" and self.api_keys["finnhub"]:
                    news = await self._download_news_finnhub(symbols, start_date, end_date, limit)
                elif source == "newsapi" and self.api_keys["newsapi"]:
                    news = await self._download_news_newsapi(symbols, start_date, end_date, limit)
                elif source == "yfinance":
                    news = await self._download_news_yfinance(symbols, start_date, end_date, limit)
                
                # 记录每个数据源的结果
                if news and len(news) > 0:
                    results[source] = {"success": True, "data": news, "count": len(news)}
                    if successful_source is None:
                        successful_source = source
                        if not test_mode:
                            # 非测试模式下，找到第一个成功的数据源就保存并返回
                            all_news.extend(news)
                            unique_news = self._deduplicate_news(all_news)
                            output_path = os.path.join(self.output_dir, "news_data.json")
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(unique_news, f, ensure_ascii=False, indent=2)
                            print(f"✅ 从 {source} 获取到 {len(news)} 条新闻，共保存 {len(unique_news)} 条去重后的新闻")
                            return
                else:
                    results[source] = {"success": False, "data": [], "count": 0}
                    
            except Exception as e:
                print(f"❌ {source} 下载新闻数据失败: {e}")
                results[source] = {"success": False, "data": [], "count": 0, "error": str(e)}
                continue
        
        # 测试模式下，显示所有数据源的结果
        if test_mode:
            print(f"📊 新闻数据测试结果:")
            for source, result in results.items():
                status = "✅ 成功" if result["success"] else "❌ 失败"
                count = result.get("count", 0)
                error = result.get("error", "")
                print(f"  {source}: {status} ({count} 条)" + (f" - {error}" if error else ""))
            
            # 如果有成功的数据源，保存第一个成功的数据
            if successful_source:
                news = results[successful_source]["data"]
                all_news.extend(news)
                unique_news = self._deduplicate_news(all_news)
                output_path = os.path.join(self.output_dir, "news_data.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(unique_news, f, ensure_ascii=False, indent=2)
                print(f"💾 保存 {successful_source} 的数据作为最终结果，共 {len(unique_news)} 条去重后的新闻")
                return results
        else:
            if successful_source:
                news = results[successful_source]["data"]
                all_news.extend(news)
                unique_news = self._deduplicate_news(all_news)
                output_path = os.path.join(self.output_dir, "news_data.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(unique_news, f, ensure_ascii=False, indent=2)
                print(f"✅ 共保存 {len(unique_news)} 条去重后的新闻")
            else:
                print(f"❌ 所有数据源都无法获取新闻数据")
        
        return results
    
    async def _download_news_finnhub(self, symbols: List[str], start_date: str, end_date: str, limit: int) -> List[Dict]:
        """使用Finnhub下载新闻数据"""
        await self._rate_limit("finnhub")
        
        all_news = []
        
        try:
            for symbol in symbols:
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    "symbol": symbol,
                    "from": start_date,
                    "to": end_date,
                    "token": self.api_keys["finnhub"]
                }
                
                async with aiohttp.ClientSession() as session:
                    data = await self._make_api_request(session, url, params)
                    
                    if data:
                        for article in data[:limit // len(symbols)]:
                            news_item = {
                                "id": str(article.get("id", "")),
                                "title": article.get("headline", ""),
                                "description": article.get("summary", ""),
                                "url": article.get("url", ""),
                                "published_date": article.get("datetime", ""),
                                "source": article.get("source", ""),
                                "tags": article.get("category", "").split(",") if article.get("category") else [],
                                "tickers": [symbol],
                                "source_api": "finnhub"
                            }
                            all_news.append(news_item)
                
                # 避免API速率限制
                await asyncio.sleep(1)
            
            return all_news
            
        except Exception as e:
            logger.error(f"Finnhub新闻数据下载失败: {e}")
            return []
    
    async def _download_news_newsapi(self, symbols: List[str], start_date: str, end_date: str, limit: int) -> List[Dict]:
        """使用NewsAPI下载新闻数据"""
        await self._rate_limit("newsapi")
        
        all_news = []
        
        try:
            for symbol in symbols:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": symbol,
                    "from": start_date,
                    "to": end_date,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "apiKey": self.api_keys["newsapi"]
                }
                
                async with aiohttp.ClientSession() as session:
                    data = await self._make_api_request(session, url, params)
                    
                    if data and data.get("status") == "ok":
                        articles = data.get("articles", [])
                        for article in articles[:limit // len(symbols)]:
                            news_item = {
                                "id": article.get("url", "")[:50],
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "url": article.get("url", ""),
                                "published_date": article.get("publishedAt", ""),
                                "source": article.get("source", {}).get("name", ""),
                                "tags": [],
                                "tickers": [symbol],
                                "source_api": "newsapi"
                            }
                            all_news.append(news_item)
                
                # 避免API速率限制
                await asyncio.sleep(1)
            
            return all_news
            
        except Exception as e:
            logger.error(f"NewsAPI新闻数据下载失败: {e}")
            return []
    
    async def _download_news_yfinance(self, symbols: List[str], start_date: str, end_date: str, limit: int) -> List[Dict]:
        """使用yfinance下载新闻数据"""
        all_news = []
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                for article in news[:limit // len(symbols)]:
                    if 'providerPublishTime' in article:
                        published_date = datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat()
                    else:
                        published_date = datetime.now().isoformat()
                    
                    news_item = {
                        "id": str(article.get("uuid", "")),
                        "title": article.get("title", ""),
                        "description": article.get("summary", ""),
                        "url": article.get("link", ""),
                        "published_date": published_date,
                        "source": article.get("publisher", ""),
                        "tags": [],
                        "tickers": article.get("relatedTickers", []) if article.get("relatedTickers") else [symbol],
                        "source_api": "yfinance"
                    }
                    all_news.append(news_item)
            
            return all_news
            
        except Exception as e:
            logger.error(f"yfinance新闻数据下载失败: {e}")
            return []
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """去重新闻数据"""
        seen_ids = set()
        unique_news = []
        
        for news in news_list:
            news_id = news.get("id", "")
            if news_id and news_id not in seen_ids:
                seen_ids.add(news_id)
                unique_news.append(news)
        
        return unique_news
    
    def check_data_exists(self, symbols: List[str], include_news: bool = True, include_financials: bool = True) -> Dict[str, bool]:
        """检查数据文件是否存在
        
        Args:
            symbols: 股票代码列表
            include_news: 是否检查新闻数据
            include_financials: 是否检查财务数据
            
        Returns:
            包含各类数据文件是否存在的字典
        """
        result = {
            "price_data": {},
            "market_info": {},
            "financial_data": {},
            "news_data": False
        }
        
        for symbol in symbols:
            # 检查价格数据
            price_file = os.path.join(self.output_dir, f"{symbol}_prices.csv")
            result["price_data"][symbol] = os.path.exists(price_file)
            
            # 检查市场信息
            info_file = os.path.join(self.output_dir, f"{symbol}_info.json")
            result["market_info"][symbol] = os.path.exists(info_file)
            
            # 检查财务数据
            if include_financials:
                financial_file = os.path.join(self.output_dir, f"{symbol}_financials.json")
                result["financial_data"][symbol] = os.path.exists(financial_file)
        
        # 检查新闻数据
        if include_news:
            news_file = os.path.join(self.output_dir, "news_data.json")
            result["news_data"] = os.path.exists(news_file)
        
        return result


async def main():
    """主函数示例"""
    # 创建下载器
    downloader = DataDownloader(output_dir="backtest_data")
    
    # 定义股票代码列表
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # 定义日期范围
    start_date = "2025-03-01"
    end_date = "2025-07-31"
    
    # 下载所有数据
    await downloader.download_all_data(symbols, start_date, end_date)


if __name__ == "__main__":
    asyncio.run(main()) 