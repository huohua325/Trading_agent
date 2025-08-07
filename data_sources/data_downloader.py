import os
import json
import pandas as pd
import asyncio
import aiohttp
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union


class DataDownloader:
    """数据下载工具，支持从多个免费数据源下载历史数据并保存为回测所需的格式"""
    
    def __init__(self, output_dir: str = "backtest_data"):
        """初始化数据下载器
        
        Args:
            output_dir: 数据保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def download_all_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        include_news: bool = True,
        include_financials: bool = True,
        force_download: bool = False
    ):
        """下载所有需要的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            include_news: 是否包含新闻数据
            include_financials: 是否包含财务数据
            force_download: 是否强制重新下载数据（即使文件已存在）
        """
        print(f"开始下载数据: {len(symbols)} 个股票, 时间范围: {start_date} 到 {end_date}")
        
        # 创建任务列表
        tasks = []
        
        # 下载价格数据
        for symbol in symbols:
            # 检查价格数据文件是否已存在
            price_file = os.path.join(self.output_dir, f"{symbol}_prices.csv")
            if force_download or not os.path.exists(price_file):
                tasks.append(self.download_price_data(symbol, start_date, end_date))
            else:
                print(f"✅ {symbol} 价格数据文件已存在，跳过下载")
            
            # 检查市场信息文件是否已存在
            info_file = os.path.join(self.output_dir, f"{symbol}_info.json")
            if force_download or not os.path.exists(info_file):
                tasks.append(self.download_market_info(symbol))
            else:
                print(f"✅ {symbol} 市场信息文件已存在，跳过下载")
            
            # 检查财务数据文件是否已存在
            if include_financials:
                financial_file = os.path.join(self.output_dir, f"{symbol}_financials.json")
                if force_download or not os.path.exists(financial_file):
                    tasks.append(self.download_financial_data(symbol))
                else:
                    print(f"✅ {symbol} 财务数据文件已存在，跳过下载")
        
        # 下载新闻数据
        if include_news:
            news_file = os.path.join(self.output_dir, "news_data.json")
            if force_download or not os.path.exists(news_file):
                tasks.append(self.download_news_data(symbols, start_date, end_date))
            else:
                print(f"✅ 新闻数据文件已存在，跳过下载")
        
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks)
            print("所有数据下载完成!")
        else:
            print("所有数据文件已存在，无需下载")
            
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
    
    async def download_price_data(self, symbol: str, start_date: str, end_date: str):
        """下载股票价格数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        print(f"下载 {symbol} 价格数据...")
        
        try:
            # 使用yfinance下载数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                print(f"警告: 无法获取 {symbol} 的价格数据")
                return
            
            # 重命名列以保持一致性
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # 保存到CSV文件
            output_path = os.path.join(self.output_dir, f"{symbol}_prices.csv")
            df.to_csv(output_path)
            
            print(f"✅ {symbol} 价格数据已保存: {len(df)} 行")
            
        except Exception as e:
            print(f"❌ 下载 {symbol} 价格数据失败: {e}")
    
    async def download_market_info(self, symbol: str):
        """下载股票市场信息
        
        Args:
            symbol: 股票代码
        """
        print(f"下载 {symbol} 市场信息...")
        
        try:
            # 使用yfinance获取市场信息
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                print(f"警告: 无法获取 {symbol} 的市场信息")
                return
            
            # 提取关键信息
            market_info = {
                "symbol": symbol,
                "name": info.get("shortName", ""),
                "description": info.get("longBusinessSummary", ""),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "website": info.get("website", "")
            }
            
            # 保存到JSON文件
            output_path = os.path.join(self.output_dir, f"{symbol}_info.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(market_info, f, ensure_ascii=False, indent=2)
            
            print(f"✅ {symbol} 市场信息已保存")
            
        except Exception as e:
            print(f"❌ 下载 {symbol} 市场信息失败: {e}")
    
    async def download_news_data(self, symbols: List[str], start_date: str, end_date: str, limit: int = 1000):
        """下载新闻数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            limit: 每个股票的最大新闻数量
        """
        print(f"下载新闻数据...")
        
        all_news = []
        
        try:
            # 为每个股票下载新闻
            for symbol in symbols:
                try:
                    print(f"获取 {symbol} 的新闻...")
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    
                    # 处理新闻数据
                    for article in news[:limit // len(symbols)]:
                        # 转换时间戳为ISO格式日期
                        if 'providerPublishTime' in article:
                            published_date = datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat()
                        else:
                            published_date = datetime.now().isoformat()
                        
                        # 转换为标准格式
                        news_item = {
                            "id": str(article.get("uuid", "")),
                            "title": article.get("title", ""),
                            "description": article.get("summary", ""),
                            "url": article.get("link", ""),
                            "published_date": published_date,
                            "source": article.get("publisher", ""),
                            "tags": [],  # YFinance不提供标签
                            "tickers": article.get("relatedTickers", []) if article.get("relatedTickers") else [symbol]
                        }
                        
                        all_news.append(news_item)
                    
                    print(f"✅ 获取到 {symbol} 的 {len(news)} 条新闻")
                    
                except Exception as e:
                    print(f"❌ 获取 {symbol} 新闻失败: {e}")
            
            # 保存所有新闻到一个文件
            output_path = os.path.join(self.output_dir, "news_data.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_news, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 共保存 {len(all_news)} 条新闻")
            
        except Exception as e:
            print(f"❌ 下载新闻数据失败: {e}")
    
    async def download_financial_data(self, symbol: str):
        """下载财务数据
        
        Args:
            symbol: 股票代码
        """
        print(f"下载 {symbol} 财务数据...")
        
        try:
            # 使用yfinance获取财务数据
            ticker = yf.Ticker(symbol)
            
            # 获取基本财务指标
            info = ticker.info
            
            # 提取关键财务指标
            key_metrics = {
                "pe_ratio": info.get("trailingPE"),
                "eps_ttm": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "market_cap": info.get("marketCap"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "beta": info.get("beta"),
                "avg_volume": info.get("averageVolume")
            }
            
            # 获取盈利数据
            earnings_data = ticker.earnings_dates
            earnings_surprises = []
            
            if earnings_data is not None and not earnings_data.empty:
                # 转换为列表格式
                for date, row in earnings_data.iterrows():
                    # 确保日期是字符串格式
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
                # 转换为列表格式
                for date, row in recommendations.iterrows():
                    # 确保日期是字符串格式
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
            
            # 组合所有财务数据
            financial_data = {
                "key_metrics": key_metrics,
                "earnings_surprises": earnings_surprises,
                "recommendation_trends": recommendation_trends
            }
            
            # 保存到JSON文件
            output_path = os.path.join(self.output_dir, f"{symbol}_financials.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(financial_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✅ {symbol} 财务数据已保存")
            
        except Exception as e:
            print(f"❌ 下载 {symbol} 财务数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def download_alternative_price_data(self, symbol: str, start_date: str, end_date: str):
        """使用Alpha Vantage API下载价格数据（备用方法）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        print(f"使用Alpha Vantage下载 {symbol} 价格数据...")
        
        try:
            # Alpha Vantage免费API密钥（每天限制500次请求）
            api_key = "demo"  # 替换为您的API密钥
            
            # 构建API URL
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": api_key,
                "datatype": "json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        print(f"警告: Alpha Vantage API返回错误: {response.status}")
                        return
                    
                    data = await response.json()
                    
                    if "Error Message" in data:
                        print(f"警告: Alpha Vantage API错误: {data['Error Message']}")
                        return
                    
                    if "Time Series (Daily)" not in data:
                        print(f"警告: Alpha Vantage API未返回数据")
                        return
                    
                    # 解析数据
                    time_series = data["Time Series (Daily)"]
                    df_data = []
                    
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
                    
                    if not df_data:
                        print(f"警告: 在指定日期范围内没有找到 {symbol} 的数据")
                        return
                    
                    # 创建DataFrame
                    df = pd.DataFrame(df_data)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    df.sort_index(inplace=True)
                    
                    # 保存到CSV文件
                    output_path = os.path.join(self.output_dir, f"{symbol}_prices.csv")
                    df.to_csv(output_path)
                    
                    print(f"✅ {symbol} 价格数据已保存 (Alpha Vantage): {len(df)} 行")
        
        except Exception as e:
            print(f"❌ 使用Alpha Vantage下载 {symbol} 价格数据失败: {e}")
    
    async def download_alternative_news_data(self, symbols: List[str], start_date: str, end_date: str):
        """使用NewsAPI下载新闻数据（备用方法）
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        print(f"使用NewsAPI下载新闻数据...")
        
        try:
            # NewsAPI免费API密钥（每天限制100次请求）
            api_key = "YOUR_API_KEY"  # 替换为您的API密钥
            
            all_news = []
            
            for symbol in symbols:
                # 构建API URL
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": symbol,
                    "from": start_date,
                    "to": end_date,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "apiKey": api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            print(f"警告: NewsAPI返回错误: {response.status}")
                            continue
                        
                        data = await response.json()
                        
                        if data.get("status") != "ok":
                            print(f"警告: NewsAPI错误: {data.get('message', 'Unknown error')}")
                            continue
                        
                        articles = data.get("articles", [])
                        
                        for article in articles:
                            news_item = {
                                "id": article.get("url", "")[:50],  # 使用URL前50个字符作为ID
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "url": article.get("url", ""),
                                "published_date": article.get("publishedAt", ""),
                                "source": article.get("source", {}).get("name", ""),
                                "tags": [],
                                "tickers": [symbol]
                            }
                            
                            all_news.append(news_item)
                
                print(f"✅ 获取到 {symbol} 的 {len(articles)} 条新闻 (NewsAPI)")
                
                # 避免API速率限制
                await asyncio.sleep(1)
            
            # 保存所有新闻到一个文件
            output_path = os.path.join(self.output_dir, "news_data.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_news, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 共保存 {len(all_news)} 条新闻 (NewsAPI)")
            
        except Exception as e:
            print(f"❌ 使用NewsAPI下载新闻数据失败: {e}")


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