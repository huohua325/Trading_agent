#!/usr/bin/env python3
"""
多数据源数据下载示例

这个脚本展示了如何使用多数据源DataDownloader来下载历史数据，
确保数据的完整性和质量。

使用方法:
1. 复制 config_example.env 为 .env 并填入你的API密钥
2. 运行: python examples/multi_source_data_download_example.py
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from data_sources.data_downloader import DataDownloader


async def main():
    """主函数示例"""
    
    # 加载环境变量
    load_dotenv()
    
    print("🚀 多数据源数据下载示例")
    print("=" * 50)
    
    # 检查API密钥配置
    api_keys = {
        "FINNHUB_API_KEY": os.getenv("FINNHUB_API_KEY"),
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY"),
        "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "TIINGO_API_KEY": os.getenv("TIINGO_API_KEY"),
        "QUANDL_API_KEY": os.getenv("QUANDL_API_KEY"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY")
    }
    
    print("📋 API密钥配置状态:")
    for key, value in api_keys.items():
        status = "✅ 已配置" if value and value != "your_" + key.lower() + "_here" else "❌ 未配置"
        print(f"  {key}: {status}")
    
    print("\n" + "=" * 50)
    
    # 创建下载器
    downloader = DataDownloader(output_dir="backtest_data")
    
    # 定义股票代码列表
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # 定义日期范围
    start_date = "2025-03-01"
    end_date = "2025-07-31"
    
    print(f"📊 开始下载数据:")
    print(f"  股票: {', '.join(symbols)}")
    print(f"  时间范围: {start_date} 到 {end_date}")
    print(f"  输出目录: {downloader.output_dir}")
    
    print("\n" + "=" * 50)
    
    try:
        # 检查是否要测试所有API
        test_all_apis = input("🧪 是否要测试所有API并对比数据质量? (y/N): ").lower().strip() == 'y'
        
        if test_all_apis:
            print("\n🔬 启动测试所有API模式...")
            await downloader.download_all_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                include_news=True,
                include_financials=True,
                force_download=False,
                test_all_apis=True
            )
        else:
            print("\n📥 启动正常下载模式...")
            await downloader.download_all_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                include_news=True,
                include_financials=True,
                force_download=False,
                test_all_apis=False
            )
            
            # 检查下载结果
            print("\n📋 数据文件检查:")
            result = downloader.check_data_exists(symbols, include_news=True, include_financials=True)
            
            for symbol in symbols:
                print(f"\n  {symbol}:")
                print(f"    价格数据: {'✅' if result['price_data'].get(symbol, False) else '❌'}")
                print(f"    市场信息: {'✅' if result['market_info'].get(symbol, False) else '❌'}")
                print(f"    财务数据: {'✅' if result['financial_data'].get(symbol, False) else '❌'}")
            
            print(f"    新闻数据: {'✅' if result['news_data'] else '❌'}")
        
        print("\n" + "=" * 50)
        print("🎉 多数据源下载示例完成!")
        
    except Exception as e:
        print(f"\n❌ 下载过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


async def download_single_stock_example():
    """单个股票下载示例"""
    print("\n🔍 单个股票下载示例")
    print("=" * 30)
    
    downloader = DataDownloader(output_dir="backtest_data")
    
    # 只下载AAPL的数据
    symbol = "AAPL"
    start_date = "2025-03-01"
    end_date = "2025-07-31"
    
    print(f"下载 {symbol} 的价格数据...")
    await downloader.download_price_data_multi_source(symbol, start_date, end_date)
    
    print(f"下载 {symbol} 的市场信息...")
    await downloader.download_market_info_multi_source(symbol)
    
    print(f"下载 {symbol} 的财务数据...")
    await downloader.download_financial_data_multi_source(symbol)
    
    print("✅ 单个股票下载完成!")


async def data_source_priority_example():
    """数据源优先级示例"""
    print("\n⚡ 数据源优先级示例")
    print("=" * 30)
    
    downloader = DataDownloader(output_dir="backtest_data")
    
    print("价格数据源优先级:")
    for i, source in enumerate(downloader.data_sources["price"], 1):
        print(f"  {i}. {source}")
    
    print("\n新闻数据源优先级:")
    for i, source in enumerate(downloader.data_sources["news"], 1):
        print(f"  {i}. {source}")
    
    print("\n财务数据源优先级:")
    for i, source in enumerate(downloader.data_sources["financials"], 1):
        print(f"  {i}. {source}")
    
    print("\n市场信息数据源优先级:")
    for i, source in enumerate(downloader.data_sources["market_info"], 1):
        print(f"  {i}. {source}")


if __name__ == "__main__":
    # 运行主示例
    asyncio.run(main())
    
    # 运行其他示例
    asyncio.run(download_single_stock_example())
    asyncio.run(data_source_priority_example()) 