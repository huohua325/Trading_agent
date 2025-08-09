#!/usr/bin/env python3
"""
全面的API测试脚本
测试所有数据源的功能，确保在测试模式下能够测试所有API
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_sources.data_downloader import DataDownloader

async def test_all_apis():
    """测试所有API的功能"""
    
    # 加载环境变量
    load_dotenv()
    
    # 创建下载器
    downloader = DataDownloader(output_dir="api_test_results")
    
    # 定义测试参数
    symbols = ["AAPL", "MSFT"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print("🧪 开始全面API测试...")
    print(f"📊 测试股票: {', '.join(symbols)}")
    print(f"📅 时间范围: {start_date} 到 {end_date}")
    print("=" * 60)
    
    # 测试所有API
    await downloader.download_all_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        include_news=True,
        include_financials=True,
        force_download=True,
        test_all_apis=True
    )
    
    print("\n" + "=" * 60)
    print("🎉 全面API测试完成!")
    print("📁 测试结果保存在 api_test_results/test_* 目录中")

async def test_single_api_comprehensive():
    """测试单个API的全面功能（测试模式）"""
    
    # 加载环境变量
    load_dotenv()
    
    # 创建下载器
    downloader = DataDownloader(output_dir="single_api_test")
    
    # 定义测试参数
    symbols = ["AAPL"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    print("🧪 开始单个API全面测试...")
    print(f"📊 测试股票: {', '.join(symbols)}")
    print(f"📅 时间范围: {start_date} 到 {end_date}")
    print("=" * 60)
    
    # 测试价格数据（测试模式）
    print("\n📊 测试价格数据下载（测试模式）...")
    for symbol in symbols:
        results = await downloader.download_price_data_multi_source(
            symbol, start_date, end_date, test_mode=True
        )
        print(f"📈 {symbol} 价格数据测试完成")
    
    # 测试市场信息（测试模式）
    print("\n📋 测试市场信息下载（测试模式）...")
    for symbol in symbols:
        results = await downloader.download_market_info_multi_source(
            symbol, test_mode=True
        )
        print(f"📈 {symbol} 市场信息测试完成")
    
    # 测试财务数据（测试模式）
    print("\n💰 测试财务数据下载（测试模式）...")
    for symbol in symbols:
        results = await downloader.download_financial_data_multi_source(
            symbol, test_mode=True
        )
        print(f"📈 {symbol} 财务数据测试完成")
    
    # 测试新闻数据（测试模式）
    print("\n📰 测试新闻数据下载（测试模式）...")
    results = await downloader.download_news_data_multi_source(
        symbols, start_date, end_date, limit=10, test_mode=True
    )
    print("📈 新闻数据测试完成")
    
    print("\n" + "=" * 60)
    print("🎉 单个API全面测试完成!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API测试工具")
    parser.add_argument("--mode", choices=["all", "single"], default="all",
                       help="测试模式: all=测试所有API, single=测试单个API的全面功能")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        asyncio.run(test_all_apis())
    else:
        asyncio.run(test_single_api_comprehensive()) 