#!/usr/bin/env python3
"""
测试所有API数据源脚本

这个脚本会测试所有配置的API，并为每个API创建单独的目录，
方便用户对比不同数据源的数据质量和完整性。

使用方法:
python trading_agent/test_all_apis.py
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from data_sources.data_downloader import DataDownloader


async def test_all_apis():
    """测试所有API数据源"""
    
    # 加载环境变量
    load_dotenv()
    
    print("🧪 测试所有API数据源")
    print("=" * 60)
    
    # 创建下载器
    downloader = DataDownloader(output_dir="api_test_results")
    
    # 定义测试参数
    symbols = ["AAPL", "MSFT"]  # 只测试2个股票以节省时间
    start_date = "2025-03-01"
    end_date = "2025-03-10"  # 只测试10天数据
    
    print(f"📊 测试配置:")
    print(f"  股票: {', '.join(symbols)}")
    print(f"  时间范围: {start_date} 到 {end_date}")
    print(f"  输出目录: {downloader.output_dir}")
    
    print("\n" + "=" * 60)
    
    try:
        # 启动测试所有API模式
        await downloader.download_all_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_news=True,
            include_financials=True,
            force_download=True,  # 强制重新下载
            test_all_apis=True
        )
        
        print("\n" + "=" * 60)
        print("📊 测试结果汇总:")
        
        # 读取并显示每个API的测试结果
        for api in ["yfinance", "finnhub", "polygon", "alpha_vantage", "tiingo"]:
            result_file = os.path.join(downloader.output_dir, f"test_{api}", "test_results.json")
            if os.path.exists(result_file):
                import json
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                print(f"\n  {result['api_name']}:")
                print(f"    成功率: {result['success_rate']:.1f}% ({result['successful_tests']}/{result['total_tests']})")
                
                # 显示详细结果
                for data_type, status in result['results'].items():
                    if isinstance(status, dict):
                        success_count = sum(1 for s in status.values() if s)
                        total_count = len(status)
                        print(f"    {data_type}: {success_count}/{total_count} 成功")
                    else:
                        print(f"    {data_type}: {'✅' if status else '❌'}")
        
        print("\n" + "=" * 60)
        print("🎉 API测试完成!")
        print(f"📁 详细数据保存在: {downloader.output_dir}/test_* 目录中")
        print("📊 你可以手动对比不同API的数据文件")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


async def compare_data_quality():
    """对比不同API的数据质量"""
    
    print("\n🔍 数据质量对比分析")
    print("=" * 40)
    
    # 这里可以添加数据质量对比的逻辑
    # 比如对比价格数据的完整性、准确性等
    
    print("📈 数据质量对比功能开发中...")
    print("💡 你可以手动查看 test_* 目录中的数据文件进行对比")


if __name__ == "__main__":
    # 运行API测试
    asyncio.run(test_all_apis())
    
    # 运行数据质量对比
    asyncio.run(compare_data_quality()) 