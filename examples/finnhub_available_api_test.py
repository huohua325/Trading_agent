import os
import asyncio
import json
from dotenv import load_dotenv
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from trading_agent.data_sources.finnhub_data_source import FinnhubDataSource

# 加载环境变量
load_dotenv()

# 获取API密钥
api_key = os.getenv("FINNHUB_API_KEY")
if not api_key:
    raise ValueError("请在.env文件中设置FINNHUB_API_KEY")

# 初始化数据源
config = {"finnhub_api_key": api_key}
data_source = FinnhubDataSource(config)

# 测试的股票代码
symbol = "AAPL"

async def test_basic_financials():
    """测试基本财务数据API"""
    print("\n1. 测试基本财务数据:")
    try:
        financials = await data_source.get_basic_financials(symbol)
        if "error" in financials:
            print(f"  获取失败: {financials['error']}")
            return
        
        metrics = financials.get("metric", {})
        print(f"  指标数量: {len(metrics)}")
        
        # 打印部分关键指标
        key_metrics = {
            "市盈率": metrics.get("peBasicExclExtraTTM", "N/A"),
            "每股收益": metrics.get("epsBasicExclExtraItemsTTM", "N/A"),
            "股息收益率": metrics.get("dividendYieldIndicatedAnnual", "N/A"),
            "市值": metrics.get("marketCapitalization", "N/A"),
            "52周高点": metrics.get("52WeekHigh", "N/A"),
            "52周低点": metrics.get("52WeekLow", "N/A")
        }
        
        print("  关键指标:")
        for name, value in key_metrics.items():
            print(f"    {name}: {value}")
    except Exception as e:
        print(f"  测试失败: {e}")

async def test_reported_financials():
    """测试已报告的财务报表API"""
    print("\n2. 测试已报告的财务报表:")
    try:
        financials = await data_source.get_reported_financials(symbol)
        if "error" in financials:
            print(f"  获取失败: {financials['error']}")
            return
        
        data = financials.get("data", [])
        print(f"  报表数量: {len(data)}")
        
        if data:
            latest = data[0]
            print(f"  最新报表: {latest.get('year')}年 Q{latest.get('quarter', 0)}, 表格: {latest.get('form')}")
            
            # 打印资产负债表的部分项目
            if "report" in latest and "bs" in latest["report"]:
                bs_items = latest["report"]["bs"][:5]  # 只显示前5项
                print("  资产负债表部分项目:")
                for item in bs_items:
                    print(f"    {item.get('label')}: {item.get('value')}")
    except Exception as e:
        print(f"  测试失败: {e}")

async def test_earnings_surprises():
    """测试盈利惊喜API"""
    print("\n3. 测试盈利惊喜数据:")
    try:
        earnings = await data_source.get_earnings_surprises(symbol)
        if not earnings:
            print("  无盈利惊喜数据")
            return
        
        print(f"  数据点数量: {len(earnings)}")
        
        if earnings:
            latest = earnings[0]
            print(f"  最新季度: {latest.get('period')}")
            print(f"  预期EPS: ${latest.get('estimate')}")
            print(f"  实际EPS: ${latest.get('actual')}")
            print(f"  惊喜百分比: {latest.get('surprisePercent')}%")
    except Exception as e:
        print(f"  测试失败: {e}")

async def test_recommendation_trends():
    """测试推荐趋势API"""
    print("\n4. 测试推荐趋势:")
    try:
        trends = await data_source.get_recommendation_trends(symbol)
        if not trends:
            print("  无推荐趋势数据")
            return
        
        print(f"  数据点数量: {len(trends)}")
        
        if trends:
            latest = trends[0]
            print(f"  最新月份: {latest.get('period')}")
            print(f"  强烈买入: {latest.get('strongBuy')}")
            print(f"  买入: {latest.get('buy')}")
            print(f"  持有: {latest.get('hold')}")
            print(f"  卖出: {latest.get('sell')}")
            print(f"  强烈卖出: {latest.get('strongSell')}")
    except Exception as e:
        print(f"  测试失败: {e}")

async def test_financial_metrics():
    """测试关键财务指标API"""
    print("\n5. 测试关键财务指标:")
    try:
        metrics = await data_source.get_financial_metrics(symbol)
        if "error" in metrics:
            print(f"  获取失败: {metrics['error']}")
            return
        
        print("  关键指标:")
        for name, value in metrics.items():
            print(f"    {name}: {value}")
    except Exception as e:
        print(f"  测试失败: {e}")

async def test_company_financials():
    """测试综合财务数据API"""
    print("\n6. 测试综合财务数据:")
    try:
        financials = await data_source.get_company_financials(symbol)
        
        print("  API状态:")
        for api_name, data in financials.items():
            status = "成功" if "error" not in data else f"失败: {data['error']}"
            print(f"    {api_name}: {status}")
    except Exception as e:
        print(f"  测试失败: {e}")

async def main():
    print(f"测试FinnhubDataSource集成的可用API - 使用股票: {symbol}")
    print("-" * 50)
    
    # 测试连接
    connected = await data_source.test_connection()
    print(f"API连接测试: {'成功' if connected else '失败'}")
    
    if not connected:
        print("无法连接到Finnhub API，请检查API密钥")
        return
    
    # 运行所有测试
    await test_basic_financials()
    await test_reported_financials()
    await test_earnings_surprises()
    await test_recommendation_trends()
    await test_financial_metrics()
    await test_company_financials()
    
    print("\n测试完成!")

if __name__ == "__main__":
    asyncio.run(main()) 