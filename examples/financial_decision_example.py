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
from trading_agent.llm.gpt4o_llm import GPT4oLLM
from trading_agent.actions.action_types import TradingAction

# 加载环境变量
load_dotenv()

# 获取API密钥
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not finnhub_api_key:
    raise ValueError("请在.env文件中设置FINNHUB_API_KEY")
if not openai_api_key:
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")

# 初始化数据源和LLM
finnhub_config = {"finnhub_api_key": finnhub_api_key}
data_source = FinnhubDataSource(finnhub_config)

llm_config = {
    "openai_api_key": openai_api_key,
    "openai_model": "gpt-4o",
    "temperature": 0.1
}
llm = GPT4oLLM(llm_config)

# 测试的股票代码
symbol = "AAPL"

async def get_market_data(symbol):
    """获取市场数据"""
    try:
        price_data = await data_source.get_real_time_price(symbol)
        return {symbol: price_data}
    except Exception as e:
        print(f"获取市场数据失败: {e}")
        return {}

async def get_news_data(symbol):
    """获取新闻数据"""
    try:
        news = await data_source.get_news(symbol=symbol, limit=5)
        return news
    except Exception as e:
        print(f"获取新闻数据失败: {e}")
        return []

async def get_financial_data(symbol):
    """获取财务数据"""
    try:
        # 获取综合财务数据
        financials = await data_source.get_company_financials(symbol)
        
        # 获取关键财务指标
        key_metrics = await data_source.get_financial_metrics(symbol)
        financials["key_metrics"] = key_metrics
        
        return financials
    except Exception as e:
        print(f"获取财务数据失败: {e}")
        return {}

async def generate_trading_decision_with_financials():
    """生成包含财务数据的交易决策"""
    print(f"为 {symbol} 生成交易决策...")
    
    # 1. 获取市场数据
    market_data = await get_market_data(symbol)
    if not market_data:
        print("无法获取市场数据")
        return None
    
    # 2. 创建模拟投资组合状态
    portfolio = {
        "cash": 10000.0,
        "total_value": 15000.0,
        "positions": {
            symbol: {
                "quantity": 10,
                "value": 5000.0,
                "avg_price": 150.0
            }
        }
    }
    
    # 3. 获取新闻数据
    news_data = await get_news_data(symbol)
    print(f"获取到 {len(news_data)} 条新闻")
    
    # 4. 获取财务数据
    financial_data = await get_financial_data(symbol)
    
    # 5. 生成决策
    print("正在调用LLM生成决策...")
    decision = await llm.generate_trading_decision(
        market_data=market_data,
        portfolio_status=portfolio,
        news_data=news_data,
        historical_context=None,
        financial_data=financial_data
    )
    
    return decision

async def generate_trading_decision_without_financials():
    """生成不包含财务数据的交易决策"""
    print(f"为 {symbol} 生成交易决策 (不含财务数据)...")
    
    # 1. 获取市场数据
    market_data = await get_market_data(symbol)
    if not market_data:
        print("无法获取市场数据")
        return None
    
    # 2. 创建模拟投资组合状态
    portfolio = {
        "cash": 10000.0,
        "total_value": 15000.0,
        "positions": {
            symbol: {
                "quantity": 10,
                "value": 5000.0,
                "avg_price": 150.0
            }
        }
    }
    
    # 3. 获取新闻数据
    news_data = await get_news_data(symbol)
    print(f"获取到 {len(news_data)} 条新闻")
    
    # 4. 生成决策
    print("正在调用LLM生成决策...")
    decision = await llm.generate_trading_decision(
        market_data=market_data,
        portfolio_status=portfolio,
        news_data=news_data,
        historical_context=None
    )
    
    return decision

async def main():
    print(f"测试财务数据对AI交易决策的影响 - 使用股票: {symbol}")
    print("-" * 50)
    
    # 测试连接
    connected = await data_source.test_connection()
    print(f"API连接测试: {'成功' if connected else '失败'}")
    
    if not connected:
        print("无法连接到Finnhub API，请检查API密钥")
        return
    
    # 1. 生成包含财务数据的交易决策
    print("\n===== 使用财务数据的交易决策 =====")
    decision_with_financials = await generate_trading_decision_with_financials()
    
    # 2. 生成不包含财务数据的交易决策
    print("\n===== 不使用财务数据的交易决策 =====")
    decision_without_financials = await generate_trading_decision_without_financials()
    
    # 3. 比较两种决策
    print("\n===== 决策比较 =====")
    print(f"使用财务数据的决策: {decision_with_financials}")
    print(f"不使用财务数据的决策: {decision_without_financials}")
    
    # 4. 分析差异
    print("\n===== 差异分析 =====")
    if decision_with_financials and decision_without_financials:
        if decision_with_financials.action_type != decision_without_financials.action_type:
            print(f"决策类型不同: {decision_with_financials.action_type} vs {decision_without_financials.action_type}")
        
        if decision_with_financials.quantity != decision_without_financials.quantity:
            print(f"交易数量不同: {decision_with_financials.quantity} vs {decision_without_financials.quantity}")
        
        print("决策理由比较:")
        print(f"使用财务数据: {decision_with_financials.reason}")
        print(f"不使用财务数据: {decision_without_financials.reason}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    asyncio.run(main()) 