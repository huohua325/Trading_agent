#!/usr/bin/env python3
"""
交易代理主程序

用法:
    python main.py --mode single    # 运行单个交易周期
    python main.py --mode continuous --duration 2  # 连续交易2小时
    python main.py --mode demo      # 演示模式
"""

import asyncio
import argparse
import sys
from typing import Optional

from trading_agent.utils.helpers import create_agent, check_environment, print_portfolio_summary, print_trade_result
from trading_agent.actions.action_types import TradingAction, ActionType


async def run_single_cycle():
    """运行单个交易周期"""
    print("🤖 启动交易代理 - 单周期模式")
    
    # 检查环境
    if not check_environment():
        return
    
    # 创建代理
    agent = create_agent()
    
    try:
        # 初始化
        if not await agent.initialize():
            print("❌ 代理初始化失败")
            return
        
        # 启动交易
        if not await agent.start_trading():
            print("❌ 启动交易失败")
            return
        
        # 显示初始状态
        portfolio = await agent.get_portfolio_status()
        print_portfolio_summary(portfolio)
        
        # 运行一个交易周期
        print("\n🔄 开始交易周期...")
        result = await agent.run_trading_cycle()
        
        # 显示结果
        print("\n📊 交易周期结果:")
        print(f"耗时: {result.get('cycle_duration', 0):.2f} 秒")
        
        # 显示决策和执行结果
        decision = result.get('decision', {})
        execution_result = result.get('execution_result', {})
        
        print(f"\n💭 AI决策: {decision.get('action_type', 'N/A')}")
        if decision.get('symbol'):
            print(f"股票: {decision['symbol']}")
        if decision.get('reason'):
            print(f"理由: {decision['reason']}")
        
        print_trade_result(execution_result)
        
        # 显示最终状态
        final_portfolio = result.get('portfolio_status', {})
        print_portfolio_summary(final_portfolio)
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        await agent.stop_trading()
        print("🛑 交易会话已结束")


async def run_continuous_trading(duration_hours: Optional[int] = None):
    """连续交易模式"""
    print(f"🤖 启动交易代理 - 连续模式 ({duration_hours or '无限'}小时)")
    
    # 检查环境
    if not check_environment():
        return
    
    # 创建代理
    agent = create_agent()
    
    try:
        # 初始化
        if not await agent.initialize():
            print("❌ 代理初始化失败")
            return
            
        # 显示初始状态
        portfolio = await agent.get_portfolio_status()
        print_portfolio_summary(portfolio)
        
        # 运行连续交易
        await agent.run_continuous_trading(duration_hours)
        
        # 显示最终分析
        print("\n📈 最终分析:")
        analysis = await agent.analyze_performance()
        
        metrics = analysis.get('basic_metrics', {})
        print(f"总收益: ${metrics.get('total_return', 0):,.2f}")
        print(f"收益率: {metrics.get('total_return_percent', 0):.2f}%")
        print(f"交易次数: {metrics.get('number_of_trades', 0)}")
        
        sentiment = analysis.get('market_sentiment', {})
        print(f"市场情绪: {sentiment.get('overall_sentiment', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  接收到停止信号")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        await agent.stop_trading()
        print("🛑 交易会话已结束")


async def run_demo():
    """演示模式"""
    print("🎬 交易代理演示模式")
    
    # 检查环境
    if not check_environment():
        return
    
    # 创建代理
    agent = create_agent()
    
    try:
        # 初始化
        print("1. 初始化交易代理...")
        if not await agent.initialize():
            print("❌ 代理初始化失败")
            return
        
        await agent.start_trading()
        
        # 显示初始状态
        print("\n2. 当前投资组合状态:")
        portfolio = await agent.get_portfolio_status()
        print_portfolio_summary(portfolio)
        
        # 演示获取市场数据
        print("\n3. 获取市场数据...")
        market_data = await agent.get_market_data()
        for symbol, data in list(market_data.items())[:3]:  # 只显示前3个
            print(f"  {symbol}: ${data.get('price', 'N/A')} ({data.get('change_percent', 0):+.2f}%)")
        
        # 演示获取新闻
        print("\n4. 获取市场新闻...")
        news = await agent.get_news_data()
        for i, article in enumerate(news[:3], 1):  # 只显示前3条
            print(f"  {i}. {article.get('title', 'No title')[:60]}...")
        
        # 演示AI决策
        print("\n5. AI决策生成...")
        decision = await agent.make_decision()
        print(f"  决策: {decision.action_type.value if hasattr(decision.action_type, 'value') else decision.action_type}")
        print(f"  理由: {decision.reason}")
        
        # 演示执行决策
        print("\n6. 执行决策...")
        result = await agent.execute_decision(decision)
        print_trade_result(result)
        
        # 显示最终状态
        print("\n7. 更新后的投资组合:")
        final_portfolio = await agent.get_portfolio_status()
        print_portfolio_summary(final_portfolio)
        
        print("\n✅ 演示完成！")
        
    except Exception as e:
        print(f"❌ 演示出错: {e}")
    finally:
        await agent.stop_trading()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI交易代理")
    parser.add_argument(
        "--mode", 
        choices=["single", "continuous", "demo"],
        default="demo",
        help="运行模式"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="连续模式的运行时长（小时）"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "single":
            asyncio.run(run_single_cycle())
        elif args.mode == "continuous":
            asyncio.run(run_continuous_trading(args.duration))
        elif args.mode == "demo":
            asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 