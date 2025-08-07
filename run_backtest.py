#!/usr/bin/env python3
"""
回测脚本 - 使用历史数据测试交易代理的性能

用法:
    python run_backtest.py --start_date 2025-03-01 --end_date 2025-07-31 --symbols AAPL,MSFT,GOOGL
"""

import asyncio
import argparse
import sys
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_agent.utils.helpers import check_environment
from trading_agent.data_sources.backtest_data_source import BacktestDataSource
from trading_agent.brokers.backtest_broker import BacktestBroker
from trading_agent.llm.gpt4o_llm import GPT4oLLM
from trading_agent.agents.trading_agent import TradingAgent
from trading_agent.data_sources.data_downloader import DataDownloader
from trading_agent.config.config import TradingConfig


async def download_data(symbols: List[str], start_date: str, end_date: str, force_download: bool = False):
    """下载回测所需的数据"""
    print(f"📥 下载回测数据: {symbols} ({start_date} 到 {end_date})")
    
    # 创建下载器
    downloader = DataDownloader(output_dir="backtest_data")
    
    # 检查数据是否已存在
    data_status = downloader.check_data_exists(symbols)
    
    # 检查是否所有数据都已存在
    all_data_exists = True
    for symbol in symbols:
        if not data_status["price_data"].get(symbol, False):
            all_data_exists = False
            break
        if not data_status["market_info"].get(symbol, False):
            all_data_exists = False
            break
        if not data_status["financial_data"].get(symbol, False):
            all_data_exists = False
            break
    
    if not data_status["news_data"]:
        all_data_exists = False
    
    if all_data_exists and not force_download:
        print("✅ 所有数据文件已存在，无需下载")
        return
    
    # 下载数据
    await downloader.download_all_data(symbols, start_date, end_date, force_download=force_download)
    
    print("✅ 数据下载完成")


async def run_backtest(start_date: str, end_date: str, symbols: List[str], interval: str = "1d", api_backtest: bool = False, yfinance_backtest: bool = False):
    """运行回测"""
    print(f"🤖 启动交易代理 - 回测模式 ({start_date} 到 {end_date})")
    
    # 检查环境
    if not check_environment():
        return
    
    # 创建回测专用数据源和Broker
    data_source_config = {
        "start_date": start_date,
        "end_date": end_date,
        "data_dir": "backtest_data"
    }

    if yfinance_backtest:
        from trading_agent.data_sources.yfinance_backtest_data_source import YFinanceBacktestDataSource
        data_source = YFinanceBacktestDataSource(data_source_config)
    elif api_backtest:
        from trading_agent.data_sources.finnhub_backtest_data_source import FinnhubBacktestDataSource
        data_source = FinnhubBacktestDataSource(data_source_config | {
            "finnhub_api_key": os.getenv("FINNHUB_API_KEY")
        })
    else:
        from trading_agent.data_sources.backtest_data_source import BacktestDataSource
        data_source = BacktestDataSource(data_source_config)

    broker_config = {
        "initial_balance": 100000.0,
        "commission_rate": 0.001,  # 0.1%佣金率
        "slippage": 0.001  # 0.1%滑点
    }
    
    broker = BacktestBroker(broker_config)
    
    # 创建回测专用代理
    config = {
        "trading_symbols": symbols,
        "trading_interval": 0  # 回测模式下不需要等待
    }
    
    # 创建代理
    # 使用TradingConfig中的配置
    trading_config = TradingConfig()
    
    # 创建LLM实例
    llm_config = {
        "openai_api_key": trading_config.openai_api_key,
        "openai_api_base": trading_config.openai_api_base,
        "openai_model": trading_config.openai_model,
        "max_tokens": trading_config.max_tokens,
        "temperature": trading_config.temperature
    }
    
    llm = GPT4oLLM(llm_config)
    
    agent = TradingAgent(broker=broker, data_source=data_source, llm=llm, config=config)
    
    # 设置Broker的价格数据源
    broker.set_price_data_source(data_source)
    
    try:
        # 初始化
        if not await agent.initialize():
            print("❌ 代理初始化失败")
            return
        
        # 加载回测数据
        print("📊 加载回测数据...")
        await data_source.load_data(symbols)
        
        # 启动交易
        if not await agent.start_trading():
            print("❌ 启动交易失败")
            return
        
        # 显示初始状态
        print("\n📊 初始投资组合状态:")
        data_source.set_current_date(start_date)
        broker.set_current_date(start_date)
        initial_portfolio = await agent.get_portfolio_status()
        initial_value = initial_portfolio.get('total_value', broker_config["initial_balance"])
        print(f"初始资金: ${initial_value:,.2f}")
        
        # 回测主循环
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        cycle_count = 0
        
        while current_date <= end_datetime:
            print(f"\n📅 当前回测日期: {current_date.strftime('%Y-%m-%d')}")
            
            # 设置当前回测日期
            data_source.set_current_date(current_date)
            broker.set_current_date(current_date)
            
            # 运行一个交易周期
            result = await agent.run_trading_cycle()
            cycle_count += 1
            
            # 显示结果
            execution_result = result.get('execution_result', {})
            if execution_result.get('action') in ['buy', 'sell']:
                action = execution_result.get('action')
                symbol = execution_result.get('symbol')
                quantity = execution_result.get('quantity')
                price = execution_result.get('price')
                
                if action == 'buy':
                    print(f"买入: {quantity} 股 {symbol} @ ${price:.2f}")
                elif action == 'sell':
                    profit_loss = execution_result.get('profit_loss', 0)
                    profit_loss_percent = execution_result.get('profit_loss_percent', 0)
                    print(f"卖出: {quantity} 股 {symbol} @ ${price:.2f}, 盈亏: ${profit_loss:+,.2f} ({profit_loss_percent:+.2f}%)")
            
            # 更新日期
            if interval == "1d":
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:         # 5/6 = Sat/Sun
                    current_date += timedelta(days=1)
            elif interval == "1h":
                current_date += timedelta(hours=1)
            elif interval == "1w":
                current_date += timedelta(days=7)
            else:
                current_date += timedelta(days=1)
            
            # 每10个周期打印一次状态报告
            if cycle_count % 10 == 0:
                portfolio = await agent.get_portfolio_status()
                print(f"\n状态报告 #{cycle_count // 10}")
                print(f"已完成 {cycle_count} 个交易周期")
                print(f"当前投资组合价值: ${portfolio.get('total_value', 0):,.2f}")
                print(f"收益率: {portfolio.get('return_percent', 0):+.2f}%")
        
        # 回测结束，显示性能分析
        print("\n" + "=" * 50)
        print("📈 回测结果分析")
        print("=" * 50)
        
        # 获取性能分析
        analysis = await agent.analyze_performance()
        
        # 显示基本指标
        print("\n📊 基本绩效指标:")
        metrics = analysis.get('basic_metrics', {})
        print(f"  总收益: ${metrics.get('total_return', 0):,.2f}")
        print(f"  收益率: {metrics.get('total_return_percent', 0):+.2f}%")
        print(f"  交易次数: {metrics.get('number_of_trades', 0)}")
        print(f"  成功交易: {metrics.get('successful_trades', 0)}")
        print(f"  胜率: {metrics.get('win_rate', 0):.2f}%")
        
        # 显示风险指标
        print("\n📉 风险指标:")
        risk_metrics = metrics.get('max_drawdown', 0)
        print(f"  最大回撤: {risk_metrics:.2f}%")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  索提诺比率: {metrics.get('sortino_ratio', 0):.2f}")
        
        # 显示最终投资组合状态
        print("\n💼 最终投资组合状态:")
        final_portfolio = await agent.get_portfolio_status()
        final_value = final_portfolio.get('total_value', 0)
        print(f"  总资产: ${final_value:,.2f}")
        print(f"  现金: ${final_portfolio.get('cash', 0):,.2f}")
        
        # 显示持仓
        holdings = final_portfolio.get('holdings', [])
        if holdings:
            print("  持仓:")
            for holding in holdings:
                symbol = holding.get('symbol')
                quantity = holding.get('quantity')
                avg_price = holding.get('avg_price')
                current_price = holding.get('current_price')
                position_value = holding.get('position_value')
                unrealized_pl = holding.get('unrealized_pl')
                unrealized_pl_percent = holding.get('unrealized_pl_percent')
                
                print(f"    {symbol}: {quantity} 股 @ ${avg_price:.2f}, 当前价格: ${current_price:.2f}, 价值: ${position_value:,.2f}, 盈亏: ${unrealized_pl:+,.2f} ({unrealized_pl_percent:+.2f}%)")
        
        # 显示收益变化
        if initial_value > 0:
            change = final_value - initial_value
            change_percent = (change / initial_value) * 100
            print(f"\n💹 总收益: ${change:+,.2f} ({change_percent:+.2f}%)")
        
        # 保存回测结果
        backtest_result = {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "initial_balance": broker_config["initial_balance"],
            "final_balance": final_value,
            "total_return": metrics.get('total_return', 0),
            "total_return_percent": metrics.get('total_return_percent', 0),
            "number_of_trades": metrics.get('number_of_trades', 0),
            "successful_trades": metrics.get('successful_trades', 0),
            "win_rate": metrics.get('win_rate', 0),
            "max_drawdown": risk_metrics,
            "sharpe_ratio": metrics.get('sharpe_ratio', 0),
            "sortino_ratio": metrics.get('sortino_ratio', 0),
            "portfolio_value_history": broker.portfolio_value_history,
            "trade_history": broker.trade_history
        }
        
        # 确保logs目录存在
        os.makedirs("logs", exist_ok=True)
        
        # 保存回测结果
        result_file = os.path.join("logs", "backtest_result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(backtest_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ 回测完成! 结果已保存到 {result_file}")
        
        # 生成回测图表
        generate_backtest_charts(broker.portfolio_value_history, broker.trade_history, symbols, start_date, end_date)
        
        return backtest_result
        
    except Exception as e:
        print(f"❌ 回测出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.stop_trading()
        print("🛑 回测会话已结束")


def generate_backtest_charts(portfolio_history: Dict[str, float], trade_history: List[Dict[str, Any]], symbols: List[str], start_date: str, end_date: str):
    """生成回测图表"""
    print("\n📊 生成回测图表...")
    
    # 确保logs/charts目录存在
    charts_dir = os.path.join("logs", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # 1. 投资组合价值曲线
    try:
        # 转换为DataFrame
        dates = sorted(portfolio_history.keys())
        values = [portfolio_history[date] for date in dates]
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'value': values
        })
        df.set_index('date', inplace=True)
        
        # 计算基准指数（使用第一个股票作为基准）
        benchmark_df = None
        if symbols:
            try:
                benchmark_symbol = symbols[0]
                benchmark_file = os.path.join("backtest_data", f"{benchmark_symbol}_prices.csv")
                if os.path.exists(benchmark_file):
                    benchmark_df = pd.read_csv(benchmark_file, index_col=0, parse_dates=True)
                    
                    # 计算基准收益率
                    initial_price = benchmark_df['Close'].iloc[0]
                    initial_value = df.iloc[0]['value']
                    
                    benchmark_df = benchmark_df[benchmark_df.index >= df.index[0]]
                    benchmark_df = benchmark_df[benchmark_df.index <= df.index[-1]]
                    
                    # 归一化基准价格
                    benchmark_df['normalized'] = initial_value * (benchmark_df['Close'] / initial_price)
            except Exception as e:
                print(f"无法加载基准数据: {e}")
        
        # 绘制投资组合价值曲线
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['value'], label='Portfolio Value', linewidth=2)
        
        # 添加基准指数
        if benchmark_df is not None:
            plt.plot(benchmark_df.index, benchmark_df['normalized'], label=f'{benchmark_symbol} (Benchmark)', linewidth=1, linestyle='--')
        
        # 添加交易标记
        buy_dates = []
        buy_values = []
        sell_dates = []
        sell_values = []
        
        for trade in trade_history:
            trade_date = datetime.fromtimestamp(trade['timestamp'])
            trade_value = trade['cash_after']
            
            if trade['action'] == 'buy':
                buy_dates.append(trade_date)
                buy_values.append(trade_value)
            elif trade['action'] == 'sell':
                sell_dates.append(trade_date)
                sell_values.append(trade_value)
        
        plt.scatter(buy_dates, buy_values, color='green', marker='^', label='Buy', alpha=0.7)
        plt.scatter(sell_dates, sell_values, color='red', marker='v', label='Sell', alpha=0.7)
        
        # 添加标题和标签
        plt.title(f'Portfolio Value Over Time ({start_date} to {end_date})')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        portfolio_chart_file = os.path.join(charts_dir, "portfolio_value.png")
        plt.savefig(portfolio_chart_file)
        print(f"✅ 投资组合价值曲线已保存到 {portfolio_chart_file}")
        
        # 2. 收益率曲线
        plt.figure(figsize=(12, 6))
        
        # 计算收益率
        df['return'] = df['value'].pct_change().fillna(0).cumsum() * 100
        
        # 计算基准收益率
        if benchmark_df is not None:
            benchmark_df['return'] = benchmark_df['normalized'].pct_change().fillna(0).cumsum() * 100
            plt.plot(benchmark_df.index, benchmark_df['return'], label=f'{benchmark_symbol} (Benchmark)', linewidth=1, linestyle='--')
        
        plt.plot(df.index, df['return'], label='Portfolio Return', linewidth=2)
        
        # 添加标题和标签
        plt.title(f'Cumulative Return (%) ({start_date} to {end_date})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        return_chart_file = os.path.join(charts_dir, "cumulative_return.png")
        plt.savefig(return_chart_file)
        print(f"✅ 收益率曲线已保存到 {return_chart_file}")
        
        # 3. 回撤曲线
        plt.figure(figsize=(12, 6))
        
        # 计算回撤
        df['peak'] = df['value'].cummax()
        df['drawdown'] = (df['value'] / df['peak'] - 1) * 100
        
        plt.plot(df.index, df['drawdown'], label='Drawdown', linewidth=2, color='red')
        
        # 添加标题和标签
        plt.title(f'Portfolio Drawdown (%) ({start_date} to {end_date})')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        drawdown_chart_file = os.path.join(charts_dir, "drawdown.png")
        plt.savefig(drawdown_chart_file)
        print(f"✅ 回撤曲线已保存到 {drawdown_chart_file}")
        
    except Exception as e:
        print(f"❌ 生成图表失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI交易代理回测工具")
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="回测开始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="回测结束日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,MSFT,GOOGL",
        help="回测股票代码，逗号分隔"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1d", "1h", "1w"],
        help="回测时间间隔"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="下载回测数据"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载数据（即使文件已存在）"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查数据是否存在，不运行回测"
    )
    parser.add_argument(
        "--api_backtest",
        action="store_true",
        help="使用Finnhub API进行在线回测（不读取本地CSV）"
    )
    parser.add_argument(
        "--yfinance_backtest",
        action="store_true",
        help="使用yfinance在线拉取历史数据进行回测"
    )
    
    args = parser.parse_args()
    
    # 解析股票代码列表
    symbols = args.symbols.split(",")
    
    try:
        # 创建下载器用于检查数据
        downloader = DataDownloader(output_dir="backtest_data")
        # 若选择 API 回测则跳过文件检查/下载
        if args.api_backtest or args.yfinance_backtest:
             await run_backtest(args.start_date, args.end_date, symbols, args.interval, args.api_backtest, args.yfinance_backtest)
             return

        # 以下逻辑仅在本地文件回测时执行
        if args.check:
            print(f"检查数据文件是否存在: {symbols}")
            data_status = downloader.check_data_exists(symbols)
            
            print("\n价格数据:")
            for symbol, exists in data_status["price_data"].items():
                status = "✅ 存在" if exists else "❌ 不存在"
                print(f"  {symbol}: {status}")
            
            print("\n市场信息:")
            for symbol, exists in data_status["market_info"].items():
                status = "✅ 存在" if exists else "❌ 不存在"
                print(f"  {symbol}: {status}")
            
            print("\n财务数据:")
            for symbol, exists in data_status["financial_data"].items():
                status = "✅ 存在" if exists else "❌ 不存在"
                print(f"  {symbol}: {status}")
            
            print("\n新闻数据:")
            status = "✅ 存在" if data_status["news_data"] else "❌ 不存在"
            print(f"  {status}")
            
            # 检查是否所有数据都已存在
            all_data_exists = True
            for symbol in symbols:
                if not data_status["price_data"].get(symbol, False):
                    all_data_exists = False
                    break
                if not data_status["market_info"].get(symbol, False):
                    all_data_exists = False
                    break
                if not data_status["financial_data"].get(symbol, False):
                    all_data_exists = False
                    break
            
            if not data_status["news_data"]:
                all_data_exists = False
            
            if all_data_exists:
                print("\n✅ 所有数据文件已存在，可以直接运行回测")
            else:
                print("\n❌ 部分数据文件不存在，需要先下载数据")
                print("   运行命令: python run_backtest.py --start_date {} --end_date {} --symbols {} --download".format(
                    args.start_date, args.end_date, args.symbols
                ))
            
            return
        
        # 下载数据（如果需要）
        if args.download:
            await download_data(symbols, args.start_date, args.end_date, args.force)
        else:
            # 检查数据是否存在
            data_status = downloader.check_data_exists(symbols)
            all_data_exists = True
            for symbol in symbols:
                if not data_status["price_data"].get(symbol, False):
                    all_data_exists = False
                    break
                if not data_status["market_info"].get(symbol, False):
                    all_data_exists = False
                    break
                if not data_status["financial_data"].get(symbol, False):
                    all_data_exists = False
                    break
            
            if not data_status["news_data"]:
                all_data_exists = False
            
            if not all_data_exists:
                print("❌ 部分数据文件不存在，需要先下载数据")
                print("   运行命令: python run_backtest.py --start_date {} --end_date {} --symbols {} --download".format(
                    args.start_date, args.end_date, args.symbols
                ))
                return
        
        # 运行回测
        await run_backtest(args.start_date, args.end_date, symbols, args.interval, args.api_backtest, args.yfinance_backtest)
        
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 