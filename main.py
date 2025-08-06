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
import os
import json
import pickle
from typing import Optional
from datetime import datetime

from trading_agent.utils.helpers import create_agent, check_environment, print_portfolio_summary, print_trade_result, print_financial_data_summary
from trading_agent.actions.action_types import TradingAction, ActionType


# 保存和恢复会话的函数
def save_session_state(agent, start_time, cycle_count, filename="trading_session.pkl"):
    """保存交易会话状态"""
    try:
        session_data = {
            "start_time": start_time,
            "cycle_count": cycle_count,
            "portfolio": agent.broker.portfolio if hasattr(agent.broker, "portfolio") else {},
            "market_sentiment": agent.market_sentiment,
            "financial_data": agent.financial_data,
            "trading_symbols": agent.trading_symbols,
            "timestamp": datetime.now().isoformat()
        }
        
        # 确保logs目录存在
        os.makedirs("logs", exist_ok=True)
        
        # 保存会话数据
        with open(os.path.join("logs", filename), "wb") as f:
            pickle.dump(session_data, f)
        
        # 保存可读的JSON版本
        json_data = {k: v for k, v in session_data.items() if k not in ["start_time"]}
        json_data["start_time_iso"] = start_time.isoformat() if start_time else None
        
        with open(os.path.join("logs", "session_state.json"), "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
            
        return True
    except Exception as e:
        print(f"保存会话状态失败: {e}")
        return False


def load_session_state(filename="trading_session.pkl"):
    """加载交易会话状态"""
    try:
        filepath = os.path.join("logs", filename)
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, "rb") as f:
            session_data = pickle.load(f)
        
        return session_data
    except Exception as e:
        print(f"加载会话状态失败: {e}")
        return None


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
        print("\n📊 初始投资组合状态:")
        portfolio = await agent.get_portfolio_status()
        print_portfolio_summary(portfolio)
        
        # 获取市场数据
        print("\n📈 获取市场数据...")
        market_data = await agent.get_market_data()
        for symbol, data in list(market_data.items())[:3]:  # 只显示前3个
            print(f"  {symbol}: ${data.get('price', 'N/A')} ({data.get('change_percent', 0):+.2f}%)")
        
        # 获取新闻数据
        print("\n📰 获取市场新闻...")
        news = await agent.get_news_data()
        for i, article in enumerate(news[:3], 1):  # 只显示前3条
            print(f"  {i}. {article.get('title', 'No title')[:60]}...")
        
        # 市场情绪分析
        print("\n🧠 分析市场情绪...")
        if not agent.market_sentiment:  # 如果没有缓存的市场情绪数据
            try:
                if news:
                    # 如果有交易股票，针对第一个股票进行情绪分析
                    symbol = None
                    if agent.trading_symbols:
                        symbol = agent.trading_symbols[0] if isinstance(agent.trading_symbols, list) else agent.trading_symbols
                    
                    agent.market_sentiment = await agent.llm.analyze_market_sentiment(news, symbol)
            except Exception as e:
                print(f"  市场情绪分析失败: {e}")
                # 如果分析失败，使用中性情绪
                agent.market_sentiment = {
                    "overall_sentiment": "neutral",
                    "confidence": 0.5,
                    "risk_level": "medium",
                    "key_factors": ["分析失败"],
                    "recommendation": "谨慎观察"
                }
        else:
            print("  使用缓存的市场情绪数据")
        
        # 打印市场情绪分析结果
        market_sentiment = agent.market_sentiment
        if market_sentiment:
            sentiment_map = {
                "positive": "积极", "negative": "消极", "neutral": "中性",
                "bullish": "看涨", "bearish": "看跌"
            }
            risk_map = {"low": "低", "medium": "中", "high": "高"}
            
            overall = market_sentiment.get("overall_sentiment", "neutral")
            translated_sentiment = sentiment_map.get(overall, overall)
            risk_level = market_sentiment.get("risk_level", "medium")
            translated_risk = risk_map.get(risk_level, risk_level)
            
            print(f"  整体情绪: {translated_sentiment}")
            print(f"  信心指数: {market_sentiment.get('confidence', 0):.2f}")
            print(f"  风险水平: {translated_risk}")
            
            # 打印关键因素
            key_factors = market_sentiment.get('key_factors', [])
            if key_factors:
                print("  关键因素:")
                for factor in key_factors[:3]:  # 只显示前3个因素
                    print(f"    - {factor}")
            
            # 打印建议
            if "recommendation" in market_sentiment:
                print(f"  建议: {market_sentiment['recommendation']}")
        
        # 获取财务数据
        print("\n💰 获取财务数据...")
        if not agent.financial_data:  # 如果没有缓存的财务数据
            financial_data = {}
            if hasattr(agent.data_source, 'get_company_financials') and agent.trading_symbols:
                # 只获取第一个交易股票的财务数据作为示例
                symbol = agent.trading_symbols[0] if isinstance(agent.trading_symbols, list) else agent.trading_symbols
                print(f"  正在获取 {symbol} 的财务数据...")
                
                try:
                    # 获取基本财务指标
                    if hasattr(agent.data_source, 'get_financial_metrics'):
                        metrics = await agent.data_source.get_financial_metrics(symbol)
                        financial_data["key_metrics"] = metrics
                    
                    # 获取盈利惊喜
                    earnings = await agent.data_source.get_earnings_surprises(symbol)
                    if earnings and len(earnings) > 0:
                        financial_data["earnings_surprises"] = earnings
                    
                    # 获取分析师推荐
                    trends = await agent.data_source.get_recommendation_trends(symbol)
                    if trends and len(trends) > 0:
                        financial_data["recommendation_trends"] = trends
                    
                    agent.financial_data = financial_data
                    print("  财务数据获取成功")
                except Exception as e:
                    print(f"  获取财务数据失败: {e}")
            else:
                print("  数据源不支持获取财务数据或未设置交易股票")
        else:
            print("  使用缓存的财务数据")
        
        # 使用辅助函数打印财务数据摘要
        if agent.financial_data:
            print_financial_data_summary(agent.financial_data)
        
        # 运行一个交易周期
        print("\n🔄 开始交易周期...")
        result = await agent.run_trading_cycle()
        
        # 显示结果
        print("\n📊 交易周期结果:")
        print(f"耗时: {result.get('cycle_duration', 0):.2f} 秒")
        
        # 显示决策和执行结果
        decision = result.get('decision', {})
        execution_result = result.get('execution_result', {})
        
        # 增强的决策展示
        print("\n🧩 AI决策详情:")
        action_type_value = decision.get('action_type', 'N/A')
        if isinstance(action_type_value, dict) and 'value' in action_type_value:
            action_type_value = action_type_value['value']
        
        action_map = {
            "buy": "买入", "sell": "卖出", "hold": "持有",
            "get_info": "获取信息", "get_news": "获取新闻"
        }
        translated_action = action_map.get(str(action_type_value).lower(), str(action_type_value))
        
        print(f"  决策类型: {translated_action}")
        
        if decision.get('symbol'):
            print(f"  交易标的: {decision['symbol']}")
        if decision.get('quantity'):
            print(f"  交易数量: {decision['quantity']}")
        if decision.get('price'):
            print(f"  交易价格: ${decision['price']:.2f}")
        if decision.get('reason'):
            print(f"  决策理由: {decision['reason']}")
        
        # 获取决策解释（如果可用）
        if hasattr(agent.llm, 'explain_decision') and action_type_value in ["buy", "sell"]:
            print("\n🔍 决策深度分析:")
            try:
                explanation = await agent.llm.explain_decision(
                    TradingAction(**decision),
                    {"market_data": market_data, "market_sentiment": agent.market_sentiment}
                )
                
                # 打印详细解释
                if isinstance(explanation, dict):
                    if "market_analysis" in explanation:
                        print(f"  市场分析: {explanation['market_analysis']}")
                    if "risk_assessment" in explanation:
                        print(f"  风险评估: {explanation['risk_assessment']}")
                    if "expected_outcome" in explanation:
                        print(f"  预期结果: {explanation['expected_outcome']}")
                    if "confidence_level" in explanation:
                        print(f"  信心水平: {explanation['confidence_level']}")
                    if "alternative_strategies" in explanation:
                        print(f"  替代策略: {explanation['alternative_strategies']}")
                else:
                    print(f"  {explanation}")
            except Exception as e:
                print(f"  无法获取详细解释: {e}")
        
        # 打印执行结果
        print("\n📋 执行结果:")
        print_trade_result(execution_result)
        
        # 显示最终状态
        print("\n📈 更新后的投资组合状态:")
        final_portfolio = result.get('portfolio_status', {})
        print_portfolio_summary(final_portfolio)
        
        # 显示收益变化
        if portfolio and final_portfolio:
            initial_value = portfolio.get('total_value', 0)
            final_value = final_portfolio.get('total_value', 0)
            if initial_value > 0:
                change = final_value - initial_value
                change_percent = (change / initial_value) * 100
                print(f"\n💹 本次交易收益: ${change:+,.2f} ({change_percent:+.2f}%)")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        await agent.stop_trading()
        print("🛑 交易会话已结束")


async def run_continuous_trading(duration_hours: Optional[int] = None, resume: bool = False):
    """连续交易模式"""
    print(f"🤖 启动交易代理 - 连续模式 ({duration_hours or '无限'}小时)")
    
    # 检查环境
    if not check_environment():
        return
    
    # 创建代理
    agent = create_agent()
    
    # 恢复变量
    start_time = None
    cycle_count = 0
    initial_portfolio = None
    
    # 尝试恢复会话
    if resume:
        print("尝试恢复之前的交易会话...")
        session_data = load_session_state()
        
        if session_data:
            start_time = session_data.get("start_time")
            cycle_count = session_data.get("cycle_count", 0)
            
            # 恢复市场情绪和财务数据
            agent.market_sentiment = session_data.get("market_sentiment", {})
            agent.financial_data = session_data.get("financial_data", {})
            
            # 计算已经过去的时间
            if start_time and duration_hours:
                elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                remaining_hours = duration_hours - elapsed_hours
                
                if remaining_hours <= 0:
                    print("恢复的会话已经超过了设定的持续时间")
                    return
                else:
                    print(f"会话将继续运行 {remaining_hours:.2f} 小时")
                    duration_hours = remaining_hours
            
            print(f"成功恢复会话 - 已完成 {cycle_count} 个交易周期")
        else:
            print("没有找到可恢复的会话，将开始新的会话")
            resume = False
    
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
        print("\n📊 初始投资组合状态:")
        initial_portfolio = await agent.get_portfolio_status()
        print_portfolio_summary(initial_portfolio)
        
        # 定义状态报告函数
        async def print_status_report(cycle_count: int, start_time):
            """打印状态报告"""
            current_portfolio = await agent.get_portfolio_status()
            
            # 计算运行时间
            elapsed_time = datetime.now() - start_time
            hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "=" * 50)
            print(f"📋 状态报告 #{cycle_count // 10}")
            print("=" * 50)
            print(f"已运行: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
            print(f"已完成交易周期: {cycle_count}")
            
            # 显示投资组合状态
            print("\n📈 当前投资组合状态:")
            print_portfolio_summary(current_portfolio)
            
            # 计算收益变化
            if initial_portfolio:
                initial_value = initial_portfolio.get('total_value', 0)
                current_value = current_portfolio.get('total_value', 0)
                if initial_value > 0:
                    change = current_value - initial_value
                    change_percent = (change / initial_value) * 100
                    print(f"\n💹 总收益: ${change:+,.2f} ({change_percent:+.2f}%)")
            
            # 显示最近的交易
            print("\n🔄 最近交易:")
            try:
                recent_trades = await agent.broker.get_trade_history(limit=5)
                if recent_trades:
                    for i, trade in enumerate(recent_trades, 1):
                        trade_time = trade.get('timestamp', 'N/A')
                        if isinstance(trade_time, (int, float)):
                            from datetime import datetime
                            trade_time = datetime.fromtimestamp(trade_time).strftime('%Y-%m-%d %H:%M:%S')
                        
                        action = trade.get('action', 'N/A')
                        symbol = trade.get('symbol', 'N/A')
                        quantity = trade.get('quantity', 0)
                        price = trade.get('price', 0)
                        
                        print(f"  {i}. [{trade_time}] {action} {quantity} {symbol} @ ${price:.2f}")
                else:
                    print("  无最近交易记录")
            except Exception as e:
                print(f"  无法获取最近交易: {e}")
            
            # 显示当前市场情绪
            if agent.market_sentiment:
                print("\n🧠 当前市场情绪:")
                sentiment_map = {
                    "positive": "积极", "negative": "消极", "neutral": "中性",
                    "bullish": "看涨", "bearish": "看跌"
                }
                overall = agent.market_sentiment.get("overall_sentiment", "neutral")
                translated_sentiment = sentiment_map.get(overall, overall)
                print(f"  整体情绪: {translated_sentiment}")
                print(f"  信心指数: {agent.market_sentiment.get('confidence', 0):.2f}")
                
                # 打印建议
                if "recommendation" in agent.market_sentiment:
                    print(f"  建议: {agent.market_sentiment['recommendation']}")
            
            print("=" * 50)
            
            # 保存会话状态
            save_session_state(agent, start_time, cycle_count)
            print("✅ 会话状态已保存")
        
        # 运行连续交易
        if not start_time:  # 如果没有恢复会话，则设置开始时间
            start_time = datetime.now()
        
        last_report_time = datetime.now()
        hourly_report_interval = 3600  # 1小时
        
        while True:
            # 检查运行时间限制
            if duration_hours:
                elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                if elapsed_hours >= duration_hours:
                    print(f"\n⏱️ 达到运行时间限制: {duration_hours}小时")
                    break
            
            # 运行交易周期
            result = await agent.run_trading_cycle()
            cycle_count += 1
            
            # 每10个周期打印一次详细状态报告
            if cycle_count % 10 == 0:
                await print_status_report(cycle_count, start_time)
            
            # 每小时打印一次简要状态
            current_time = datetime.now()
            if (current_time - last_report_time).total_seconds() >= hourly_report_interval:
                portfolio = await agent.get_portfolio_status()
                print(f"\n⏰ 每小时报告 - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"已完成 {cycle_count} 个交易周期")
                print(f"当前投资组合价值: ${portfolio.get('total_value', 0):,.2f}")
                
                # 保存会话状态
                save_session_state(agent, start_time, cycle_count)
                print("✅ 会话状态已保存")
                
                # 更新最后报告时间
                last_report_time = current_time
            
            # 等待下一个周期
            await asyncio.sleep(agent.trading_interval)
        
        # 显示最终分析
        print("\n" + "=" * 50)
        print("📈 最终交易分析报告")
        print("=" * 50)
        
        # 获取性能分析
        analysis = await agent.analyze_performance()
        
        # 1. 基本指标
        print("\n📊 基本绩效指标:")
        metrics = analysis.get('basic_metrics', {})
        print(f"  总收益: ${metrics.get('total_return', 0):,.2f}")
        print(f"  收益率: {metrics.get('total_return_percent', 0):.2f}%")
        print(f"  交易次数: {metrics.get('number_of_trades', 0)}")
        print(f"  成功交易: {metrics.get('successful_trades', 0)}")
        print(f"  胜率: {metrics.get('win_rate', 0):.2f}%")
        
        # 2. 市场情绪分析
        print("\n🧠 最终市场情绪分析:")
        sentiment = analysis.get('market_sentiment', {})
        sentiment_map = {
            "positive": "积极", "negative": "消极", "neutral": "中性",
            "bullish": "看涨", "bearish": "看跌"
        }
        overall = sentiment.get("overall_sentiment", "neutral")
        translated_sentiment = sentiment_map.get(overall, overall)
        print(f"  整体情绪: {translated_sentiment}")
        print(f"  信心指数: {sentiment.get('confidence', 0):.2f}")
        
        # 打印关键因素
        key_factors = sentiment.get('key_factors', [])
        if key_factors:
            print("  关键因素:")
            for factor in key_factors[:3]:
                print(f"    - {factor}")
        
        # 3. 财务数据分析
        print("\n💰 财务数据分析:")
        if agent.financial_data:
            print_financial_data_summary(agent.financial_data)
        else:
            # 如果没有缓存的财务数据，尝试获取
            try:
                if hasattr(agent.data_source, 'get_company_financials') and agent.trading_symbols:
                    # 只获取第一个交易股票的财务数据作为示例
                    symbol = agent.trading_symbols[0] if isinstance(agent.trading_symbols, list) else agent.trading_symbols
                    print(f"  正在获取 {symbol} 的财务数据...")
                    
                    # 获取基本财务指标
                    financial_data = {}
                    if hasattr(agent.data_source, 'get_financial_metrics'):
                        metrics = await agent.data_source.get_financial_metrics(symbol)
                        financial_data["key_metrics"] = metrics
                    
                    # 获取盈利惊喜
                    earnings = await agent.data_source.get_earnings_surprises(symbol)
                    if earnings and len(earnings) > 0:
                        financial_data["earnings_surprises"] = earnings
                    
                    # 获取分析师推荐
                    trends = await agent.data_source.get_recommendation_trends(symbol)
                    if trends and len(trends) > 0:
                        financial_data["recommendation_trends"] = trends
                    
                    print_financial_data_summary(financial_data)
                else:
                    print("  无法获取财务数据: 数据源不支持或未设置交易股票")
            except Exception as e:
                print(f"  获取财务数据失败: {e}")
        
        # 4. 交易历史摘要
        print("\n📜 交易历史摘要:")
        trade_history = analysis.get('trade_history_summary', {})
        print(f"  总交易次数: {trade_history.get('total_trades', 0)}")
        print(f"  成功交易次数: {trade_history.get('successful_trades', 0)}")
        
        # 显示最近的交易
        recent_activity = trade_history.get('recent_activity', [])
        if recent_activity:
            print("\n  最近交易:")
            for i, trade in enumerate(recent_activity[:5], 1):
                trade_time = trade.get('timestamp', 'N/A')
                if isinstance(trade_time, (int, float)):
                    from datetime import datetime
                    trade_time = datetime.fromtimestamp(trade_time).strftime('%Y-%m-%d %H:%M:%S')
                
                action = trade.get('action', 'N/A')
                symbol = trade.get('symbol', 'N/A')
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                
                print(f"    {i}. [{trade_time}] {action} {quantity} {symbol} @ ${price:.2f}")
        
        # 5. 最终投资组合状态
        print("\n💼 最终投资组合状态:")
        final_portfolio = await agent.get_portfolio_status()
        print_portfolio_summary(final_portfolio)
        
        # 计算总收益
        if initial_portfolio:
            initial_value = initial_portfolio.get('total_value', 0)
            final_value = final_portfolio.get('total_value', 0)
            if initial_value > 0:
                change = final_value - initial_value
                change_percent = (change / initial_value) * 100
                print(f"\n💹 总收益: ${change:+,.2f} ({change_percent:+.2f}%)")
        
        print("=" * 50)
        
        # 清除会话状态文件
        try:
            os.remove(os.path.join("logs", "trading_session.pkl"))
            os.remove(os.path.join("logs", "session_state.json"))
            print("✅ 会话状态文件已清除")
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n⏹️  接收到停止信号")
        
        # 保存会话状态以便之后恢复
        if save_session_state(agent, start_time, cycle_count):
            print("✅ 会话状态已保存，您可以使用 --resume 选项恢复此会话")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        
        # 尝试保存会话状态
        if start_time is not None and cycle_count > 0:
            if save_session_state(agent, start_time, cycle_count):
                print("✅ 会话状态已保存，尽管发生错误")
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
        
        # 演示市场情绪分析
        print("\n5. 分析市场情绪...")
        if not agent.market_sentiment:  # 如果没有缓存的市场情绪数据
            try:
                if news:
                    # 如果有交易股票，针对第一个股票进行情绪分析
                    symbol = None
                    if agent.trading_symbols:
                        symbol = agent.trading_symbols[0] if isinstance(agent.trading_symbols, list) else agent.trading_symbols
                    
                    agent.market_sentiment = await agent.llm.analyze_market_sentiment(news, symbol)
            except Exception as e:
                print(f"  市场情绪分析失败: {e}")
                # 如果分析失败，使用中性情绪
                agent.market_sentiment = {
                    "overall_sentiment": "neutral",
                    "confidence": 0.5,
                    "risk_level": "medium",
                    "key_factors": ["分析失败"],
                    "recommendation": "谨慎观察"
                }
        else:
            print("  使用缓存的市场情绪数据")
        
        # 打印市场情绪分析结果
        market_sentiment = agent.market_sentiment
        if market_sentiment:
            sentiment_map = {
                "positive": "积极", "negative": "消极", "neutral": "中性",
                "bullish": "看涨", "bearish": "看跌"
            }
            risk_map = {"low": "低", "medium": "中", "high": "高"}
            
            overall = market_sentiment.get("overall_sentiment", "neutral")
            translated_sentiment = sentiment_map.get(overall, overall)
            risk_level = market_sentiment.get("risk_level", "medium")
            translated_risk = risk_map.get(risk_level, risk_level)
            
            print(f"  整体情绪: {translated_sentiment}")
            print(f"  信心指数: {market_sentiment.get('confidence', 0):.2f}")
            print(f"  风险水平: {translated_risk}")
            
            # 打印关键因素
            key_factors = market_sentiment.get('key_factors', [])
            if key_factors:
                print("  关键因素:")
                for factor in key_factors[:3]:  # 只显示前3个因素
                    print(f"    - {factor}")
            
            # 打印建议
            if "recommendation" in market_sentiment:
                print(f"  建议: {market_sentiment['recommendation']}")
        
        # 演示获取财务数据
        print("\n6. 获取财务数据...")
        if not agent.financial_data:  # 如果没有缓存的财务数据
            financial_data = {}
            if hasattr(agent.data_source, 'get_company_financials') and agent.trading_symbols:
                # 只获取第一个交易股票的财务数据作为示例
                symbol = agent.trading_symbols[0] if isinstance(agent.trading_symbols, list) else agent.trading_symbols
                print(f"  正在获取 {symbol} 的财务数据...")
                
                try:
                    # 获取基本财务指标
                    if hasattr(agent.data_source, 'get_financial_metrics'):
                        metrics = await agent.data_source.get_financial_metrics(symbol)
                        financial_data["key_metrics"] = metrics
                    
                    # 获取盈利惊喜
                    earnings = await agent.data_source.get_earnings_surprises(symbol)
                    if earnings and len(earnings) > 0:
                        financial_data["earnings_surprises"] = earnings
                    
                    # 获取分析师推荐
                    trends = await agent.data_source.get_recommendation_trends(symbol)
                    if trends and len(trends) > 0:
                        financial_data["recommendation_trends"] = trends
                    
                    agent.financial_data = financial_data
                    print("  财务数据获取成功")
                except Exception as e:
                    print(f"  获取财务数据失败: {e}")
            else:
                print("  数据源不支持获取财务数据或未设置交易股票")
        else:
            print("  使用缓存的财务数据")
        
        # 使用辅助函数打印财务数据摘要
        if agent.financial_data:
            print_financial_data_summary(agent.financial_data)
        
        # 演示AI决策
        print("\n7. AI决策生成...")
        decision = await agent.make_decision()
        print(f"  决策: {decision.action_type.value if hasattr(decision.action_type, 'value') else decision.action_type}")
        print(f"  理由: {decision.reason}")
        
        # 演示执行决策
        print("\n8. 执行决策...")
        result = await agent.execute_decision(decision)
        print_trade_result(result)
        
        # 显示最终状态
        print("\n9. 更新后的投资组合:")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复之前中断的交易会话"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "single":
            asyncio.run(run_single_cycle())
        elif args.mode == "continuous":
            asyncio.run(run_continuous_trading(args.duration, args.resume))
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