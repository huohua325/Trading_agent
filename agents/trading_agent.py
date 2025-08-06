import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from ..actions.action_types import TradingAction, ActionType
from ..brokers.base_broker import BaseBroker
from ..data_sources.base_data_source import BaseDataSource
from ..llm.base_llm import BaseLLM


class TradingAgent(BaseAgent):
    """交易代理具体实现"""
    
    def __init__(
        self,
        broker: BaseBroker,
        data_source: BaseDataSource,
        llm: BaseLLM,
        config: Dict[str, Any]
    ):
        super().__init__(broker, data_source, llm, config)
        
        # 交易周期设置
        self.trading_interval = config.get("trading_interval", 300)  # 5分钟
        self.max_trades_per_day = config.get("max_trades_per_day", 10)
        self.daily_trade_count = 0
        
        # 市场条件追踪
        self.last_market_data = {}

        # 市场情绪缓存
        self.market_sentiment = {}
        self.market_sentiment_timestamp = None
        
        # 财务数据缓存
        self.financial_data = {}
        self.financial_data_timestamp = None
        
        # Finnhub配置
        self.finnhub_historical_days = config.get("finnhub_historical_days", 365)
        self.finnhub_price_resolution = config.get("finnhub_price_resolution", "D")
        self.finnhub_financial_quarters = config.get("finnhub_financial_quarters", 4)
        self.finnhub_earnings_limit = config.get("finnhub_earnings_limit", 4)
        self.finnhub_cache_duration = config.get("finnhub_cache_duration", 3600)  # 默认1小时
        self.finnhub_data_cache_enabled = config.get("finnhub_data_cache_enabled", True)
        
        # API调用限制管理
        self.api_calls_this_minute = 0
        self.api_call_reset_time = datetime.now() + timedelta(minutes=1)
        self.finnhub_api_calls_per_minute = config.get("finnhub_api_calls_per_minute", 45)
        
    async def initialize(self) -> bool:
        """初始化代理"""
        try:
            # 初始化各个组件
            broker_init = await self.broker.initialize()
            if not broker_init:
                print("经纪人初始化失败")
                return False
            
            # 测试数据源连接
            if hasattr(self.data_source, 'test_connection'):
                data_source_test = await self.data_source.test_connection()
                if not data_source_test:
                    print("数据源连接测试失败")
                    return False
            
            print("交易代理初始化成功")
            return True
            
        except Exception as e:
            print(f"代理初始化失败: {e}")
            return False
    
    async def make_decision(self) -> TradingAction:
        """做出交易决策"""
        try:
            print("\n===== 开始生成交易决策 =====")
            
            # 1. 收集市场数据
            market_data = await self.get_market_data()
            if not market_data:
                print("无法获取市场数据，返回HOLD决策")
                return TradingAction(
                    action_type="hold",
                    reason="无法获取市场数据"
                )
            
            # 2. 获取投资组合状态
            portfolio = await self.get_portfolio_status()
            print(f"获取到投资组合状态: 现金 ${portfolio.get('cash', 0):.2f}, 总价值 ${portfolio.get('total_value', 0):.2f}")
            
            # 3. 获取新闻数据
            try:
                news_data = await self.get_news_data(self.trading_symbols)
                print(f"获取到 {len(news_data)} 条新闻")
            except Exception as e:
                print(f"获取新闻数据失败: {e}")
                news_data = []  # 如果新闻获取失败，继续使用空列表
            
            # 4. 分析市场情绪（检查缓存是否有效）
            if not self._is_cache_valid(self.market_sentiment_timestamp):
                self.market_sentiment = {}
                try:
                    if news_data:
                        # 如果有交易股票，针对第一个股票进行情绪分析
                        symbol = None
                        if self.trading_symbols:
                            symbol = self.trading_symbols[0] if isinstance(self.trading_symbols, list) else self.trading_symbols
                        
                        print(f"分析市场情绪{' (针对 ' + symbol + ')' if symbol else ''}...")
                        market_sentiment = await self.llm.analyze_market_sentiment(news_data, symbol)
                        self.market_sentiment = market_sentiment  # 更新缓存的市场情绪
                        self.market_sentiment_timestamp = datetime.now()  # 更新时间戳
                        
                        print(f"市场情绪: {self.market_sentiment.get('overall_sentiment', 'unknown')}")
                        print(f"信心指数: {self.market_sentiment.get('confidence', 0)}")
                        print(f"风险水平: {self.market_sentiment.get('risk_level', 'unknown')}")
                        
                        # 打印关键因素
                        key_factors = self.market_sentiment.get('key_factors', [])
                        if key_factors:
                            print("关键因素:")
                            for factor in key_factors[:3]:  # 只显示前3个因素
                                print(f"  - {factor}")
                except Exception as e:
                    print(f"市场情绪分析失败: {e}")
                    # 如果分析失败，使用中性情绪
                    self.market_sentiment = {
                        "overall_sentiment": "neutral",
                        "confidence": 0.5,
                        "risk_level": "medium",
                        "key_factors": ["分析失败"],
                        "recommendation": "谨慎观察"
                    }
                    self.market_sentiment_timestamp = datetime.now()  # 更新时间戳
            else:
                print("使用缓存的市场情绪数据")
                
            # 5. 准备历史上下文
            historical_context = self.get_recent_performance()
            
            # 6. 获取财务数据（检查缓存是否有效）
            if not self._is_cache_valid(self.financial_data_timestamp):
                self.financial_data = {}
                try:
                    if hasattr(self.data_source, 'get_company_financials') and self.trading_symbols:
                        # 只获取第一个交易股票的财务数据作为示例
                        symbol = self.trading_symbols[0] if isinstance(self.trading_symbols, list) else self.trading_symbols
                        print(f"获取 {symbol} 的财务数据...")
                        
                        # 传递配置参数
                        self.financial_data = await self.data_source.get_company_financials(
                            symbol, 
                            quarters=self.finnhub_financial_quarters,
                            earnings_limit=self.finnhub_earnings_limit
                        )
                        
                        # 获取关键财务指标
                        if hasattr(self.data_source, 'get_financial_metrics'):
                            key_metrics = await self.data_source.get_financial_metrics(symbol)
                            self.financial_data["key_metrics"] = key_metrics
                            
                        self.financial_data_timestamp = datetime.now()  # 更新时间戳
                        print(f"成功获取财务数据")
                except Exception as e:
                    print(f"获取财务数据失败: {e}")
                    self.financial_data = {}  # 如果财务数据获取失败，使用空字典
            else:
                print("使用缓存的财务数据")
            
            # 7. 生成决策
            print("正在调用LLM生成决策...")
            decision = await self.llm.generate_trading_decision(
                market_data=market_data,
                portfolio_status=portfolio,
                news_data=news_data,
                historical_context=historical_context,
                financial_data=self.financial_data,
                market_sentiment=self.market_sentiment  # 使用类属性self.market_sentiment
            )
            print(f"LLM生成的决策: {decision}")
            
            # 8. 验证决策
            if not self.validate_decision(decision):
                print("决策验证失败，返回HOLD决策")
                return TradingAction(
                    action_type="hold",
                    reason="决策验证失败"
                )
            
            # 9. 检查每日交易限制
            if self._check_daily_trade_limit(decision):
                print("已达到每日交易限制，返回HOLD决策")
                return TradingAction(
                    action_type="hold",
                    reason="已达到每日交易限制"
                )
            
            print("决策生成完成")
            print("=========================\n")
            return decision
            
        except Exception as e:
            print(f"决策生成失败: {e}")
            return TradingAction(
                action_type="hold",
                reason=f"决策生成错误: {str(e)}"
            )
    
    def _is_cache_valid(self, timestamp) -> bool:
        """检查缓存是否有效"""
        # 如果缓存禁用，总是返回False（需要刷新）
        if not self.finnhub_data_cache_enabled:
            return False
            
        # 如果没有时间戳，缓存无效
        if timestamp is None:
            return False
            
        # 检查是否超过缓存时间
        elapsed_seconds = (datetime.now() - timestamp).total_seconds()
        return elapsed_seconds < self.finnhub_cache_duration
    
    async def check_api_rate_limit(self):
        """检查API调用频率限制，必要时等待"""
        current_time = datetime.now()
        
        # 如果到了重置时间，重置计数器
        if current_time >= self.api_call_reset_time:
            self.api_calls_this_minute = 0
            self.api_call_reset_time = current_time + timedelta(minutes=1)
        
        # 如果已达到限制，等待到下一个重置时间
        if self.api_calls_this_minute >= self.finnhub_api_calls_per_minute:
            wait_time = (self.api_call_reset_time - current_time).total_seconds()
            print(f"已达到API调用频率限制，等待 {wait_time:.2f} 秒...")
            await asyncio.sleep(wait_time)
            self.api_calls_this_minute = 0
            self.api_call_reset_time = datetime.now() + timedelta(minutes=1)
        
        # 增加API调用计数
        self.api_calls_this_minute += 1
    
    async def get_historical_data(self, symbol: str, days: int = None) -> Dict[str, Any]:
        """获取历史价格数据，应用配置的限制"""
        await self.check_api_rate_limit()
        
        # 使用配置的天数或默认值
        if days is None:
            days = self.finnhub_historical_days
        
        # 计算开始和结束日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 调用数据源的方法，传递配置的分辨率
        return await self.data_source.get_historical_data(
            symbol, 
            start_date, 
            end_date, 
            interval=self.finnhub_price_resolution
        )
    
    async def get_news_data(self, symbols=None) -> List[Dict[str, Any]]:
        """获取新闻数据，应用配置的限制"""
        await self.check_api_rate_limit()
        
        # 使用配置的新闻限制
        limit = self.config.get("news_limit", 10)
        days_back = self.config.get("news_days_back", 7)
        
        # 如果提供了多个股票符号，获取所有相关新闻
        if isinstance(symbols, list) and len(symbols) > 0:
            all_news = []
            for symbol in symbols[:3]:  # 限制最多查询3个股票的新闻以避免过多API调用
                news = await self.data_source.get_news(
                    symbol=symbol,
                    limit=limit,
                    days_back=days_back
                )
                all_news.extend(news)
            return all_news[:limit]  # 返回合并后的新闻，但仍然限制总数
        else:
            # 单个股票或一般市场新闻
            symbol = symbols if not isinstance(symbols, list) else (symbols[0] if symbols else None)
            return await self.data_source.get_news(
                symbol=symbol,
                limit=limit,
                days_back=days_back
            )
    
    async def execute_decision(self, action: TradingAction) -> Dict[str, Any]:
        """执行交易决策"""
        try:
            # 1. 风险检查
            risk_check = await self.risk_check(action)
            if not risk_check.get("approved", False):
                return {
                    "success": False,
                    "message": f"风险检查失败: {risk_check.get('reason', 'Unknown')}",
                    "risk_assessment": risk_check.get("risk_assessment", {})
                }
            
            # 获取action_type的值，处理可能是枚举或字符串的情况
            action_type_value = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
            action_type_value = action_type_value.lower() if isinstance(action_type_value, str) else action_type_value
            
            # 2. 处理不同类型的行为
            if action_type_value == "get_info":
                result = await self._handle_get_info(action)
            elif action_type_value == "get_news":
                result = await self._handle_get_news(action)
            elif action_type_value in ["buy", "sell"]:
                # 添加实时价格信息
                if action.symbol and not action.price:
                    try:
                        await self.check_api_rate_limit()  # 检查API调用限制
                        price_data = await self.data_source.get_real_time_price(action.symbol)
                        action.price = price_data.get("price", 0)
                    except:
                        pass
                
                result = await self.broker.execute_action(action)
                
                # 更新交易计数
                if result.get("success", False):
                    self.daily_trade_count += 1
            else:
                result = await self.broker.execute_action(action)
            
            # 3. 记录决策和结果
            self.record_decision(action, result)
            
            # 4. 获取决策解释
            if action_type_value in ["buy", "sell"]:
                try:
                    explanation = await self.llm.explain_decision(
                        action,
                        {"market_summary": "当前市场状况"}
                    )
                    result["explanation"] = explanation
                except Exception as e:
                    result["explanation"] = f"解释生成失败: {str(e)}"
            
            print(f"执行结果: {result}")
            return result
            
        except Exception as e:
            print(f"决策执行失败: {e}")
            return {
                "success": False,
                "message": f"执行错误: {str(e)}"
            }
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """分析交易表现"""
        try:
            # 1. 获取基础绩效指标
            metrics = await self.broker.get_performance_metrics()
            
            # 2. 获取交易历史
            trade_history = await self.broker.get_trade_history(limit=100)
            
            # 3. 计算额外指标
            additional_metrics = self._calculate_additional_metrics(trade_history)
            
            # 4. 分析市场情绪
            recent_news = await self.get_news_data()
            sentiment_analysis = await self.llm.analyze_market_sentiment(recent_news)
            
            # 5. 组合分析结果
            analysis = {
                "basic_metrics": metrics,
                "additional_metrics": additional_metrics,
                "market_sentiment": sentiment_analysis,
                "trade_history_summary": {
                    "total_trades": len(trade_history),
                    "successful_trades": len([t for t in trade_history if t.get("success", False)]),
                    "recent_activity": trade_history[:10] if trade_history else []
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"性能分析失败: {e}")
            return {
                "error": str(e),
                "basic_metrics": {},
                "additional_metrics": {},
                "market_sentiment": {},
                "trade_history_summary": {}
            }
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """运行一个交易周期"""
        cycle_start = datetime.now()
        
        try:
            print(f"开始交易周期: {cycle_start}")
            
            # 1. 做出决策
            decision = await self.make_decision()
            
            # 2. 执行决策
            execution_result = await self.execute_decision(decision)
            
            # 3. 更新市场数据缓存
            self.last_market_data = await self.get_market_data()
            
            # 4. 准备周期结果
            cycle_result = {
                "cycle_start": cycle_start.isoformat(),
                "cycle_duration": (datetime.now() - cycle_start).total_seconds(),
                "decision": decision.dict(),
                "execution_result": execution_result,
                "portfolio_status": await self.get_portfolio_status(),
                "daily_trade_count": self.daily_trade_count
            }
            
            print(f"交易周期完成，耗时: {cycle_result['cycle_duration']:.2f}秒")
            return cycle_result
            
        except Exception as e:
            print(f"交易周期执行失败: {e}")
            return {
                "cycle_start": cycle_start.isoformat(),
                "error": str(e),
                "success": False
            }
    
    async def start_trading(self) -> bool:
        """开始交易"""
        if self.is_running:
            print("交易代理已在运行中")
            return True
        
        try:
            # 不再重复初始化，假设initialize()已经在外部调用过
            # if not await self.initialize():
            #     return False
            
            # 启动经纪人
            if not await self.broker.start_trading():
                print("经纪人启动失败")
                return False
            
            self.is_running = True
            self.daily_trade_count = 0
            
            print("交易代理已启动")
            return True
            
        except Exception as e:
            print(f"启动交易失败: {e}")
            return False
    
    async def stop_trading(self) -> bool:
        """停止交易"""
        if not self.is_running:
            print("交易代理未在运行")
            return True
        
        try:
            # 停止经纪人
            await self.broker.stop_trading()
            
            self.is_running = False
            
            print("交易代理已停止")
            return True
            
        except Exception as e:
            print(f"停止交易失败: {e}")
            return False
    
    async def run_continuous_trading(self, duration_hours: Optional[int] = None):
        """连续交易模式"""
        if not await self.start_trading():
            return
        
        start_time = datetime.now()
        cycle_count = 0
        
        try:
            while self.is_running:
                # 检查运行时间限制
                if duration_hours:
                    elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed_hours >= duration_hours:
                        print(f"达到运行时间限制: {duration_hours}小时")
                        break
                
                # 运行交易周期
                cycle_result = await self.run_trading_cycle()
                cycle_count += 1
                
                # 每10个周期打印一次状态
                if cycle_count % 10 == 0:
                    portfolio = await self.get_portfolio_status()
                    print(f"已完成 {cycle_count} 个交易周期")
                    print(f"当前投资组合价值: ${portfolio.get('total_value', 0):,.2f}")
                
                # 等待下一个周期
                await asyncio.sleep(self.trading_interval)
                
        except KeyboardInterrupt:
            print("接收到停止信号")
        except Exception as e:
            print(f"连续交易过程中出错: {e}")
        finally:
            await self.stop_trading()
            print(f"交易会话结束，共完成 {cycle_count} 个周期")
    
    async def _handle_get_info(self, action: TradingAction) -> Dict[str, Any]:
        """处理获取信息行为"""
        try:
            if action.symbol:
                info = await self.data_source.get_market_info(action.symbol)
                return {
                    "success": True,
                    "message": "成功获取股票信息",
                    "data": info
                }
            else:
                market_data = await self.get_market_data()
                return {
                    "success": True,
                    "message": "成功获取市场信息",
                    "data": market_data
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"获取信息失败: {str(e)}"
            }
    
    async def _handle_get_news(self, action: TradingAction) -> Dict[str, Any]:
        """处理获取新闻行为"""
        try:
            news = await self.get_news_data(
                symbol=action.symbol
            )
            return {
                "success": True,
                "message": f"成功获取 {len(news)} 条新闻",
                "data": news
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"获取新闻失败: {str(e)}"
            }
    
    def _check_daily_trade_limit(self, action: TradingAction) -> bool:
        """检查每日交易限制
        
        返回值:
            bool: 如果达到或超过每日交易限制返回True，否则返回False
        """
        # 获取action_type的值，处理可能是枚举或字符串的情况
        action_type_value = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
        action_type_value = action_type_value.lower() if isinstance(action_type_value, str) else action_type_value
        
        if action_type_value in ["buy", "sell"]:
            return self.daily_trade_count >= self.max_trades_per_day
        return False
    
    def _calculate_additional_metrics(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算额外的性能指标"""
        if not trade_history:
            return {}
        
        # 计算胜率
        successful_trades = [t for t in trade_history if t.get("success", False)]
        win_rate = len(successful_trades) / len(trade_history) * 100 if trade_history else 0
        
        # 计算平均交易间隔
        avg_trade_interval = 0
        if len(trade_history) > 1:
            intervals = []
            for i in range(1, len(trade_history)):
                # 这里需要根据实际的时间戳格式来计算
                pass
        
        return {
            "win_rate": win_rate,
            "total_trades": len(trade_history),
            "successful_trades": len(successful_trades),
            "avg_trade_interval_minutes": avg_trade_interval
        } 