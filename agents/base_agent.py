from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..actions.action_types import TradingAction
from ..brokers.base_broker import BaseBroker
from ..data_sources.base_data_source import BaseDataSource
from ..llm.base_llm import BaseLLM


class BaseAgent(ABC):
    """交易代理基础抽象类"""
    
    def __init__(
        self,
        broker: BaseBroker,
        data_source: BaseDataSource,
        llm: BaseLLM,
        config: Dict[str, Any]
    ):
        """初始化代理"""
        self.broker = broker
        self.data_source = data_source
        self.llm = llm
        self.config = config
        
        # 代理状态
        self.is_running = False
        self.trading_symbols = config.get("trading_symbols", [])
        self.max_position_size = config.get("max_position_size", 0.2)
        self.risk_tolerance = config.get("risk_tolerance", 0.02)
        
        # 历史记录
        self.decision_history = []
        self.performance_history = []
        
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化代理"""
        pass
    
    @abstractmethod
    async def make_decision(self) -> TradingAction:
        """做出交易决策"""
        pass
    
    @abstractmethod
    async def execute_decision(self, action: TradingAction) -> Dict[str, Any]:
        """执行交易决策"""
        pass
    
    @abstractmethod
    async def analyze_performance(self) -> Dict[str, Any]:
        """分析交易表现"""
        pass
    
    @abstractmethod
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """运行一个交易周期"""
        pass
    
    @abstractmethod
    async def start_trading(self) -> bool:
        """开始交易"""
        pass
    
    @abstractmethod
    async def stop_trading(self) -> bool:
        """停止交易"""
        pass
    
    async def get_market_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """获取市场数据"""
        if not symbols:
            symbols = self.trading_symbols
        
        market_data = {}
        for symbol in symbols:
            try:
                price_data = await self.data_source.get_real_time_price(symbol)
                market_data[symbol] = price_data
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
        
        return market_data
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        return await self.broker.get_portfolio_status()
    
    async def get_news_data(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取新闻数据"""
        all_news = []
        
        if symbols:
            for symbol in symbols:
                try:
                    news = await self.data_source.get_news(
                        symbol=symbol,
                        limit=self.config.get("news_limit", 5),
                        days_back=self.config.get("news_days_back", 3)
                    )
                    all_news.extend(news)
                except Exception as e:
                    print(f"获取 {symbol} 新闻失败: {e}")
        else:
            # 获取一般市场新闻
            try:
                news = await self.data_source.get_news(
                    limit=self.config.get("news_limit", 10),
                    days_back=self.config.get("news_days_back", 7)
                )
                all_news.extend(news)
            except Exception as e:
                print(f"获取市场新闻失败: {e}")
        
        return all_news
    
    def validate_decision(self, action: TradingAction) -> bool:
        """验证交易决策"""
        # 基本验证
        if not action:
            print("验证失败: action为空")
            return False
        
        # 获取action_type的值，处理可能是枚举或字符串的情况
        action_type_value = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
        action_type_value = action_type_value.lower() if isinstance(action_type_value, str) else action_type_value
        
        print(f"\n===== 决策验证 =====")
        print(f"action_type: {action_type_value}")
        print(f"symbol: {action.symbol}")
        print(f"quantity: {action.quantity}")
        print(f"trading_symbols: {self.trading_symbols}")
        
        # 检查交易行为类型
        if action_type_value in ["buy", "sell"]:
            if not action.symbol:
                print(f"验证失败: {action_type_value}操作缺少symbol")
                return False
            if action.symbol not in self.trading_symbols:
                print(f"验证失败: symbol '{action.symbol}' 不在允许的交易符号列表中")
                return False
            if not action.quantity:
                print(f"验证失败: {action_type_value}操作缺少quantity")
                return False
            if action.quantity <= 0:
                print(f"验证失败: quantity必须大于0，当前值: {action.quantity}")
                return False
        
        print("决策验证通过")
        print("===================\n")
        return True
    
    async def risk_check(self, action: TradingAction) -> Dict[str, Any]:
        """风险检查"""
        portfolio = await self.get_portfolio_status()
        risk_assessment = await self.llm.risk_assessment(portfolio, action)
        
        # 根据风险评估决定是否继续
        risk_level = risk_assessment.get("risk_level", "medium")
        risk_score = risk_assessment.get("risk_score", 0.5)
        
        # 如果风险太高，建议暂停
        if risk_level == "high" or risk_score > 0.8:
            return {
                "approved": False,
                "reason": "风险过高",
                "risk_assessment": risk_assessment
            }
        
        return {
            "approved": True,
            "risk_assessment": risk_assessment
        }
    
    def record_decision(self, action: TradingAction, result: Dict[str, Any]):
        """记录决策历史"""
        decision_record = {
            "timestamp": self._get_current_time(),
            "action": action.dict(),
            "result": result,
            "market_conditions": {},  # 可以添加市场条件记录
        }
        
        self.decision_history.append(decision_record)
        
        # 限制历史记录长度
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """获取最近表现"""
        # 这里可以实现获取最近几天的表现数据
        
        # 计算交易次数时，考虑action_type可能是字符串或枚举
        trades_count = 0
        for decision in self.decision_history:
            action_type = decision["action"].get("action_type", "")
            if isinstance(action_type, str) and action_type.lower() in ["buy", "sell"]:
                trades_count += 1
                
        return {
            "period": f"最近{days}天",
            "trades_count": trades_count,
            "success_rate": 0.0,  # 需要根据实际情况计算
            "average_return": 0.0  # 需要根据实际情况计算
        } 