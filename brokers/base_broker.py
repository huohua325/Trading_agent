from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..actions.action_types import TradingAction


class BaseBroker(ABC):
    """经纪人基础抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化经纪人"""
        self.config = config
        self.initial_cash = config.get("initial_cash", 100000.0)
        self.commission = config.get("commission", 0.001)
        self.is_running = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化经纪人"""
        pass
    
    @abstractmethod
    async def execute_action(self, action: TradingAction) -> Dict[str, Any]:
        """执行交易行为
        
        Args:
            action: 交易行为对象
            
        Returns:
            执行结果字典
        """
        pass
    
    @abstractmethod
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态
        
        Returns:
            投资组合状态字典
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Any]:
        """获取当前持仓
        
        Returns:
            持仓字典
        """
        pass
    
    @abstractmethod
    async def get_cash_balance(self) -> float:
        """获取现金余额
        
        Returns:
            现金余额
        """
        pass
    
    @abstractmethod
    async def get_total_value(self) -> float:
        """获取投资组合总价值
        
        Returns:
            总价值
        """
        pass
    
    @abstractmethod
    async def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取交易历史
        
        Args:
            limit: 记录数限制
            
        Returns:
            交易历史列表
        """
        pass
    
    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取绩效指标
        
        Returns:
            绩效指标字典
        """
        pass
    
    @abstractmethod
    async def start_trading(self) -> bool:
        """开始交易"""
        pass
    
    @abstractmethod
    async def stop_trading(self) -> bool:
        """停止交易"""
        pass
    
    @abstractmethod
    async def reset(self) -> bool:
        """重置经纪人状态"""
        pass
    
    def validate_action(self, action: TradingAction) -> bool:
        """验证交易行为"""
        if not action:
            return False
        
        # 基本验证
        if action.action_type in ["BUY", "SELL"]:
            if not action.symbol or not action.quantity:
                return False
            if action.quantity <= 0:
                return False
        
        return True
    
    def calculate_commission(self, action: TradingAction) -> float:
        """计算佣金"""
        if action.action_type in ["BUY", "SELL"] and action.quantity and action.price:
            trade_value = action.quantity * action.price
            return trade_value * self.commission
        return 0.0
    
    def format_portfolio_summary(self, portfolio: Dict[str, Any]) -> str:
        """格式化投资组合摘要"""
        cash = portfolio.get('cash', 0)
        total_value = portfolio.get('total_value', 0)
        positions = portfolio.get('positions', {})
        
        summary = f"投资组合摘要:\n"
        summary += f"现金: ${cash:,.2f}\n"
        summary += f"总价值: ${total_value:,.2f}\n"
        summary += f"收益: ${total_value - self.initial_cash:,.2f}\n"
        summary += f"收益率: {((total_value - self.initial_cash) / self.initial_cash) * 100:.2f}%\n"
        
        if positions:
            summary += f"持仓数量: {len(positions)}\n"
            for symbol, position in positions.items():
                summary += f"  {symbol}: {position.get('quantity', 0)} 股\n"
        
        return summary 