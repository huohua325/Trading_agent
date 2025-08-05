import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .base_broker import BaseBroker
from ..actions.action_types import TradingAction, ActionType


class BacktraderBroker(BaseBroker):
    """Backtrader经纪人实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Backtrader相关设置
        self.cerebro = None
        self.strategy = None
        self.data_feeds = {}
        self.trade_history = []
        self.current_positions = {}
        self.current_cash = self.initial_cash
        self.total_value = self.initial_cash
        
        # 性能追踪
        self.start_time = None
        self.end_time = None
        self.returns = []
        
    async def initialize(self) -> bool:
        """初始化Backtrader"""
        try:
            self.cerebro = bt.Cerebro()
            
            # 设置初始资金
            self.cerebro.broker.setcash(self.initial_cash)
            
            # 设置佣金
            self.cerebro.broker.setcommission(commission=self.commission)
            
            # 添加策略（我们将使用自定义策略来接收外部信号）
            self.cerebro.addstrategy(TradingAgentStrategy, broker_ref=self)
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Backtrader初始化失败: {e}")
            return False
    
    async def execute_action(self, action: TradingAction) -> Dict[str, Any]:
        """执行交易行为"""
        if not self.validate_action(action):
            return {
                "success": False,
                "message": "无效的交易行为",
                "action": str(action)
            }
        
        try:
            result = {"success": True, "action": str(action)}
            
            # 获取action_type的值，处理可能是枚举或字符串的情况
            action_type_value = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
            action_type_value = action_type_value.lower() if isinstance(action_type_value, str) else action_type_value
            
            if action_type_value == "buy":
                result = await self._execute_buy(action)
            elif action_type_value == "sell":
                result = await self._execute_sell(action)
            elif action_type_value == "hold":
                result = {
                    "success": True,
                    "message": "持有当前仓位",
                    "action": str(action)
                }
            else:
                result = {
                    "success": True,
                    "message": f"非交易操作: {action_type_value}",
                    "action": str(action)
                }
            
            # 记录交易历史
            self._record_trade(action, result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "message": f"执行交易失败: {str(e)}",
                "action": str(action)
            }
    
    async def _execute_buy(self, action: TradingAction) -> Dict[str, Any]:
        """执行买入操作"""
        symbol = action.symbol
        quantity = action.quantity
        price = action.price or 0  # 如果没有指定价格，使用市价
        
        # 计算所需资金
        estimated_cost = quantity * price if price > 0 else quantity * 100  # 如果没有价格，估算100美元每股
        commission = self.calculate_commission(action)
        total_cost = estimated_cost + commission
        
        # 检查资金是否充足
        if total_cost > self.current_cash:
            return {
                "success": False,
                "message": f"资金不足，需要 ${total_cost:,.2f}，可用 ${self.current_cash:,.2f}",
                "action": str(action)
            }
        
        # 执行买入
        self.current_cash -= total_cost
        
        # 更新持仓
        if symbol in self.current_positions:
            old_quantity = self.current_positions[symbol]["quantity"]
            old_avg_price = self.current_positions[symbol]["avg_price"]
            new_quantity = old_quantity + quantity
            new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
            
            self.current_positions[symbol] = {
                "quantity": new_quantity,
                "avg_price": new_avg_price,
                "current_price": price,
                "value": new_quantity * price
            }
        else:
            self.current_positions[symbol] = {
                "quantity": quantity,
                "avg_price": price,
                "current_price": price,
                "value": quantity * price
            }
        
        # 更新总价值
        await self._update_total_value()
        
        return {
            "success": True,
            "message": f"成功买入 {quantity} 股 {symbol}，价格 ${price:.2f}",
            "action": str(action),
            "cost": total_cost,
            "remaining_cash": self.current_cash
        }
    
    async def _execute_sell(self, action: TradingAction) -> Dict[str, Any]:
        """执行卖出操作"""
        symbol = action.symbol
        quantity = action.quantity
        price = action.price or 0
        
        # 检查是否有足够的持仓
        if symbol not in self.current_positions:
            return {
                "success": False,
                "message": f"没有 {symbol} 的持仓",
                "action": str(action)
            }
        
        current_quantity = self.current_positions[symbol]["quantity"]
        if quantity > current_quantity:
            return {
                "success": False,
                "message": f"持仓不足，拥有 {current_quantity} 股，尝试卖出 {quantity} 股",
                "action": str(action)
            }
        
        # 执行卖出
        proceeds = quantity * price
        commission = self.calculate_commission(action)
        net_proceeds = proceeds - commission
        
        self.current_cash += net_proceeds
        
        # 更新持仓
        remaining_quantity = current_quantity - quantity
        if remaining_quantity <= 0:
            # 清空持仓
            del self.current_positions[symbol]
        else:
            # 更新剩余持仓
            self.current_positions[symbol]["quantity"] = remaining_quantity
            self.current_positions[symbol]["value"] = remaining_quantity * price
            self.current_positions[symbol]["current_price"] = price
        
        # 更新总价值
        await self._update_total_value()
        
        return {
            "success": True,
            "message": f"成功卖出 {quantity} 股 {symbol}，价格 ${price:.2f}",
            "action": str(action),
            "proceeds": net_proceeds,
            "current_cash": self.current_cash
        }
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        await self._update_total_value()
        
        return {
            "cash": self.current_cash,
            "positions": self.current_positions.copy(),
            "total_value": self.total_value,
            "initial_cash": self.initial_cash,
            "unrealized_pnl": self.total_value - self.initial_cash,
            "unrealized_pnl_percent": ((self.total_value - self.initial_cash) / self.initial_cash) * 100,
            "position_count": len(self.current_positions)
        }
    
    async def get_positions(self) -> Dict[str, Any]:
        """获取当前持仓"""
        return self.current_positions.copy()
    
    async def get_cash_balance(self) -> float:
        """获取现金余额"""
        return self.current_cash
    
    async def get_total_value(self) -> float:
        """获取投资组合总价值"""
        await self._update_total_value()
        return self.total_value
    
    async def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取交易历史"""
        history = self.trade_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取绩效指标"""
        await self._update_total_value()
        
        total_return = self.total_value - self.initial_cash
        total_return_percent = (total_return / self.initial_cash) * 100
        
        # 计算其他指标
        metrics = {
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "current_value": self.total_value,
            "initial_value": self.initial_cash,
            "cash_balance": self.current_cash,
            "positions_value": sum(pos["value"] for pos in self.current_positions.values()),
            "number_of_positions": len(self.current_positions),
            "number_of_trades": len(self.trade_history)
        }
        
        return metrics
    
    async def start_trading(self) -> bool:
        """开始交易"""
        if not self.cerebro:
            await self.initialize()
        
        self.is_running = True
        self.start_time = datetime.now()
        return True
    
    async def stop_trading(self) -> bool:
        """停止交易"""
        self.is_running = False
        self.end_time = datetime.now()
        return True
    
    async def reset(self) -> bool:
        """重置经纪人状态"""
        self.current_cash = self.initial_cash
        self.total_value = self.initial_cash
        self.current_positions = {}
        self.trade_history = []
        self.returns = []
        self.start_time = None
        self.end_time = None
        
        return await self.initialize()
    
    async def _update_total_value(self):
        """更新投资组合总价值"""
        positions_value = sum(pos["value"] for pos in self.current_positions.values())
        self.total_value = self.current_cash + positions_value
    
    def _record_trade(self, action: TradingAction, result: Dict[str, Any]):
        """记录交易历史"""
        # 获取action_type的值，处理可能是枚举或字符串的情况
        action_value = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
        action_value = action_value.lower() if isinstance(action_value, str) else action_value
            
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action_value,
            "symbol": action.symbol,
            "quantity": action.quantity,
            "price": action.price,
            "reason": action.reason,
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "cash_after": self.current_cash,
            "total_value_after": self.total_value
        }
        
        self.trade_history.append(trade_record)
    
    def add_data_feed(self, symbol: str, data: pd.DataFrame):
        """添加数据源"""
        # 将pandas DataFrame转换为Backtrader数据源
        bt_data = bt.feeds.PandasData(dataname=data)
        self.data_feeds[symbol] = bt_data
        
        if self.cerebro:
            self.cerebro.adddata(bt_data, name=symbol)


class TradingAgentStrategy(bt.Strategy):
    """交易代理策略类"""
    
    params = (
        ('broker_ref', None),
    )
    
    def __init__(self):
        self.broker_ref = self.params.broker_ref
    
    def next(self):
        """策略的主要逻辑（由外部代理控制）"""
        # 这里可以添加一些基础的逻辑
        # 主要的交易决策由外部的AI agent来做
        pass
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            else:
                print(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')
    
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
        
        print(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}') 