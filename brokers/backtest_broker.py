import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from .base_broker import BaseBroker
from ..actions.action_types import TradingAction, ActionType


class BacktestBroker(BaseBroker):
    """回测专用Broker，用于模拟交易执行"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化回测Broker
        
        Args:
            config: 配置字典，包含初始资金等设置
        """
        super().__init__(config)
        
        # 回测特定配置
        self.initial_balance = config.get("initial_balance", 100000.0)
        self.commission_rate = config.get("commission_rate", 0.001)  # 默认0.1%佣金率
        self.slippage = config.get("slippage", 0.0)  # 默认无滑点
        self.current_date = None
        
        # 投资组合状态
        self.cash = self.initial_balance
        self.portfolio = {}  # {symbol: {'quantity': 100, 'avg_price': 150.0}}
        
        # 交易历史
        self.trade_history = []
        
        # 回测性能指标
        self.performance_metrics = {
            "total_return": 0.0,
            "total_return_percent": 0.0,
            "number_of_trades": 0,
            "successful_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "daily_returns": [],
            "cagr": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0
        }
        
        # 每日投资组合价值历史
        self.portfolio_value_history = {}  # {date: total_value}
        
        # 初始化状态
        self.is_running = False
    
    def set_current_date(self, date: Union[str, datetime]):
        """设置当前回测日期"""
        if isinstance(date, str):
            self.current_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            self.current_date = date
    
    async def initialize(self) -> bool:
        """初始化Broker"""
        try:
            # 回测Broker不需要实际连接，始终返回成功
            return True
        except Exception as e:
            print(f"回测Broker初始化失败: {e}")
            return False
    
    async def start_trading(self) -> bool:
        """启动交易"""
        self.is_running = True
        return True
    
    async def stop_trading(self) -> bool:
        """停止交易"""
        self.is_running = False
        return True
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        return {
            "cash": self.cash,
            "portfolio_value": self.get_portfolio_value(),
            "total_value": self.cash + self.get_portfolio_value(),
            "initial_balance": self.initial_balance,
            "return_percent": (self.cash + self.get_portfolio_value() - self.initial_balance) / self.initial_balance * 100 if self.initial_balance > 0 else 0
        }
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        holdings = []
        total_value = self.cash
        
        for symbol, position in self.portfolio.items():
            # 获取当前价格
            current_price = await self._get_current_price(symbol)
            
            # 计算持仓价值和收益
            quantity = position["quantity"]
            avg_price = position["avg_price"]
            position_value = quantity * current_price
            total_value += position_value
            
            # 计算收益
            unrealized_pl = position_value - (quantity * avg_price)
            unrealized_pl_percent = (unrealized_pl / (quantity * avg_price)) * 100 if quantity * avg_price > 0 else 0
            
            holdings.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "position_value": position_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_percent": unrealized_pl_percent
            })
        
        # 更新投资组合价值历史
        if self.current_date:
            date_str = self.current_date.strftime("%Y-%m-%d")
            self.portfolio_value_history[date_str] = total_value
        
        return {
            "cash": self.cash,
            "holdings": holdings,
            "total_value": total_value,
            "return_percent": (total_value - self.initial_balance) / self.initial_balance * 100 if self.initial_balance > 0 else 0
        }
    
    async def execute_action(self, action: TradingAction) -> Dict[str, Any]:
        """执行交易行为"""
        if not self.is_running:
            return {"success": False, "message": "Broker未启动"}
        
        if not self.current_date:
            return {"success": False, "message": "未设置当前回测日期"}
        
        # 获取action_type的值，处理可能是枚举或字符串的情况
        action_type_value = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
        action_type_value = action_type_value.lower() if isinstance(action_type_value, str) else action_type_value
        
        # 根据行为类型执行不同操作
        if action_type_value == "buy":
            return await self._execute_buy(action)
        elif action_type_value == "sell":
            return await self._execute_sell(action)
        elif action_type_value == "hold":
            return {"success": True, "message": "保持当前仓位", "action": "hold"}
        else:
            return {"success": False, "message": f"不支持的行为类型: {action_type_value}"}
    
    async def _execute_buy(self, action: TradingAction) -> Dict[str, Any]:
        """执行买入操作"""
        try:
            symbol = action.symbol
            quantity = action.quantity
            
            # 如果没有指定价格，获取当前价格
            price = action.price
            if not price:
                price = await self._get_current_price(symbol)
            
            # 考虑滑点
            execution_price = price * (1 + self.slippage)
            
            # 计算交易成本
            trade_cost = quantity * execution_price
            commission = trade_cost * self.commission_rate
            total_cost = trade_cost + commission
            
            # 检查资金是否足够
            if total_cost > self.cash:
                return {
                    "success": False,
                    "message": f"资金不足: 需要 ${total_cost:.2f}, 可用 ${self.cash:.2f}",
                    "action": "buy",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": execution_price
                }
            
            # 执行交易
            self.cash -= total_cost
            
            # 更新投资组合
            if symbol in self.portfolio:
                # 更新现有持仓的平均价格
                current_position = self.portfolio[symbol]
                current_quantity = current_position["quantity"]
                current_avg_price = current_position["avg_price"]
                
                new_quantity = current_quantity + quantity
                new_avg_price = (current_quantity * current_avg_price + quantity * execution_price) / new_quantity
                
                self.portfolio[symbol] = {
                    "quantity": new_quantity,
                    "avg_price": new_avg_price
                }
            else:
                # 添加新持仓
                self.portfolio[symbol] = {
                    "quantity": quantity,
                    "avg_price": execution_price
                }
            
            # 记录交易
            trade_record = {
                "timestamp": self.current_date.timestamp(),
                "date": self.current_date.strftime("%Y-%m-%d"),
                "action": "buy",
                "symbol": symbol,
                "quantity": quantity,
                "price": execution_price,
                "commission": commission,
                "total_cost": total_cost,
                "cash_after": self.cash
            }
            
            self.trade_history.append(trade_record)
            self.performance_metrics["number_of_trades"] += 1
            
            return {
                "success": True,
                "message": f"买入 {quantity} 股 {symbol} @ ${execution_price:.2f}",
                "action": "buy",
                "symbol": symbol,
                "quantity": quantity,
                "price": execution_price,
                "commission": commission,
                "total_cost": total_cost,
                "cash_after": self.cash
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"买入执行错误: {str(e)}",
                "action": "buy",
                "symbol": action.symbol
            }
    
    async def _execute_sell(self, action: TradingAction) -> Dict[str, Any]:
        """执行卖出操作"""
        try:
            symbol = action.symbol
            quantity = action.quantity
            
            # 检查是否持有该股票
            if symbol not in self.portfolio:
                return {
                    "success": False,
                    "message": f"未持有股票: {symbol}",
                    "action": "sell",
                    "symbol": symbol
                }
            
            # 检查持仓数量是否足够
            current_position = self.portfolio[symbol]
            current_quantity = current_position["quantity"]
            
            if quantity > current_quantity:
                return {
                    "success": False,
                    "message": f"持仓不足: 需要 {quantity} 股, 持有 {current_quantity} 股",
                    "action": "sell",
                    "symbol": symbol,
                    "quantity": quantity
                }
            
            # 如果没有指定价格，获取当前价格
            price = action.price
            if not price:
                price = await self._get_current_price(symbol)
            
            # 考虑滑点
            execution_price = price * (1 - self.slippage)
            
            # 计算交易收益
            trade_value = quantity * execution_price
            commission = trade_value * self.commission_rate
            net_value = trade_value - commission
            
            # 更新现金
            self.cash += net_value
            
            # 更新投资组合
            avg_price = current_position["avg_price"]
            new_quantity = current_quantity - quantity
            
            if new_quantity > 0:
                # 更新持仓
                self.portfolio[symbol]["quantity"] = new_quantity
            else:
                # 清空持仓
                del self.portfolio[symbol]
            
            # 计算盈亏
            profit_loss = (execution_price - avg_price) * quantity - commission
            profit_loss_percent = (profit_loss / (avg_price * quantity)) * 100 if avg_price * quantity > 0 else 0
            
            # 记录交易
            trade_record = {
                "timestamp": self.current_date.timestamp(),
                "date": self.current_date.strftime("%Y-%m-%d"),
                "action": "sell",
                "symbol": symbol,
                "quantity": quantity,
                "price": execution_price,
                "commission": commission,
                "net_value": net_value,
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_percent,
                "cash_after": self.cash
            }
            
            self.trade_history.append(trade_record)
            self.performance_metrics["number_of_trades"] += 1
            
            # 更新成功交易计数
            if profit_loss > 0:
                self.performance_metrics["successful_trades"] += 1
            
            return {
                "success": True,
                "message": f"卖出 {quantity} 股 {symbol} @ ${execution_price:.2f}, 盈亏: ${profit_loss:.2f} ({profit_loss_percent:.2f}%)",
                "action": "sell",
                "symbol": symbol,
                "quantity": quantity,
                "price": execution_price,
                "commission": commission,
                "net_value": net_value,
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_percent,
                "cash_after": self.cash
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"卖出执行错误: {str(e)}",
                "action": "sell",
                "symbol": action.symbol
            }
    
    async def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取交易历史"""
        if limit:
            return self.trade_history[-limit:]
        return self.trade_history
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        # 更新性能指标
        await self._update_performance_metrics()
        return self.performance_metrics
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        # 计算总收益
        current_value = self.cash + self.get_portfolio_value()
        total_return = current_value - self.initial_balance
        total_return_percent = (total_return / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # 计算胜率
        if self.performance_metrics["number_of_trades"] > 0:
            win_rate = (self.performance_metrics["successful_trades"] / self.performance_metrics["number_of_trades"]) * 100
        else:
            win_rate = 0
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # 计算夏普比率和索提诺比率
        daily_returns = self._calculate_daily_returns()
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        # 计算CAGR
        cagr = 0.0
        if self.portfolio_value_history:
            dates_sorted = sorted(self.portfolio_value_history.keys())
            start_dt = datetime.strptime(dates_sorted[0], "%Y-%m-%d")
            end_dt = datetime.strptime(dates_sorted[-1], "%Y-%m-%d")
            years = (end_dt - start_dt).days / 365.0
            if years > 0:
                cagr = (current_value / self.initial_balance) ** (1/years) - 1

        # 交易级指标: profit_factor & expectancy
        total_profit = sum(t["profit_loss"] for t in self.trade_history if t.get("profit_loss", 0) > 0)
        total_loss = sum(abs(t["profit_loss"]) for t in self.trade_history if t.get("profit_loss", 0) < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

        avg_win = (total_profit / len([t for t in self.trade_history if t.get("profit_loss", 0) > 0])) if total_profit > 0 else 0.0
        avg_loss = (total_loss / len([t for t in self.trade_history if t.get("profit_loss", 0) < 0])) if total_loss > 0 else 0.0
        win_rate_ratio = win_rate / 100  # convert to 0-1
        loss_rate_ratio = 1 - win_rate_ratio
        expectancy = (avg_win * win_rate_ratio) - (avg_loss * loss_rate_ratio)
        
        # 更新指标
        self.performance_metrics.update({
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "daily_returns": daily_returns,
            "cagr": cagr,
            "profit_factor": profit_factor,
            "expectancy": expectancy
        })
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.portfolio_value_history:
            return 0.0
        
        # 将历史价值转换为列表并排序
        dates = sorted(self.portfolio_value_history.keys())
        values = [self.portfolio_value_history[date] for date in dates]
        
        # 计算最大回撤
        max_drawdown = 0
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_daily_returns(self) -> List[float]:
        """计算每日收益率"""
        if not self.portfolio_value_history:
            return []
        
        # 将历史价值转换为列表并排序
        dates = sorted(self.portfolio_value_history.keys())
        values = [self.portfolio_value_history[date] for date in dates]
        
        # 计算每日收益率
        daily_returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                daily_return = (values[i] - values[i-1]) / values[i-1]
                daily_returns.append(daily_return)
        
        return daily_returns
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02/252) -> float:
        """计算夏普比率"""
        if not daily_returns:
            return 0.0
        
        # 计算年化收益率和标准差
        avg_return = np.mean(daily_returns)
        std_dev = np.std(daily_returns)
        
        if std_dev == 0:
            return 0.0
        
        # 计算夏普比率（年化）
        sharpe_ratio = (avg_return - risk_free_rate) / std_dev * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_sortino_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02/252) -> float:
        """计算索提诺比率"""
        if not daily_returns:
            return 0.0
        
        # 计算年化收益率和下行标准差
        avg_return = np.mean(daily_returns)
        
        # 只考虑负收益率
        negative_returns = [r for r in daily_returns if r < 0]
        
        if not negative_returns:
            return 0.0
        
        downside_std_dev = np.std(negative_returns)
        
        if downside_std_dev == 0:
            return 0.0
        
        # 计算索提诺比率（年化）
        sortino_ratio = (avg_return - risk_free_rate) / downside_std_dev * np.sqrt(252)
        
        return sortino_ratio
    
    def get_portfolio_value(self) -> float:
        """获取投资组合价值"""
        total_value = 0.0
        
        for symbol, position in self.portfolio.items():
            try:
                # 使用同步方式获取价格
                price = self._get_current_price_sync(symbol)
                position_value = position["quantity"] * price
                total_value += position_value
            except Exception:
                # 如果无法获取价格，使用平均价格
                position_value = position["quantity"] * position["avg_price"]
                total_value += position_value
        
        return total_value
    
    async def _get_current_price(self, symbol: str) -> float:
        """获取当前价格（异步）"""
        # 从数据源获取价格
        if hasattr(self, 'price_data_source') and self.price_data_source:
            try:
                # 调用数据源的方法获取价格
                if hasattr(self.price_data_source, '_get_current_price'):
                    return await self.price_data_source._get_current_price(symbol)
                elif hasattr(self.price_data_source, 'get_real_time_price'):
                    price_data = await self.price_data_source.get_real_time_price(symbol)
                    return price_data.get('price', 0)
            except Exception as e:
                print(f"无法从数据源获取价格: {e}")
        
        # 如果无法从数据源获取，返回默认值
        print(f"警告: 使用默认价格 100.0 用于 {symbol}")
        return 100.0
    
    def _get_current_price_sync(self, symbol: str) -> float:
        """获取当前价格（同步）"""
        # 在同步环境中，我们只能返回默认值或缓存的价格
        # 实际应用中，可能需要更复杂的解决方案
        if symbol in self.portfolio:
            return self.portfolio[symbol].get("last_price", 100.0)
        return 100.0
    
    def set_price_data_source(self, price_data_source):
        """设置价格数据源"""
        self.price_data_source = price_data_source 

    async def get_cash_balance(self) -> float:
        """获取当前现金余额"""
        return self.cash
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取当前持仓"""
        positions = {}
        for symbol, position in self.portfolio.items():
            positions[symbol] = {
                "quantity": position["quantity"],
                "avg_price": position["avg_price"]
            }
        return positions
    
    async def get_total_value(self) -> float:
        """获取投资组合总价值（现金 + 持仓市值）"""
        return self.cash + self.get_portfolio_value()
    
    async def reset(self) -> bool:
        """重置Broker状态"""
        try:
            # 重置投资组合状态
            self.cash = self.initial_balance
            self.portfolio = {}
            
            # 重置交易历史
            self.trade_history = []
            
            # 重置性能指标
            self.performance_metrics = {
                "total_return": 0.0,
                "total_return_percent": 0.0,
                "number_of_trades": 0,
                "successful_trades": 0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "daily_returns": [],
                "cagr": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0
            }
            
            # 重置投资组合价值历史
            self.portfolio_value_history = {}
            
            return True
        except Exception as e:
            print(f"重置Broker失败: {e}")
            return False 