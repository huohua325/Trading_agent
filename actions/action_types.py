from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel


class ActionType(Enum):
    """交易行为类型枚举"""
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"
    GET_INFO = "get_info"
    GET_NEWS = "get_news"


class TradingAction(BaseModel):
    """交易行为数据模型"""
    action_type: ActionType
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    reason: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic配置"""
        use_enum_values = True
        
    def __str__(self) -> str:
        """字符串表示"""
        action_value = self.action_type
        if isinstance(action_value, str):
            action_value = action_value.lower()
        else:
            action_value = action_value.value
            
        if action_value in ["buy", "sell"]:
            return f"{action_value.upper()}: {self.quantity} shares of {self.symbol} at ${self.price}"
        elif action_value == "hold":
            return f"HOLD: {self.symbol if self.symbol else 'current position'}"
        elif action_value == "get_info":
            return f"GET_INFO: {self.symbol if self.symbol else 'market data'}"
        elif action_value == "get_news":
            return f"GET_NEWS: {self.symbol if self.symbol else 'market news'}"
        return f"{action_value.upper()}" 