from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


class BaseDataSource(ABC):
    """数据源基础抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化数据源"""
        self.config = config
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1D"
    ) -> pd.DataFrame:
        """获取历史价格数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔 (1D, 1H, etc.)
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        pass
    
    @abstractmethod
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """获取实时价格数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含实时价格信息的字典
        """
        pass
    
    @abstractmethod
    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取市场信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含市场信息的字典
        """
        pass
    
    @abstractmethod
    async def get_news(
        self, 
        symbol: Optional[str] = None,
        limit: int = 10,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """获取新闻数据
        
        Args:
            symbol: 股票代码，如果为None则获取一般市场新闻
            limit: 新闻数量限制
            days_back: 获取多少天前的新闻
            
        Returns:
            新闻列表
        """
        pass
    
    @abstractmethod
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票代码
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            匹配的股票列表
        """
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证股票代码格式"""
        if not symbol or not isinstance(symbol, str):
            return False
        return len(symbol.strip()) > 0
    
    def format_datetime(self, dt: datetime) -> str:
        """格式化日期时间"""
        return dt.strftime("%Y-%m-%d") 