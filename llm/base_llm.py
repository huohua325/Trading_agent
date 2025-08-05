from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..actions.action_types import TradingAction


class BaseLLM(ABC):
    """LLM基础抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化LLM"""
        self.config = config
    
    @abstractmethod
    async def generate_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_status: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        historical_context: Optional[Dict[str, Any]] = None
    ) -> TradingAction:
        """生成交易决策
        
        Args:
            market_data: 市场数据
            portfolio_status: 投资组合状态
            news_data: 新闻数据
            historical_context: 历史上下文
            
        Returns:
            交易行为对象
        """
        pass
    
    @abstractmethod
    async def analyze_market_sentiment(
        self,
        news_data: List[Dict[str, Any]],
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """分析市场情绪
        
        Args:
            news_data: 新闻数据
            symbol: 股票代码（可选）
            
        Returns:
            情绪分析结果
        """
        pass
    
    @abstractmethod
    async def explain_decision(
        self,
        action: TradingAction,
        context: Dict[str, Any]
    ) -> str:
        """解释交易决策
        
        Args:
            action: 交易行为
            context: 上下文信息
            
        Returns:
            决策解释文本
        """
        pass
    
    @abstractmethod
    async def risk_assessment(
        self,
        portfolio_status: Dict[str, Any],
        proposed_action: TradingAction
    ) -> Dict[str, Any]:
        """风险评估
        
        Args:
            portfolio_status: 投资组合状态
            proposed_action: 拟议的交易行为
            
        Returns:
            风险评估结果
        """
        pass
    
    def format_market_data_prompt(self, market_data: Dict[str, Any]) -> str:
        """格式化市场数据为提示文本"""
        prompt = "当前市场数据:\n"
        for symbol, data in market_data.items():
            prompt += f"- {symbol}: 价格 ${data.get('price', 'N/A')}, "
            prompt += f"变化 {data.get('change_percent', 0):.2f}%, "
            prompt += f"成交量 {data.get('volume', 'N/A')}\n"
        return prompt
    
    def format_portfolio_prompt(self, portfolio: Dict[str, Any]) -> str:
        """格式化投资组合为提示文本"""
        prompt = "当前投资组合:\n"
        prompt += f"- 现金: ${portfolio.get('cash', 0):,.2f}\n"
        prompt += f"- 总价值: ${portfolio.get('total_value', 0):,.2f}\n"
        
        positions = portfolio.get('positions', {})
        if positions:
            prompt += "持仓:\n"
            for symbol, position in positions.items():
                prompt += f"  - {symbol}: {position.get('quantity', 0)} 股, "
                prompt += f"价值 ${position.get('value', 0):,.2f}\n"
        else:
            prompt += "- 无持仓\n"
        
        return prompt
    
    def format_news_prompt(self, news_data: List[Dict[str, Any]]) -> str:
        """格式化新闻数据为提示文本"""
        if not news_data:
            return "无相关新闻\n"
        
        prompt = "相关新闻:\n"
        for i, news in enumerate(news_data[:5], 1):  # 只显示前5条新闻
            prompt += f"{i}. {news.get('title', 'No title')}\n"
            if news.get('description'):
                prompt += f"   摘要: {news['description'][:100]}...\n"
            prompt += f"   来源: {news.get('source', 'Unknown')}\n"
            prompt += f"   日期: {news.get('published_date', 'Unknown')}\n\n"
        
        return prompt 