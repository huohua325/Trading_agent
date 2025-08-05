import os
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class TradingConfig(BaseModel):
    """交易配置类"""
    
    # 通用配置
    initial_cash: float = 100000.0
    trading_symbols: list[str] = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    # LLM配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_model: str = "gpt-4o"
    max_tokens: int = 1000
    temperature: float = 0.1
    
    # 数据源配置
    data_source_type: Literal["tiingo", "finnhub"] = "finnhub"  # 默认使用Finnhub
    
    # Tiingo配置
    tiingo_api_key: str = os.getenv("TIINGO_API_KEY", "")
    tiingo_base_url: str = "https://api.tiingo.com/tiingo"
    
    # Finnhub配置
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    
    # 模拟盘配置
    broker_type: str = "backtrader"
    commission: float = 0.001  # 0.1% 手续费
    
    # 交易配置
    max_position_size: float = 0.2  # 单只股票最大仓位20%
    risk_tolerance: float = 0.02  # 2%的风险容忍度
    
    # 新闻配置
    news_limit: int = 10
    news_days_back: int = 7
    
    class Config:
        """Pydantic配置"""
        extra = "allow"
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()
        
    def validate_config(self) -> bool:
        """验证配置有效性"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        if self.data_source_type == "tiingo" and not self.tiingo_api_key:
            raise ValueError("Tiingo API key is required when using Tiingo data source")
            
        if self.data_source_type == "finnhub" and not self.finnhub_api_key:
            raise ValueError("Finnhub API key is required when using Finnhub data source")
            
        if self.initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
            
        return True 