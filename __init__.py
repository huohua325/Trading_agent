"""
AI交易代理框架

一个模块化的AI驱动交易代理框架，支持多种数据源、经纪人和LLM的组合。
"""

__version__ = "1.0.0"
__author__ = "Trading Agent Team"

# 导入主要组件
from .agents import TradingAgent, BaseAgent
from .brokers import BacktraderBroker, BaseBroker
from .data_sources import TiingoDataSource, BaseDataSource
from .llm import GPT4oLLM, BaseLLM
from .actions import TradingAction, ActionType
from .config import TradingConfig
from .utils import create_agent, load_config_from_env

__all__ = [
    # 代理
    'TradingAgent', 'BaseAgent',
    # 经纪人
    'BacktraderBroker', 'BaseBroker',
    # 数据源
    'TiingoDataSource', 'BaseDataSource',
    # LLM
    'GPT4oLLM', 'BaseLLM',
    # 行为
    'TradingAction', 'ActionType',
    # 配置
    'TradingConfig',
    # 工具
    'create_agent', 'load_config_from_env'
] 