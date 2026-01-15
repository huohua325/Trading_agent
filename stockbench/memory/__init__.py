"""
StockBench 记忆系统

提供三层记忆架构：
- CacheStore: 缓存层，兼容现有 storage/cache/
- WorkingMemory: 工作记忆，运行时上下文
- EpisodicMemory: 情景记忆，决策历史

使用示例:
    from stockbench.memory import MemoryStore, DecisionEpisode
    
    memory = MemoryStore(base_path="storage")
    
    # 存储决策
    ep = DecisionEpisode(
        symbol="AAPL",
        action="increase",
        reasoning="技术面看涨"
    )
    memory.episodes.add(ep)
    
    # 获取历史
    history = memory.episodes.get_for_prompt("AAPL", n=5)
"""

from .store import MemoryStore
from .schemas import MemoryItem, DecisionEpisode
from .layers.cache import CacheStore
from .layers.working import WorkingMemory
from .layers.episodic import EpisodicMemory

__all__ = [
    "MemoryStore",
    "MemoryItem",
    "DecisionEpisode",
    "CacheStore",
    "WorkingMemory",
    "EpisodicMemory",
]
