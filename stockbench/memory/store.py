"""
记忆系统统一入口

整合三层记忆：
1. cache - 缓存层（兼容现有 storage/cache/）
2. working - 工作记忆（运行时上下文）
3. episodes - 情景记忆（决策历史）
"""

from typing import Optional
from pathlib import Path
from loguru import logger

from .layers.cache import CacheStore
from .layers.working import WorkingMemory
from .layers.episodic import EpisodicMemory
from .backends.file_backend import FileBackend
from .schemas import DecisionEpisode


class MemoryStore:
    """
    记忆系统统一入口
    
    使用示例:
        memory = MemoryStore(base_path="storage")
        
        # 缓存
        memory.cache.get("llm", key)
        memory.cache.set("llm", key, value)
        
        # 工作记忆
        memory.working.add("当前分析结果...", importance=0.7)
        memory.working.search("BTC 突破")
        
        # 情景记忆
        memory.episodes.add(decision_episode)
        memory.episodes.query(symbol="BTC", days=3)
        memory.episodes.search("高波动 止损")
        memory.episodes.get_for_prompt("AAPL", n=5)
    """
    
    def __init__(
        self,
        base_path: str = "storage",
        working_memory_capacity: int = 50,
        working_memory_ttl_minutes: int = 60,
        episode_max_days: int = 30,
    ):
        """
        Args:
            base_path: 存储根目录
            working_memory_capacity: 工作记忆容量
            working_memory_ttl_minutes: 工作记忆过期时间（分钟）
            episode_max_days: 情景记忆保留天数
        """
        self.base_path = Path(base_path)
        self.backend = FileBackend(self.base_path)
        
        # 配置参数
        self._config = {
            "working_memory_capacity": working_memory_capacity,
            "working_memory_ttl_minutes": working_memory_ttl_minutes,
            "episode_max_days": episode_max_days,
        }
        
        # 初始化三层记忆
        self._cache = CacheStore(
            backend=self.backend,
            cache_dir=self.base_path / "cache"
        )
        self._working = WorkingMemory(
            max_capacity=working_memory_capacity,
            ttl_minutes=working_memory_ttl_minutes
        )
        self._episodes = EpisodicMemory(
            backend=self.backend,
            data_dir=self.base_path / "memory" / "episodes",
            max_days=episode_max_days
        )
        
        logger.debug(f"[Memory] Initialized MemoryStore at {self.base_path}")
    
    @property
    def cache(self) -> CacheStore:
        """缓存层 - 兼容现有 storage/cache/"""
        return self._cache
    
    @property
    def working(self) -> WorkingMemory:
        """工作记忆 - 运行时上下文"""
        return self._working
    
    @property
    def episodes(self) -> EpisodicMemory:
        """情景记忆 - 决策历史（history 升级版）"""
        return self._episodes
    
    def clear_working(self):
        """清空工作记忆（通常在运行结束时调用）"""
        self._working.clear()
        logger.debug("[Memory] Cleared working memory")
    
    def commit_working_to_episodes(self):
        """
        将工作记忆中的重要内容提交到情景记忆
        （记忆整合 - 短期 → 长期）
        """
        important_memories = self._working.get_important(threshold=0.7)
        count = 0
        
        for mem in important_memories:
            # 只转换带有决策标签的记忆
            if mem.metadata.get("is_decision", False):
                episode = DecisionEpisode(
                    reasoning=mem.content,
                    importance=mem.importance,
                    tags=mem.metadata.get("tags", []),
                    symbol=mem.metadata.get("symbol", ""),
                    action=mem.metadata.get("action", ""),
                )
                self._episodes.add(episode)
                count += 1
        
        if count > 0:
            logger.debug(f"[Memory] Committed {count} important memories to episodes")
    
    def get_stats(self) -> dict:
        """获取记忆系统统计信息"""
        return {
            "working_memory_count": len(self._working),
            "episode_count": self._episodes.count(),
            "cache_namespaces": self._cache.list_namespaces(),
            "config": self._config,
        }
    
    def cleanup(self):
        """执行清理任务（清理过期情景记忆）"""
        self._episodes.cleanup()


__all__ = ["MemoryStore"]
