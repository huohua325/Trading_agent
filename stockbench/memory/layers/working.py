"""
工作记忆 - 运行时上下文管理

特点:
- 容量有限（默认50条）
- TTL 自动清理
- 纯内存存储
- 简单关键词搜索
- 重要性排序
"""

from typing import List, Optional
from datetime import datetime, timedelta

from ..schemas import MemoryItem


class WorkingMemory:
    """
    工作记忆 - 运行时上下文
    
    用于存储当前运行周期内的临时信息，如：
    - 分析结论
    - 中间计算结果
    - 上下文状态
    
    生命周期：单次运行，运行结束后清空
    """
    
    def __init__(self, max_capacity: int = 50, ttl_minutes: int = 60):
        """
        Args:
            max_capacity: 最大容量
            ttl_minutes: 过期时间（分钟）
        """
        self.max_capacity = max_capacity
        self.ttl_minutes = ttl_minutes
        self._memories: List[MemoryItem] = []
    
    def __len__(self) -> int:
        self._expire_old()
        return len(self._memories)
    
    def add(
        self,
        content: str,
        importance: float = 0.5,
        **metadata
    ) -> str:
        """
        添加工作记忆
        
        Args:
            content: 记忆内容
            importance: 重要性 0.0-1.0
            **metadata: 元数据
            
        Returns:
            记忆ID
        """
        self._expire_old()
        
        # 容量管理：移除最不重要的记忆
        if len(self._memories) >= self.max_capacity:
            self._remove_lowest_importance()
        
        item = MemoryItem(
            content=content,
            importance=importance,
            metadata=metadata
        )
        self._memories.append(item)
        return item.id
    
    def search(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0
    ) -> List[MemoryItem]:
        """
        关键词搜索
        
        评分算法: 关键词匹配分 × 时间衰减 × 重要性权重
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            min_importance: 最低重要性阈值
            
        Returns:
            匹配的记忆列表（按相关性排序）
        """
        self._expire_old()
        
        query_words = query.lower().split()
        if not query_words:
            return []
        
        scored = []
        
        for mem in self._memories:
            if mem.importance < min_importance:
                continue
            
            # 1. 关键词匹配评分
            content_lower = mem.content.lower()
            keyword_score = sum(1 for word in query_words if word in content_lower)
            
            if keyword_score == 0:
                continue
            
            # 2. 时间衰减（最近的权重高）
            age_minutes = (datetime.now() - mem.timestamp).total_seconds() / 60
            time_decay = max(0.3, 1.0 - age_minutes / self.ttl_minutes)
            
            # 3. 重要性权重 [0.8, 1.2]
            importance_weight = 0.8 + (mem.importance * 0.4)
            
            # 4. 综合评分
            final_score = keyword_score * time_decay * importance_weight
            scored.append((final_score, mem))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]
    
    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """获取最近的记忆"""
        self._expire_old()
        return sorted(self._memories, key=lambda m: m.timestamp, reverse=True)[:limit]
    
    def get_important(self, threshold: float = 0.7) -> List[MemoryItem]:
        """获取重要记忆（用于记忆整合）"""
        self._expire_old()
        return [m for m in self._memories if m.importance >= threshold]
    
    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆"""
        self._expire_old()
        return list(self._memories)
    
    def clear(self):
        """清空工作记忆"""
        self._memories.clear()
    
    def _expire_old(self):
        """清理过期记忆"""
        cutoff = datetime.now() - timedelta(minutes=self.ttl_minutes)
        self._memories = [m for m in self._memories if m.timestamp > cutoff]
    
    def _remove_lowest_importance(self):
        """移除最不重要的记忆"""
        if not self._memories:
            return
        self._memories.sort(key=lambda m: m.importance)
        self._memories.pop(0)


__all__ = ["WorkingMemory"]
