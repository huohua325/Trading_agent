"""
情景记忆 - 决策历史存储

这是对现有 history 的升级版，增强点：
1. 结构化存储（DecisionEpisode）
2. 多维检索（时间/品种/策略/标签）
3. 简单关键词搜索
4. 结果回填机制（形成闭环）
5. 滑动窗口管理
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from ..schemas import DecisionEpisode
from ..backends.base import StorageBackend


class EpisodicMemory:
    """
    情景记忆 - 决策历史
    
    用于存储和检索历史决策记录，支持：
    - 按品种/策略/时间/标签多维查询
    - 关键词搜索
    - 结果回填（形成闭环学习）
    
    存储格式: JSONL (按月份分文件)
    """
    
    def __init__(
        self,
        backend: StorageBackend,
        data_dir: Path,
        max_days: int = 30
    ):
        """
        Args:
            backend: 存储后端
            data_dir: 数据目录
            max_days: 保留天数（滑动窗口）
        """
        self.backend = backend
        self.data_dir = Path(data_dir)
        self.max_days = max_days
        
        # 内存缓存（加速检索）
        self._cache: List[DecisionEpisode] = []
        self._index: Dict[str, List[str]] = {}  # symbol -> [episode_ids]
        
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载现有数据到缓存
        self._load_recent_to_cache()
    
    def add(self, episode: DecisionEpisode) -> str:
        """
        添加决策记录
        
        Args:
            episode: 决策情景记录
            
        Returns:
            记录ID
        """
        # 写入存储（按月份分文件）
        file_path = self.data_dir / f"decisions_{episode.date[:7]}.jsonl"
        self.backend.append_jsonl(file_path, episode.to_dict())
        
        # 更新内存缓存和索引
        self._cache.append(episode)
        self._index.setdefault(episode.symbol, []).append(episode.id)
        
        logger.debug(f"[Memory] Added episode {episode.id}: {episode.symbol} {episode.action}")
        return episode.id
    
    def query(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        days: int = 3,
        tags: Optional[List[str]] = None,
        action: Optional[str] = None,
        limit: int = 10
    ) -> List[DecisionEpisode]:
        """
        多维查询
        
        Args:
            symbol: 品种筛选
            strategy: 策略筛选
            days: 最近N天
            tags: 标签筛选（任一匹配）
            action: 动作筛选
            limit: 返回数量限制
            
        Returns:
            匹配的决策记录列表（按时间倒序）
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        results = []
        
        for ep in self._cache:
            # 时间过滤
            if ep.date < cutoff_date:
                continue
            
            # 品种过滤
            if symbol and ep.symbol != symbol:
                continue
            
            # 策略过滤
            if strategy and ep.strategy != strategy:
                continue
            
            # 动作过滤
            if action and ep.action != action:
                continue
            
            # 标签过滤（任一匹配）
            if tags and not any(t in ep.tags for t in tags):
                continue
            
            results.append(ep)
        
        # 按时间倒序
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[DecisionEpisode]:
        """
        关键词搜索
        
        评分算法: 关键词匹配分 × 时间衰减 × 重要性权重
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            min_importance: 最低重要性阈值
            
        Returns:
            匹配的决策记录（按相关性排序）
        """
        query_words = query.lower().split()
        if not query_words:
            return []
        
        scored = []
        
        for ep in self._cache:
            if ep.importance < min_importance:
                continue
            
            # 1. 关键词匹配评分
            searchable = ep.get_searchable_text()
            keyword_score = sum(1 for word in query_words if word in searchable)
            
            if keyword_score == 0:
                continue
            
            # 2. 时间衰减（7天内高权重）
            days_ago = (datetime.now() - ep.timestamp).days
            time_decay = max(0.3, 1.0 - days_ago * 0.1)
            
            # 3. 重要性权重 [0.8, 1.2]
            importance_weight = 0.8 + (ep.importance * 0.4)
            
            # 4. 综合评分
            final_score = keyword_score * time_decay * importance_weight
            scored.append((final_score, ep))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]
    
    def get_for_prompt(self, symbol: str, n: int = 5) -> str:
        """
        生成用于 prompt 的历史摘要（文本格式）
        
        兼容现有的 history 用法，可直接替换原有代码。
        
        Args:
            symbol: 品种
            n: 返回数量
            
        Returns:
            格式化的历史文本
        """
        episodes = self.query(symbol=symbol, limit=n)
        
        if not episodes:
            return "No previous decisions for this symbol."
        
        lines = []
        for ep in episodes:
            result_str = f" -> {ep.actual_result:+.1f}%" if ep.actual_result is not None else ""
            lines.append(f"- {ep.date}: {ep.action} ${ep.target_amount:.0f}{result_str}")
            if ep.reasoning:
                reason_preview = ep.reasoning[:100] + "..." if len(ep.reasoning) > 100 else ep.reasoning
                lines.append(f"  Reason: {reason_preview}")
        
        return "\n".join(lines)
    
    def get_history_for_prompt_dict(self, symbols: List[str], n: int = 7) -> Dict[str, List[Dict]]:
        """
        生成用于 prompt 的历史记录（字典格式）
        
        符合 input_prompt 中 history 字段的格式要求：
        {
            "SYMBOL": [
                {
                    "date": "YYYY-MM-DD",
                    "action": "increase|hold|decrease|close",
                    "cash_change": 1000.0,
                    "target_cash_amount": 5000.0,
                    "shares": 10.0,
                    "reasons": ["reason1", "reason2"],
                    "confidence": 0.85
                },
                ...
            ]
        }
        
        Args:
            symbols: 品种列表
            n: 每个品种返回的历史记录数量
            
        Returns:
            按 symbol 组织的历史记录字典
        """
        history_dict = {}
        
        for symbol in symbols:
            episodes = self.query(symbol=symbol, limit=n)
            
            if episodes:
                history_dict[symbol] = [
                    {
                        "date": ep.date,
                        "action": ep.action,
                        "cash_change": ep.cash_change,
                        "target_cash_amount": ep.target_amount,
                        "shares": ep.shares,
                        "reasons": ep.reasons,
                        "confidence": ep.confidence
                    }
                    for ep in episodes
                ]
            else:
                # 无历史记录时返回空列表
                history_dict[symbol] = []
        
        return history_dict
    
    def update_result(
        self,
        episode_id: str,
        result: float,
        note: Optional[str] = None
    ) -> bool:
        """
        回填决策结果 - 形成闭环
        
        Args:
            episode_id: 决策记录ID
            result: 实际收益率
            note: 结果备注
            
        Returns:
            是否更新成功
        """
        for ep in self._cache:
            if ep.id == episode_id:
                ep.actual_result = result
                ep.outcome_note = note
                ep.filled_at = datetime.now()
                
                # 更新存储
                self._update_in_storage(ep)
                logger.debug(f"[Memory] Updated result for {episode_id}: {result:+.1f}%")
                return True
        
        return False
    
    def get_by_id(self, episode_id: str) -> Optional[DecisionEpisode]:
        """根据ID获取决策记录"""
        for ep in self._cache:
            if ep.id == episode_id:
                return ep
        return None
    
    def query_unfilled(
        self,
        days: int = 7,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[DecisionEpisode]:
        """
        查询未回填结果的决策记录
        
        Args:
            days: 最近N天
            symbol: 品种筛选（可选）
            limit: 返回数量限制
            
        Returns:
            未回填结果的决策记录列表（按时间倒序）
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        results = []
        
        for ep in self._cache:
            # 时间过滤
            if ep.date < cutoff_date:
                continue
            
            # 品种过滤
            if symbol and ep.symbol != symbol:
                continue
            
            # 只返回未回填的记录
            if ep.actual_result is not None:
                continue
            
            results.append(ep)
        
        # 按时间倒序
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]
    
    def count(self) -> int:
        """获取记录总数"""
        return len(self._cache)
    
    def cleanup(self, keep_days: Optional[int] = None):
        """
        清理过期数据（滑动窗口）
        
        Args:
            keep_days: 保留天数，默认使用 max_days
        """
        keep_days = keep_days or self.max_days
        cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d")
        
        # 清理内存缓存
        self._cache = [ep for ep in self._cache if ep.date >= cutoff_date]
        
        # 重建索引
        self._index.clear()
        for ep in self._cache:
            self._index.setdefault(ep.symbol, []).append(ep.id)
        
        logger.debug(f"[Memory] Cleaned up episodes older than {cutoff_date}")
    
    def _load_recent_to_cache(self):
        """加载最近数据到内存缓存"""
        cutoff_date = (datetime.now() - timedelta(days=self.max_days)).strftime("%Y-%m-%d")
        
        # 遍历 JSONL 文件
        for file_path in self.data_dir.glob("decisions_*.jsonl"):
            try:
                lines = self.backend.read_jsonl(file_path)
                for data in lines:
                    ep = DecisionEpisode.from_dict(data)
                    if ep.date >= cutoff_date:
                        self._cache.append(ep)
                        self._index.setdefault(ep.symbol, []).append(ep.id)
            except Exception as e:
                logger.warning(f"[Memory] Failed to load {file_path}: {e}")
        
        # 按时间排序
        self._cache.sort(key=lambda e: e.timestamp, reverse=True)
        logger.debug(f"[Memory] Loaded {len(self._cache)} episodes to cache")
    
    def _update_in_storage(self, episode: DecisionEpisode):
        """更新存储中的记录"""
        file_path = self.data_dir / f"decisions_{episode.date[:7]}.jsonl"
        
        # 读取所有记录
        lines = self.backend.read_jsonl(file_path)
        
        # 更新目标记录
        updated_lines = []
        for data in lines:
            if data.get("id") == episode.id:
                updated_lines.append(episode.to_dict())
            else:
                updated_lines.append(data)
        
        # 写回文件
        self.backend.write_jsonl(file_path, updated_lines)


__all__ = ["EpisodicMemory"]
