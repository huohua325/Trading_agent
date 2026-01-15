"""
记忆系统数据结构定义

包含：
- MemoryItem: 基础记忆项（WorkingMemory 使用）
- DecisionEpisode: 决策情景记录（EpisodicMemory 使用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4


@dataclass
class MemoryItem:
    """
    基础记忆项 - WorkingMemory 使用
    
    Attributes:
        id: 唯一标识
        content: 记忆内容
        timestamp: 创建时间
        importance: 重要性 0.0-1.0
        metadata: 元数据
    """
    id: str = field(default_factory=lambda: f"mem_{uuid4().hex[:8]}")
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            id=data.get("id", f"mem_{uuid4().hex[:8]}"),
            content=data.get("content", ""),
            timestamp=timestamp or datetime.now(),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DecisionEpisode:
    """
    决策情景记录 - EpisodicMemory 使用
    
    这是对原有 history 的结构化升级，增强点：
    1. 结构化 reasoning（不只是动作，还有为什么）
    2. 市场上下文快照（便于事后分析）
    3. 标签系统（支持检索）
    4. 结果回填（形成闭环）
    5. 关联消息（与 Message 系统打通）
    """
    
    # 基本信息
    id: str = field(default_factory=lambda: f"ep_{datetime.now().strftime('%Y%m%d')}_{uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.now)
    date: str = ""  # 交易日期 YYYY-MM-DD
    
    # 决策内容
    symbol: str = ""
    strategy: str = ""          # 策略名称
    action: str = ""            # increase | decrease | hold | close
    target_amount: float = 0.0
    cash_change: float = 0.0    # 现金变化量（正为买入，负为卖出）
    shares: float = 0.0         # 当前持仓股数（累计）
    confidence: float = 0.5     # 置信度 0.0-1.0
    
    # 决策依据
    reasoning: str = ""                                    # LLM 给出的理由
    reasons: List[str] = field(default_factory=list)       # 结构化理由列表
    market_context: Dict[str, Any] = field(default_factory=dict)  # 市场状态快照
    signals: Dict[str, Any] = field(default_factory=dict)  # 触发的信号
    
    # 结果（事后回填）
    actual_result: Optional[float] = None   # 收益率
    outcome_note: Optional[str] = None      # 结果备注
    filled_at: Optional[datetime] = None    # 回填时间
    
    # 索引标签
    tags: List[str] = field(default_factory=list)  # ["高波动", "突破", "止损"]
    importance: float = 0.5  # 重要性权重
    
    # 关联消息（与 Message 系统打通）
    related_messages: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.date:
            self.date = self.timestamp.strftime("%Y-%m-%d")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "date": self.date,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "action": self.action,
            "target_amount": self.target_amount,
            "cash_change": self.cash_change,
            "shares": self.shares,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "reasons": self.reasons,
            "market_context": self.market_context,
            "signals": self.signals,
            "actual_result": self.actual_result,
            "outcome_note": self.outcome_note,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "tags": self.tags,
            "importance": self.importance,
            "related_messages": self.related_messages,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionEpisode':
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        filled_at = data.get("filled_at")
        if isinstance(filled_at, str):
            filled_at = datetime.fromisoformat(filled_at)
            
        return cls(
            id=data.get("id", ""),
            timestamp=timestamp or datetime.now(),
            date=data.get("date", ""),
            symbol=data.get("symbol", ""),
            strategy=data.get("strategy", ""),
            action=data.get("action", ""),
            target_amount=data.get("target_amount", 0.0),
            cash_change=data.get("cash_change", 0.0),
            shares=data.get("shares", 0.0),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            reasons=data.get("reasons", []),
            market_context=data.get("market_context", {}),
            signals=data.get("signals", {}),
            actual_result=data.get("actual_result"),
            outcome_note=data.get("outcome_note"),
            filled_at=filled_at,
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            related_messages=data.get("related_messages", []),
        )
    
    def get_searchable_text(self) -> str:
        """获取可搜索文本（用于关键词检索）"""
        parts = [
            self.symbol,
            self.action,
            self.strategy,
            self.reasoning,
            " ".join(self.reasons),
            " ".join(self.tags),
            self.outcome_note or "",
        ]
        return " ".join(parts).lower()


__all__ = ["MemoryItem", "DecisionEpisode"]
