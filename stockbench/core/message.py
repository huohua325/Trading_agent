"""
消息系统 (Message System)

为 StockBench Agent 提供标准化的消息格式，用于：
1. Agent 与 LLM 之间的通信
2. 对话历史的存储和传递
3. 记忆系统的基础数据单元

与 Memory 系统的关系：
- Message 是记忆系统存储的基本单元之一
- ConversationMemory 存储 List[Message]
- EpisodicMemory 中的 DecisionEpisode 可关联 Message

使用示例：
    # 创建消息
    msg = Message.user("请分析 AAPL 的走势")
    msg = Message.assistant("根据技术指标...")
    msg = Message.system("你是一个交易分析助手")
    
    # 转换为 LLM API 格式
    messages = [msg.to_api_dict() for msg in history]
    
    # 序列化/反序列化
    data = msg.to_dict()
    msg = Message.from_dict(data)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Union
from enum import Enum
import json
import uuid


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class Message:
    """
    标准化消息格式
    
    Attributes:
        role: 消息角色 (system/user/assistant/tool)
        content: 消息内容
        timestamp: 消息创建时间
        id: 消息唯一ID
        metadata: 可选元数据（如 symbol, date, agent_name 等）
        tool_call_id: 工具调用ID（仅当 role=tool 时使用）
        name: 工具名称（仅当 role=tool 时使用）
    """
    role: Union[MessageRole, str]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    def __post_init__(self):
        # 确保 role 是字符串
        if isinstance(self.role, MessageRole):
            self.role = self.role.value
    
    # ==================== 工厂方法 ====================
    
    @classmethod
    def system(cls, content: str, **metadata) -> Message:
        """创建系统消息"""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            metadata=metadata if metadata else None
        )
    
    @classmethod
    def user(cls, content: str, **metadata) -> Message:
        """创建用户消息"""
        return cls(
            role=MessageRole.USER,
            content=content,
            metadata=metadata if metadata else None
        )
    
    @classmethod
    def assistant(cls, content: str, **metadata) -> Message:
        """创建助手消息"""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata=metadata if metadata else None
        )
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str, **metadata) -> Message:
        """创建工具响应消息"""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            metadata=metadata if metadata else None
        )
    
    # ==================== 序列化方法 ====================
    
    def to_api_dict(self) -> Dict[str, str]:
        """
        转换为 OpenAI API 格式（用于 LLM 调用）
        
        Returns:
            {"role": "user", "content": "..."}
        """
        result = {"role": str(self.role), "content": self.content}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为完整字典格式（用于持久化）
        
        Returns:
            包含所有字段的字典
        """
        return {
            "id": self.id,
            "role": str(self.role),
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """
        从字典创建 Message（用于反序列化）
        
        Args:
            data: to_dict() 输出的字典
            
        Returns:
            Message 实例
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
            
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> Message:
        """从 JSON 字符串创建"""
        return cls.from_dict(json.loads(json_str))
    
    # ==================== 辅助方法 ====================
    
    def with_metadata(self, **kwargs) -> Message:
        """
        添加元数据，返回新 Message（不可变模式）
        
        Example:
            msg = Message.user("分析 AAPL").with_metadata(symbol="AAPL", date="2025-01-01")
        """
        new_metadata = {**(self.metadata or {}), **kwargs}
        return Message(
            role=self.role,
            content=self.content,
            timestamp=self.timestamp,
            id=self.id,
            metadata=new_metadata,
            tool_call_id=self.tool_call_id,
            name=self.name,
        )
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role}, content='{content_preview}')"


# ==================== 辅助函数 ====================

def messages_to_api_format(messages: List[Message]) -> List[Dict[str, str]]:
    """
    将消息列表转换为 OpenAI API 格式
    
    Args:
        messages: Message 列表
        
    Returns:
        [{"role": "...", "content": "..."}, ...]
    """
    return [msg.to_api_dict() for msg in messages]


def messages_from_api_format(api_messages: List[Dict[str, str]]) -> List[Message]:
    """
    从 OpenAI API 格式创建消息列表
    
    Args:
        api_messages: [{"role": "...", "content": "..."}, ...]
        
    Returns:
        Message 列表
    """
    return [
        Message(role=msg["role"], content=msg["content"])
        for msg in api_messages
    ]


def build_conversation(
    system_prompt: str,
    history: Optional[List[Message]] = None,
    current_user_content: str = None
) -> List[Message]:
    """
    构建完整对话消息列表
    
    Args:
        system_prompt: 系统提示词
        history: 历史对话消息
        current_user_content: 当前用户输入
        
    Returns:
        完整的 Message 列表
    """
    messages = [Message.system(system_prompt)]
    if history:
        messages.extend(history)
    if current_user_content:
        messages.append(Message.user(current_user_content))
    return messages


def truncate_history(
    messages: List[Message],
    max_messages: int = 20,
    preserve_system: bool = True
) -> List[Message]:
    """
    截断历史消息（保留最近的）
    
    Args:
        messages: 消息列表
        max_messages: 最大消息数
        preserve_system: 是否保留系统消息
        
    Returns:
        截断后的消息列表
    """
    if len(messages) <= max_messages:
        return messages
    
    result = []
    remaining = max_messages
    
    if preserve_system and messages and messages[0].role == "system":
        result.append(messages[0])
        messages = messages[1:]
        remaining -= 1
    
    result.extend(messages[-remaining:])
    return result


def estimate_tokens(messages: List[Message]) -> int:
    """
    估算消息列表的 token 数量
    
    简单估算：中文字符 1 token，英文单词 1 token，每条消息 +4 开销
    
    Args:
        messages: 消息列表
        
    Returns:
        估算的 token 数量
    """
    total = 0
    for msg in messages:
        content = msg.content
        chinese = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        non_cn = ''.join(' ' if '\u4e00' <= c <= '\u9fff' else c for c in content)
        english = len(non_cn.split())
        total += chinese + english + 4
    return total


# ==================== 导出 ====================

__all__ = [
    "Message",
    "MessageRole",
    "messages_to_api_format",
    "messages_from_api_format",
    "build_conversation",
    "truncate_history",
    "estimate_tokens",
]
