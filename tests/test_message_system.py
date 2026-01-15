"""
Message 系统单元测试

测试覆盖:
- Message: 创建、序列化、工厂方法
- 辅助函数: build_conversation, truncate_history, estimate_tokens
- API 格式转换
"""

import pytest
from datetime import datetime

from stockbench.core.message import (
    Message, MessageRole,
    messages_to_api_format, messages_from_api_format,
    build_conversation, truncate_history, estimate_tokens
)


# ==================== Message Tests ====================

class TestMessage:
    """Message 类测试"""
    
    def test_system_factory(self):
        """测试系统消息工厂方法"""
        msg = Message.system("You are a helpful assistant")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant"
    
    def test_user_factory(self):
        """测试用户消息工厂方法"""
        msg = Message.user("Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
    
    def test_assistant_factory(self):
        """测试助手消息工厂方法"""
        msg = Message.assistant("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
    
    def test_tool_factory(self):
        """测试工具消息工厂方法"""
        msg = Message.tool("Result", tool_call_id="tc_123", name="search")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_123"
        assert msg.name == "search"
    
    def test_with_metadata(self):
        """测试添加元数据"""
        msg = Message.user("Analyze AAPL").with_metadata(symbol="AAPL", date="2025-01-01")
        assert msg.metadata["symbol"] == "AAPL"
        assert msg.metadata["date"] == "2025-01-01"
    
    def test_to_api_dict(self):
        """测试转换为 API 格式"""
        msg = Message.user("Hello")
        api_dict = msg.to_api_dict()
        
        assert api_dict["role"] == "user"
        assert api_dict["content"] == "Hello"
    
    def test_to_dict(self):
        """测试完整序列化"""
        msg = Message.user("Hello", symbol="AAPL")
        d = msg.to_dict()
        
        assert "id" in d
        assert "timestamp" in d
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert d["metadata"]["symbol"] == "AAPL"
    
    def test_from_dict(self):
        """测试反序列化"""
        data = {
            "id": "msg_test",
            "role": "user",
            "content": "Test message",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"key": "value"}
        }
        msg = Message.from_dict(data)
        
        assert msg.id == "msg_test"
        assert msg.role == "user"
        assert msg.content == "Test message"
        assert msg.metadata["key"] == "value"
    
    def test_repr(self):
        """测试字符串表示"""
        msg = Message.user("This is a test message")
        repr_str = repr(msg)
        
        assert "Message" in repr_str
        assert "user" in repr_str


# ==================== API Format Tests ====================

class TestApiFormat:
    """API 格式转换测试"""
    
    def test_messages_to_api_format(self):
        """测试转换为 API 格式"""
        messages = [
            Message.system("System prompt"),
            Message.user("User question"),
            Message.assistant("Assistant answer")
        ]
        
        api_messages = messages_to_api_format(messages)
        
        assert len(api_messages) == 3
        assert api_messages[0]["role"] == "system"
        assert api_messages[1]["role"] == "user"
        assert api_messages[2]["role"] == "assistant"
    
    def test_messages_from_api_format(self):
        """测试从 API 格式转换"""
        api_messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"}
        ]
        
        messages = messages_from_api_format(api_messages)
        
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"


# ==================== Helper Functions Tests ====================

class TestBuildConversation:
    """build_conversation 测试"""
    
    def test_basic_conversation(self):
        """测试基本对话构建"""
        messages = build_conversation(
            system_prompt="You are an analyst",
            current_user_content="Analyze AAPL"
        )
        
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
    
    def test_with_history(self):
        """测试包含历史的对话构建"""
        history = [
            Message.user("Previous question"),
            Message.assistant("Previous answer")
        ]
        
        messages = build_conversation(
            system_prompt="System",
            history=history,
            current_user_content="New question"
        )
        
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "user"
    
    def test_without_current_content(self):
        """测试不含当前内容的构建"""
        messages = build_conversation(
            system_prompt="System",
            history=[Message.user("Old")],
            current_user_content=None
        )
        
        assert len(messages) == 2


class TestTruncateHistory:
    """truncate_history 测试"""
    
    def test_no_truncation_needed(self):
        """测试不需要截断的情况"""
        messages = [Message.user("1"), Message.assistant("2")]
        result = truncate_history(messages, max_messages=10)
        
        assert len(result) == 2
    
    def test_truncation_preserves_system(self):
        """测试截断时保留系统消息"""
        messages = [
            Message.system("System"),
            Message.user("1"),
            Message.assistant("2"),
            Message.user("3"),
            Message.assistant("4"),
            Message.user("5")
        ]
        
        result = truncate_history(messages, max_messages=3, preserve_system=True)
        
        assert len(result) == 3
        assert result[0].role == "system"
    
    def test_truncation_without_preserve_system(self):
        """测试不保留系统消息的截断"""
        messages = [
            Message.system("System"),
            Message.user("1"),
            Message.assistant("2"),
            Message.user("3")
        ]
        
        result = truncate_history(messages, max_messages=2, preserve_system=False)
        
        assert len(result) == 2


class TestEstimateTokens:
    """estimate_tokens 测试"""
    
    def test_english_only(self):
        """测试纯英文估算"""
        messages = [Message.user("Hello world how are you")]
        tokens = estimate_tokens(messages)
        
        assert tokens > 0
        assert tokens < 20  # 合理范围
    
    def test_chinese_only(self):
        """测试纯中文估算"""
        messages = [Message.user("你好世界")]
        tokens = estimate_tokens(messages)
        
        assert tokens > 0
    
    def test_mixed_content(self):
        """测试中英混合估算"""
        messages = [Message.user("分析 AAPL 股票的技术面")]
        tokens = estimate_tokens(messages)
        
        assert tokens > 0
    
    def test_multiple_messages(self):
        """测试多条消息估算"""
        messages = [
            Message.system("You are an analyst"),
            Message.user("Analyze AAPL"),
            Message.assistant("AAPL is looking good")
        ]
        tokens = estimate_tokens(messages)
        
        # 每条消息有 +4 开销
        assert tokens >= 12


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
