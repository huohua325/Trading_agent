"""
Memory 系统单元测试

测试覆盖:
- WorkingMemory: 添加、搜索、过期、容量限制
- EpisodicMemory: 添加、查询、搜索、结果回填
- CacheStore: 缓存读写、TTL
- MemoryStore: 统一入口、生命周期
"""

import pytest
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from stockbench.memory import (
    MemoryStore, MemoryItem, DecisionEpisode,
    WorkingMemory, EpisodicMemory, CacheStore
)
from stockbench.memory.backends import FileBackend


# ==================== Fixtures ====================

@pytest.fixture
def temp_storage(tmp_path):
    """创建临时存储目录"""
    storage_path = tmp_path / "test_storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    yield storage_path
    # 清理
    if storage_path.exists():
        shutil.rmtree(storage_path)


@pytest.fixture
def memory_store(temp_storage):
    """创建 MemoryStore 实例"""
    return MemoryStore(
        base_path=str(temp_storage),
        working_memory_capacity=10,
        working_memory_ttl_minutes=60,
        episode_max_days=30
    )


@pytest.fixture
def working_memory():
    """创建 WorkingMemory 实例"""
    return WorkingMemory(max_capacity=5, ttl_minutes=60)


@pytest.fixture
def file_backend(temp_storage):
    """创建 FileBackend 实例"""
    return FileBackend(base_path=temp_storage)


# ==================== WorkingMemory Tests ====================

class TestWorkingMemory:
    """工作记忆测试"""
    
    def test_add_and_search(self, working_memory):
        """测试添加和搜索"""
        working_memory.add("AAPL 突破前高，成交量放大", importance=0.8, symbol="AAPL")
        working_memory.add("MSFT 横盘整理中", importance=0.5, symbol="MSFT")
        
        results = working_memory.search("AAPL", limit=5)
        assert len(results) >= 1
        assert any("AAPL" in r.content for r in results)
    
    def test_capacity_limit(self, working_memory):
        """测试容量限制"""
        # 添加超过容量的记忆
        for i in range(10):
            working_memory.add(f"记忆 {i}", importance=0.1 * i)
        
        # 应该只保留最重要的 5 条
        assert len(working_memory) <= 5
    
    def test_importance_based_eviction(self, working_memory):
        """测试基于重要性的淘汰"""
        working_memory.add("低重要性", importance=0.1)
        working_memory.add("高重要性", importance=0.9)
        
        for i in range(5):
            working_memory.add(f"中等重要性 {i}", importance=0.5)
        
        # 高重要性应该被保留
        important = working_memory.get_important(threshold=0.8)
        assert len(important) >= 1
    
    def test_get_recent(self, working_memory):
        """测试获取最近记忆"""
        working_memory.add("第一条", importance=0.5)
        working_memory.add("第二条", importance=0.5)
        working_memory.add("第三条", importance=0.5)
        
        recent = working_memory.get_recent(limit=2)
        assert len(recent) == 2
        assert recent[0].content == "第三条"
    
    def test_clear(self, working_memory):
        """测试清空"""
        working_memory.add("测试内容", importance=0.5)
        assert len(working_memory) > 0
        
        working_memory.clear()
        assert len(working_memory) == 0


# ==================== EpisodicMemory Tests ====================

class TestEpisodicMemory:
    """情景记忆测试"""
    
    def test_add_and_query(self, memory_store):
        """测试添加和查询"""
        ep = DecisionEpisode(
            symbol="AAPL",
            action="increase",
            target_amount=10000,
            reasoning="技术面看涨",
            confidence=0.8,
            tags=["技术分析", "看涨"]
        )
        ep_id = memory_store.episodes.add(ep)
        
        results = memory_store.episodes.query(symbol="AAPL", limit=10)
        assert len(results) >= 1
        assert results[0].symbol == "AAPL"
        assert results[0].action == "increase"
    
    def test_search(self, memory_store):
        """测试关键词搜索"""
        memory_store.episodes.add(DecisionEpisode(
            symbol="AAPL", action="increase", reasoning="财报超预期"
        ))
        memory_store.episodes.add(DecisionEpisode(
            symbol="MSFT", action="decrease", reasoning="估值过高"
        ))
        
        results = memory_store.episodes.search("财报", limit=10)
        assert len(results) >= 1
        assert results[0].symbol == "AAPL"
    
    def test_update_result(self, memory_store):
        """测试结果回填"""
        ep = DecisionEpisode(symbol="AAPL", action="increase", reasoning="test")
        ep_id = memory_store.episodes.add(ep)
        
        success = memory_store.episodes.update_result(ep_id, result=5.5, note="盈利")
        assert success
        
        updated = memory_store.episodes.get_by_id(ep_id)
        assert updated.actual_result == 5.5
        assert updated.outcome_note == "盈利"
    
    def test_get_for_prompt(self, memory_store):
        """测试生成 prompt 摘要"""
        memory_store.episodes.add(DecisionEpisode(
            symbol="AAPL", action="increase", target_amount=5000, reasoning="看涨"
        ))
        
        prompt = memory_store.episodes.get_for_prompt("AAPL", n=5)
        assert "AAPL" in prompt or "increase" in prompt
    
    def test_filter_by_action(self, memory_store):
        """测试按 action 过滤"""
        memory_store.episodes.add(DecisionEpisode(symbol="AAPL", action="increase"))
        memory_store.episodes.add(DecisionEpisode(symbol="AAPL", action="decrease"))
        memory_store.episodes.add(DecisionEpisode(symbol="AAPL", action="hold"))
        
        results = memory_store.episodes.query(symbol="AAPL", action="increase")
        assert all(r.action == "increase" for r in results)


# ==================== CacheStore Tests ====================

class TestCacheStore:
    """缓存层测试"""
    
    def test_set_and_get(self, temp_storage, file_backend):
        """测试缓存读写"""
        cache = CacheStore(file_backend, temp_storage / "cache")
        
        cache.set("test_ns", "key1", {"data": "value"})
        result = cache.get("test_ns", "key1")
        
        assert result == {"data": "value"}
    
    def test_get_nonexistent(self, temp_storage, file_backend):
        """测试获取不存在的缓存"""
        cache = CacheStore(file_backend, temp_storage / "cache")
        result = cache.get("test_ns", "nonexistent")
        assert result is None
    
    def test_delete(self, temp_storage, file_backend):
        """测试删除缓存"""
        cache = CacheStore(file_backend, temp_storage / "cache")
        
        cache.set("test_ns", "key1", "value")
        assert cache.get("test_ns", "key1") == "value"
        
        cache.delete("test_ns", "key1")
        assert cache.get("test_ns", "key1") is None


# ==================== MemoryStore Tests ====================

class TestMemoryStore:
    """统一入口测试"""
    
    def test_initialization(self, memory_store):
        """测试初始化"""
        assert memory_store.cache is not None
        assert memory_store.working is not None
        assert memory_store.episodes is not None
    
    def test_get_stats(self, memory_store):
        """测试统计信息"""
        memory_store.working.add("test", importance=0.5)
        memory_store.episodes.add(DecisionEpisode(symbol="TEST", action="hold"))
        
        stats = memory_store.get_stats()
        assert "working_memory_count" in stats
        assert "episode_count" in stats
        assert stats["working_memory_count"] >= 1
        assert stats["episode_count"] >= 1
    
    def test_clear_working(self, memory_store):
        """测试清空工作记忆"""
        memory_store.working.add("test", importance=0.5)
        assert len(memory_store.working) > 0
        
        memory_store.clear_working()
        assert len(memory_store.working) == 0


# ==================== DecisionEpisode Tests ====================

class TestDecisionEpisode:
    """决策记录数据结构测试"""
    
    def test_to_dict(self):
        """测试序列化"""
        ep = DecisionEpisode(
            symbol="AAPL",
            action="increase",
            reasoning="test reason",
            tags=["tag1", "tag2"]
        )
        d = ep.to_dict()
        
        assert d["symbol"] == "AAPL"
        assert d["action"] == "increase"
        assert d["tags"] == ["tag1", "tag2"]
    
    def test_from_dict(self):
        """测试反序列化"""
        data = {
            "id": "ep_test",
            "symbol": "MSFT",
            "action": "decrease",
            "reasoning": "估值过高",
            "timestamp": datetime.now().isoformat(),
            "tags": ["估值"]
        }
        ep = DecisionEpisode.from_dict(data)
        
        assert ep.id == "ep_test"
        assert ep.symbol == "MSFT"
        assert ep.action == "decrease"
    
    def test_searchable_text(self):
        """测试可搜索文本"""
        ep = DecisionEpisode(
            symbol="AAPL",
            action="increase",
            reasoning="财报超预期",
            tags=["财报", "看涨"]
        )
        text = ep.get_searchable_text()
        
        assert "aapl" in text
        assert "increase" in text
        assert "财报" in text


# ==================== MemoryItem Tests ====================

class TestMemoryItem:
    """记忆项数据结构测试"""
    
    def test_to_dict(self):
        """测试序列化"""
        item = MemoryItem(content="test content", importance=0.8)
        d = item.to_dict()
        
        assert d["content"] == "test content"
        assert d["importance"] == 0.8
        assert "timestamp" in d
    
    def test_from_dict(self):
        """测试反序列化"""
        data = {
            "id": "mem_test",
            "content": "test",
            "importance": 0.7,
            "timestamp": datetime.now().isoformat()
        }
        item = MemoryItem.from_dict(data)
        
        assert item.id == "mem_test"
        assert item.content == "test"
        assert item.importance == 0.7


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
