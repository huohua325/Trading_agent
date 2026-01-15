"""
PipelineContext 集成测试

测试覆盖:
- PipelineContext 与 Memory 集成
- 对话历史管理
- 生命周期钩子
"""

import pytest
import shutil
from pathlib import Path

from stockbench.core.pipeline_context import PipelineContext, AgentTrace, AgentStep
from stockbench.core.message import Message


# ==================== Fixtures ====================

@pytest.fixture
def temp_storage(tmp_path):
    """创建临时存储目录"""
    storage_path = tmp_path / "test_storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    yield storage_path
    if storage_path.exists():
        shutil.rmtree(storage_path)


@pytest.fixture
def pipeline_ctx(temp_storage):
    """创建 PipelineContext 实例"""
    return PipelineContext(
        run_id="test_run_001",
        date="2025-01-01",
        llm_client=None,
        llm_config=None,
        config={
            "memory": {
                "enabled": True,
                "storage_path": str(temp_storage),
                "working_memory": {"capacity": 50, "ttl_minutes": 60},
                "episodic_memory": {"max_days": 30}
            },
            "conversation_history": {
                "max_messages": 10
            }
        }
    )


# ==================== PipelineContext Memory Tests ====================

class TestPipelineContextMemory:
    """PipelineContext 记忆集成测试"""
    
    def test_memory_enabled(self, pipeline_ctx):
        """测试记忆系统启用状态"""
        assert pipeline_ctx.memory_enabled == True
    
    def test_memory_disabled(self, temp_storage):
        """测试记忆系统禁用状态"""
        ctx = PipelineContext(
            run_id="test", date="2025-01-01",
            llm_client=None, llm_config=None,
            config={"memory": {"enabled": False}}
        )
        assert ctx.memory_enabled == False
    
    def test_memory_lazy_initialization(self, pipeline_ctx):
        """测试记忆系统延迟初始化"""
        # 初始时 _memory_store 应该是 None
        assert pipeline_ctx._memory_store is None
        
        # 访问 memory 属性后应该初始化
        memory = pipeline_ctx.memory
        assert memory is not None
        assert pipeline_ctx._memory_store is not None
    
    def test_memory_working_operations(self, pipeline_ctx):
        """测试工作记忆操作"""
        pipeline_ctx.memory.working.add("AAPL 突破前高", importance=0.8)
        assert len(pipeline_ctx.memory.working) >= 1
        
        results = pipeline_ctx.memory.working.search("AAPL")
        assert len(results) >= 1


# ==================== Conversation History Tests ====================

class TestConversationHistory:
    """对话历史测试"""
    
    def test_initial_empty(self, pipeline_ctx):
        """测试初始为空"""
        assert len(pipeline_ctx.conversation_history) == 0
    
    def test_add_to_history(self, pipeline_ctx):
        """测试添加消息到历史"""
        pipeline_ctx.add_to_history(Message.user("问题1"))
        pipeline_ctx.add_to_history(Message.assistant("回答1"))
        
        assert len(pipeline_ctx.conversation_history) == 2
        assert pipeline_ctx.conversation_history[0].role == "user"
        assert pipeline_ctx.conversation_history[1].role == "assistant"
    
    def test_history_truncation(self, pipeline_ctx):
        """测试历史自动截断"""
        # 添加超过最大限制的消息
        for i in range(15):
            pipeline_ctx.add_to_history(Message.user(f"消息 {i}"))
        
        # 应该被截断到 max_messages
        assert len(pipeline_ctx.conversation_history) <= 10
    
    def test_clear_history(self, pipeline_ctx):
        """测试清空历史"""
        pipeline_ctx.add_to_history(Message.user("test"))
        assert len(pipeline_ctx.conversation_history) > 0
        
        pipeline_ctx.clear_history()
        assert len(pipeline_ctx.conversation_history) == 0


# ==================== Lifecycle Tests ====================

class TestLifecycle:
    """生命周期测试"""
    
    def test_on_run_complete(self, pipeline_ctx):
        """测试运行完成钩子"""
        # 添加一些数据
        pipeline_ctx.memory.working.add("test", importance=0.8)
        
        # 调用完成钩子
        pipeline_ctx.on_run_complete()
        
        # 工作记忆应该被清空
        assert len(pipeline_ctx.memory.working) == 0


# ==================== Data Bus Tests ====================

class TestDataBus:
    """数据总线测试"""
    
    def test_put_and_get(self, pipeline_ctx):
        """测试存取数据"""
        pipeline_ctx.put("test_key", {"data": "value"}, agent_name="test_agent")
        
        result = pipeline_ctx.get("test_key")
        assert result == {"data": "value"}
    
    def test_get_source(self, pipeline_ctx):
        """测试获取数据来源"""
        pipeline_ctx.put("key1", "value1", agent_name="agent_a")
        
        source = pipeline_ctx.get_source("key1")
        assert source == "agent_a"
    
    def test_has_key(self, pipeline_ctx):
        """测试检查键存在"""
        assert pipeline_ctx.has("nonexistent") == False
        
        pipeline_ctx.put("exists", "value")
        assert pipeline_ctx.has("exists") == True
    
    def test_keys(self, pipeline_ctx):
        """测试获取所有键"""
        pipeline_ctx.put("key1", "v1")
        pipeline_ctx.put("key2", "v2")
        
        keys = pipeline_ctx.keys()
        assert "key1" in keys
        assert "key2" in keys


# ==================== Agent Trace Tests ====================

class TestAgentTrace:
    """Agent 追踪测试"""
    
    def test_start_and_finish_agent(self, pipeline_ctx):
        """测试开始和结束追踪"""
        step = pipeline_ctx.start_agent("test_agent", input_summary="测试输入")
        
        assert step.agent_name == "test_agent"
        assert step.status == "running"
        
        pipeline_ctx.finish_agent(step, status="success", output_summary="测试输出")
        
        assert step.status == "success"
        assert step.duration_ms >= 0
    
    def test_get_failed_agents(self, pipeline_ctx):
        """测试获取失败的 Agent"""
        step1 = pipeline_ctx.start_agent("agent1")
        pipeline_ctx.finish_agent(step1, status="success")
        
        step2 = pipeline_ctx.start_agent("agent2")
        pipeline_ctx.finish_agent(step2, status="failed", error="test error")
        
        failed = pipeline_ctx.get_failed_agents()
        assert "agent2" in failed
        assert "agent1" not in failed
    
    def test_get_summary(self, pipeline_ctx):
        """测试获取执行摘要"""
        step = pipeline_ctx.start_agent("test")
        pipeline_ctx.finish_agent(step, status="success")
        
        summary = pipeline_ctx.get_summary()
        
        assert summary["run_id"] == "test_run_001"
        assert summary["total_agents"] >= 1
        assert summary["success"] >= 1


# ==================== Backfill Tests ====================

class TestBackfill:
    """结果回填测试"""
    
    def test_backfill_with_portfolio_history(self, pipeline_ctx, tmp_path):
        """测试使用持仓历史进行回填"""
        from stockbench.memory import DecisionEpisode
        from datetime import datetime, timedelta
        
        # 创建一个未回填的决策记录
        decision_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        episode = DecisionEpisode(
            symbol="AAPL",
            action="increase",
            target_amount=10000,
            reasoning="Test decision",
            confidence=0.8
        )
        episode.date = decision_date
        
        # 添加到 memory
        pipeline_ctx.memory.episodes.add(episode)
        
        # 构建模拟持仓历史
        eval_date = (datetime.strptime(decision_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")
        portfolio_history = [
            {
                "date": decision_date,
                "positions": {"AAPL": {"value": 10000, "pnl": 0}}
            },
            {
                "date": eval_date,
                "positions": {"AAPL": {"value": 10500, "pnl": 500}}  # +5%
            }
        ]
        
        # 调用 on_run_complete 触发回填
        pipeline_ctx.on_run_complete(portfolio_history=portfolio_history)
        
        # 验证回填结果
        updated_episode = pipeline_ctx.memory.episodes.get_by_id(episode.id)
        assert updated_episode is not None
        assert updated_episode.actual_result == 5.0  # 5% 收益
    
    def test_query_unfilled(self, pipeline_ctx):
        """测试查询未回填的记录"""
        from stockbench.memory import DecisionEpisode
        
        # 添加已回填的记录
        ep1 = DecisionEpisode(symbol="AAPL", action="increase", target_amount=1000)
        ep1.actual_result = 5.0
        pipeline_ctx.memory.episodes.add(ep1)
        
        # 添加未回填的记录
        ep2 = DecisionEpisode(symbol="MSFT", action="decrease", target_amount=2000)
        pipeline_ctx.memory.episodes.add(ep2)
        
        # 查询未回填的记录
        unfilled = pipeline_ctx.memory.episodes.query_unfilled(days=7)
        
        # 只应该返回未回填的
        unfilled_ids = [e.id for e in unfilled]
        assert ep2.id in unfilled_ids
        assert ep1.id not in unfilled_ids


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
