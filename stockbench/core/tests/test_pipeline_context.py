"""
Pipeline Context 测试

测试 PipelineContext、AgentTrace 和 @traced_agent 装饰器的功能。

运行方式:
    pytest stockbench/core/tests/test_pipeline_context.py -v
"""
import pytest
import time
import sys
import os
from unittest.mock import MagicMock

# 直接导入模块文件，避免触发 stockbench.core.__init__.py 的重依赖
# 这样可以独立测试 pipeline_context 和 decorators，无需 LLM 等依赖
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)

# 直接使用 importlib 导入特定文件，避开 __init__.py
import importlib.util

def _import_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_core_dir = os.path.join(_root, "stockbench", "core")
_pipeline_ctx_mod = _import_from_file("pipeline_context", os.path.join(_core_dir, "pipeline_context.py"))
_decorators_mod = _import_from_file("decorators", os.path.join(_core_dir, "decorators.py"))

PipelineContext = _pipeline_ctx_mod.PipelineContext
AgentTrace = _pipeline_ctx_mod.AgentTrace
AgentStep = _pipeline_ctx_mod.AgentStep
traced_agent = _decorators_mod.traced_agent
_make_summary = _decorators_mod._make_summary


class TestAgentStep:
    """测试 AgentStep"""
    
    def test_step_creation(self):
        """测试步骤创建"""
        from datetime import datetime
        step = AgentStep(agent_name="test_agent", started_at=datetime.now())
        assert step.agent_name == "test_agent"
        assert step.status == "running"
        assert step.error is None
    
    def test_step_finish_success(self):
        """测试步骤成功完成"""
        from datetime import datetime
        step = AgentStep(agent_name="test_agent", started_at=datetime.now())
        time.sleep(0.01)  # 小延迟
        step.finish("success", output_summary="Done")
        
        assert step.status == "success"
        assert step.output_summary == "Done"
        assert step.duration_ms > 0
        assert step.error is None
    
    def test_step_finish_failed(self):
        """测试步骤失败"""
        from datetime import datetime
        step = AgentStep(agent_name="test_agent", started_at=datetime.now())
        step.finish("failed", error="Something went wrong")
        
        assert step.status == "failed"
        assert step.error == "Something went wrong"


class TestAgentTrace:
    """测试 AgentTrace"""
    
    def test_trace_creation(self):
        """测试追踪器创建"""
        trace = AgentTrace(run_id="test_run")
        assert trace.run_id == "test_run"
        assert len(trace.steps) == 0
    
    def test_start_and_finish_agent(self):
        """测试开始和完成 Agent 追踪"""
        trace = AgentTrace(run_id="test_run")
        
        step = trace.start_agent("agent1", "input_data")
        assert len(trace.steps) == 1
        assert step.agent_name == "agent1"
        assert step.status == "running"
        
        trace.finish_agent(step, "success", "output_data")
        assert step.status == "success"
    
    def test_get_failed_agents(self):
        """测试获取失败的 Agent"""
        trace = AgentTrace(run_id="test_run")
        
        step1 = trace.start_agent("agent1")
        trace.finish_agent(step1, "success")
        
        step2 = trace.start_agent("agent2")
        trace.finish_agent(step2, "failed", error="Error")
        
        step3 = trace.start_agent("agent3")
        trace.finish_agent(step3, "success")
        
        failed = trace.get_failed_agents()
        assert failed == ["agent2"]
    
    def test_to_summary(self):
        """测试生成执行摘要"""
        trace = AgentTrace(run_id="test_run")
        
        step1 = trace.start_agent("agent1")
        trace.finish_agent(step1, "success")
        
        step2 = trace.start_agent("agent2")
        trace.finish_agent(step2, "failed", error="Error")
        
        summary = trace.to_summary()
        assert summary["run_id"] == "test_run"
        assert summary["total_agents"] == 2
        assert summary["success"] == 1
        assert summary["failed"] == 1
        assert len(summary["steps"]) == 2


class TestPipelineContext:
    """测试 PipelineContext"""
    
    def test_context_creation(self):
        """测试上下文创建"""
        mock_llm = MagicMock()
        mock_config = MagicMock()
        
        ctx = PipelineContext(
            run_id="test_run",
            date="2025-01-01",
            llm_client=mock_llm,
            llm_config=mock_config,
            config={"key": "value"}
        )
        
        assert ctx.run_id == "test_run"
        assert ctx.date == "2025-01-01"
        assert ctx.config["key"] == "value"
        assert ctx.trace is not None
    
    def test_data_bus_put_get(self):
        """测试数据总线存取"""
        ctx = PipelineContext(
            run_id="test",
            date="2025-01-01",
            llm_client=MagicMock(),
            llm_config=MagicMock()
        )
        
        ctx.put("key1", "value1")
        ctx.put("key2", {"nested": "data"}, agent_name="agent1")
        
        assert ctx.get("key1") == "value1"
        assert ctx.get("key2") == {"nested": "data"}
        assert ctx.get("nonexistent") is None
        assert ctx.get("nonexistent", "default") == "default"
    
    def test_data_bus_source_tracking(self):
        """测试数据来源追踪"""
        ctx = PipelineContext(
            run_id="test",
            date="2025-01-01",
            llm_client=MagicMock(),
            llm_config=MagicMock()
        )
        
        ctx.put("result", {"data": 123}, agent_name="producer_agent")
        
        assert ctx.get_source("result") == "producer_agent"
        assert ctx.get_source("nonexistent") is None
    
    def test_data_bus_keys(self):
        """测试获取所有数据键"""
        ctx = PipelineContext(
            run_id="test",
            date="2025-01-01",
            llm_client=MagicMock(),
            llm_config=MagicMock()
        )
        
        ctx.put("key1", "value1")
        ctx.put("key2", "value2", agent_name="agent1")
        
        keys = ctx.keys()
        assert "key1" in keys
        assert "key2" in keys
        assert "_source_key2" not in keys  # 元数据不应出现
    
    def test_agent_tracing_via_context(self):
        """测试通过上下文追踪 Agent"""
        ctx = PipelineContext(
            run_id="test",
            date="2025-01-01",
            llm_client=MagicMock(),
            llm_config=MagicMock()
        )
        
        step = ctx.start_agent("my_agent", "input_summary")
        ctx.finish_agent(step, "success", "output_summary")
        
        assert len(ctx.trace.steps) == 1
        assert ctx.trace.steps[0].agent_name == "my_agent"
        assert ctx.trace.steps[0].status == "success"


class TestTracedAgentDecorator:
    """测试 @traced_agent 装饰器"""
    
    def test_decorator_with_ctx_kwarg(self):
        """测试装饰器通过关键字参数传入 ctx"""
        @traced_agent("test_agent")
        def my_func(data, ctx=None):
            return {"result": data * 2}
        
        ctx = PipelineContext(
            run_id="test",
            date="2025-01-01",
            llm_client=MagicMock(),
            llm_config=MagicMock()
        )
        
        result = my_func(5, ctx=ctx)
        
        assert result == {"result": 10}
        assert len(ctx.trace.steps) == 1
        assert ctx.trace.steps[0].agent_name == "test_agent"
        assert ctx.trace.steps[0].status == "success"
    
    def test_decorator_without_ctx(self):
        """测试装饰器不传 ctx（向后兼容）"""
        @traced_agent("test_agent")
        def my_func(data):
            return data * 2
        
        result = my_func(5)
        assert result == 10  # 应该正常工作，无追踪
    
    def test_decorator_with_exception(self):
        """测试装饰器处理异常"""
        @traced_agent("failing_agent")
        def failing_func(data, ctx=None):
            raise ValueError("Test error")
        
        ctx = PipelineContext(
            run_id="test",
            date="2025-01-01",
            llm_client=MagicMock(),
            llm_config=MagicMock()
        )
        
        with pytest.raises(ValueError):
            failing_func("data", ctx=ctx)
        
        assert len(ctx.trace.steps) == 1
        assert ctx.trace.steps[0].status == "failed"
        assert ctx.trace.steps[0].error == "Test error"
    
    def test_decorator_preserves_function_name(self):
        """测试装饰器保留函数名"""
        @traced_agent("my_agent")
        def original_func(data, ctx=None):
            return data
        
        assert original_func.__name__ == "original_func"
        assert original_func.agent_name == "my_agent"


class TestMakeSummary:
    """测试 _make_summary 辅助函数"""
    
    def test_summary_none(self):
        assert _make_summary(None) == "None"
    
    def test_summary_list(self):
        assert "List[3" in _make_summary([1, 2, 3])
        assert "List[0" in _make_summary([])
    
    def test_summary_dict(self):
        result = _make_summary({"a": 1, "b": 2})
        assert "Dict[2" in result
        assert "a" in result
    
    def test_summary_dict_many_keys(self):
        data = {f"key{i}": i for i in range(10)}
        result = _make_summary(data)
        assert "Dict[10" in result
        assert "..." in result  # 应该被截断
    
    def test_summary_string(self):
        assert "str" in _make_summary("hello")
        # 长字符串应该被截断
        long_str = "a" * 100
        result = _make_summary(long_str)
        assert "..." in result
    
    def test_summary_number(self):
        assert _make_summary(42) == "42"
        assert _make_summary(3.14) == "3.14"


class TestPipelineIntegration:
    """集成测试：模拟完整的 Agent 流水线"""
    
    def test_multi_agent_pipeline(self):
        """测试多 Agent 流水线"""
        
        @traced_agent("filter_agent")
        def filter_agent(features, ctx=None):
            # 模拟过滤逻辑
            result = {"needs_analysis": ["AAPL", "GOOGL"]}
            if ctx:
                ctx.put("filter_result", result, agent_name="filter_agent")
            return result
        
        @traced_agent("decision_agent")
        def decision_agent(features, ctx=None):
            # 从数据总线获取上游结果
            filter_result = ctx.get("filter_result") if ctx else None
            # 模拟决策逻辑
            decisions = {
                "AAPL": {"action": "hold", "confidence": 0.8},
                "GOOGL": {"action": "increase", "confidence": 0.7}
            }
            if ctx:
                ctx.put("decisions", decisions, agent_name="decision_agent")
            return decisions
        
        # 创建上下文
        ctx = PipelineContext(
            run_id="pipeline_test",
            date="2025-01-01",
            llm_client=MagicMock(),
            llm_config=MagicMock()
        )
        
        # 执行流水线
        features = [{"symbol": "AAPL"}, {"symbol": "GOOGL"}]
        
        filter_result = filter_agent(features, ctx=ctx)
        decisions = decision_agent(features, ctx=ctx)
        
        # 验证结果
        assert filter_result["needs_analysis"] == ["AAPL", "GOOGL"]
        assert "AAPL" in decisions
        assert "GOOGL" in decisions
        
        # 验证追踪
        summary = ctx.get_summary()
        assert summary["total_agents"] == 2
        assert summary["success"] == 2
        assert summary["failed"] == 0
        
        # 验证数据总线
        assert ctx.get("filter_result") == filter_result
        assert ctx.get("decisions") == decisions
        assert ctx.get_source("filter_result") == "filter_agent"
        assert ctx.get_source("decisions") == "decision_agent"


def run_tests_standalone():
    """直接运行测试（不通过 pytest，避免包导入问题）"""
    import traceback
    
    test_classes = [
        TestAgentStep,
        TestAgentTrace, 
        TestPipelineContext,
        TestTracedAgentDecorator,
        TestMakeSummary,
        TestPipelineIntegration,
    ]
    
    passed = 0
    failed = 0
    
    for cls in test_classes:
        print(f"\n{'='*50}")
        print(f"Running: {cls.__name__}")
        print('='*50)
        
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    traceback.print_exc()
                    failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print('='*50)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests_standalone()
    sys.exit(0 if success else 1)
