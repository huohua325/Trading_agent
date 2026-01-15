"""
PipelineContext 使用示例

展示如何使用新的 PipelineContext 运行 Agent 流水线。

运行方式:
    python -m stockbench.examples.pipeline_example
"""

from stockbench.core import (
    PipelineContext,
    AgentTrace,
    traced_agent,
)


# ============================================
# 示例 1: 基础用法 - 创建 PipelineContext
# ============================================

def example_basic_context():
    """基础用法示例"""
    print("=" * 60)
    print("示例 1: 基础 PipelineContext 用法")
    print("=" * 60)
    
    # 创建上下文（不需要真实的 LLM 客户端来演示）
    ctx = PipelineContext(
        run_id="demo_2025_01_01",
        date="2025-01-01",
        llm_client=None,  # 实际使用时传入 LLMClient
        llm_config=None,  # 实际使用时传入 LLMConfig
        config={"portfolio": {"total_cash": 100000}}
    )
    
    # 使用数据总线存取数据
    ctx.put("input_features", [{"symbol": "AAPL"}, {"symbol": "GOOGL"}])
    ctx.put("analysis_results", {"AAPL": {"score": 0.85}}, agent_name="analyzer")
    
    # 获取数据
    features = ctx.get("input_features")
    print(f"存入的 features: {features}")
    
    # 追踪数据来源
    source = ctx.get_source("analysis_results")
    print(f"analysis_results 来源: {source}")
    
    # 列出所有数据键
    print(f"数据总线中的键: {ctx.keys()}")
    
    return ctx


# ============================================
# 示例 2: 使用 @traced_agent 装饰器
# ============================================

@traced_agent("example_filter")
def example_filter_agent(data, ctx=None):
    """示例过滤 Agent"""
    result = {
        "needs_analysis": ["AAPL"],
        "skip_analysis": ["GOOGL"]
    }
    if ctx:
        ctx.put("filter_result", result, agent_name="example_filter")
    return result


@traced_agent("example_decision")
def example_decision_agent(data, ctx=None):
    """示例决策 Agent"""
    # 从数据总线获取上游结果
    filter_result = ctx.get("filter_result") if ctx else None
    
    decisions = {
        "AAPL": {"action": "increase", "confidence": 0.8},
        "GOOGL": {"action": "hold", "confidence": 0.6}
    }
    
    if ctx:
        ctx.put("decisions", decisions, agent_name="example_decision")
    
    return decisions


def example_traced_agents():
    """装饰器用法示例"""
    print("\n" + "=" * 60)
    print("示例 2: @traced_agent 装饰器用法")
    print("=" * 60)
    
    ctx = PipelineContext(
        run_id="traced_demo",
        date="2025-01-01",
        llm_client=None,
        llm_config=None
    )
    
    # 运行 Agent 流水线
    features = [{"symbol": "AAPL"}, {"symbol": "GOOGL"}]
    
    print("\n运行 filter_agent...")
    filter_result = example_filter_agent(features, ctx=ctx)
    
    print("\n运行 decision_agent...")
    decisions = example_decision_agent(features, ctx=ctx)
    
    # 查看执行摘要
    print("\n执行摘要:")
    summary = ctx.trace.to_summary()
    print(f"  总 Agent 数: {summary['total_agents']}")
    print(f"  成功: {summary['success']}")
    print(f"  失败: {summary['failed']}")
    print(f"  总耗时: {summary['total_duration_ms']:.2f}ms")
    
    print("\n各步骤详情:")
    for step in summary['steps']:
        print(f"  - {step['agent']}: {step['status']} ({step['duration_ms']:.2f}ms)")
    
    return ctx


# ============================================
# 示例 3: 错误处理和追踪
# ============================================

@traced_agent("failing_agent")
def example_failing_agent(data, ctx=None):
    """故意失败的 Agent，用于演示错误追踪"""
    raise ValueError("模拟的 Agent 错误")


def example_error_handling():
    """错误处理示例"""
    print("\n" + "=" * 60)
    print("示例 3: 错误处理和追踪")
    print("=" * 60)
    
    ctx = PipelineContext(
        run_id="error_demo",
        date="2025-01-01",
        llm_client=None,
        llm_config=None
    )
    
    try:
        example_failing_agent([], ctx=ctx)
    except ValueError as e:
        print(f"捕获到错误: {e}")
    
    # 查看失败的 Agent
    failed = ctx.get_failed_agents()
    print(f"失败的 Agent: {failed}")
    
    # 查看详细错误
    for step in ctx.trace.steps:
        if step.status == "failed":
            print(f"错误详情: {step.error}")
    
    return ctx


# ============================================
# 示例 4: 向后兼容 - 不使用 ctx 调用
# ============================================

def example_backward_compatibility():
    """向后兼容示例"""
    print("\n" + "=" * 60)
    print("示例 4: 向后兼容（无 ctx 调用）")
    print("=" * 60)
    
    # 不传 ctx，函数仍然正常工作
    features = [{"symbol": "AAPL"}]
    
    result = example_filter_agent(features)  # 不传 ctx
    print(f"结果（无追踪）: {result}")
    
    return result


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("PipelineContext 使用示例\n")
    
    # 运行所有示例
    example_basic_context()
    example_traced_agents()
    example_error_handling()
    example_backward_compatibility()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
