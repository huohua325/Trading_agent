"""
Agent 装饰器

提供 @traced_agent 装饰器，自动追踪 Agent 函数的执行。

使用示例:
    from stockbench.core import traced_agent, PipelineContext
    
    @traced_agent("fundamental_filter")
    def filter_stocks_needing_fundamental(features_list, ctx=None, **kwargs):
        # 自动记录输入摘要、执行耗时、输出摘要
        # 如果发生异常，自动记录错误
        ...
        return result
"""

from functools import wraps
from typing import Callable, Any


def traced_agent(name: str):
    """
    Agent 追踪装饰器
    
    自动记录 Agent 函数的输入/输出/耗时/错误。
    
    Args:
        name: Agent 名称，用于日志和追踪
        
    使用:
        @traced_agent("my_agent")
        def my_agent_function(input_data, ctx=None, **kwargs):
            ...
            return result
    
    注意:
        - 函数必须接受 ctx 参数（可以是位置参数或关键字参数）
        - 如果 ctx 为 None，装饰器不会进行追踪，直接执行原函数
        - 异常会被记录后重新抛出
        - 自动使用 ctx.logger 进行带上下文的日志记录
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试获取 ctx 参数
            ctx = _extract_ctx(args, kwargs)
            
            # 如果没有 ctx，直接执行原函数（向后兼容）
            if ctx is None:
                return func(*args, **kwargs)
            
            # 生成输入摘要
            input_data = args[0] if args else None
            input_summary = _make_summary(input_data)
            
            # 获取带上下文的 logger
            agent_logger = ctx.get_agent_logger(name) if hasattr(ctx, 'get_agent_logger') else None
            
            # 开始追踪
            step = ctx.start_agent(name, input_summary)
            
            if agent_logger:
                agent_logger.info(
                    f"[AGENT_EXEC] {name} executing",
                    input_summary=input_summary
                )
            
            try:
                # 执行原函数
                result = func(*args, **kwargs)
                
                # 记录成功
                output_summary = _make_summary(result)
                ctx.finish_agent(step, "success", output_summary)
                
                return result
            except Exception as e:
                # 记录失败
                ctx.finish_agent(step, "failed", error=str(e))
                
                if agent_logger:
                    agent_logger.error(
                        f"[AGENT_ERROR] {name} execution failed",
                        error=str(e)
                    )
                raise
        
        # 保存原始函数的 agent_name 属性，方便调试
        wrapper.agent_name = name
        return wrapper
    return decorator


def _extract_ctx(args: tuple, kwargs: dict):
    """
    从参数中提取 PipelineContext
    
    支持以下方式传入 ctx:
    1. 关键字参数: func(..., ctx=ctx)
    2. 位置参数（最后一个）: func(data, ctx)
    """
    # 优先从关键字参数获取
    ctx = kwargs.get("ctx")
    if ctx is not None and hasattr(ctx, 'trace'):
        return ctx
    
    # 尝试从位置参数获取（检查最后一个参数）
    if args:
        last_arg = args[-1]
        if hasattr(last_arg, 'trace'):
            return last_arg
    
    return None


def _make_summary(data: Any) -> str:
    """
    生成数据摘要（避免日志过大）
    
    Args:
        data: 任意数据
        
    Returns:
        简短的摘要字符串
    """
    if data is None:
        return "None"
    
    if isinstance(data, list):
        if len(data) == 0:
            return "List[0 items]"
        # 尝试获取第一个元素的类型
        first_type = type(data[0]).__name__
        return f"List[{len(data)} {first_type}s]"
    
    if isinstance(data, dict):
        keys = list(data.keys())
        if len(keys) <= 5:
            return f"Dict[{len(keys)} keys: {keys}]"
        return f"Dict[{len(keys)} keys: {keys[:5]}...]"
    
    if isinstance(data, str):
        if len(data) <= 50:
            return f"str({len(data)}): {repr(data)}"
        return f"str({len(data)}): {repr(data[:50])}..."
    
    if isinstance(data, (int, float, bool)):
        return str(data)
    
    # 其他类型
    type_name = type(data).__name__
    return f"{type_name}: {str(data)[:100]}"


# 导出
__all__ = ["traced_agent"]
