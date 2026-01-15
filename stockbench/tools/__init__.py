"""
StockBench 工具系统

提供可扩展的工具抽象，支持：
- Tool 基类定义
- ToolRegistry 工具注册中心
- 数据获取工具集
- OpenAI Function Calling 格式转换

使用示例:
    from stockbench.tools import ToolRegistry, Tool, ToolResult
    
    # 获取默认注册中心
    registry = ToolRegistry.default()
    
    # 执行工具
    result = registry.execute("get_price_data", symbol="AAPL", start_date="2025-01-01", end_date="2025-01-10")
    if result.success:
        df = result.data
        print(f"获取到 {len(df)} 条价格数据")
    
    # 获取 OpenAI Function Calling 格式
    tools = registry.to_openai_tools()
"""

# 基类和类型
from .base import (
    Tool,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)

# 注册中心
from .registry import (
    ToolRegistry,
    get_default_registry,
    execute_tool,
)

# 数据工具
from .data_tools import (
    PriceDataTool,
    NewsDataTool,
    FinancialsTool,
    SnapshotTool,
    DividendsTool,
    TickerDetailsTool,
    SplitsTool,
)


__all__ = [
    # 基类和类型
    "Tool",
    "ToolParameter",
    "ToolParameterType",
    "ToolResult",
    
    # 注册中心
    "ToolRegistry",
    "get_default_registry",
    "execute_tool",
    
    # 数据工具
    "PriceDataTool",
    "NewsDataTool",
    "FinancialsTool",
    "SnapshotTool",
    "DividendsTool",
    "TickerDetailsTool",
    "SplitsTool",
]
