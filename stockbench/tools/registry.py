"""
ToolRegistry 工具注册中心

提供工具的统一管理：
- 工具注册/注销
- 工具查找/执行
- OpenAI Function Calling 格式批量转换
- 按标签筛选工具

使用示例:
    # 获取默认注册中心（包含内置工具）
    registry = ToolRegistry.default()
    
    # 执行工具
    result = registry.execute("get_price_data", symbol="AAPL", start_date="2025-01-01")
    
    # 注册自定义工具
    registry.register(MyCustomTool())
    
    # 获取 OpenAI 格式
    tools = registry.to_openai_tools()
"""

from typing import Dict, List, Optional, Callable, Any
from loguru import logger

# 使用绝对导入以支持独立测试
try:
    from .base import Tool, ToolResult
except ImportError:
    from stockbench.tools.base import Tool, ToolResult


class ToolRegistry:
    """
    工具注册中心
    
    管理所有可用的工具，提供统一的执行接口。
    
    Attributes:
        _tools: 已注册的工具字典 {name: Tool}
        _default_instance: 单例默认实例
    """
    
    _default_instance: Optional["ToolRegistry"] = None
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._initialized = False
    
    @classmethod
    def default(cls) -> "ToolRegistry":
        """
        获取默认注册中心（单例）
        
        首次调用时会自动注册内置工具。
        
        Returns:
            ToolRegistry: 默认注册中心实例
        """
        if cls._default_instance is None:
            cls._default_instance = cls()
            cls._default_instance._register_builtin_tools()
        return cls._default_instance
    
    @classmethod
    def reset_default(cls):
        """重置默认注册中心（主要用于测试）"""
        cls._default_instance = None
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        if self._initialized:
            return
        
        try:
            # 延迟导入避免循环依赖
            from .data_tools import (
                PriceDataTool,
                NewsDataTool,
                FinancialsTool,
                SnapshotTool,
                DividendsTool,
                SplitsTool,
            )
            
            self.register(PriceDataTool())
            self.register(NewsDataTool())
            self.register(FinancialsTool())
            self.register(SnapshotTool())
            self.register(DividendsTool())
            self.register(SplitsTool())
            
            self._initialized = True
            logger.debug(f"[ToolRegistry] Registered {len(self._tools)} builtin tools")
            
        except ImportError as e:
            logger.warning(f"[ToolRegistry] Failed to import builtin tools: {e}")
    
    def register(self, tool: Tool) -> "ToolRegistry":
        """
        注册工具
        
        Args:
            tool: 工具实例
            
        Returns:
            self: 支持链式调用
        """
        if tool.name in self._tools:
            logger.warning(f"[ToolRegistry] Overwriting existing tool: {tool.name}")
        
        self._tools[tool.name] = tool
        logger.debug(f"[ToolRegistry] Registered tool: {tool.name}")
        return self
    
    def unregister(self, name: str) -> bool:
        """
        注销工具
        
        Args:
            name: 工具名称
            
        Returns:
            bool: 是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"[ToolRegistry] Unregistered tool: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """
        获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            Tool 或 None
        """
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools
    
    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())
    
    def get_tools_by_tag(self, tag: str) -> List[Tool]:
        """
        按标签获取工具
        
        Args:
            tag: 标签名
            
        Returns:
            List[Tool]: 匹配的工具列表
        """
        return [t for t in self._tools.values() if tag in t.tags]
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            ToolResult: 执行结果
        """
        tool = self.get(name)
        if not tool:
            return ToolResult.fail(f"Tool '{name}' not found")
        
        logger.debug(f"[ToolRegistry] Executing tool: {name}")
        return tool.safe_run(**kwargs)
    
    def execute_many(self, calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """
        批量执行工具
        
        Args:
            calls: 工具调用列表，每项格式为 {"name": "tool_name", "args": {...}}
            
        Returns:
            List[ToolResult]: 执行结果列表
        """
        results = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {})
            result = self.execute(name, **args)
            results.append(result)
        return results
    
    def to_openai_tools(self, names: Optional[List[str]] = None) -> List[Dict]:
        """
        转换为 OpenAI Function Calling 格式
        
        Args:
            names: 可选，指定要转换的工具名称列表。为 None 时转换所有工具。
            
        Returns:
            List[Dict]: OpenAI tools 格式列表
        """
        if names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[n] for n in names if n in self._tools]
        
        return [tool.to_openai_schema() for tool in tools]
    
    def get_tools_description(self) -> str:
        """
        获取所有工具的文本描述（用于 prompt）
        
        Returns:
            str: 工具描述文本
        """
        lines = ["Available tools:"]
        for name, tool in self._tools.items():
            params = ", ".join(p.name for p in tool.get_parameters())
            lines.append(f"  - {name}({params}): {tool.description}")
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __iter__(self):
        return iter(self._tools.values())
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# 便捷函数
def get_default_registry() -> ToolRegistry:
    """获取默认注册中心"""
    return ToolRegistry.default()


def execute_tool(name: str, **kwargs) -> ToolResult:
    """使用默认注册中心执行工具"""
    return ToolRegistry.default().execute(name, **kwargs)


# 导出
__all__ = [
    "ToolRegistry",
    "get_default_registry",
    "execute_tool",
]
