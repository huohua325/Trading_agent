"""
Tool 基类和类型定义

提供工具系统的基础抽象：
- ToolParameterType: 参数类型枚举
- ToolParameter: 工具参数定义
- ToolResult: 工具执行结果
- Tool: 工具抽象基类

所有数据获取工具都应继承 Tool 基类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime


class ToolParameterType(str, Enum):
    """
    工具参数类型枚举
    
    对应 JSON Schema 类型，用于 OpenAI Function Calling 格式转换。
    """
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """
    工具参数定义
    
    Attributes:
        name: 参数名称
        type: 参数类型
        description: 参数描述
        required: 是否必需，默认 True
        default: 默认值
        enum: 可选的枚举值列表
    """
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None
    
    def to_json_schema(self) -> Dict[str, Any]:
        """转换为 JSON Schema 格式"""
        schema = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolResult:
    """
    工具执行结果
    
    Attributes:
        success: 是否成功
        data: 返回数据（成功时）
        error: 错误信息（失败时）
        metadata: 附加元数据（耗时、缓存命中等）
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any, **metadata) -> "ToolResult":
        """创建成功结果"""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, **metadata) -> "ToolResult":
        """创建失败结果"""
        return cls(success=False, error=error, metadata=metadata)
    
    def __bool__(self) -> bool:
        """支持 if result: 语法"""
        return self.success


class Tool(ABC):
    """
    工具抽象基类
    
    所有工具都应继承此类并实现：
    - run(): 执行工具逻辑
    - get_parameters(): 定义工具参数
    
    Attributes:
        name: 工具名称（唯一标识）
        description: 工具描述（用于 LLM 理解工具功能）
        version: 工具版本
        tags: 工具标签（用于分类）
    
    使用示例:
        class MyTool(Tool):
            def __init__(self):
                super().__init__(
                    name="my_tool",
                    description="我的工具描述"
                )
            
            def get_parameters(self) -> List[ToolParameter]:
                return [
                    ToolParameter("param1", ToolParameterType.STRING, "参数1描述"),
                ]
            
            def run(self, param1: str, **kwargs) -> ToolResult:
                return ToolResult.ok({"result": param1})
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.version = version
        self.tags = tags or []
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolResult: 执行结果
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """
        获取工具参数定义
        
        Returns:
            List[ToolParameter]: 参数定义列表
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        验证参数
        
        Args:
            **kwargs: 待验证的参数
            
        Returns:
            None 如果验证通过，否则返回错误信息
        """
        params = {p.name: p for p in self.get_parameters()}
        
        # 检查必需参数
        for name, param in params.items():
            if param.required and name not in kwargs:
                return f"Missing required parameter: {name}"
        
        # 检查枚举值
        for name, value in kwargs.items():
            if name in params and params[name].enum:
                if value not in params[name].enum:
                    return f"Invalid value for {name}: {value}, must be one of {params[name].enum}"
        
        return None
    
    def safe_run(self, **kwargs) -> ToolResult:
        """
        安全执行工具（带参数验证和异常捕获）
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolResult: 执行结果
        """
        # 验证参数
        error = self.validate_parameters(**kwargs)
        if error:
            return ToolResult.fail(error)
        
        # 执行工具
        start_time = datetime.now()
        try:
            result = self.run(**kwargs)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            result.metadata["duration_ms"] = duration_ms
            result.metadata["tool_name"] = self.name
            return result
        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return ToolResult.fail(
                str(e),
                duration_ms=duration_ms,
                tool_name=self.name,
                exception_type=type(e).__name__
            )
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换为 OpenAI Function Calling 格式
        
        Returns:
            Dict: OpenAI function schema
        """
        parameters = self.get_parameters()
        
        properties = {}
        required = []
        
        for param in parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    def __repr__(self) -> str:
        return f"Tool(name={self.name}, version={self.version})"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


# 导出
__all__ = [
    "ToolParameterType",
    "ToolParameter",
    "ToolResult",
    "Tool",
]
