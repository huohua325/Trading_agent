"""
工具系统测试

测试 Tool 基类、ToolRegistry 和数据工具的功能。

运行方式:
    python stockbench/tools/tests/test_tools.py
"""

import sys
import os

# 直接导入模块文件，避免触发重依赖
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)

import importlib.util

def _import_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_tools_dir = os.path.join(_root, "stockbench", "tools")
_base_mod = _import_from_file("base", os.path.join(_tools_dir, "base.py"))

# 导入基础组件
Tool = _base_mod.Tool
ToolParameter = _base_mod.ToolParameter
ToolParameterType = _base_mod.ToolParameterType
ToolResult = _base_mod.ToolResult


class TestToolParameter:
    """测试 ToolParameter"""
    
    def test_parameter_creation(self):
        """测试参数创建"""
        param = ToolParameter(
            name="symbol",
            type=ToolParameterType.STRING,
            description="股票代码"
        )
        assert param.name == "symbol"
        assert param.type == ToolParameterType.STRING
        assert param.required == True
    
    def test_parameter_with_default(self):
        """测试带默认值的参数"""
        param = ToolParameter(
            name="limit",
            type=ToolParameterType.INTEGER,
            description="返回数量",
            required=False,
            default=10
        )
        assert param.required == False
        assert param.default == 10
    
    def test_parameter_to_json_schema(self):
        """测试转换为 JSON Schema"""
        param = ToolParameter(
            name="action",
            type=ToolParameterType.STRING,
            description="操作类型",
            enum=["buy", "sell", "hold"]
        )
        schema = param.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "操作类型"
        assert schema["enum"] == ["buy", "sell", "hold"]


class TestToolResult:
    """测试 ToolResult"""
    
    def test_result_ok(self):
        """测试成功结果"""
        result = ToolResult.ok({"price": 150.0}, symbol="AAPL")
        assert result.success == True
        assert result.data["price"] == 150.0
        assert result.metadata["symbol"] == "AAPL"
        assert result.error is None
    
    def test_result_fail(self):
        """测试失败结果"""
        result = ToolResult.fail("API timeout", retries=3)
        assert result.success == False
        assert result.error == "API timeout"
        assert result.metadata["retries"] == 3
        assert result.data is None
    
    def test_result_bool(self):
        """测试布尔转换"""
        ok_result = ToolResult.ok("data")
        fail_result = ToolResult.fail("error")
        
        assert bool(ok_result) == True
        assert bool(fail_result) == False
        
        # if result: 语法
        if ok_result:
            passed = True
        else:
            passed = False
        assert passed == True


class TestTool:
    """测试 Tool 基类"""
    
    def test_custom_tool(self):
        """测试自定义工具"""
        
        class EchoTool(Tool):
            def __init__(self):
                super().__init__(
                    name="echo",
                    description="Echo the input"
                )
            
            def get_parameters(self):
                return [
                    ToolParameter("message", ToolParameterType.STRING, "Message to echo")
                ]
            
            def run(self, message: str, **kwargs):
                return ToolResult.ok(message)
        
        tool = EchoTool()
        assert tool.name == "echo"
        assert len(tool.get_parameters()) == 1
        
        result = tool.run(message="hello")
        assert result.success
        assert result.data == "hello"
    
    def test_safe_run_with_exception(self):
        """测试 safe_run 异常处理"""
        
        class FailingTool(Tool):
            def __init__(self):
                super().__init__(name="failing", description="Always fails")
            
            def get_parameters(self):
                return []
            
            def run(self, **kwargs):
                raise ValueError("Intentional error")
        
        tool = FailingTool()
        result = tool.safe_run()
        
        assert result.success == False
        assert "Intentional error" in result.error
        assert result.metadata.get("exception_type") == "ValueError"
    
    def test_validate_parameters(self):
        """测试参数验证"""
        
        class StrictTool(Tool):
            def __init__(self):
                super().__init__(name="strict", description="Strict tool")
            
            def get_parameters(self):
                return [
                    ToolParameter("required_param", ToolParameterType.STRING, "必需参数"),
                    ToolParameter("optional_param", ToolParameterType.STRING, "可选参数", required=False),
                ]
            
            def run(self, **kwargs):
                return ToolResult.ok("ok")
        
        tool = StrictTool()
        
        # 缺少必需参数
        error = tool.validate_parameters()
        assert error is not None
        assert "required_param" in error
        
        # 提供必需参数
        error = tool.validate_parameters(required_param="value")
        assert error is None
    
    def test_to_openai_schema(self):
        """测试 OpenAI Schema 转换"""
        
        class SearchTool(Tool):
            def __init__(self):
                super().__init__(
                    name="search_stocks",
                    description="Search for stocks by criteria"
                )
            
            def get_parameters(self):
                return [
                    ToolParameter("query", ToolParameterType.STRING, "Search query"),
                    ToolParameter("limit", ToolParameterType.INTEGER, "Max results", required=False, default=10),
                ]
            
            def run(self, **kwargs):
                return ToolResult.ok([])
        
        tool = SearchTool()
        schema = tool.to_openai_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_stocks"
        assert "parameters" in schema["function"]
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]
        assert "limit" not in schema["function"]["parameters"]["required"]


class TestToolRegistry:
    """测试 ToolRegistry（不依赖 data_hub）"""
    
    def test_registry_creation(self):
        """测试注册中心创建"""
        # 直接导入 registry 模块
        _registry_mod = _import_from_file("registry", os.path.join(_tools_dir, "registry.py"))
        ToolRegistry = _registry_mod.ToolRegistry
        
        registry = ToolRegistry()
        assert len(registry) == 0
    
    def test_register_and_get(self):
        """测试注册和获取"""
        _registry_mod = _import_from_file("registry", os.path.join(_tools_dir, "registry.py"))
        ToolRegistry = _registry_mod.ToolRegistry
        
        class DummyTool(Tool):
            def __init__(self):
                super().__init__(name="dummy", description="Dummy tool")
            def get_parameters(self):
                return []
            def run(self, **kwargs):
                return ToolResult.ok("dummy result")
        
        registry = ToolRegistry()
        registry.register(DummyTool())
        
        assert registry.has("dummy")
        assert not registry.has("nonexistent")
        
        tool = registry.get("dummy")
        assert tool is not None
        assert tool.name == "dummy"
    
    def test_execute(self):
        """测试执行工具"""
        _registry_mod = _import_from_file("registry", os.path.join(_tools_dir, "registry.py"))
        ToolRegistry = _registry_mod.ToolRegistry
        
        class AddTool(Tool):
            def __init__(self):
                super().__init__(name="add", description="Add two numbers")
            def get_parameters(self):
                return [
                    ToolParameter("a", ToolParameterType.NUMBER, "First number"),
                    ToolParameter("b", ToolParameterType.NUMBER, "Second number"),
                ]
            def run(self, a: float, b: float, **kwargs):
                return ToolResult.ok(a + b)
        
        registry = ToolRegistry()
        registry.register(AddTool())
        
        result = registry.execute("add", a=3, b=5)
        assert result.success
        assert result.data == 8
        
        # 测试不存在的工具
        result = registry.execute("nonexistent")
        assert not result.success
        assert "not found" in result.error
    
    def test_to_openai_tools(self):
        """测试批量转换为 OpenAI 格式"""
        _registry_mod = _import_from_file("registry", os.path.join(_tools_dir, "registry.py"))
        ToolRegistry = _registry_mod.ToolRegistry
        
        class Tool1(Tool):
            def __init__(self):
                super().__init__(name="tool1", description="Tool 1")
            def get_parameters(self):
                return []
            def run(self, **kwargs):
                return ToolResult.ok(1)
        
        class Tool2(Tool):
            def __init__(self):
                super().__init__(name="tool2", description="Tool 2")
            def get_parameters(self):
                return []
            def run(self, **kwargs):
                return ToolResult.ok(2)
        
        registry = ToolRegistry()
        registry.register(Tool1())
        registry.register(Tool2())
        
        tools = registry.to_openai_tools()
        assert len(tools) == 2
        
        names = [t["function"]["name"] for t in tools]
        assert "tool1" in names
        assert "tool2" in names


def run_tests_standalone():
    """直接运行测试"""
    import traceback
    
    test_classes = [
        TestToolParameter,
        TestToolResult,
        TestTool,
        TestToolRegistry,
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
    success = run_tests_standalone()
    sys.exit(0 if success else 1)
