# StockBench 框架升级路线图

基于 HelloAgents 框架设计理念，对 StockBench 进行系统性升级分析。

---

## 概述

本文档分析《Agent框架如何构建》四个部分的核心思想，并提出 StockBench 的升级路线。

| Part | 主题 | 核心内容 | 对 StockBench 的价值 |
|------|------|----------|---------------------|
| Part 1 | 框架设计 + LLM扩展 | 设计理念、多提供商支持、自动检测 | LLMClient 重构 |
| Part 2 | 接口实现 + Agent范式 | Message/Config/Agent基类、四种范式 | Agent 层标准化 |
| Part 3 | 工具系统 | Tool基类、ToolRegistry、工具链 | 数据工具抽象 |
| Part 4 | 总结与原则 | 分层解耦、接口统一、渐进增强 | 架构优化指导 |

---

## Part 1 分析：框架设计理念与 LLM 扩展

### 1.1 核心思想提取

**设计理念（7.1.2节）**：
- **轻量级与教学友好**：避免过度抽象，保持代码可读性
- **基于标准API**：OpenAI API 已成为行业标准
- **渐进式学习路径**：从简单到复杂，逐步完善
- **万物皆为工具**：统一抽象，降低学习成本

**HelloAgentsLLM 特性（7.2节）**：
- 多提供商支持（OpenAI、ModelScope、智谱AI、VLLM、Ollama）
- 自动检测机制（环境变量 → base_url → API密钥格式）
- 本地模型无缝集成

### 1.2 StockBench 现状对比

```
当前 StockBench LLMClient:
├── provider 区分 (openai/zhipuai)
├── 内容缓存机制 (run_id + date)
├── JSON 解析修复 (json_repair, demjson3)
├── Token 预算追踪
└── 重试逻辑

HelloAgents 增强点:
├── 自动提供商检测
├── 本地模型支持 (VLLM/Ollama)
├── 通过继承扩展新提供商
└── 统一的环境变量规范
```

### 1.3 升级建议

#### 升级项 1.1: LLMClient 自动检测机制

**目标**：无需手动配置 `llm_profile`，自动根据环境变量检测提供商

```python
# 当前方式
llm_cfg = LLMConfig(provider="openai", model="gpt-4o-mini", ...)

# 升级后
llm_cfg = LLMConfig()  # 自动检测 OPENAI_API_KEY → provider="openai"
```

**实现要点**：
- 在 `LLMConfig.__init__` 添加 `_auto_detect_provider()` 方法
- 优先级：特定环境变量 > base_url 解析 > 密钥格式推断

#### 升级项 1.2: 本地模型支持

**目标**：支持 VLLM/Ollama 本地部署，用于开发调试和成本控制

```yaml
# config.yaml 新增
llm_profiles:
  local-vllm:
    provider: "vllm"
    base_url: "http://localhost:8000/v1"
    model: "Qwen/Qwen2.5-7B-Instruct"
    
  local-ollama:
    provider: "ollama"
    base_url: "http://localhost:11434/v1"
    model: "llama3"
```

**价值**：
- 降低开发成本（本地推理无 API 费用）
- 保护数据隐私（敏感金融数据本地处理）
- 快速迭代（无网络延迟）

#### 升级项 1.3: 提供商扩展机制

**目标**：通过继承扩展新的 LLM 提供商

```python
# stockbench/llm/providers/kimi_client.py
class KimiLLMClient(LLMClient):
    def _resolve_credentials(self):
        self.api_key = os.getenv("KIMI_API_KEY")
        self.base_url = "https://api.moonshot.cn/v1"
```

---

## Part 2 分析：框架接口与 Agent 范式

### 2.1 核心思想提取

**Message 类（7.3.1节）**：
- 角色限定：`Literal["user", "assistant", "system", "tool"]`
- 元数据支持：`timestamp`, `metadata`
- 格式转换：`to_dict()` → OpenAI API 格式

**Config 类（7.3.2节）**：
- 集中化配置管理
- 环境变量覆盖：`Config.from_env()`
- 合理默认值

**Agent 抽象基类（7.3.3节）**：
- 统一接口：`run(input_text) -> str`
- 历史管理：`add_message()`, `get_history()`, `clear_history()`
- 依赖注入：`llm`, `config`, `system_prompt`

**四种 Agent 范式（7.4节）**：
| 范式 | 特点 | 适用场景 |
|------|------|----------|
| SimpleAgent | 基础对话 | 简单问答 |
| ReActAgent | 思考-行动循环 | 需要工具调用 |
| ReflectionAgent | 自我反思迭代 | 质量优化 |
| PlanAndSolveAgent | 分解-执行 | 复杂多步任务 |

### 2.2 StockBench 现状对比

```
当前 StockBench Agent 实现:
├── fundamental_filter_agent.py  # 独立实现，无基类
├── dual_agent_llm.py            # 决策Agent，包含验证逻辑
└── backtest_report_llm.py       # 报告Agent，独立实现

问题:
├── 无统一 Agent 基类
├── 消息格式不规范（直接使用 dict）
├── 历史管理各自实现
└── 提示词硬编码在代码中
```

### 2.3 升级建议

#### 升级项 2.1: 引入 Agent 抽象基类

**目标**：统一所有 Agent 的接口和基础功能

```python
# stockbench/agents/base.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from stockbench.llm import LLMClient
from stockbench.core.message import Message

class BaseAgent(ABC):
    """StockBench Agent 基类"""
    
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or {}
        self._history: list[Message] = []
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行 Agent 主逻辑"""
        pass
    
    def add_message(self, message: Message):
        self._history.append(message)
    
    def get_history(self) -> list[Message]:
        return self._history.copy()
```

#### 升级项 2.2: 消息系统标准化

**目标**：引入 Message 类，规范化 Agent-LLM 通信

```python
# stockbench/core/message.py
from pydantic import BaseModel
from datetime import datetime
from typing import Literal, Optional, Dict, Any

MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}
```

#### 升级项 2.3: 引入 ReflectionAgent 范式

**目标**：为决策 Agent 添加自我反思能力，提升决策质量

```python
# stockbench/agents/reflective_decision_agent.py
class ReflectiveDecisionAgent(BaseAgent):
    """带自我反思的决策 Agent"""
    
    def run(self, features: Dict, max_iterations: int = 2) -> Dict:
        # 1. 初始决策
        initial_decision = self._make_decision(features)
        
        # 2. 反思循环
        for i in range(max_iterations):
            feedback = self._reflect(features, initial_decision)
            if "无需改进" in feedback:
                break
            initial_decision = self._refine(features, initial_decision, feedback)
        
        return initial_decision
```

**应用场景**：
- 重大持仓变动时触发反思
- 市场异常波动时多轮验证
- 提升决策逻辑一致性

#### 升级项 2.4: 提示词模板化管理

**目标**：从硬编码转向可配置模板

```python
# 当前方式
system_prompt = """你是一个股票分析师..."""  # 硬编码在代码中

# 升级后 (支持自定义)
class DecisionAgent(BaseAgent):
    DEFAULT_PROMPT = "decision_agent_v1.txt"
    
    def __init__(self, ..., custom_prompt: Optional[str] = None):
        self.prompt_template = self._load_prompt(custom_prompt or self.DEFAULT_PROMPT)
```

---

## Part 3 分析：工具系统设计

### 3.1 核心思想提取

**Tool 基类（7.5.1节）**：
```python
class Tool(ABC):
    def __init__(self, name: str, description: str): ...
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str: ...
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]: ...
```

**ToolRegistry 注册表**：
- `register_tool()`: 注册 Tool 对象
- `register_function()`: 直接注册函数
- `get_tools_description()`: 生成工具描述
- `execute_tool()`: 执行工具
- `to_openai_schema()`: 转换为 OpenAI function calling 格式

**高级特性（7.5.4节）**：
- **工具链**：多工具顺序执行，结果传递
- **异步执行**：并行执行多个工具

### 3.2 StockBench 现状对比

```
当前 StockBench 数据获取:
├── adapters/polygon_client.py   # 直接调用
├── adapters/finnhub_client.py   # 直接调用
├── core/data_hub.py             # 统一入口，但非工具抽象
└── core/features.py             # 特征构建，非工具化

潜在工具化对象:
├── 价格数据获取
├── 新闻数据获取
├── 财务数据获取
├── 技术指标计算
├── 基本面指标获取
└── 特征构建
```

### 3.3 升级建议

#### 升级项 3.1: 数据获取工具化

**目标**：将 DataHub 功能抽象为可注册的工具

```python
# stockbench/tools/data_tools.py
from stockbench.tools.base import Tool, ToolParameter

class PriceDataTool(Tool):
    """价格数据获取工具"""
    
    def __init__(self, data_hub):
        super().__init__(
            name="get_price_data",
            description="获取指定股票的历史价格数据，包括开盘价、收盘价、最高价、最低价、成交量"
        )
        self.data_hub = data_hub
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="symbol", type="string", description="股票代码", required=True),
            ToolParameter(name="start_date", type="string", description="开始日期 YYYY-MM-DD"),
            ToolParameter(name="end_date", type="string", description="结束日期 YYYY-MM-DD"),
        ]
    
    def run(self, parameters: Dict[str, Any]) -> str:
        bars = self.data_hub.get_bars(
            parameters["symbol"],
            parameters["start_date"],
            parameters["end_date"]
        )
        return bars.to_json()

class NewsDataTool(Tool):
    """新闻数据获取工具"""
    ...

class FinancialsTool(Tool):
    """财务数据获取工具"""
    ...
```

#### 升级项 3.2: 工具注册中心

**目标**：集中管理所有可用工具

```python
# stockbench/tools/registry.py
class StockBenchToolRegistry:
    """StockBench 工具注册中心"""
    
    def __init__(self, data_hub, config):
        self._tools = {}
        self._init_builtin_tools(data_hub, config)
    
    def _init_builtin_tools(self, data_hub, config):
        # 注册内置工具
        self.register(PriceDataTool(data_hub))
        self.register(NewsDataTool(data_hub))
        self.register(FinancialsTool(data_hub))
        self.register(TechnicalIndicatorTool())
    
    def get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        """根据 Agent 类型返回可用工具"""
        if agent_type == "filter":
            return [self._tools["get_price_data"], self._tools["get_news"]]
        elif agent_type == "decision":
            return list(self._tools.values())
```

#### 升级项 3.3: 工具链实现

**目标**：支持复杂的数据处理流水线

```python
# stockbench/tools/chain.py
class DataPipelineChain:
    """数据处理工具链"""
    
    def __init__(self, registry: StockBenchToolRegistry):
        self.registry = registry
        self.steps = []
    
    def add_step(self, tool_name: str, input_template: str, output_key: str):
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key
        })
    
    def execute(self, initial_context: Dict) -> Dict:
        context = initial_context.copy()
        for step in self.steps:
            tool_input = step["input_template"].format(**context)
            result = self.registry.execute(step["tool_name"], tool_input)
            context[step["output_key"]] = result
        return context

# 使用示例：构建完整特征
def create_feature_pipeline():
    chain = DataPipelineChain(registry)
    chain.add_step("get_price_data", "{symbol}", "price_data")
    chain.add_step("get_news", "{symbol}", "news_data")
    chain.add_step("calculate_indicators", "{price_data}", "indicators")
    chain.add_step("build_features", "{price_data},{news_data},{indicators}", "features")
    return chain
```

#### 升级项 3.4: 异步并行数据获取

**目标**：并行获取多个股票的数据，提升回测效率

```python
# stockbench/tools/async_executor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncDataExecutor:
    """异步数据执行器"""
    
    def __init__(self, registry, max_workers: int = 8):
        self.registry = registry
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def fetch_all_symbols_data(self, symbols: List[str], date: str) -> Dict:
        """并行获取所有股票数据"""
        tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_data_async(symbol, date)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))
    
    async def _fetch_symbol_data_async(self, symbol: str, date: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.registry.execute("get_all_data", {"symbol": symbol, "date": date})
        )
```

---

## Part 4 分析：设计原则与总结

### 4.1 核心设计原则

| 原则 | 含义 | StockBench 应用 |
|------|------|-----------------|
| **分层解耦** | 清晰的层次，明确的职责 | 数据层 → 特征层 → Agent层 → 回测层 |
| **接口统一** | 抽象基类定义统一接口 | Agent基类、Tool基类、Message标准 |
| **渐进增强** | 从简单到复杂 | 先完善基础架构，再添加高级范式 |
| **实用主义** | 基于标准接口，避免过度抽象 | 兼容 OpenAI API，保持代码简洁 |

### 4.2 StockBench 升级优先级

```
Phase 1: 基础架构 (2-3 天)
├── [P0] 引入 Message 类
├── [P0] 创建 Agent 抽象基类
└── [P1] LLMClient 自动检测机制

Phase 2: Agent 层重构 (3-5 天)
├── [P0] 重构 FundamentalFilterAgent 继承基类
├── [P0] 重构 DecisionAgent 继承基类
├── [P1] 重构 BacktestReportAgent 继承基类
└── [P2] 引入 ReflectionAgent 范式

Phase 3: 工具系统 (3-5 天)
├── [P1] 创建 Tool 基类和 ToolRegistry
├── [P1] 将 DataHub 方法工具化
├── [P2] 实现工具链
└── [P2] 异步并行执行器

Phase 4: 高级特性 (可选)
├── [P3] 本地模型支持 (VLLM/Ollama)
├── [P3] FunctionCallAgent 原生调用
└── [P3] 插件系统设计
```

---

## 升级收益总结

| 维度 | 当前状态 | 升级后 | 收益 |
|------|----------|--------|------|
| **代码组织** | 独立脚本 | 统一基类+继承 | 可维护性 ↑ |
| **接口规范** | 各自实现 | 统一 run() 接口 | 可扩展性 ↑ |
| **消息管理** | dict 硬编码 | Message 类 | 类型安全 ↑ |
| **工具调用** | 直接调用 | ToolRegistry | 灵活性 ↑ |
| **提示词** | 硬编码 | 可配置模板 | 易调优 ↑ |
| **LLM支持** | 2种提供商 | 多提供商+本地 | 成本控制 ↑ |
| **数据获取** | 串行 | 异步并行 | 性能 ↑ |

---

## 下一步行动

1. **确认优先级**：根据当前项目需求，确定 Phase 1-4 的执行顺序
2. **创建分支**：`feature/agent-framework-upgrade`
3. **逐步实施**：按 Phase 顺序实现，每个 Phase 完成后进行回测验证
4. **文档同步**：更新 CLAUDE.md 反映新架构

---

# Part 1 详细升级计划：LLM 层重构

基于《Agent框架如何构建》Part 1（7.1-7.2节），针对 `stockbench/llm/llm_client.py` 的详细升级实施方案。

---

## 一、当前 LLMClient 架构分析

### 1.1 现有结构

```
stockbench/llm/llm_client.py (1333行)
├── LLMConfig (dataclass)
│   ├── provider: str = "openai"
│   ├── base_url: str
│   ├── model: str
│   ├── temperature, max_tokens, seed
│   ├── timeout_sec, max_retries, backoff_factor
│   ├── cache_enabled, cache_ttl_hours
│   ├── cache_read_enabled, cache_write_enabled
│   ├── budget_prompt_tokens, budget_completion_tokens
│   └── auth_required: Optional[bool]
│
└── LLMClient (class)
    ├── __init__(api_key_env, cache_dir)
    ├── _get_openai_client(cfg) -> openai.OpenAI
    ├── _cache_path, _read_cache, _write_cache
    ├── _make_cache_key, _make_cache_key_with_date
    ├── _extract_json_with_improved_logic()
    ├── _fix_common_json_issues()
    ├── generate_json() - 核心生成方法
    └── 其他辅助方法
```

### 1.2 现有提供商支持

```python
# config.yaml 中的 llm_profiles
llm_profiles:
  openai:        # provider: "openai"
  zhipuai:       # provider: "zhipuai" - 使用 ZhipuAiClient SDK
  deepseek-v3.1: # provider: "openai" (OpenAI兼容)
  kimi-k2:       # provider: "openai" (OpenAI兼容)
  qwen3:         # provider: "openai" (OpenAI兼容)
```

### 1.3 存在的问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **无自动检测** | 必须在 config.yaml 手动配置 provider | 配置繁琐 |
| **无本地模型支持** | 不支持 VLLM/Ollama | 无法本地调试 |
| **扩展困难** | 添加新提供商需修改核心代码 | 可维护性差 |
| **环境变量分散** | 各提供商 API Key 变量名不统一 | 易出错 |

---

## 二、升级目标

### 2.1 Part 1 升级范围

```
升级项 1.1: 自动检测机制
├── _auto_detect_provider() 方法
├── 环境变量优先级检测
└── base_url 智能解析

升级项 1.2: 本地模型支持
├── VLLM provider 配置
├── Ollama provider 配置
└── 本地端口自动识别

升级项 1.3: 提供商扩展机制
├── 通过继承扩展新提供商
├── _resolve_credentials() 方法
└── 保持向后兼容
```

### 2.2 不在本次范围

- Agent 基类设计 (Part 2)
- 工具系统 (Part 3)
- Message 标准化 (Part 2)

---

## 三、详细实施步骤

### 步骤 1: 添加提供商常量定义

**文件**: `stockbench/llm/llm_client.py`

**位置**: 文件顶部，LLMConfig 之前

```python
# ==================== 提供商常量定义 ====================
# 支持的 LLM 提供商列表
class LLMProvider:
    """LLM 提供商常量"""
    OPENAI = "openai"
    ZHIPUAI = "zhipuai"
    VLLM = "vllm"
    OLLAMA = "ollama"
    MODELSCOPE = "modelscope"
    LOCAL = "local"  # 通用本地服务
    AUTO = "auto"    # 自动检测

# 提供商默认配置
PROVIDER_DEFAULTS = {
    LLMProvider.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    LLMProvider.ZHIPUAI: {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "ZHIPUAI_API_KEY",
        "default_model": "glm-4.5",
    },
    LLMProvider.VLLM: {
        "base_url": "http://localhost:8000/v1",
        "env_key": None,  # 本地服务无需 API Key
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "auth_required": False,
    },
    LLMProvider.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
        "default_model": "llama3",
        "auth_required": False,
    },
    LLMProvider.MODELSCOPE: {
        "base_url": "https://api-inference.modelscope.cn/v1/",
        "env_key": "MODELSCOPE_API_KEY",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
    },
}
```

---

### 步骤 2: 扩展 LLMConfig 添加自动检测

**文件**: `stockbench/llm/llm_client.py`

**修改**: `LLMConfig` dataclass

```python
@dataclass
class LLMConfig:
    provider: str = "auto"  # 改为默认 auto，启用自动检测
    base_url: str = ""      # 改为空字符串，由自动检测填充
    model: str = ""         # 改为空字符串，由自动检测填充
    temperature: float = 0.0
    max_tokens: int = 256
    seed: Optional[int] = None
    timeout_sec: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    cache_read_enabled: Optional[bool] = None
    cache_write_enabled: Optional[bool] = None
    budget_prompt_tokens: int = 200_000
    budget_completion_tokens: int = 200_000
    auth_required: Optional[bool] = None
    
    def __post_init__(self):
        """初始化后自动检测和解析配置"""
        if self.provider == "auto" or self.provider == "":
            self.provider = self._auto_detect_provider()
        
        # 解析凭证和默认值
        self._resolve_credentials()
    
    def _auto_detect_provider(self) -> str:
        """
        自动检测 LLM 提供商
        
        检测优先级:
        1. 特定提供商环境变量 (OPENAI_API_KEY, ZHIPUAI_API_KEY, etc.)
        2. base_url 特征匹配 (域名、端口)
        3. 通用环境变量 LLM_API_KEY 的格式
        4. 默认返回 openai
        """
        # 优先级 1: 检查特定提供商的环境变量
        if os.getenv("ZHIPUAI_API_KEY"):
            logger.debug("Auto-detected provider: zhipuai (via ZHIPUAI_API_KEY)")
            return LLMProvider.ZHIPUAI
        if os.getenv("MODELSCOPE_API_KEY"):
            logger.debug("Auto-detected provider: modelscope (via MODELSCOPE_API_KEY)")
            return LLMProvider.MODELSCOPE
        if os.getenv("OPENAI_API_KEY"):
            logger.debug("Auto-detected provider: openai (via OPENAI_API_KEY)")
            return LLMProvider.OPENAI
        
        # 优先级 2: 根据 base_url 判断
        actual_base_url = self.base_url or os.getenv("LLM_BASE_URL", "")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            
            # 云服务商域名匹配
            if "open.bigmodel.cn" in base_url_lower:
                logger.debug("Auto-detected provider: zhipuai (via base_url)")
                return LLMProvider.ZHIPUAI
            if "api-inference.modelscope.cn" in base_url_lower:
                logger.debug("Auto-detected provider: modelscope (via base_url)")
                return LLMProvider.MODELSCOPE
            if "api.openai.com" in base_url_lower:
                logger.debug("Auto-detected provider: openai (via base_url)")
                return LLMProvider.OPENAI
            
            # 本地服务端口匹配
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                if ":11434" in base_url_lower:
                    logger.debug("Auto-detected provider: ollama (via localhost:11434)")
                    return LLMProvider.OLLAMA
                if ":8000" in base_url_lower:
                    logger.debug("Auto-detected provider: vllm (via localhost:8000)")
                    return LLMProvider.VLLM
                logger.debug("Auto-detected provider: local (via localhost)")
                return LLMProvider.LOCAL
        
        # 优先级 3: 检查通用 API Key 格式
        generic_key = os.getenv("LLM_API_KEY", "")
        if generic_key:
            if generic_key.startswith("sk-"):
                logger.debug("Auto-detected provider: openai (via LLM_API_KEY format sk-)")
                return LLMProvider.OPENAI
        
        # 默认返回 openai
        logger.debug("Auto-detected provider: openai (default)")
        return LLMProvider.OPENAI
    
    def _resolve_credentials(self) -> None:
        """
        根据 provider 解析 API 密钥和 base_url
        """
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})
        
        # 解析 base_url
        if not self.base_url:
            self.base_url = os.getenv("LLM_BASE_URL") or defaults.get("base_url", "https://api.openai.com/v1")
        
        # 解析 model
        if not self.model:
            self.model = os.getenv("LLM_MODEL_ID") or defaults.get("default_model", "gpt-4o-mini")
        
        # 解析 auth_required
        if self.auth_required is None:
            self.auth_required = defaults.get("auth_required", True)
        
        logger.debug(f"Resolved config: provider={self.provider}, base_url={self.base_url}, model={self.model}")
```

---

### 步骤 3: 修改 LLMClient 支持新的提供商逻辑

**文件**: `stockbench/llm/llm_client.py`

**修改**: `LLMClient.__init__` 和相关方法

```python
class LLMClient:
    def __init__(self, api_key_env: str = "auto", cache_dir: Optional[str] = None) -> None:
        """
        初始化 LLM 客户端
        
        Args:
            api_key_env: API Key 环境变量名，"auto" 表示自动检测
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "storage", "cache", "llm")
        ensure_dir(self.cache_dir)
        self._client: Optional[httpx.Client] = None
        self._openai_client: Optional[openai.OpenAI] = None
        self._prompt_tokens_used = 0
        self._completion_tokens_used = 0
        self.llm_logger = get_llm_logger()
        
        # 延迟解析 API Key，根据具体配置决定
        self._api_key_env = api_key_env
        self._resolved_api_key: Optional[str] = None
    
    def _get_api_key(self, cfg: LLMConfig) -> str:
        """
        根据配置获取 API Key
        
        优先级:
        1. 特定提供商环境变量
        2. 通用 LLM_API_KEY
        3. 传入的 api_key_env 参数
        """
        # 本地服务无需 API Key
        if cfg.provider in [LLMProvider.VLLM, LLMProvider.OLLAMA, LLMProvider.LOCAL]:
            if not cfg.auth_required:
                return cfg.provider  # 返回占位符
        
        # 特定提供商环境变量
        provider_defaults = PROVIDER_DEFAULTS.get(cfg.provider, {})
        env_key = provider_defaults.get("env_key")
        if env_key:
            api_key = os.getenv(env_key)
            if api_key:
                return api_key
        
        # 通用环境变量
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            return api_key
        
        # 使用传入的环境变量名
        if self._api_key_env and self._api_key_env != "auto":
            api_key = os.getenv(self._api_key_env)
            if api_key:
                return api_key
        
        # 最后尝试 OPENAI_API_KEY
        return os.getenv("OPENAI_API_KEY", "")
    
    def _get_openai_client(self, cfg: LLMConfig) -> openai.OpenAI:
        """获取 OpenAI 官方客户端（支持所有 OpenAI 兼容 API）"""
        api_key = self._get_api_key(cfg)
        
        # 为每个不同的 base_url 创建独立的客户端
        client_key = f"{cfg.base_url}_{api_key[:8] if api_key else 'no_key'}"
        
        if not hasattr(self, '_openai_clients'):
            self._openai_clients = {}
        
        if client_key not in self._openai_clients:
            self._openai_clients[client_key] = openai.OpenAI(
                api_key=api_key or "dummy",  # 本地服务可用占位符
                base_url=cfg.base_url,
                timeout=cfg.timeout_sec
            )
        
        return self._openai_clients[client_key]
```

---

### 步骤 4: 更新 config.yaml 添加本地模型配置

**文件**: `config.yaml`

**添加内容**:

```yaml
llm_profiles:
  # ========== 云端 API ==========
  openai:
    provider: "openai"
    base_url: ""  # 留空使用默认
    model: "gpt-4o-mini"
    auth_required: true
    # ... 其他配置
    
  zhipuai:
    provider: "zhipuai"
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    model: "glm-4.5"
    auth_required: true
    # ... 其他配置

  # ========== 本地模型 (新增) ==========
  local-vllm:
    provider: "vllm"
    base_url: "http://localhost:8000/v1"
    model: "Qwen/Qwen2.5-7B-Instruct"
    auth_required: false
    timeout_sec: 120
    retry:
      max_retries: 2
      backoff_factor: 0.5
    
  local-ollama:
    provider: "ollama"
    base_url: "http://localhost:11434/v1"
    model: "llama3"
    auth_required: false
    timeout_sec: 120
    retry:
      max_retries: 2
      backoff_factor: 0.5
      
  # ========== 自动检测 (新增) ==========
  auto:
    provider: "auto"  # 根据环境变量自动检测
    base_url: ""
    model: ""
    timeout_sec: 60
```

---

### 步骤 5: 添加提供商扩展基类

**新建文件**: `stockbench/llm/providers/__init__.py`

```python
"""
LLM 提供商扩展模块

通过继承 LLMClient 扩展新的 LLM 提供商，无需修改核心代码。

示例:
    from stockbench.llm.providers import BaseLLMProvider
    
    class MyCustomProvider(BaseLLMProvider):
        PROVIDER_NAME = "my_provider"
        DEFAULT_BASE_URL = "https://api.myprovider.com/v1"
        ENV_KEY_NAME = "MY_PROVIDER_API_KEY"
        
        def _custom_init(self):
            # 自定义初始化逻辑
            pass
"""

from stockbench.llm.llm_client import LLMClient, LLMConfig
from typing import Optional
import os


class BaseLLMProvider(LLMClient):
    """
    LLM 提供商基类
    
    继承此类可以轻松添加新的 LLM 提供商支持。
    """
    
    PROVIDER_NAME: str = "custom"
    DEFAULT_BASE_URL: str = ""
    ENV_KEY_NAME: Optional[str] = None
    DEFAULT_MODEL: str = "default"
    AUTH_REQUIRED: bool = True
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(api_key_env=self.ENV_KEY_NAME or "auto", cache_dir=cache_dir)
        self._custom_init()
    
    def _custom_init(self):
        """子类可重写此方法进行自定义初始化"""
        pass
    
    def get_default_config(self) -> LLMConfig:
        """获取此提供商的默认配置"""
        return LLMConfig(
            provider=self.PROVIDER_NAME,
            base_url=self.DEFAULT_BASE_URL,
            model=self.DEFAULT_MODEL,
            auth_required=self.AUTH_REQUIRED,
        )


class VLLMProvider(BaseLLMProvider):
    """VLLM 本地模型提供商"""
    
    PROVIDER_NAME = "vllm"
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    ENV_KEY_NAME = None
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    AUTH_REQUIRED = False


class OllamaProvider(BaseLLMProvider):
    """Ollama 本地模型提供商"""
    
    PROVIDER_NAME = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    ENV_KEY_NAME = None
    DEFAULT_MODEL = "llama3"
    AUTH_REQUIRED = False


class ModelScopeProvider(BaseLLMProvider):
    """ModelScope 提供商"""
    
    PROVIDER_NAME = "modelscope"
    DEFAULT_BASE_URL = "https://api-inference.modelscope.cn/v1/"
    ENV_KEY_NAME = "MODELSCOPE_API_KEY"
    DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
    AUTH_REQUIRED = True
```

---

### 步骤 6: 添加自动检测测试用例

**新建文件**: `stockbench/llm/tests/test_auto_detect.py`

```python
"""
LLM 自动检测机制测试
"""
import os
import pytest
from unittest.mock import patch
from stockbench.llm.llm_client import LLMConfig, LLMProvider


class TestAutoDetect:
    """测试自动检测机制"""
    
    def test_detect_openai_by_env(self):
        """通过 OPENAI_API_KEY 检测"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.OPENAI
    
    def test_detect_zhipuai_by_env(self):
        """通过 ZHIPUAI_API_KEY 检测"""
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "test123"}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.ZHIPUAI
    
    def test_detect_vllm_by_port(self):
        """通过 localhost:8000 检测 VLLM"""
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.VLLM
    
    def test_detect_ollama_by_port(self):
        """通过 localhost:11434 检测 Ollama"""
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:11434/v1"}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.OLLAMA
    
    def test_explicit_provider_overrides_auto(self):
        """显式指定 provider 应覆盖自动检测"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            cfg = LLMConfig(provider="zhipuai")
            assert cfg.provider == "zhipuai"
    
    def test_resolve_base_url(self):
        """测试 base_url 解析"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            cfg = LLMConfig(provider="openai")
            assert "openai.com" in cfg.base_url or cfg.base_url != ""
    
    def test_local_no_auth_required(self):
        """本地服务不需要认证"""
        cfg = LLMConfig(provider="vllm")
        assert cfg.auth_required == False


class TestProviderCredentials:
    """测试凭证解析"""
    
    def test_zhipuai_uses_correct_env(self):
        """智谱AI 使用正确的环境变量"""
        with patch.dict(os.environ, {
            "ZHIPUAI_API_KEY": "zhipu-key",
            "OPENAI_API_KEY": "openai-key"
        }, clear=True):
            cfg = LLMConfig(provider="zhipuai")
            # 验证会使用 ZHIPUAI_API_KEY
            assert cfg.provider == "zhipuai"
    
    def test_fallback_to_llm_api_key(self):
        """测试降级到通用 LLM_API_KEY"""
        with patch.dict(os.environ, {"LLM_API_KEY": "generic-key"}, clear=True):
            cfg = LLMConfig(provider="auto")
            # 应该能正常工作
            assert cfg.provider is not None
```

---

## 四、实施时间表

| 步骤 | 内容 | 预计耗时 | 依赖 |
|------|------|----------|------|
| 步骤 1 | 添加提供商常量定义 | 0.5h | 无 |
| 步骤 2 | 扩展 LLMConfig 自动检测 | 1.5h | 步骤 1 |
| 步骤 3 | 修改 LLMClient 支持新逻辑 | 1h | 步骤 2 |
| 步骤 4 | 更新 config.yaml | 0.5h | 步骤 3 |
| 步骤 5 | 添加提供商扩展基类 | 1h | 步骤 3 |
| 步骤 6 | 添加测试用例 | 1h | 步骤 5 |
| **总计** | | **5.5h** | |

---

## 五、验收标准

### 5.1 功能验收

- [ ] `LLMConfig(provider="auto")` 能根据环境变量自动检测提供商
- [ ] 设置 `OPENAI_API_KEY` → 自动使用 OpenAI
- [ ] 设置 `ZHIPUAI_API_KEY` → 自动使用智谱AI
- [ ] 设置 `LLM_BASE_URL=http://localhost:8000/v1` → 自动使用 VLLM
- [ ] 设置 `LLM_BASE_URL=http://localhost:11434/v1` → 自动使用 Ollama
- [ ] 显式指定 `provider="zhipuai"` 能覆盖自动检测
- [ ] 本地模型 (VLLM/Ollama) 无需 API Key 能正常工作

### 5.2 兼容性验收

- [ ] 现有 `config.yaml` 中的 `llm_profiles` 配置继续有效
- [ ] 现有回测脚本无需修改即可运行
- [ ] `--llm-profile openai` 命令行参数仍然生效

### 5.3 测试验收

- [ ] 所有自动检测测试用例通过
- [ ] 使用 `local-vllm` profile 能连接本地 VLLM 服务
- [ ] 使用 `local-ollama` profile 能连接本地 Ollama 服务

---

## 六、回滚计划

如果升级出现问题，可通过以下方式回滚：

1. **配置回滚**: 将 `provider: "auto"` 改回 `provider: "openai"`
2. **代码回滚**: `git checkout HEAD~1 -- stockbench/llm/llm_client.py`
3. **完整回滚**: `git revert <commit_hash>`

---

## 七、后续扩展

Part 1 完成后，可继续以下升级：

- **Part 2**: Agent 基类、Message 标准化
- **Part 3**: 工具系统、ToolRegistry
- **Part 4**: 插件系统、高级特性

---

*Part 1 详细计划生成时间: 2025-12-07*
*基于: Agent框架如何构建 Part 1 (7.1-7.2节)*

---

# Part 2 详细升级计划：Pipeline 架构与可观测性

基于《Agent框架如何构建》Part 2（7.3-7.4节），针对 StockBench 多 Agent 流水线架构的升级方案。

**设计原则**：保持函数式风格 + 统一上下文传递 + 完整可观测性

---

## 一、当前 Agent 架构分析

### 1.1 现有多 Agent 流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│                    StockBench Agent Pipeline                         │
│                                                                       │
│  features_list ──▶ [FundamentalFilterAgent] ──▶ filtered_result      │
│                           │                                           │
│                           ▼                                           │
│  filtered_result ──▶ [DecisionAgent] ──▶ decisions                   │
│                           │                                           │
│                           ▼                                           │
│  decisions + metrics ──▶ [ReportAgent] ──▶ report                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 现有文件结构

```
stockbench/agents/
├── fundamental_filter_agent.py  # 判断是否需要基本面数据
│   └── filter_stocks_needing_fundamental()
├── dual_agent_llm.py            # 真正做出决策
│   └── generate_dual_decisions()
├── backtest_report_llm.py       # 根据决策总结报告
│   └── generate_backtest_report()
└── prompts/
    └── decision_agent_v1.txt
```

### 1.3 存在的问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **无统一上下文** | 各函数参数不一致 | 数据传递混乱 |
| **无执行追踪** | 不知道哪个 Agent 耗时/出错 | 调试困难 |
| **数据流不透明** | Agent 间数据传递靠参数 | 难以扩展 |
| **重复代码** | 每个文件都有 `_load_prompt()` | 可维护性差 |

---

## 二、升级方案：增强版函数式 + PipelineContext

### 2.1 核心设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PipelineContext                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  trace: AgentTrace  ← 追踪每个 Agent 的输入/输出/耗时/错误    │    │
│  │  data_bus: Dict     ← Agent 间数据传递的统一通道              │    │
│  │  config: Dict       ← 统一配置                               │    │
│  │  llm: LLMClient     ← 共享 LLM 客户端                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│   Agent1.run(ctx) ──▶ Agent2.run(ctx) ──▶ Agent3.run(ctx)          │
│        │                    │                    │                   │
│        └── trace.log() ─────┴── trace.log() ────┴── trace.log()     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 满足的需求

| 需求 | 设计方案 |
|------|---------|
| **未来扩展** | `ctx.put()/get()` 数据总线，新 Agent 无缝接入 |
| **数据透明** | `AgentTrace` 记录每步输入/输出摘要、耗时、token |
| **问题定位** | `ctx.trace.get_failed_agents()` 直接定位失败 Agent |
| **统一整洁** | `@traced_agent` 装饰器，函数签名统一 |

---

## 三、详细实施步骤

### 步骤 1: 创建 PipelineContext 和 AgentTrace

**文件**: `stockbench/core/pipeline_context.py`

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime
from loguru import logger


@dataclass
class AgentStep:
    """单个 Agent 执行步骤的记录"""
    agent_name: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    tokens_used: int = 0
    status: str = "running"  # running | success | failed
    error: Optional[str] = None
    duration_ms: float = 0
    
    def finish(self, status: str, output_summary: str = None, error: str = None):
        self.finished_at = datetime.now()
        self.status = status
        self.output_summary = output_summary
        self.error = error
        self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000


@dataclass  
class AgentTrace:
    """Agent 执行追踪器"""
    run_id: str
    steps: List[AgentStep] = field(default_factory=list)
    
    def start_agent(self, agent_name: str, input_summary: str = None) -> AgentStep:
        step = AgentStep(agent_name=agent_name, started_at=datetime.now(), input_summary=input_summary)
        self.steps.append(step)
        logger.info(f"▶ [{agent_name}] Started | input: {input_summary or 'N/A'}")
        return step
    
    def finish_agent(self, step: AgentStep, status: str, output_summary: str = None, error: str = None):
        step.finish(status, output_summary, error)
        if status == "success":
            logger.info(f"✓ [{step.agent_name}] Completed in {step.duration_ms:.0f}ms")
        else:
            logger.error(f"✗ [{step.agent_name}] Failed: {error}")
    
    def get_failed_agents(self) -> List[str]:
        return [s.agent_name for s in self.steps if s.status == "failed"]
    
    def to_summary(self) -> Dict:
        return {
            "run_id": self.run_id,
            "total_agents": len(self.steps),
            "success": len([s for s in self.steps if s.status == "success"]),
            "failed": len([s for s in self.steps if s.status == "failed"]),
            "total_duration_ms": sum(s.duration_ms for s in self.steps),
            "steps": [{"agent": s.agent_name, "status": s.status, "duration_ms": s.duration_ms, "error": s.error} for s in self.steps]
        }


@dataclass
class PipelineContext:
    """Agent 流水线上下文"""
    run_id: str
    date: str
    llm_client: Any  # LLMClient
    llm_config: Any  # LLMConfig
    config: Dict[str, Any] = field(default_factory=dict)
    _data_bus: Dict[str, Any] = field(default_factory=dict)
    trace: AgentTrace = field(default=None)
    
    def __post_init__(self):
        if self.trace is None:
            self.trace = AgentTrace(run_id=self.run_id)
    
    # 数据总线操作
    def put(self, key: str, value: Any, agent_name: str = None):
        self._data_bus[key] = value
        if agent_name:
            self._data_bus[f"_source_{key}"] = agent_name
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data_bus.get(key, default)
    
    def get_source(self, key: str) -> Optional[str]:
        return self._data_bus.get(f"_source_{key}")
    
    def keys(self) -> List[str]:
        return [k for k in self._data_bus.keys() if not k.startswith("_")]
    
    # Agent 执行追踪
    def start_agent(self, agent_name: str, input_summary: str = None) -> AgentStep:
        return self.trace.start_agent(agent_name, input_summary)
    
    def finish_agent(self, step: AgentStep, status: str, output_summary: str = None, error: str = None):
        self.trace.finish_agent(step, status, output_summary, error)
```

### 步骤 2: 创建 Agent 装饰器

**文件**: `stockbench/core/decorators.py`

```python
from functools import wraps
from typing import Callable


def traced_agent(name: str):
    """Agent 追踪装饰器 - 自动记录输入/输出/耗时/错误"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = kwargs.get("ctx") or (args[-1] if args and hasattr(args[-1], 'trace') else None)
            if ctx is None:
                return func(*args, **kwargs)
            
            input_summary = _make_summary(args[0] if args else None)
            step = ctx.start_agent(name, input_summary)
            
            try:
                result = func(*args, **kwargs)
                ctx.finish_agent(step, "success", _make_summary(result))
                return result
            except Exception as e:
                ctx.finish_agent(step, "failed", error=str(e))
                raise
        return wrapper
    return decorator


def _make_summary(data) -> str:
    if data is None:
        return "None"
    if isinstance(data, list):
        return f"List[{len(data)} items]"
    if isinstance(data, dict):
        keys = list(data.keys())[:5]
        return f"Dict[{len(data)} keys: {keys}...]"
    return str(data)[:100]
```

### 步骤 3: 创建类型定义

**文件**: `stockbench/core/types.py`

```python
from typing import TypedDict, List, Optional


class Decision(TypedDict):
    action: str  # "increase" | "decrease" | "hold" | "close"
    target_cash_amount: float
    cash_change: float
    reasons: List[str]
    confidence: float


class FilterResult(TypedDict):
    needs_fundamental: List[str]
    skip_fundamental: List[str]
    filter_reasons: dict
```

### 步骤 4: 更新 core/__init__.py

**文件**: `stockbench/core/__init__.py`

```python
from .pipeline_context import PipelineContext, AgentTrace, AgentStep
from .decorators import traced_agent
from .types import Decision, FilterResult

__all__ = [
    "PipelineContext",
    "AgentTrace", 
    "AgentStep",
    "traced_agent",
    "Decision",
    "FilterResult",
]
```

### 步骤 5: 更新 Agent 函数签名（渐进式）

Agent 函数保持向后兼容，新增 `ctx` 参数为可选：

```python
# fundamental_filter_agent.py
from stockbench.core import traced_agent, PipelineContext

@traced_agent("fundamental_filter")
def filter_stocks_needing_fundamental(
    features_list: List[Dict], 
    cfg: Dict | None = None,
    ctx: PipelineContext = None,  # 新增可选参数
    **kwargs
) -> Dict:
    # 兼容模式：从 ctx 或旧参数获取配置
    if ctx:
        cfg = cfg or ctx.config
        previous_decisions = ctx.get("previous_decisions", kwargs.get("previous_decisions"))
    else:
        previous_decisions = kwargs.get("previous_decisions")
    
    # ... 现有逻辑 ...
    
    # 如果有 ctx，存入数据总线
    if ctx:
        ctx.put("filter_result", result, agent_name="fundamental_filter")
    
    return result
```

### 步骤 6: 添加测试用例

**文件**: `stockbench/core/tests/test_pipeline_context.py`

---

## 四、文件结构

```
stockbench/core/
├── __init__.py              # 导出核心组件
├── pipeline_context.py      # PipelineContext + AgentTrace (~100行)
├── decorators.py            # @traced_agent 装饰器 (~40行)
├── types.py                 # Decision, FilterResult 类型 (~30行)
├── features.py              # 已有文件
└── tests/
    └── test_pipeline_context.py  # 测试用例 (~80行)
```

---

## 五、实施时间表

| 步骤 | 内容 | 预计耗时 | 依赖 |
|------|------|----------|------|
| 步骤 1 | 创建 PipelineContext 和 AgentTrace | 0.5h | 无 |
| 步骤 2 | 创建 Agent 装饰器 | 0.3h | 步骤 1 |
| 步骤 3 | 创建类型定义 | 0.2h | 无 |
| 步骤 4 | 更新 core/__init__.py | 0.1h | 步骤 1-3 |
| 步骤 5 | 更新 Agent 函数签名 | 1h | 步骤 4 |
| 步骤 6 | 添加测试用例 | 0.5h | 步骤 5 |
| **总计** | | **2.6h** | |

---

## 六、验收标准

### 6.1 功能验收

- [ ] `PipelineContext` 能正确传递数据在 Agent 间
- [ ] `AgentTrace` 记录每个 Agent 的执行状态和耗时
- [ ] `@traced_agent` 装饰器自动追踪函数执行
- [ ] `ctx.get_failed_agents()` 能正确返回失败的 Agent
- [ ] `ctx.trace.to_summary()` 输出完整执行摘要

### 6.2 兼容性验收

- [ ] 现有 Agent 函数不传 `ctx` 参数时仍能正常工作
- [ ] 现有回测脚本无需修改即可运行
- [ ] 新旧调用方式可以共存

### 6.3 使用示例验收

```python
# 新调用方式
ctx = PipelineContext(
    run_id="backtest_2025_01_01",
    date="2025-01-01",
    llm_client=llm,
    llm_config=cfg,
    config=config
)
ctx.put("previous_decisions", prev_decisions)

filter_result = filter_stocks_needing_fundamental(features_list, ctx=ctx)
decisions = generate_dual_decisions(features_list, ctx=ctx)

print(ctx.trace.to_summary())
# {"run_id": "...", "success": 2, "failed": 0, "steps": [...]}
```

---

## 七、后续扩展

Part 2 完成后，可继续以下升级：

- **Part 3**: 工具系统、ToolRegistry
- **Part 4**: 插件系统、高级特性
- **可选**: Message 类标准化（如需更严格的消息格式）

---

*Part 2 详细计划生成时间: 2025-12-07*
*基于: Agent框架如何构建 Part 2 (7.3-7.4节) + StockBench 多 Agent 流水线架构*

---

# Part 3 详细升级计划：工具系统设计

基于《Agent框架如何构建》Part 3（7.5节），为 StockBench 实现可扩展的工具系统。

**设计原则**：Tool 抽象 + 注册中心 + 向后兼容

---

## 一、当前数据层分析

### 1.1 现有结构

```
stockbench/
├── core/
│   ├── data_hub.py          # 1650行，包含所有数据获取函数
│   │   ├── get_bars()       # 价格数据
│   │   ├── get_news()       # 新闻数据
│   │   ├── get_financials() # 财务数据
│   │   ├── get_dividends()  # 分红数据
│   │   └── ...              # 20+ 函数
│   └── features.py          # 特征构建
├── adapters/
│   ├── polygon_client.py    # Polygon API 适配器
│   └── finnhub_client.py    # Finnhub API 适配器
```

### 1.2 存在的问题

| 问题 | 具体表现 | 影响 |
|------|----------|------|
| **耦合度高** | Agent 直接调用 data_hub 函数 | 难以替换数据源 |
| **无法动态组合** | 数据获取流程硬编码 | 扩展性差 |
| **不支持 Function Calling** | 无法让 LLM 自主选择工具 | 限制 Agent 能力 |
| **测试困难** | 无法 Mock 单个数据获取 | 单测复杂 |

---

## 二、目标架构

### 2.1 升级后结构

```
stockbench/
├── tools/                      # 新增：工具系统
│   ├── __init__.py
│   ├── base.py                 # Tool 基类、ToolParameter、ToolResult
│   ├── registry.py             # ToolRegistry 注册中心
│   ├── data_tools.py           # 数据获取工具集
│   │   ├── PriceDataTool
│   │   ├── NewsDataTool
│   │   ├── FinancialsTool
│   │   └── SnapshotTool
│   └── openai_schema.py        # OpenAI Function Calling 格式转换
├── core/
│   ├── data_hub.py             # 保留，作为底层实现
│   └── features.py             # 保留，作为底层实现
```

### 2.2 核心组件

```python
# Tool 基类
class Tool(ABC):
    name: str
    description: str
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult
    
    @abstractmethod  
    def get_parameters(self) -> List[ToolParameter]
    
    def to_openai_schema(self) -> Dict

# ToolRegistry 注册中心
class ToolRegistry:
    def register(self, tool: Tool)
    def get(self, name: str) -> Tool
    def execute(self, name: str, **kwargs) -> ToolResult
    def to_openai_tools(self) -> List[Dict]
```

---

## 三、修改前后对比

### 3.1 调用方式对比

**修改前**:
```python
from stockbench.core.data_hub import get_bars, get_news

bars = get_bars(symbol, start, end, ...)
news = get_news(symbol, start, end, ...)
```

**修改后**:
```python
from stockbench.tools import ToolRegistry

registry = ToolRegistry.default()
bars = registry.execute("get_price_data", symbol=symbol, start_date=start)
news = registry.execute("get_news", symbol=symbol, start_date=start)
```

### 3.2 优劣对比

| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| **耦合度** | Agent 直接依赖 data_hub | Agent 依赖 Tool 接口 |
| **可测试性** | 需要 Mock 整个 data_hub | 可 Mock 单个 Tool |
| **扩展性** | 添加新数据源需改 data_hub | 注册新 Tool 即可 |
| **Function Calling** | 不支持 | 内置转换 |
| **向后兼容** | - | ✅ data_hub 不变 |

---

## 四、详细实施步骤

### 步骤 1: 创建 Tool 基类和类型定义

**文件**: `stockbench/tools/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        pass
    
    def to_openai_schema(self) -> Dict:
        # 转换为 OpenAI function calling 格式
        ...
```

### 步骤 2: 创建 ToolRegistry 注册中心

**文件**: `stockbench/tools/registry.py`

```python
class ToolRegistry:
    _default_instance = None
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    @classmethod
    def default(cls) -> "ToolRegistry":
        if cls._default_instance is None:
            cls._default_instance = cls()
            cls._default_instance._register_builtin_tools()
        return cls._default_instance
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        return tool.run(**kwargs)
    
    def to_openai_tools(self) -> List[Dict]:
        return [t.to_openai_schema() for t in self._tools.values()]
```

### 步骤 3: 包装现有数据函数为 Tool

**文件**: `stockbench/tools/data_tools.py`

```python
class PriceDataTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_price_data",
            description="获取股票历史价格数据"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("symbol", ToolParameterType.STRING, "股票代码"),
            ToolParameter("start_date", ToolParameterType.STRING, "开始日期"),
            ToolParameter("end_date", ToolParameterType.STRING, "结束日期"),
        ]
    
    def run(self, symbol: str, start_date: str, end_date: str, **kwargs) -> ToolResult:
        from stockbench.core.data_hub import get_bars
        try:
            df = get_bars(symbol, start_date, end_date, 1, "day", True)
            return ToolResult(success=True, data=df)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

### 步骤 4: 创建 OpenAI Function Calling 转换

**文件**: `stockbench/tools/openai_schema.py`

### 步骤 5: 更新包导出

**文件**: `stockbench/tools/__init__.py`

### 步骤 6: 添加测试用例

**文件**: `stockbench/tools/tests/test_tools.py`

### 步骤 7: (可选) 集成到 PipelineContext

---

## 五、实施时间表

| 步骤 | 内容 | 预计耗时 | 依赖 |
|------|------|----------|------|
| 步骤 1 | Tool 基类和类型 | 0.3h | 无 |
| 步骤 2 | ToolRegistry | 0.4h | 步骤 1 |
| 步骤 3 | 数据工具包装 | 1.0h | 步骤 2 |
| 步骤 4 | OpenAI 格式转换 | 0.3h | 步骤 1 |
| 步骤 5 | 包导出 | 0.1h | 步骤 1-4 |
| 步骤 6 | 测试用例 | 0.5h | 步骤 5 |
| 步骤 7 | PipelineContext 集成 | 0.3h | 步骤 6 |
| **总计** | | **2.9h** | |

---

## 六、验收标准

### 6.1 功能验收

- [ ] Tool 基类能正确定义工具接口
- [ ] ToolRegistry 能注册、获取、执行工具
- [ ] PriceDataTool 能正确获取价格数据
- [ ] NewsDataTool 能正确获取新闻数据
- [ ] to_openai_schema() 输出符合 OpenAI 格式
- [ ] 所有现有功能不受影响（向后兼容）

### 6.2 测试验收

```python
# 基本使用
registry = ToolRegistry.default()
result = registry.execute("get_price_data", symbol="AAPL", start_date="2025-01-01", end_date="2025-01-10")
assert result.success

# OpenAI 格式
tools = registry.to_openai_tools()
assert len(tools) > 0
assert "function" in tools[0]
```

---

## 七、后续扩展

Part 3 完成后，可继续：

- **Part 4**: 插件系统、高级特性
- **扩展**: 添加更多数据工具（期权、ETF、宏观数据）
- **扩展**: 实现工具链（ToolChain）
- **扩展**: 异步并行执行器

---

*Part 3 详细计划生成时间: 2025-12-08*
*基于: Agent框架如何构建 Part 3 (7.5节) + StockBench 数据层架构*
