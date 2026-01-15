# Agent 框架如何构建

## 7.1 框架整体架构设计

### 7.1.1 为何需要自建Agent框架

在智能体技术快速发展的今天，市面上已经存在众多成熟的Agent框架。那么，为什么我们还要从零开始构建一个新的框架呢？

#### （1）市面框架的快速迭代与局限性

智能体领域是一个快速发展的领域，随时会有新的概念产生，对于智能体的设计每个框架都有自己的定位和理解，不过智能体的核心知识点是一致的。

- **过度抽象的复杂性**：许多框架为了追求通用性，引入了大量抽象层和配置选项。以LangChain为例，其链式调用机制虽然灵活，但对初学者而言学习曲线陡峭，往往需要理解大量概念才能完成简单任务。

- **快速迭代带来的不稳定性**：商业化框架为了抢占市场，API接口变更频繁。开发者经常面临版本升级后代码无法运行的困扰，维护成本居高不下。

- **黑盒化的实现逻辑**：许多框架将核心逻辑封装得过于严密，开发者难以理解Agent的内部工作机制，缺乏深度定制能力。遇到问题时只能依赖文档和社区支持，尤其是如果社区不够活跃，可能一个反馈意见会非常久也没有人推进，影响后续的开发效率。

- **依赖关系的复杂性**：成熟框架往往携带大量依赖包，安装包体积庞大，在需要与别的项目代码配合的下可能出现依赖冲突问题。

#### （2）从使用者到构建者的能力跃迁

构建自己的Agent框架，实际上是一个从"使用者"向"构建者"转变的过程。这种转变带来的价值是长远的。

- **深度理解Agent工作原理**：通过亲手实现每个组件，开发者能够真正理解Agent的思考过程、工具调用机制、以及各种设计模式的好坏与区别。

- **获得完全的控制权**：自建框架意味着对每一行代码都有完全的掌控，可以根据具体需求进行精确调优，而不受第三方框架设计理念的束缚。

- **培养系统设计能力**：框架构建过程涉及模块化设计、接口抽象、错误处理等软件工程核心技能，这些能力对开发者的长期成长具有重要价值。

#### （3）定制化需求与深度掌握的必要性

在实际应用中，不同场景对智能体的需求差异巨大，往往都需要在通用框架基础上做二次开发。

- **特定领域的优化需求**：金融、医疗、教育等垂直领域往往需要针对性的提示词模板、特殊的工具集成、以及定制化的安全策略。

- **性能与资源的精确控制**：生产环境中，对响应时间、内存占用、并发处理能力都有严格要求，通用框架的"一刀切"方案往往无法满足精细化需求。

- **学习与教学的透明性要求**：在我们的教学场景中，学习者更期待的是清晰地看到智能体的每一步构建过程，理解不同范式的工作机制，这要求框架具有高度的可观测性和可解释性。

---

### 7.1.2 HelloAgents框架的设计理念

构建一个新的Agent框架，关键不在于功能的多少，而在于设计理念是否能真正解决现有框架的痛点。HelloAgents框架的设计围绕着一个核心问题展开：如何让学习者既能快速上手，又能深入理解Agent的工作原理？

当你初次接触任何成熟的框架时，可能会被其丰富的功能所吸引，但很快就会发现一个问题：要完成一个简单的任务，往往需要理解Chain、Agent、Tool、Memory、Retriever等十几个不同的概念。每个概念都有自己的抽象层，学习曲线变得异常陡峭。这种复杂性虽然带来了强大的功能，但也成为了初学者的障碍。HelloAgents框架试图在功能完整性和学习友好性之间找到平衡点，形成了四个核心的设计理念。

#### （1）轻量级与教学友好的平衡

一个优秀的学习框架应该具备完整的可读性。HelloAgents将核心代码按照章节区分开，这是基于一个简单的原则：任何有一定编程基础的开发者都应该能够在合理的时间内完全理解框架的工作原理。在依赖管理方面，框架采用了极简主义的策略。除了OpenAI的官方SDK和几个必要的基础库外，不引入任何重型依赖。如果遇到问题时，我们可以直接定位到框架本身的代码，而不需要在复杂的依赖关系中寻找答案。

#### （2）基于标准API的务实选择

OpenAI的API已经成为了行业标准，几乎所有主流的LLM提供商都在努力兼容这套接口。HelloAgents选择在这个标准之上构建，而不是重新发明一套抽象接口。这个决定主要是出于几点动机。首先是兼容性的保证，当你掌握了HelloAgents的使用方法后，迁移到其他框架或将其集成到现有项目中时，底层的API调用逻辑是完全一致的。其次是学习成本的降低。你不需要学习新的概念模型，因为所有的操作都基于你已经熟悉的标准接口。

#### （3）渐进式学习路径的精心设计

HelloAgents提供了一条清晰的学习路径。我们将会把每一章的学习代码，保存为一个可以pip下载的历史版本，因此无需担心代码的使用成本，因为每一个核心的功能都将会是你自己编写的。这种设计让你能够按照自己的需求和节奏前进。每一步的升级都是自然而然的，不会产生概念上的跳跃或理解上的断层。值得一提的是，我们这一章的内容，也是基于前六章的内容来完善的。同样，这一章也是为后续高级知识学习部分打下框架基础。

#### （4）统一的"工具"抽象：万物皆为工具

为了彻底贯彻轻量级与教学友好的理念，HelloAgents在架构上做出了一个关键的简化：除了核心的Agent类，一切皆为Tools。在许多其他框架中需要独立学习的Memory（记忆）、RAG（检索增强生成）、RL（强化学习）、MCP（协议）等模块，在HelloAgents中都被统一抽象为一种"工具"。这种设计的初衷是消除不必要的抽象层，让学习者可以回归到最直观的"智能体调用工具"这一核心逻辑上，从而真正实现快速上手和深入理解的统一。

---

### 7.1.3 本章学习目标

让我们先看看第七章的核心学习内容：

```
hello-agents/
├── hello_agents/
│   │
│   ├── core/                     # 核心框架层
│   │   ├── agent.py              # Agent基类
│   │   ├── llm.py                # HelloAgentsLLM统一接口
│   │   ├── message.py            # 消息系统
│   │   ├── config.py             # 配置管理
│   │   └── exceptions.py         # 异常体系
│   │
│   ├── agents/                   # Agent实现层
│   │   ├── simple_agent.py       # SimpleAgent实现
│   │   ├── react_agent.py        # ReActAgent实现
│   │   ├── reflection_agent.py   # ReflectionAgent实现
│   │   └── plan_solve_agent.py   # PlanAndSolveAgent实现
│   │
│   ├── tools/                    # 工具系统层
│   │   ├── base.py               # 工具基类
│   │   ├── registry.py           # 工具注册机制
│   │   ├── chain.py              # 工具链管理系统
│   │   ├── async_executor.py     # 异步工具执行器
│   │   └── builtin/              # 内置工具集
│   │       ├── calculator.py     # 计算工具
│   │       └── search.py         # 搜索工具
```

在开始编写具体代码之前，我们需要先建立一个清晰的架构蓝图。HelloAgents的架构设计遵循了"分层解耦、职责单一、接口统一"的核心原则，这样既保持了代码的组织性，也便于按照章节扩展内容。

#### 快速开始：安装HelloAgents框架

为了让读者能够快速体验本章的完整功能，我们提供了可直接安装的Python包。你可以通过以下命令安装本章对应的版本：

```bash
# python版本需要>=3.10
pip install "hello-agents==0.1.1"
```

本章的学习可以采用两种方式：

- **体验式学习**：直接使用pip安装框架，运行示例代码，快速体验各种功能
- **深度学习**：跟随本章内容，从零开始实现每个组件，深入理解框架的设计思想和实现细节

我们建议采用"先体验，后实现"的学习路径。在本章中，我们提供了完整的测试文件，你可以重写核心函数并运行测试，以检验你的实现是否正确。这种学习方式既保证了实践性，又确保了学习效果。如果你想深入了解框架的实现细节，或者希望参与到框架的开发中来，可以访问这个GitHub仓库。

在开始之前，让我们用30秒体验使用Hello-agents构建简单智能体！

```python
# 配置好同级文件夹下.env中的大模型API, 可参考code文件夹配套的.env.example，也可以拿前几章的案例的.env文件复用。
from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建LLM实例 - 框架自动检测provider
llm = HelloAgentsLLM()

# 或手动指定provider（可选）
# llm = HelloAgentsLLM(provider="modelscope")

# 创建SimpleAgent
agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手"
)

# 基础对话
response = agent.run("你好！请介绍一下自己")
print(response)

# 添加工具功能（可选）
from hello_agents.tools import CalculatorTool
calculator = CalculatorTool()
# 需要实现7.4.1的MySimpleAgent进行调用，后续章节会支持此类调用方式
# agent.add_tool(calculator)

# 现在可以使用工具了
response = agent.run("请帮我计算 2 + 3 * 4")
print(response)

# 查看对话历史
print(f"历史消息数: {len(agent.get_history())}")
```

---

## 7.2 HelloAgentsLLM扩展

本节内容将在第 4.1.3 节创建的 HelloAgentsLLM 基础上进行迭代升级。我们将把这个基础客户端，改造为一个更具适应性的模型调用中枢。本次升级主要围绕以下三个目标展开：

- **多提供商支持**：实现对 OpenAI、ModelScope、智谱 AI 等多种主流 LLM 服务商的无缝切换，避免框架与特定供应商绑定。
- **本地模型集成**：引入 VLLM 和 Ollama 这两种高性能本地部署方案，作为对第 3.2.3 节中 Hugging Face Transformers 方案的生产级补充，满足数据隐私和成本控制的需求。
- **自动检测机制**：建立一套自动识别机制，使框架能根据环境信息智能推断所使用的 LLM 服务类型，简化用户的配置过程。

### 7.2.1 支持多提供商

我们之前定义的 HelloAgentsLLM 类，已经能够通过 api_key 和 base_url 这两个核心参数，连接任何兼容 OpenAI 接口的服务。这在理论上保证了通用性，但在实际应用中，不同的服务商在环境变量命名、默认 API 地址和推荐模型等方面都存在差异。如果每次切换服务商都需要用户手动查询并修改代码，会极大影响开发效率。为了解决这一问题，我们引入 provider。其改进思路是：让 HelloAgentsLLM 在内部处理不同服务商的配置细节，从而为用户提供一个统一、简洁的调用体验。具体的实现细节我们将在7.2.3节"自动检测机制"中详细阐述，在这里，我们首先关注如何利用这一机制来扩展框架。

下面，我们将演示如何通过继承 HelloAgentsLLM，来增加对 ModelScope 平台的支持。我们希望读者不仅学会如何"使用"框架，更能掌握如何"扩展"框架。直接修改已安装的库源码是一种不被推荐的做法，因为它会使后续的库升级变得困难。

#### （1）创建自定义LLM类并继承

假设我们的项目目录中有一个 `my_llm.py` 文件。我们首先从 `hello_agents` 库中导入 HelloAgentsLLM 基类，然后创建一个名为 MyLLM 的新类继承它。

```python
# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对ModelScope的支持。
    """
    pass # 暂时留空
```

#### （2）重写 __init__ 方法以支持新供应商

接下来，我们在 MyLLM 类中重写 `__init__` 方法。我们的目标是：当用户传入 `provider="modelscope"` 时，执行我们自定义的逻辑；否则，就调用父类 HelloAgentsLLM 的原始逻辑，使其能够继续支持 OpenAI 等其他内置的供应商。

```python
class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
            
            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"
            
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            
            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)
```

这段代码展示了"重写"的思想：我们拦截了 `provider="modelscope"` 的情况并进行了特殊处理，对于其他所有情况，则通过 `super().__init__(...)` 交还给父类，保留了原有框架的全部功能。

#### （3）使用自定义的 MyLLM 类

现在，我们可以在项目的业务逻辑中，像使用原生 HelloAgentsLLM 一样使用我们自己的 MyLLM 类。

首先，在 `.env` 文件中配置 ModelScope 的 API 密钥：

```bash
# .env file
MODELSCOPE_API_KEY="your-modelscope-api-key"
```

然后，在主程序中导入并使用 MyLLM：

```python
# my_main.py
from dotenv import load_dotenv
from my_llm import MyLLM # 注意:这里导入我们自己的类

# 加载环境变量
load_dotenv()

# 实例化我们重写的客户端，并指定provider
llm = MyLLM(provider="modelscope") 

# 准备消息
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用，think等方法都已从父类继承，无需重写
response_stream = llm.think(messages)

# 打印响应
print("ModelScope Response:")
for chunk in response_stream:
    # chunk 已经是文本片段，可以直接使用
    print(chunk, end="", flush=True)
```

通过以上步骤，我们就在不修改 hello-agents 库源码的前提下，成功为其扩展了新的功能。这种方法不仅保证了代码的整洁和可维护性，也使得未来升级 hello-agents 库时，我们的定制化功能不会丢失。

---

### 7.2.2 本地模型调用

在第 3.2.3 节，我们学习了如何使用 Hugging Face Transformers 库在本地直接运行开源模型。该方法非常适合入门学习和功能验证，但其底层实现在处理高并发请求时性能有限，通常不作为生产环境的首选方案。

为了在本地实现高性能、生产级的模型推理服务，社区涌现出了 VLLM 和 Ollama 等优秀工具。它们通过连续批处理、PagedAttention 等技术，显著提升了模型的吞吐量和运行效率，并将模型封装为兼容 OpenAI 标准的 API 服务。这意味着，我们可以将它们无缝地集成到 HelloAgentsLLM 中。

#### VLLM

VLLM 是一个为 LLM 推理设计的高性能 Python 库。它通过 PagedAttention 等先进技术，可以实现比标准 Transformers 实现高出数倍的吞吐量。下面是在本地部署一个 VLLM 服务的完整步骤：

首先，需要根据你的硬件环境（特别是 CUDA 版本）安装 VLLM。推荐遵循其官方文档进行安装，以避免版本不匹配问题。

```bash
pip install vllm
```

安装完成后，使用以下命令即可启动一个兼容 OpenAI 的 API 服务。VLLM 会自动从 Hugging Face Hub 下载指定的模型权重（如果本地不存在）。我们依然以 Qwen1.5-0.5B-Chat 模型为例：

```bash
# 启动 VLLM 服务，并加载 Qwen1.5-0.5B-Chat 模型
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-0.5B-Chat \
    --host 0.0.0.0 \
    --port 8000
```

服务启动后，便会在 `http://localhost:8000/v1` 地址上提供与 OpenAI 兼容的 API。

#### Ollama

Ollama 进一步简化了本地模型的管理和部署，它将模型下载、配置和服务启动等步骤封装到了一条命令中，非常适合快速上手。访问 Ollama 官方网站下载并安装适用于你操作系统的客户端。

安装后，打开终端，执行以下命令即可下载并运行一个模型（以 Llama 3 为例）。Ollama 会自动处理模型的下载、服务封装和硬件加速配置。

```bash
# 首次运行会自动下载模型，之后会直接启动服务
ollama run llama3
```

当你在终端看到模型的交互提示时，即表示服务已经成功在后台启动。Ollama 默认会在 `http://localhost:11434/v1` 地址上暴露 OpenAI 兼容的 API 接口。

#### 接入 HelloAgentsLLM

由于 VLLM 和 Ollama 都遵循了行业标准 API，将它们接入 HelloAgentsLLM 的过程非常简单。我们只需在实例化客户端时，将它们视为一个新的 provider 即可。

例如，连接本地运行的 VLLM 服务：

```python
llm_client = HelloAgentsLLM(
    provider="vllm",
    model="Qwen/Qwen1.5-0.5B-Chat", # 需与服务启动时指定的模型一致
    base_url="http://localhost:8000/v1",
    api_key="vllm" # 本地服务通常不需要真实API Key，可填任意非空字符串
)
```

或者，通过设置环境变量并让客户端自动检测，实现代码的零修改：

```bash
# 在 .env 文件中设置
LLM_BASE_URL="http://localhost:8000/v1"
LLM_API_KEY="vllm"
```

```python
# Python 代码中直接实例化即可
llm_client = HelloAgentsLLM() # 将自动检测为 vllm
```

同理，连接本地的 Ollama 服务也一样简单：

```python
llm_client = HelloAgentsLLM(
    provider="ollama",
    model="llama3", # 需与 ollama run 指定的模型一致
    base_url="http://localhost:11434/v1",
    api_key="ollama" # 本地服务同样不需要真实 Key
)
```

通过这种统一的设计，我们的智能体核心代码无需任何修改，就可以在云端 API 和本地模型之间自由切换。这为后续应用的开发、部署、成本控制以及保护数据隐私提供了极大的灵活性。

---

### 7.2.3 自动检测机制

为了尽可能减少用户的配置负担并遵循"约定优于配置"的原则，HelloAgentsLLM 内部设计了两个核心辅助方法：`_auto_detect_provider` 和 `_resolve_credentials`。它们协同工作，`_auto_detect_provider` 负责根据环境信息推断服务商，而 `_resolve_credentials` 则根据推断结果完成具体的参数配置。

`_auto_detect_provider` 方法负责根据环境信息，按照下述优先级顺序，尝试自动推断服务商：

**最高优先级：检查特定服务商的环境变量**

这是最直接、最可靠的判断依据。框架会依次检查 MODELSCOPE_API_KEY, OPENAI_API_KEY, ZHIPU_API_KEY 等环境变量是否存在。一旦发现任何一个，就会立即确定对应的服务商。

**次高优先级：根据 base_url 进行判断**

如果用户没有设置特定服务商的密钥，但设置了通用的 LLM_BASE_URL，框架会转而解析这个 URL。

- **域名匹配**：通过检查 URL 中是否包含 "api-inference.modelscope.cn", "api.openai.com" 等特征字符串来识别云服务商。
- **端口匹配**：通过检查 URL 中是否包含 :11434 (Ollama), :8000 (VLLM) 等本地服务的标准端口来识别本地部署方案。

**辅助判断：分析 API 密钥的格式**

在某些情况下，如果上述两种方式都无法确定，框架会尝试分析通用环境变量 LLM_API_KEY 的格式。例如，某些服务商的 API 密钥有固定的前缀或独特的编码格式。不过，由于这种方式可能存在模糊性（例如多个服务商的密钥格式相似），因此它的优先级较低，仅作为辅助手段。

其部分关键代码如下：

```python
def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
    """
    自动检测LLM提供商
    """
    # 1. 检查特定提供商的环境变量 (最高优先级)
    if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
    if os.getenv("OPENAI_API_KEY"): return "openai"
    if os.getenv("ZHIPU_API_KEY"): return "zhipu"
    # ... 其他服务商的环境变量检查

    # 获取通用的环境变量
    actual_api_key = api_key or os.getenv("LLM_API_KEY")
    actual_base_url = base_url or os.getenv("LLM_BASE_URL")

    # 2. 根据 base_url 判断
    if actual_base_url:
        base_url_lower = actual_base_url.lower()
        if "api-inference.modelscope.cn" in base_url_lower: return "modelscope"
        if "open.bigmodel.cn" in base_url_lower: return "zhipu"
        if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
            if ":11434" in base_url_lower: return "ollama"
            if ":8000" in base_url_lower: return "vllm"
            return "local" # 其他本地端口

    # 3. 根据 API 密钥格式辅助判断
    if actual_api_key:
        if actual_api_key.startswith("ms-"): return "modelscope"
        # ... 其他密钥格式判断

    # 4. 默认返回 'auto'，使用通用配置
    return "auto"
```

一旦 provider 被确定（无论是用户指定还是自动检测），`_resolve_credentials` 方法便会接手处理服务商的差异化配置。它会根据 provider 的值，去主动查找对应的环境变量，并为其设置默认的 base_url。其部分关键实现如下：

```python
def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
    """根据provider解析API密钥和base_url"""
    if self.provider == "openai":
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        return resolved_api_key, resolved_base_url

    elif self.provider == "modelscope":
        resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
        return resolved_api_key, resolved_base_url
    
    # ... 其他服务商的逻辑
```

让我们通过一个简单的例子来感受自动检测带来的便利。假设一个用户想要使用本地的 Ollama 服务，他只需在 `.env` 文件中进行如下配置：

```bash
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL_ID="llama3"
```

他完全不需要配置 LLM_API_KEY 或在代码中指定 provider。然后，在 Python 代码中，他只需简单地实例化 HelloAgentsLLM 即可：

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

load_dotenv()

# 无需传入 provider，框架会自动检测
llm = HelloAgentsLLM() 
# 框架内部日志会显示检测到 provider 为 'ollama'

# 后续调用方式完全不变
messages = [{"role": "user", "content": "你好！"}]
for chunk in llm.think(messages):
    print(chunk, end="")
```

在这个过程中，`_auto_detect_provider` 方法通过解析 LLM_BASE_URL 中的 "localhost" 和 :11434，成功地将 provider 推断为 "ollama"。随后，`_resolve_credentials` 方法会为 Ollama 设置正确的默认参数。

相比于4.1.3节的基础实现，现在的HelloAgentsLLM具有以下显著优势：

**表 7.1 HelloAgentLLM不同版本特性对比**

| 特性 | 基础版本(4.1.3) | 扩展版本(7.2) |
|------|----------------|--------------|
| 支持的提供商 | 仅OpenAI兼容接口 | 多提供商+本地模型 |
| 配置方式 | 手动指定所有参数 | 自动检测+智能默认 |
| 扩展性 | 需修改源码 | 通过继承扩展 |
| 易用性 | 需理解底层细节 | 开箱即用 |

如上表7.1所示，这种演进体现了框架设计的重要原则：从简单开始，逐步完善。我们在保持接口简洁的同时，增强了功能的完整性。
