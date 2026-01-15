# InvestorBench 相关工作总结

> **论文**: INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent  
> **GitHub**: https://github.com/felis33/INVESTOR-BENCH  
> **发布时间**: 2024-11

---

## 1. 概述

InvestorBench 是一个用于评估 LLM-based Agent 在金融决策任务中表现的基准框架。该框架将金融决策任务形式化为**部分可观测马尔可夫决策过程（POMDP）**。

---

## 2. Agent 架构

InvestorBench 采用 **LLM-modulo 框架**，设计目标是匹配或超越专业人类投资者的能力。框架包含以下互联模块：

### 2.1 Brain/Backbone (LLM)

- **核心模块**：LLM 本身作为 Agent 的核心
- **功能**：
  - 理解、处理和生成自然语言
  - 支持复杂决策过程
  - 解释市场相关信息
  - 生成预测分析
  - 反思过去的投资决策

### 2.2 Perception（感知模块）

- **功能**：将原始市场数据转换为 LLM 兼容的结构化格式
- **输入类型**：
  - 数值信息
  - 文本信息
  - 视觉信息

### 2.3 Profile（配置模块）

**双重功能**：

1. **角色描述**：
   - 定义 Agent 为经验丰富的投资者
   - 具有专家级知识
   - 自适应风险偏好（基于历史市场动量动态调整）

2. **任务背景**：
   - 目标资产的关键特征
   - 股票历史表现
   - 价格波动
   - 行业信息

### 2.4 Memory（记忆模块）⭐

**核心创新**：基于 FinMem 的分层记忆架构

#### 2.4.1 Working Memory（工作记忆）

- **功能**：观察、总结、反思
- **反思机制**：
  - **即时反思（Immediate Reflection）**：整合当前市场指标与 Top-K 长期记忆事件
  - **扩展反思（Extended Reflection）**：输出交易方向和理由

- **阶段差异**：
  | 阶段 | 重点 |
  |------|------|
  | Warm-up | 理解市场趋势，提高预测准确性 |
  | Evaluation | 输出交易方向（Buy/Sell/Hold）和理由 |

#### 2.4.2 Layered Long-Term Memory（分层长期记忆）

- **灵感来源**：人类认知系统的信息衰减速度差异
- **实现**：向量数据库存储
- **特点**：
  - 深层：较小衰减率，保留更长时间
  - 浅层：较大衰减率，处理瞬态数据
- **优势**：适应更广泛的金融任务和数据源

### 2.5 Action（动作模块）

- **输出**：`{"Buy", "Sell", "Hold"}`
- **输入整合**：
  - 历史 PnL
  - 扩展反思结果
  - Top-K 检索记忆

**阶段差异**：

| 阶段 | 数据访问 | 功能 |
|------|---------|------|
| Warm-up | 可访问每日调整价格差 | 校准决策策略 |
| Evaluation | 仅历史数据 | 依赖认知处理能力 |

---

## 3. POMDP 形式化

### 3.1 数学定义

金融决策过程建模为**无限时域 POMDP**：

- **时间索引**：$\mathbb{T} = \{0, 1, 2, \cdots\}$
- **折扣因子**：$\alpha \in (0, 1]$

### 3.2 POMDP 组件

| 组件 | 符号 | 描述 |
|------|------|------|
| **状态空间** | $\mathcal{X} \times \mathcal{Y}$ | $\mathcal{X}$: 可观测部分，$\mathcal{Y}$: 不可观测部分 |
| **动作空间** | $\mathcal{A}$ | `{"Buy", "Sell", "Hold"}` |
| **奖励函数** | $R(o, b, a)$ | 每日 PnL |
| **观测过程** | $\{O_t\}_{t \in \mathbb{T}} \subseteq \mathcal{X}$ | 多维过程 |
| **反思过程** | $\{B_t\}_{t \in \mathbb{T}} \subseteq \mathcal{Y}$ | Agent 自我反思，每日更新 |
| **动作** | $A_t \sim \pi(\cdot | \text{prompt})$ | 语言条件策略驱动 |

### 3.3 优化目标

$$\max_{\pi \in \Pi} \mathbb{E}\left[\sum_{t \in \mathbb{T}} \alpha^t R_t^\pi\right]$$

其中：
- $R_t^\pi = R(O_t, B_t, A_t)$：每日 PnL
- $\Pi = \{\pi(\cdot | \text{prompt})\}$：所有可接受的语言条件策略集合

---

## 4. 与 StockBench 对比

| 维度 | InvestorBench | StockBench |
|------|---------------|------------|
| **形式化** | POMDP | 无明确形式化 |
| **记忆架构** | 分层长期记忆 + 工作记忆 | EpisodicMemory（单层） |
| **反思机制** | 即时反思 + 扩展反思 | 无 |
| **动作空间** | Buy/Sell/Hold | increase/decrease/hold/close |
| **风险偏好** | 自适应 | 固定 |
| **Warm-up 阶段** | 有 | 无 |

---

## 5. 可借鉴的设计

### 5.1 分层记忆架构

```python
# InvestorBench 的分层记忆概念
class LayeredLongTermMemory:
    def __init__(self):
        self.layers = {
            "shallow": VectorDB(decay_rate=0.9),   # 快速衰减
            "medium": VectorDB(decay_rate=0.5),    # 中等衰减
            "deep": VectorDB(decay_rate=0.1),      # 慢速衰减
        }
    
    def store(self, event, importance):
        # 根据重要性决定存储层
        layer = self._select_layer(importance)
        self.layers[layer].add(event)
    
    def retrieve(self, query, top_k=5):
        # 从所有层检索 Top-K
        results = []
        for layer in self.layers.values():
            results.extend(layer.search(query, top_k))
        return self._rank(results)[:top_k]
```

### 5.2 反思机制

```python
# InvestorBench 的反思机制概念
class ReflectionModule:
    def immediate_reflection(self, current_indicators, top_k_memories):
        """即时反思：整合当前指标和历史记忆"""
        prompt = f"""
        Current Market Indicators: {current_indicators}
        Relevant Historical Events: {top_k_memories}
        
        Analyze the current situation and provide reasoning.
        """
        return self.llm.generate(prompt)
    
    def extended_reflection(self, immediate_result, task_context):
        """扩展反思：输出最终决策"""
        prompt = f"""
        Immediate Analysis: {immediate_result}
        Task Context: {task_context}
        
        Output: Trading Direction (Buy/Sell/Hold) and Rationale
        """
        return self.llm.generate(prompt)
```

### 5.3 POMDP 框架

```python
# 将 StockBench 升级为 POMDP 框架
@dataclass
class POMDPState:
    observable: Dict[str, Any]      # 价格、新闻、基本面
    unobservable: Dict[str, Any]    # 市场情绪、隐藏因素
    belief: np.ndarray              # 对不可观测状态的信念

class FinancialPOMDP:
    def __init__(self, discount_factor=0.99):
        self.alpha = discount_factor
        self.action_space = ["Buy", "Sell", "Hold"]
    
    def step(self, state, action):
        # 执行动作，返回奖励和新状态
        reward = self._calculate_pnl(state, action)
        next_state = self._transition(state, action)
        return next_state, reward
    
    def optimize(self, policy):
        # 优化目标：最大化折扣累积回报
        return sum(self.alpha**t * r for t, r in enumerate(rewards))
```

---

## 6. 基准组成

InvestorBench 由四个主要组件组成：

### 6.1 数据源与市场环境

- **数据来源**：开源数据 + 第三方 API
- **API 集成**：
  - Yahoo Finance
  - SEC EDGAR
- **数据仓库**：综合多模态市场环境

### 6.2 LLM Agent

- **模块**：Brain、Perception、Profile、Memory、Action
- **外部工具**：
  - 表格数据读取器
  - API 调用器
- **数据操作**：
  - 向量数据库管理
  - 信息增强
  - 检索

### 6.3 金融决策任务

三种不同资产类型的决策任务（详见 6.4 交易环境）

### 6.4 评估指标

标准量化金融指标（详见第 7 节）

---

## 7. 交易环境

InvestorBench 提供三个数据集，构建针对特定任务的金融市场环境：

### 7.1 股票市场环境

| 数据类型 | 来源 | 说明 |
|---------|------|------|
| **OHLCV** | Yahoo Finance | 每日开高低收量 |
| **公司报告** | SEC EDGAR | 10-Q、10-K 季报/年报摘要 |
| **新闻数据** | Zhou et al. + Refinitiv Reuters | 2020-07-01 至 2021-05-06 |
| **情感分类** | GPT-3.5-turbo-0125 | positive/negative/neutral |

**股票列表**：
- MSFT、JNJ、UVV、HON（Zhou et al. 数据集）
- TSLA、AAPL、NIO（Refinitiv Real-Time News）

### 7.2 加密货币市场环境

| 数据类型 | 来源 | 说明 |
|---------|------|------|
| **OHLCV** | CoinMarketCap | 每日开高低收量 |
| **新闻数据** | cryptonews, cryptopotato, cointelegraph | 多源 |
| **时间范围** | Zhou et al. | 2023-02-13 至 2023-11-05 |
| **情感分类** | GPT-3.5-turbo-0125 | 同上 |

### 7.3 ETF 市场环境

| 数据类型 | 来源 | 说明 |
|---------|------|------|
| **新闻标题** | NIFTY 数据集 | News-Informed Financial Trend Yield |
| **时间范围** | Saqur et al. (2024) | 2019-07-29 至 2020-09-21 |
| **情感分类** | 预处理 | 每条新闻标题 |

### 7.4 数据集划分

| 阶段 | 用途 |
|------|------|
| **Train Set** | Warm-up 阶段，建立记忆数据库 |
| **Test Set** | 评估阶段，测试模型性能 |

---

## 8. 评估指标

### 8.1 主要指标（Primary）

| 指标 | 英文 | 说明 | 重要性 |
|------|------|------|--------|
| **累积收益** | Cumulative Return (CR) | 长期收益 | ⭐⭐⭐ |
| **夏普比率** | Sharpe Ratio (SR) | 风险调整后收益 | ⭐⭐⭐ |

### 8.2 次要指标（Secondary）

| 指标 | 英文 | 说明 |
|------|------|------|
| **年化波动率** | Annualized Volatility (AV) | 风险度量 |
| **最大回撤** | Maximum Drawdown (MDD) | 最大亏损幅度 |

### 8.3 评估重点

> CR 和 SR 被认为比 AV 和 MDD **更重要**，因为它们关注**长期收益**和**风险调整后回报**。

---

## 9. 测试的 LLM 模型

InvestorBench 评估了 13 个专有或开源 LLM：

### 9.1 专有模型

| 模型 | 版本 | 接口 |
|------|------|------|
| GPT-4 | 0613 | API |
| GPT-4o | 0806 | API |
| GPT-o1-preview | 0912 | API |

### 9.2 开源模型

| 模型 | 参数量 | 版本 |
|------|--------|------|
| DeepSeek-v2 | 15B | Lite |
| DeepSeek-llm | 67B | Chat |
| Qwen2.5-7b | 7B | Instruct |
| Qwen2.5-32b | 32B | Instruct |
| Qwen2.5-72b | 72B | Instruct |
| Llama3.1-8b | 8B | Instruct |
| Llama3.1-70b | 70B | Instruct |
| Yi-1.5-9b | 9B | Chat |
| Yi-1.5-34b | 34B | Chat |
| Palmyra-Fin | 70B | 32K |

---

## 10. 与 StockBench 详细对比

| 维度 | InvestorBench | StockBench |
|------|---------------|------------|
| **形式化** | POMDP | 无明确形式化 |
| **记忆架构** | 分层长期记忆 + 工作记忆 | EpisodicMemory（单层） |
| **反思机制** | 即时反思 + 扩展反思 | 无 |
| **动作空间** | Buy/Sell/Hold | increase/decrease/hold/close |
| **风险偏好** | 自适应 | 固定 |
| **Warm-up 阶段** | 有 | 无 |
| **资产类型** | 股票、加密货币、ETF | 仅股票 |
| **股票数量** | 7 只 | 20 只 DJIA |
| **评估周期** | ~10 个月 | 4 个月 |
| **数据源** | Yahoo Finance, SEC EDGAR, 新闻 | Finnhub, 新闻 |
| **情感分析** | GPT-3.5 生成 | 无 |
| **交易成本** | 未提及 | 未实现 |
| **主要指标** | CR, SR | CR, SR, MDD, Sortino |

---

## 11. 关键引用

```bibtex
@inproceedings{li-etal-2025-investorbench,
    title = "{INVESTORBENCH}: A Benchmark for Financial Decision-Making Tasks with {LLM}-based Agent",
    author = "Li, Haohang and Cao, Yupeng and Yu, Yangyang and ...",
    booktitle = "Proceedings of ...",
    year = "2025"
}
```

---

## 12. 参考资源

- **GitHub**: https://github.com/felis33/INVESTOR-BENCH
- **FinMem 论文**: Yu et al. (2024a) - FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory
- **POMDP 参考**: Bertsekas and Shreve (1996); Liu et al. (2020); Kabbani and Duman (2022)
- **评估指标参考**:
  - Cumulative Return: Hull (2007)
  - Sharpe Ratio: Sharpe (1994)
  - Annualized Volatility: Cochrane (1988)
  - Maximum Drawdown: Ang and Chen (2003)
- **数据集参考**:
  - 新闻数据: Zhou et al. (2021)
  - NIFTY 数据集: Saqur et al. (2024)
  - 加密货币新闻: Vanhoucke (2023)

---

## 13. 实验与讨论

### 13.1 实验设置

**基准策略**：
- **单资产交易**：Buy and Hold 策略
- **投资组合管理**：等权重投资组合

**实验参数**：
- **温度参数**：0.6（平衡响应一致性和推理创造性）
- **评估方法**：5 次重复实验的中位数 CR、SR、AV、MDD

**时间划分**：

| 任务 | Warm-up 期间 | 测试期间 |
|------|-------------|---------|
| **股票交易** | 2020-07-01 ~ 2020-09-30 | 2020-10-01 ~ 2021-05-06 |
| **加密货币交易** | 2023-02-11 ~ 2023-04-04 | 2023-04-05 ~ 2023-11-05 |
| **ETF 交易** | 2019-07-29 ~ 2019-12-30 | 2020-01-02 ~ 2020-09-21 |

**硬件配置**：

| 模型规模 | GPU 配置 |
|---------|---------|
| 小规模 (<10B) | 2× RTX A6000 (48GB) |
| 中规模 (10B-65B) | 4× RTX A6000 |
| 大规模 (>65B) | 8× A100 (80GB) |

### 13.2 股票交易结果

**测试股票**：MSFT、JNJ、UVV、HON、TSLA、AAPL、NIO

#### 关键发现

1. **专有模型表现最优**
   - GPT-4o 平均 CR: 39.03%，SR: 0.718
   - 相比开源和金融领域微调模型，专有模型展现更高且更一致的 CR 和 SR
   - 金融领域微调模型（如 Palmyra-Fin-70B）在序列决策任务中未显示决定性优势

2. **模型参数规模影响决策质量**
   - 开源模型中，>67B 参数的模型展现更优的 CR 和 SR
   - 大模型在类别内的方差明显更小
   - 验证了 LLM 推理能力与参数规模成正比的观点

3. **复杂市场条件下专有模型优势明显**
   - TSLA 和 NIO 呈现混合涨跌趋势（复杂市场）
   - 其他五只股票呈现牛市趋势（单调市场）
   - 专有模型在复杂市场中能更好地利用历史动量、当前持仓和自我反思结果

#### 股票交易性能对比（平均值）

| 模型类型 | 代表模型 | CR↑ | SR↑ | AV↓ | MDD↓ |
|---------|---------|-----|-----|-----|------|
| **基准** | Buy & Hold | 34.10% | 0.505 | 51.07 | 34.95 |
| **专有模型** | GPT-4o | 39.03% | 0.718 | 39.08 | 21.23 |
| **大规模开源** | Qwen2.5-72B | 46.15% | 0.880 | 38.42 | 19.52 |
| **中规模开源** | Yi-1.5-34B | 37.97% | 0.573 | 50.31 | 34.32 |
| **小规模开源** | Yi-1.5-9B | 22.91% | 0.330 | 49.99 | 36.95 |
| **金融领域** | Palmyra-Fin-70B | -0.45% | 0.021 | 43.93 | 36.56 |

### 13.3 加密货币交易结果

**测试资产**：Bitcoin (BTC)、Ethereum (ETH)

#### 关键发现

- **大规模开源模型和专有模型**才能有效捕捉加密货币市场的交易信号
- 加密货币市场对新闻和金融情绪高度敏感
- 中小规模开源模型的 CR 和 SR 普遍低于市场基准

#### 加密货币交易性能对比

| 模型类型 | BTC CR↑ | BTC SR↑ | ETH CR↑ | ETH SR↑ |
|---------|---------|---------|---------|---------|
| **基准** | 21.82% | 0.989 | 4.53% | 0.211 |
| **专有模型平均** | 23.60% | 1.195 | 2.89% | 0.158 |
| **开源模型平均** | 14.14% | 0.795 | 1.02% | 0.090 |
| **最佳开源** | DeepSeek-67B: 28.31% | 1.290 | Qwen2.5-72B: 11.98% | 0.846 |

### 13.4 ETF 交易结果

#### 关键发现

- ETF 投资需要**专有模型**提供丰富的预训练知识和稳健的推理支持
- ETF 交易复杂性高，需要解读跨行业的可操作信号
- 需要更具战略性的长期决策，依赖深度理解和反思

#### ETF 交易性能对比

| 模型 | CR↑ | SR↑ | AV↓ | MDD↓ |
|------|-----|-----|-----|------|
| **Buy & Hold** | 2.07% | 0.06 | 46.65 | 35.75 |
| **Palmyra-Fin-70B** | 24.76% | 1.152 | 30.42 | 8.20 |
| **GPT-o1-preview** | 21.22% | 0.849 | 43.77 | 20.05 |
| **Qwen2.5-32B** | 19.62% | 0.955 | 29.07 | 7.50 |
| **开源模型平均** | 5.83% | 0.282 | 28.84 | 14.84 |
| **专有模型平均** | 12.11% | 0.445 | 44.87 | 30.17 |

### 13.5 综合讨论

#### 核心结论

1. **模型选择至关重要**
   - 不同 LLM 在股票、加密货币、ETF 交易中表现差异显著
   - 反映了金融市场的内在复杂性
   - 强调了模型选择或微调的重要性

2. **专有模型优势**
   - 在股票交易中表现最佳，得益于丰富的金融数据集训练
   - 开源模型在波动性较大的环境（如加密货币）中难以达到同等效果

3. **适应性是关键**
   - LLM Agent 的有效性高度依赖于其适应市场波动的能力
   - 具备**高级记忆系统**和**动态风险评估能力**的 Agent 更能应对复杂市场

4. **架构特征的价值**
   - 复杂的 LLM Agent 框架架构特征在金融决策任务中具有重要价值
   - 分层记忆、反思机制等设计对提升性能有显著帮助

#### 对我们项目的启示

| 启示 | 建议 |
|------|------|
| **模型选择** | 优先考虑大规模模型（>67B）或专有模型作为 backbone |
| **记忆系统** | 实现分层记忆架构，区分短期和长期信息 |
| **反思机制** | 加入即时反思和扩展反思模块 |
| **风险评估** | 实现动态风险评估，适应市场波动 |
| **资产特性** | 针对不同资产类型（股票/加密货币/ETF）调整策略 |

---

## 14. 相关工作

### 14.1 金融领域的 LLM

通用领域语言模型的快速发展推动了金融 LM 的探索：

#### 预训练语言模型

| 模型 | 参考文献 |
|------|---------|
| **FinBERT** | Liu et al., 2021; Yang et al., 2020; Araci, 2019; Huang et al., 2023 |
| **FinBERT-MRC** | Zhang and Zhang, 2023 |
| **FLANG** | Shah et al., 2022 |

#### 金融大语言模型

| 模型 | 参考文献 | 特点 |
|------|---------|------|
| **FinGPT** | Liu et al., 2023 | 开源金融 LLM |
| **FinMA** | Xie et al., 2023 | 金融多任务 |
| **InvestLM** | Yang et al., 2023 | 投资决策 |
| **BloombergGPT** | Wu et al., 2023 | 大规模金融数据训练 |

这些模型利用多样化金融数据集（股价数据、金融新闻、分析师报告）进行训练，将 LM 能力适配到金融应用的独特需求。

#### 金融 Agent 框架

| 框架 | 参考文献 | 特点 |
|------|---------|------|
| **FinMem** | Yu et al., 2024a | 分层记忆架构 |
| **FinAgent** | Zhang et al., 2024a | 多模态金融 Agent |
| **FinRobot** | Yang et al., 2024 | 开放式金融 Agent |

**挑战**：框架设计、任务范围和数据类型的差异，使得统一评估 LLM Agent 在金融场景中的效果面临困难。

### 14.2 金融 LLM 基准

| 基准 | 参考文献 | 任务数量 | 特点 |
|------|---------|---------|------|
| **FLUE** | Shah et al., 2022 | 5 | 首个综合金融 NLP 基准（情感分析、标题分类、NER、结构边界检测、QA） |
| **Pixiu** | Xie et al., 2023 | - | 扩展至文档理解和分类，包含多模态数据集 |
| **FinBen** | Xie et al., 2024 | 24 | 36 个数据集，覆盖 24 个金融任务 |

**现有差距**：尽管有这些进展，专门为金融领域 LLM Agent 应用设计的基准仍然缺乏。

---

## 15. 结论

### 15.1 InvestorBench 的两种使用模式

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| **模式一** | 将微调后的 LLM 集成到 InvestorBench 的 Agent 框架中 | 对比自己模型与已有模型的性能 |
| **模式二** | 将 InvestorBench 的环境和评估指标集成到自己设计的 Agent 中 | 评估自己 Agent 设计的有效性 |

这种双模式方法为在 InvestorBench 生态系统中测试和增强金融决策策略提供了灵活的框架。

### 15.2 未来研究方向

**扩展信息模态**：
- **音频**：如财报电话会议录音
- **图表**：如 K 线图、交易图表

**目标**：探索这些数据类型是否能提升决策质量。

**框架设计**：InvestorBench 的基础 Agent 框架设计为可无缝适配这些模态，确保扩展后的基准易于使用且可扩展。

---

## 16. 局限性

| 局限性 | 描述 |
|--------|------|
| **单资产聚焦** | 目前仅关注单资产金融决策任务，未涉及多资产任务（如投资组合管理） |
| **数据版权限制** | 金融领域数据的版权限制可能影响数据集质量，限制模型性能评估 |

---

## 17. 总结与启示

### 17.1 InvestorBench 核心贡献

1. **POMDP 形式化**：将金融决策任务形式化为部分可观测马尔可夫决策过程
2. **分层记忆架构**：基于 FinMem 的多层记忆系统
3. **双阶段评估**：Warm-up + Evaluation 阶段
4. **多资产覆盖**：股票、加密货币、ETF 三类资产
5. **全面模型评估**：13 个 LLM 的系统性对比

### 17.2 对 Trading Agent 项目的借鉴价值

| 方面 | 借鉴点 |
|------|--------|
| **架构设计** | 采用 POMDP 框架，明确状态空间、动作空间、奖励函数 |
| **记忆系统** | 实现分层长期记忆 + 工作记忆的双层架构 |
| **反思机制** | 加入即时反思和扩展反思模块 |
| **评估体系** | 使用 CR、SR、AV、MDD 四个标准指标 |
| **模型选择** | 优先考虑大规模模型（>67B）或专有模型 |
| **Warm-up 阶段** | 在正式评估前进行预热，建立记忆数据库 |
