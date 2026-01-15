# StockBench 论文审稿意见总结

> **投稿编号**: Submission 10010  
> **审稿日期**: 2025年11月  
> **综合评分**: Reject (2/10)

---

## 📊 评审概览

| 审稿人 | 评分 | Soundness | Presentation | Contribution | 置信度 |
|--------|------|-----------|--------------|--------------|--------|
| **Reviewer YVFQ** | 2 (Reject) | 2 (Fair) | 3 (Good) | 2 (Fair) | 4/5 |
| **Reviewer FKzD** | 2 (Reject) | 3 (Good) | 3 (Good) | 2 (Fair) | 4/5 |
| **Reviewer 5AJA** | 4 (Marginally Below) | 2 (Fair) | 3 (Good) | 2 (Fair) | 4/5 |
| **Reviewer XKW2** | 2 (Reject) | 1 (Poor) | 2 (Fair) | 2 (Fair) | 4/5 |

---

## 🔴 Reviewer YVFQ 详细意见

### 优点 (Strengths)
- ✅ 模拟环境解决了**数据污染问题**（contamination-free）

### 主要问题 (Weaknesses)

#### 1. **创新性不足**
> "The novelty is quite limited."

- 已有大量类似工作评估 LLM 在股票市场的能力：
  - **InvestorBench**
  - **StockAgent**
  - **Agent Market Arena**
- 审稿人质疑：为什么不直接在**当前市场**进行实时评估？这样可以完全消除数据泄露问题

#### 2. **提出的问题**
- ❓ 为什么只选择 **DJIA 前 20 只股票**？
- ❓ 论文承认 LLM 在熊市表现不佳，评估周期影响巨大，但仍然基于**单一的 4 个月窗口**得出结论？

---

## 🔴 Reviewer FKzD 详细意见

### 优点 (Strengths)
- ✅ 提出了 **"contamination-free"** 概念，为时序领域的 LLM 评估提供了新视角
- ✅ 正确指出：在历史数据上测试可能评估的是记忆能力，而非推理能力

### 主要问题 (Weaknesses)

#### 1. **评估周期过短**
> "Short evaluation period"

- 主要评估期仅 **4 个月**
- 这期间 Buy-and-Hold 基准几乎持平（0.4% 收益）
- 这使得基准**太容易被超越**
- 论文自己的分析（Sec 4.4）显示：在下跌期间，**所有 Agent 都无法超越基准**
- **结论**：盈利能力的声明高度依赖于所选市场环境，**不具有普遍性**

#### 2. **投资组合构建理由薄弱**
> "The justification for the portfolio construction is weak"

- 选择 "DJIA 权重最高的 20 只股票"，但未解释为什么选择这种方法
- 未考虑：
  - 随机抽样
  - 行业分层抽样
- 引入了**显著的选择偏差**
- Agent 的表现可能无法推广到这 20 只股票之外

#### 3. **摘要与主要结果矛盾**
> "Major contradiction between the Abstract and the Main Results"

| 位置 | 陈述 |
|------|------|
| **摘要** | "...most LLM agents **struggle to outperform** the simple buy-and-hold baseline..." |
| **Section 3.2** | "Most tested models **outperform** the passive buy-and-hold baseline..." |

⚠️ **这是一个严重的自相矛盾！**

#### 4. **更微妙的数据污染问题被忽视**
> "Ignores a more-subtle and important form of contamination: memorized behavioral priors"

- 选择的 20 只资产是**世界上分析最多的股票**（DJIA 顶级成分股）
- LLM 很可能从训练数据中**记忆了这些公司的典型行为**：
  - 季节性模式
  - 特征性反应
  - 例如：模型可能"知道" Stock X 对油价高度敏感，或 Stock Y 通常超预期盈利
- **这也是一种数据泄露**，使用近期数据并不能解决
- 这是论文相对于现有基准的**唯一创新点**，需要更仔细的设计

### 提出的问题

> ❓ "Have you considered testing the agents on a 'shadow portfolio' of 'twin stocks'?"

建议测试**影子投资组合**：
- 选择与 20 只股票**高度相关的竞争对手**（同行业）
- 例如：
  - Bank of America 代替 JPM
  - Oracle 代替 Salesforce
- 目的：揭示 Agent 是否学到了**可泛化的行业策略**，还是仅仅是特定 20 只股票的**人工产物**

---

## 📋 问题汇总与优先级

### 🔴 必须解决的问题（Critical）

| # | 问题 | 来源 | 建议解决方案 |
|---|------|------|-------------|
| 1 | **摘要与结果矛盾** | FKzD | 统一表述，确保一致性 |
| 2 | **评估周期过短（4个月）** | YVFQ, FKzD | 扩展到多个市场周期（牛市、熊市、震荡市） |
| 3 | **创新性不足** | YVFQ | 与 InvestorBench、StockAgent 等明确区分 |
| 4 | **行为先验污染未解决** | FKzD | 设计更严格的 contamination-free 方案 |

### 🟡 应该解决的问题（Important）

| # | 问题 | 来源 | 建议解决方案 |
|---|------|------|-------------|
| 5 | **为什么只选 DJIA 前 20？** | YVFQ, FKzD | 提供选择理由，或扩展到更多股票 |
| 6 | **选择偏差** | FKzD | 增加随机抽样或行业分层抽样对比 |
| 7 | **结论不具普遍性** | FKzD | 在多个市场环境下验证 |

### 🟢 可选改进（Nice to Have）

| # | 问题 | 来源 | 建议解决方案 |
|---|------|------|-------------|
| 8 | **影子投资组合测试** | FKzD | 测试同行业竞争对手股票 |
| 9 | **实时市场评估** | YVFQ | 考虑在当前市场进行实时评估 |

---

## 🎯 修改建议

### 1. 解决摘要与结果矛盾
```diff
- Abstract: "most LLM agents struggle to outperform..."
+ Abstract: "In favorable market conditions, most LLM agents outperform the buy-and-hold baseline, 
+           but struggle in bearish markets..."
```

### 2. 扩展评估周期
- 增加至少 **3 个不同市场环境**：
  - 牛市（如 2023 Q4）
  - 熊市（如 2022 Q1-Q2）
  - 震荡市（如 2023 Q2）
- 分别报告各环境下的表现

### 3. 解决行为先验污染
- **方案 A**：使用**非 DJIA 股票**（如中小盘股、新上市公司）
- **方案 B**：使用**影子投资组合**（同行业竞争对手）
- **方案 C**：使用**合成数据**（保留统计特性但改变具体公司）

### 4. 增强创新性论述
- 明确与现有工作的区别：
  - InvestorBench：侧重于...，我们侧重于...
  - StockAgent：使用...方法，我们使用...
  - Agent Market Arena：评估...，我们评估...

### 5. 解释股票选择
```markdown
我们选择 DJIA 前 20 只股票的原因：
1. 流动性高，数据质量好
2. 覆盖多个行业
3. 便于与其他研究对比
4. 限制：可能存在选择偏差，未来工作将扩展到更广泛的股票池
```

---

## 📝 回复审稿人模板

### 回复 Reviewer YVFQ

> **Q1: 为什么只选择 DJIA 前 20 只股票？**

我们选择 DJIA 前 20 只股票是因为：(1) 高流动性确保数据质量；(2) 覆盖多个行业；(3) 便于与现有研究对比。我们承认这可能引入选择偏差，在修订版中，我们将增加 [具体方案]。

> **Q2: 为什么基于单一 4 个月窗口得出结论？**

感谢您的指出。在修订版中，我们将评估扩展到 [X] 个不同市场环境，包括牛市、熊市和震荡市，以验证结论的普遍性。

> **Q3: 与现有工作的区别？**

我们的工作与 InvestorBench、StockAgent、Agent Market Arena 的主要区别在于：[具体区别]。我们将在修订版中更清晰地阐述这些区别。

### 回复 Reviewer FKzD

> **Q1: 摘要与结果矛盾**

感谢您指出这一重要问题。这是我们的疏忽，我们将统一表述为：[修正后的表述]。

> **Q2: 行为先验污染**

这是一个非常深刻的观察。我们承认使用 DJIA 顶级成分股可能引入行为先验污染。在修订版中，我们将：(1) 增加影子投资组合测试；(2) 分析模型对不同股票的泛化能力。

> **Q3: 影子投资组合建议**

这是一个极好的建议。我们将在修订版中增加对同行业竞争对手的测试，以验证策略的可泛化性。

---

## 📅 修改计划

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| 1 | 修复摘要与结果矛盾 | 1 天 |
| 2 | 扩展评估周期（多市场环境） | 1-2 周 |
| 3 | 增加影子投资组合测试 | 1 周 |
| 4 | 增强创新性论述 | 2-3 天 |
| 5 | 解释股票选择理由 | 1 天 |
| 6 | 撰写审稿人回复 | 2-3 天 |

---

## 🔗 相关工作参考

- **InvestorBench**: [需要查阅具体论文]
- **StockAgent**: [需要查阅具体论文]
- **Agent Market Arena**: [需要查阅具体论文]
- **FinMem**: [需要查阅具体论文]
- **FinAgent**: [需要查阅具体论文]

---

## 🟡 Reviewer 5AJA 详细意见

### 优点 (Strengths)
- ✅ 研究解决了一个**有趣的现实挑战**：评估 AI Agent 在股票交易模拟中的能力
- ✅ 选择 2025 年时间段是**深思熟虑的**，有助于缓解数据泄露问题
- ✅ 论文**写作清晰**，易于理解，组织良好

### 主要问题 (Weaknesses)

#### 1. **与现有股票交易 Agent 的对比不足**
> "It would be helpful to clarify how the proposed AI agents compare to existing stock-trading agents."

- 已有多项研究探索 LLM 或 AI Agent 在股票交易中的潜力
- 审稿人对本工作的**创新性不够信服**

#### 2. **无法处理突发事件**
> "The paper does not appear to demonstrate that the proposed system can handle or adapt to such abrupt disruptions."

- 股票交易高度复杂，常受**突发或不可预测事件**影响
- 例如：COVID-19 疫情导致的剧烈市场波动
- 论文未展示系统能够**处理或适应**此类突发中断

#### 3. **时间框架可能不足**
> "I wonder if this relatively short time frame can be insufficient to capture long-term trends."

- 为避免数据泄露，研究使用 2025 年内的多个时间段
- 但这个相对较短的时间框架可能**不足以捕捉长期趋势**
- 也无法验证模型预测的**稳健性**，尤其是对于波动性较大的股票

### 提出的问题

> ❓ **Q1: 外部研究者如何访问或使用这个基准测试自己的模型？**

虽然论文将 STOCKBENCH 呈现为 AI Agent 的基准，但它更像是一个**股票交易框架**。需要澄清外部研究者如何访问和使用。

> ❓ **Q2: 基准的主要目的是什么？**

- 是用于**性能测量**（评估交易结果）？
- 还是用于 **Agent 开发**（测试新架构和策略）？

> ❓ **Q3: 是否考虑与现有 AI 交易 Agent 或传统算法方法进行比较？**

建议增加与现有方法的对比，以更好地**定位基准结果**。

---

## 🔴 Reviewer XKW2 详细意见

### 优点 (Strengths)
- ✅ 针对**静态金融 QA 与动态交易评估之间的有意义差距**
- ✅ 有趣的发现：推理调优模型（如 Qwen-Think）在实践中**并不优于**简单的指令调优版本

### 主要问题 (Weaknesses)

#### 1. **评估周期严重不足**（最严重问题）
> "The evaluation period is only four months, which is clearly insufficient for daily trading evaluation."

- 4 个月的窗口**太短**，无法得出任何**统计意义上的结论**
- 审稿人更希望看到**更长的评估周期**，即使可能引入前瞻偏差
- 现有文献已表明：在约 **20 年**的更长时间范围内测试，LLM 仍然无法跑赢市场（即使存在潜在数据泄露）
- 这将提供**更可信的基准**

#### 2. **未引用关键先前工作**
> "Key prior works such as FinMem and FinAgent are not mentioned, compared, or benchmarked."

- 论文声称评估 "LLM agents"，但未提及、对比或基准测试关键先前工作：
  - **FinMem**
  - **FinAgent**
- 没有这些参考，**Agent 能力评估的声明不令人信服**
- 演示框架**没有展示**工具使用或记忆管理的工作流程
- 这些是 **Agent 设计的核心组件**
- 整体描述**高度抽象**，缺乏足够的技术细节来支撑创新性声明

#### 3. **模拟设计不完整**（严重问题）
> "The simulation design itself appears incomplete and not properly implemented."

- 没有考虑基本的**交易现实性**：
  - ❌ 佣金费用（Commission fees）
  - ❌ 滑点（Slippage）
  - ❌ 流动性上限（Liquidity caps）
- 这些对于评估**真实市场中的盈利能力**至关重要
- 忽略这些因素导致**过于乐观**且**实际上无意义**的结果
- **削弱了评估的有效性**

#### 4. **缺乏统计稳健性检验**
> "The results are presented without any statistical robustness checks."

- 没有分析：
  - ❌ 显著性检验（Significance tests）
  - ❌ 置信区间（Confidence intervals）
  - ❌ 方差分解（Variance decomposition）
- 无法确定观察到的性能差异是**有意义的**还是**随机波动**
- LLM 输出受**采样随机性**影响（如 temperature）
- 这些**不确定性来源未被量化或控制**

### 提出的问题

> ❓ **Q1: 能否提供 Agent 输入、推理轨迹和动作输出的具体示例？**

以澄清决策是如何做出的。

> ❓ **Q2: Temperature 设置是多少？**

> ❓ **Q3: 收益差异是否具有统计显著性？Alpha 和 Beta 是多少？**

---

## 📋 更新后的问题汇总与优先级

### 🔴 必须解决的问题（Critical）

| # | 问题 | 来源 | 建议解决方案 |
|---|------|------|-------------|
| 1 | **摘要与结果矛盾** | FKzD | 统一表述，确保一致性 |
| 2 | **评估周期过短（4个月）** | YVFQ, FKzD, 5AJA, XKW2 | 扩展到多个市场周期（牛市、熊市、震荡市） |
| 3 | **创新性不足** | YVFQ, 5AJA | 与 InvestorBench、StockAgent、FinMem、FinAgent 等明确区分 |
| 4 | **行为先验污染未解决** | FKzD | 设计更严格的 contamination-free 方案 |
| 5 | **模拟设计不完整（无佣金/滑点/流动性）** | XKW2 | 添加交易成本模拟 |
| 6 | **缺乏统计稳健性检验** | XKW2 | 添加显著性检验、置信区间、方差分析 |
| 7 | **未引用关键先前工作（FinMem, FinAgent）** | XKW2 | 添加对比和引用 |

### 🟡 应该解决的问题（Important）

| # | 问题 | 来源 | 建议解决方案 |
|---|------|------|-------------|
| 8 | **为什么只选 DJIA 前 20？** | YVFQ, FKzD | 提供选择理由，或扩展到更多股票 |
| 9 | **选择偏差** | FKzD | 增加随机抽样或行业分层抽样对比 |
| 10 | **结论不具普遍性** | FKzD | 在多个市场环境下验证 |
| 11 | **无法处理突发事件** | 5AJA | 增加突发事件场景测试 |
| 12 | **缺乏工具使用/记忆管理展示** | XKW2 | 添加 Agent 工作流程图和示例 |
| 13 | **基准定位不清（框架 vs 基准）** | 5AJA | 明确定位和使用方式 |

### 🟢 可选改进（Nice to Have）

| # | 问题 | 来源 | 建议解决方案 |
|---|------|------|-------------|
| 14 | **影子投资组合测试** | FKzD | 测试同行业竞争对手股票 |
| 15 | **实时市场评估** | YVFQ | 考虑在当前市场进行实时评估 |
| 16 | **与传统算法方法对比** | 5AJA | 增加与传统量化策略的对比 |
| 17 | **提供 Agent 推理轨迹示例** | XKW2 | 添加具体的输入/推理/输出示例 |
| 18 | **报告 Temperature 设置** | XKW2 | 明确实验参数 |
| 19 | **报告 Alpha 和 Beta** | XKW2 | 添加风险调整后的收益分析 |

---

## 📝 回复审稿人模板（续）

### 回复 Reviewer 5AJA

> **Q1: 外部研究者如何访问或使用这个基准？**

感谢您的问题。STOCKBENCH 是开源的，我们将在修订版中添加详细的使用文档，包括：(1) 如何安装和配置；(2) 如何添加自定义模型；(3) 如何运行基准测试。

> **Q2: 基准的主要目的是什么？**

STOCKBENCH 的主要目的是**性能测量**，即评估 LLM Agent 在真实交易场景中的决策能力。同时，它也可以用于 Agent 开发，测试新的架构和策略。

> **Q3: 是否考虑与现有 AI 交易 Agent 对比？**

这是一个很好的建议。在修订版中，我们将增加与 FinMem、FinAgent 等现有框架的对比分析。

### 回复 Reviewer XKW2

> **Q1: 能否提供 Agent 输入、推理轨迹和动作输出的具体示例？**

感谢您的建议。在修订版中，我们将添加完整的 Agent 工作流程示例，包括：(1) 输入数据格式；(2) 推理过程；(3) 决策输出。

> **Q2: Temperature 设置是多少？**

我们使用 temperature=0.0 以确保结果的可复现性。我们将在修订版中明确说明这一点。

> **Q3: 收益差异是否具有统计显著性？**

这是一个重要的问题。在修订版中，我们将添加：(1) 显著性检验；(2) 置信区间；(3) Alpha 和 Beta 分析。

> **关于模拟设计不完整的问题**

感谢您指出这一重要问题。在修订版中，我们将添加：(1) 佣金费用模拟；(2) 滑点模型；(3) 流动性约束。这将使评估更加贴近真实市场。

> **关于未引用 FinMem 和 FinAgent**

感谢您的指出。我们将在修订版中添加对这些重要工作的引用和对比分析。

---

## 📅 更新后的修改计划

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|---------|--------|
| 1 | 修复摘要与结果矛盾 | 1 天 | 🔴 Critical |
| 2 | 添加交易成本模拟（佣金/滑点/流动性） | 3-5 天 | 🔴 Critical |
| 3 | 添加统计稳健性检验 | 2-3 天 | 🔴 Critical |
| 4 | 扩展评估周期（多市场环境） | 1-2 周 | 🔴 Critical |
| 5 | 添加 FinMem、FinAgent 引用和对比 | 2-3 天 | 🔴 Critical |
| 6 | 增强创新性论述 | 2-3 天 | 🟡 Important |
| 7 | 添加 Agent 工作流程示例 | 2 天 | 🟡 Important |
| 8 | 增加影子投资组合测试 | 1 周 | 🟢 Nice to Have |
| 9 | 解释股票选择理由 | 1 天 | 🟡 Important |
| 10 | 撰写审稿人回复 | 2-3 天 | - |
