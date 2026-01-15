# Trading Agent 项目完整应用说明书

## 📚 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 环境配置](#3-环境配置)
- [4. 快速开始](#4-快速开始)
- [5. 核心功能模块](#5-核心功能模块)
- [6. 配置详解](#6-配置详解)
- [7. 使用场景与示例](#7-使用场景与示例)
- [8. 性能分析工具](#8-性能分析工具)
- [9. 常见问题](#9-常见问题)
- [10. 高级功能](#10-高级功能)

---

## 1. 项目概述

### 1.1 项目介绍

**StockBench** 是一个基于大语言模型（LLM）的股票交易基准测试平台。它通过模拟真实的交易环境，使用历史市场数据来评估不同LLM模型在股票交易决策、风险管理和收益表现方面的能力。

### 1.2 核心特点

- **🌍 真实市场交互**：使用来自Polygon和Finnhub的高质量价格、基本面数据和及时新闻
- **🔄 连续决策制定**：多步骤工作流（投资组合 → 分析 → 交易），反映真实投资者行为
- **🔒 数据污染免疫**：使用2024年后的最新市场数据，与LLM训练语料零重叠
- **🤖 双智能体架构**：基本面筛选智能体 + 决策智能体的协同工作模式
- **📊 全面的性能分析**：包括总收益、Sortino比率、最大回撤等多维度指标

### 1.3 投资标的

选择道琼斯工业平均指数（DJIA）中权重最高的20只股票作为投资标的：

```
GS, MSFT, HD, V, SHW, CAT, MCD, UNH, AXP, AMGN,
TRV, CRM, JPM, IBM, HON, BA, AMZN, AAPL, PG, JNJ
```

---

## 2. 系统架构

### 2.1 项目目录结构

```
Trading_agent/
├── stockbench/              # 核心包
│   ├── agents/             # 交易智能体
│   │   ├── dual_agent_llm.py           # 双智能体实现
│   │   ├── fundamental_filter_agent.py  # 基本面筛选智能体
│   │   ├── backtest_report_llm.py       # 回测报告生成
│   │   └── prompts/                     # 提示词模板
│   ├── backtest/           # 回测引擎
│   │   ├── engine.py                    # 核心回测引擎
│   │   ├── metrics.py                   # 性能指标计算
│   │   ├── reports.py                   # 报告生成
│   │   ├── visualization.py             # 可视化工具
│   │   └── strategies/                  # 策略模块
│   ├── adapters/           # 数据适配器
│   │   ├── polygon_client.py            # Polygon API客户端
│   │   └── finnhub_client.py            # Finnhub API客户端
│   ├── core/               # 核心组件
│   │   ├── features.py                  # 特征构建
│   │   ├── pipeline_context.py          # 流程上下文
│   │   ├── message.py                   # 消息系统
│   │   └── decorators.py                # 装饰器工具
│   ├── llm/                # LLM客户端
│   │   └── llm_client.py                # LLM调用封装
│   ├── memory/             # 记忆系统
│   │   └── decision_memory.py           # 决策记忆管理
│   ├── tools/              # 工具函数
│   │   ├── data_tools.py                # 数据处理工具
│   │   └── registry.py                  # 工具注册表
│   └── utils/              # 通用工具
├── scripts/                # 运行脚本
│   ├── run_benchmark.sh                 # 回测运行脚本
│   ├── log_performance.py               # 性能日志分析
│   ├── log_query.py                     # 日志查询工具
│   └── log_trace.py                     # 日志追踪工具
├── storage/                # 数据存储
│   ├── cache/                           # 缓存数据
│   ├── reports/                         # 回测报告
│   └── logs/                            # 运行日志
├── docs/                   # 文档
├── config.yaml             # 主配置文件
└── requirements.txt        # 依赖包
```

### 2.2 系统工作流程

```
1. 数据获取 (Adapters)
   ↓
2. 特征构建 (Core/Features)
   ↓
3. 基本面筛选 (Fundamental Filter Agent)
   ↓
4. 交易决策 (Decision Agent)
   ↓
5. 订单执行 (Backtest Engine)
   ↓
6. 性能评估 (Metrics & Reports)
   ↓
7. 可视化展示 (Visualization)
```

---

## 3. 环境配置

### 3.1 系统要求

- **Python**: 3.10 或更高版本
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议 8GB 以上
- **磁盘空间**: 建议 10GB 以上（用于数据缓存）

### 3.2 安装步骤

#### 步骤1: 克隆项目

```bash
git clone <repository-url>
cd Trading_agent
```

#### 步骤2: 创建虚拟环境

```bash
# 使用 conda
conda create -n stockbench python=3.11
conda activate stockbench

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### 步骤3: 安装依赖

```bash
pip install -r requirements.txt
```

### 3.3 API密钥配置

需要配置以下API密钥（如果需要测试新月份或新股票）：

#### 3.3.1 环境变量方式

**Linux/macOS:**
```bash
export POLYGON_API_KEY="your_polygon_api_key"
export FINNHUB_API_KEY="your_finnhub_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export ZHIPUAI_API_KEY="your_zhipuai_api_key"
```

**Windows (PowerShell):**
```powershell
$env:POLYGON_API_KEY="your_polygon_api_key"
$env:FINNHUB_API_KEY="your_finnhub_api_key"
$env:OPENAI_API_KEY="your_openai_api_key"
$env:ZHIPUAI_API_KEY="your_zhipuai_api_key"
```

**Windows (CMD):**
```cmd
set POLYGON_API_KEY=your_polygon_api_key
set FINNHUB_API_KEY=your_finnhub_api_key
set OPENAI_API_KEY=your_openai_api_key
set ZHIPUAI_API_KEY=your_zhipuai_api_key
```

#### 3.3.2 获取免费API密钥

- **Polygon**: https://polygon.io/ (免费层可用)
- **Finnhub**: https://finnhub.io/ (免费层可用)
- **OpenAI**: https://platform.openai.com/
- **智谱AI**: https://open.bigmodel.cn/

---

## 4. 快速开始

### 4.1 第一次运行

#### 方法1: 使用脚本（推荐）

**Linux/macOS:**
```bash
bash scripts/run_benchmark.sh
```

**Windows:**
```bash
bash scripts/run_benchmark.sh
# 或者如果没有bash环境，需要手动运行Python命令
```

#### 方法2: 直接Python命令（必须先设置API Key）

**基础命令格式:**
```bash
# 设置环境变量（必需）
export OPENAI_API_KEY='your-api-key-here'

# 运行回测（--cfg 参数是必需的）
python -m stockbench.apps.run_backtest \
    --cfg config.yaml \
    --start 2025-03-01 \
    --end 2025-03-31 \
    --llm-profile openai
```

**使用离线模式（推荐，利用缓存数据）:**
```bash
export OPENAI_API_KEY='your-api-key-here'

python -m stockbench.apps.run_backtest \
    --cfg config.yaml \
    --start 2025-03-01 \
    --end 2025-03-31 \
    --llm-profile openai \
    --offline \
    --no-summary-llm
```

### 4.2 自定义回测参数

#### 使用脚本方式

编辑 `scripts/run_benchmark.sh`:

```bash
START_DATE="${START_DATE:-2025-03-01}"
END_DATE="${END_DATE:-2025-06-30}"
LLM_PROFILE="${LLM_PROFILE:-openai}"
```

或使用命令行参数：

```bash
bash scripts/run_benchmark.sh \
    --start 2025-04-01 \
    --end 2025-05-31 \
    --llm-profile deepseek-v3.1
```

#### 直接Python命令方式

**完整参数示例:**
```bash
python -m stockbench.apps.run_backtest \
    --cfg config.yaml \
    --start 2025-03-01 \
    --end 2025-03-31 \
    --llm-profile openai \
    --symbols "AAPL,MSFT,GOOGL" \
    --offline \
    --no-summary-llm \
    --benchmark-symbol SPY
```

**关键参数说明:**
- `--cfg`: 配置文件路径（**必需参数**）
- `--start`: 回测开始日期（默认：2025-03-01）
- `--end`: 回测结束日期（默认：2025-07-31）
- `--llm-profile`: LLM配置名称（如：openai, zhipuai, deepseek-v3.1）
- `--symbols`: 指定股票代码，逗号分隔（留空则使用config.yaml中的配置）
- `--offline`: 离线模式，使用缓存数据
- `--no-summary-llm`: 禁用LLM生成报告总结（加快速度）
- `--benchmark-symbol`: 基准指数（如：SPY）

### 4.3 查看结果

回测结果自动保存在：

```
storage/reports/backtest/<timestamp>_<llm_profile>/
├── backtest_report.txt          # 文本报告
├── backtest_report.json         # JSON格式报告
├── portfolio_log.json           # 投资组合日志
├── trade_log.json               # 交易日志
└── visualizations/              # 可视化图表
    ├── cumulative_return.png
    ├── drawdown.png
    └── position_heatmap.png
```

---

## 5. 核心功能模块

### 5.1 智能体系统 (Agents)

#### 5.1.1 双智能体架构

**基本面筛选智能体 (Fundamental Filter Agent)**
- **功能**: 从20只股票中筛选出需要详细分析的股票
- **输入**: 所有股票的价格数据、新闻、基本面数据
- **输出**: 筛选后的股票列表（通常5-10只）
- **配置参数**:
  - `temperature`: 0.6（较低温度确保稳定筛选）
  - `max_tokens`: 8192
  - `prompt`: "fundamental_filter_v1.txt"

**决策智能体 (Decision Agent)**
- **功能**: 对筛选后的股票做出交易决策
- **输入**: 筛选后的股票数据 + 当前投资组合状态
- **输出**: 每只股票的交易决策（buy/sell/hold）
- **配置参数**:
  - `temperature`: 0.6
  - `max_tokens`: 8192
  - `prompt`: "decision_agent_v1.txt"

#### 5.1.2 决策动作类型

- **increase**: 增加持仓
- **decrease**: 减少持仓
- **hold**: 保持不变
- **close**: 清仓

#### 5.1.3 回测报告生成

**功能**: 使用LLM生成专业的回测报告
- 分析交易策略的优劣
- 识别关键交易时机
- 提供改进建议

### 5.2 回测引擎 (Backtest)

#### 5.2.1 核心引擎 (engine.py)

**Portfolio类**: 投资组合管理
```python
- cash: 现金余额
- positions: 持仓字典 {symbol: Position}
- equity(): 计算总权益
- get_position_value(): 获取持仓市值
- update_cash(): 安全更新现金
```

**Position类**: 单个持仓
```python
- shares: 持股数量
- avg_price: 平均成本
- holding_days: 持有天数
- total_cost: 累计投资成本
```

**BacktestEngine类**: 回测执行引擎
- 处理每日交易
- 计算滑点和佣金
- 管理投资组合状态
- 记录交易历史

#### 5.2.2 性能指标 (metrics.py)

**核心指标**:
- **Total Return**: 总收益率
- **Sortino Ratio**: 索提诺比率（下行风险调整后收益）
- **Maximum Drawdown**: 最大回撤
- **Sharpe Ratio**: 夏普比率
- **Win Rate**: 胜率
- **Average Trade**: 平均交易收益

#### 5.2.3 交易成本

```yaml
commission_bps: 1.0      # 佣金 (基点，1bp = 0.01%)
slippage_bps: 2.0        # 滑点 (基点)
fill_ratio: 1.0          # 成交比例
```

### 5.3 数据适配器 (Adapters)

#### 5.3.1 Polygon客户端

**功能**:
- 获取股票日K线数据
- 获取股票基本面数据
- 支持股票分割和分红调整

**API端点**:
- `/v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}` - K线数据
- `/v3/reference/tickers/{symbol}` - 股票信息

#### 5.3.2 Finnhub客户端

**功能**:
- 获取公司新闻
- 获取基本面财务数据
- 获取市场情绪指标

**API端点**:
- `/company-news` - 公司新闻
- `/stock/metric` - 股票指标
- `/stock/profile2` - 公司档案

### 5.4 特征构建 (Core/Features)

#### 5.4.1 价格特征

- **close_7d**: 最近7个交易日收盘价序列
- **day_ret**: 日收益率
- **volatility**: 波动率
- **moving_averages**: 移动平均线（MA5, MA10, MA20）

#### 5.4.2 基本面特征

- **market_cap**: 市值
- **pe_ratio**: 市盈率
- **dividend_yield**: 股息率
- **52w_high/low**: 52周最高/最低价

#### 5.4.3 新闻特征

- **news_count**: 新闻数量
- **news_sentiment**: 新闻情绪得分
- **top_events**: 重要新闻事件（最多5条）

### 5.5 记忆系统 (Memory)

#### 5.5.1 决策记忆

**功能**: 记录历史决策，帮助智能体学习
- 记录每次交易决策
- 记录决策结果
- 支持历史回溯

**配置**:
```yaml
history:
  max_records_per_symbol: 7    # 每只股票最多保留7条历史记录
  max_history_days: 30         # 最多保留30天的历史记录
```

### 5.6 可视化工具 (Visualization)

#### 5.6.1 累计收益曲线

显示策略与基准（SPY或个股买入持有）的累计收益对比。

#### 5.6.2 回撤曲线

显示策略的历史最大回撤情况。

#### 5.6.3 持仓热力图

显示不同时间段的持仓分布。

#### 5.6.4 多周期分析

- **性能热力图**: 不同时间窗口的收益表现
- **滚动Sortino比率**: 滚动窗口的风险调整收益
- **滚动Sharpe比率**: 滚动窗口的夏普比率
- **排名变化**: 各股票表现排名变化

---

## 6. 配置详解

### 6.1 config.yaml 结构

```yaml
# 股票池配置
symbols_universe: [GS, MSFT, HD, ...]

# 数据模式配置
data:
  mode: auto  # auto | offline_only

# 特征配置
features:
  history:
    price_series_days: 7  # 价格序列天数

# 新闻配置
news:
  lookback_days: 2       # 新闻回溯天数
  page_limit: 100        # 新闻页数限制
  top_k_event_count: 5   # 选取最重要的K条新闻

# 投资组合配置
portfolio:
  total_cash: 100000     # 初始资金
  min_cash_ratio: 0.0    # 最小现金储备比例

# 智能体配置
agents:
  mode: "dual"           # 双智能体模式
  dual_agent:
    fundamental_filter:
      temperature: 0.6
      max_tokens: 8192
      prompt: "fundamental_filter_v1.txt"
    decision_agent:
      temperature: 0.6
      max_tokens: 8192
      prompt: "decision_agent_v1.txt"
  retry:
    max_attempts: 3      # 业务级重试次数

# LLM配置文件
llm_profiles:
  openai:
    provider: "openai"
    model: "oss-120b"
    timeout_sec: 360
    retry:
      max_retries: 3
      backoff_factor: 0.5

# 缓存配置
cache:
  mode: llm_write_only   # off | llm_write_only | full

# 日志配置
logging:
  console_level: INFO
  file_level: INFO

# 回测配置
backtest:
  warmup_days: 15        # 预热天数
  enable_detailed_logging: true
  commission_bps: 1.0    # 佣金(基点)
  slippage_bps: 2.0      # 滑点(基点)
  max_positions: 20      # 最大持仓数量
  summary_llm: true      # 是否生成LLM总结报告
```

### 6.2 数据模式详解

#### auto模式（推荐）
- 优先使用本地缓存
- 缓存缺失时自动调用API获取
- 适合正常使用场景

#### offline_only模式
- 仅使用本地缓存数据
- 不会调用任何外部API
- 适合离线测试或API配额有限时

### 6.3 LLM配置详解

#### 支持的LLM提供商

1. **OpenAI**
```yaml
openai:
  provider: "openai"
  base_url: ""
  model: "oss-120b"
  auth_required: true
```

2. **智谱AI (ZhipuAI)**
```yaml
zhipuai:
  provider: "zhipuai"
  base_url: "https://open.bigmodel.cn/api/paas/v4"
  model: "glm-4.5"
  auth_required: true
```

3. **DeepSeek**
```yaml
deepseek-v3.1:
  provider: "openai"
  model: "deepseek-v3.1-250821"
  timeout_sec: 180
```

4. **本地vLLM**
```yaml
vllm:
  provider: "vllm"
  base_url: "http://localhost:8000/v1"
  model: "Qwen/Qwen2.5-7B-Instruct"
  auth_required: false
```

5. **本地Ollama**
```yaml
ollama:
  provider: "ollama"
  base_url: "http://localhost:11434/v1"
  model: "llama3"
  auth_required: false
```

### 6.4 缓存策略

#### off
完全禁用缓存，每次都重新获取数据和调用LLM。

#### llm_write_only
- LLM响应仅写入缓存
- 新闻、财务数据正常读写缓存
- 适合测试不同提示词时使用

#### full
全面启用读写缓存（LLM和数据）。

---

## 7. 使用场景与示例

### 7.1 基础回测场景

**场景**: 评估GPT-4在2025年3-6月的交易表现

```bash
python -m stockbench.apps.run_backtest \
    --start-date 2025-03-01 \
    --end-date 2025-06-30 \
    --llm-profile openai
```

### 7.2 对比不同LLM

**场景**: 对比OpenAI和DeepSeek的表现

```bash
# 测试 OpenAI
bash scripts/run_benchmark.sh --llm-profile openai

# 测试 DeepSeek
bash scripts/run_benchmark.sh --llm-profile deepseek-v3.1

# 对比结果
python scripts/log_performance.py --compare
```

### 7.3 离线模式测试

**场景**: 在没有网络或API配额有限时使用缓存数据

```bash
python -m stockbench.apps.run_backtest \
    --start-date 2025-03-01 \
    --end-date 2025-06-30 \
    --llm-profile openai \
    --offline
```

### 7.4 自定义股票池

**场景**: 只测试科技股

修改 `config.yaml`:
```yaml
symbols_universe:
  - AAPL
  - MSFT
  - GOOGL
  - AMZN
  - META
```

### 7.5 调整交易成本

**场景**: 模拟低佣金券商

修改 `config.yaml`:
```yaml
backtest:
  commission_bps: 0.1    # 降低佣金
  slippage_bps: 0.5      # 降低滑点
```

---

## 8. 性能分析工具

### 8.1 日志查询工具 (log_query.py)

**功能**: 查询和过滤回测日志

```bash
# 查询特定日期的决策
python scripts/log_query.py \
    --date 2025-03-15 \
    --log-type decision

# 查询特定股票的交易
python scripts/log_query.py \
    --symbol AAPL \
    --log-type trade

# 查询性能指标
python scripts/log_query.py \
    --log-type metrics
```

### 8.2 日志追踪工具 (log_trace.py)

**功能**: 追踪单笔交易的完整流程

```bash
# 追踪特定交易
python scripts/log_trace.py \
    --trade-id 20250315_AAPL_BUY

# 追踪决策链
python scripts/log_trace.py \
    --decision-chain \
    --symbol AAPL \
    --start-date 2025-03-01
```

### 8.3 性能分析工具 (log_performance.py)

**功能**: 深度分析回测性能

```bash
# 生成完整性能报告
python scripts/log_performance.py \
    --report-path storage/reports/backtest/<timestamp>

# 对比多个回测
python scripts/log_performance.py \
    --compare \
    --reports report1 report2 report3

# 生成可视化图表
python scripts/log_performance.py \
    --visualize \
    --output-dir ./analysis_output
```

---

## 9. 常见问题

### 9.1 API相关问题

**Q: API调用超时怎么办？**

A: 增加超时时间配置：
```yaml
llm_profiles:
  openai:
    timeout_sec: 600  # 增加到10分钟
```

**Q: API配额用完了怎么办？**

A: 使用离线模式：
```bash
python -m stockbench.apps.run_backtest --offline
```

### 9.2 数据问题

**Q: 数据缺失怎么办？**

A: 检查以下内容：
1. API密钥是否配置正确
2. 日期范围是否合理（避免未来日期或太久远的日期）
3. 使用 `--data-mode auto` 让系统自动处理

**Q: 如何清除缓存重新获取数据？**

A: 删除缓存目录：
```bash
rm -rf storage/cache/*
```

### 9.3 性能问题

**Q: 回测运行太慢？**

A: 优化建议：
1. 减少股票池大小
2. 缩短回测时间范围
3. 启用缓存（cache.mode: full）
4. 使用更快的LLM提供商

**Q: 内存不足？**

A: 解决方案：
1. 减少 `max_history_days` 配置
2. 关闭详细日志 `enable_detailed_logging: false`
3. 减少并行处理数量

### 9.4 LLM问题

**Q: LLM返回格式不正确？**

A: 解决方案：
1. 检查提示词模板是否正确
2. 增加重试次数 `retry.max_attempts`
3. 调整 temperature 参数（降低以获得更稳定输出）

**Q: 如何使用本地LLM？**

A: 使用vLLM或Ollama配置：
```bash
# 启动vLLM服务
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 使用vLLM配置运行
python -m stockbench.apps.run_backtest --llm-profile vllm
```

### 9.5 结果分析问题

**Q: 如何解读Sortino比率？**

A: Sortino比率衡量下行风险调整后的收益：
- > 2.0: 优秀
- 1.0-2.0: 良好
- 0.5-1.0: 一般
- < 0.5: 较差

**Q: 最大回撤多少算正常？**

A: 取决于策略类型：
- 保守策略: < 10%
- 平衡策略: 10-20%
- 激进策略: 20-30%
- > 30%: 需要重新评估风险管理

---

## 10. 高级功能

### 10.1 自定义智能体

**创建自定义智能体**:

```python
# 在 stockbench/agents/ 下创建新文件
from stockbench.agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def make_decision(self, context):
        # 实现自定义决策逻辑
        pass
```

**注册自定义智能体**:

修改 `config.yaml`:
```yaml
agents:
  mode: "custom"
  custom_agent:
    class: "MyCustomAgent"
    module: "stockbench.agents.my_custom_agent"
```

### 10.2 自定义策略

**创建基准策略**:

```python
# 在 stockbench/backtest/strategies/ 下创建
from stockbench.backtest.strategies.base import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def generate_signals(self, data):
        # 实现动量策略逻辑
        pass
```

### 10.3 扩展数据源

**添加新的数据适配器**:

```python
# 在 stockbench/adapters/ 下创建
from stockbench.adapters.base import BaseAdapter

class CustomDataAdapter(BaseAdapter):
    def fetch_data(self, symbol, start_date, end_date):
        # 实现数据获取逻辑
        pass
```

### 10.4 高级可视化

**自定义图表**:

```python
from stockbench.backtest.visualization import Visualizer

viz = Visualizer(report_path)
viz.plot_custom_chart(
    data=custom_data,
    chart_type='line',
    title='Custom Analysis'
)
```

### 10.5 批量回测

**批量测试多个配置**:

```bash
# 创建批量测试脚本
#!/bin/bash

LLM_PROFILES=("openai" "deepseek-v3.1" "zhipuai")
DATE_RANGES=("2025-03-01,2025-03-31" "2025-04-01,2025-04-30")

for profile in "${LLM_PROFILES[@]}"; do
    for dates in "${DATE_RANGES[@]}"; do
        IFS=',' read start end <<< "$dates"
        python -m stockbench.apps.run_backtest \
            --start-date $start \
            --end-date $end \
            --llm-profile $profile
    done
done
```

### 10.6 实时交易模拟（开发中）

**注意**: 此功能正在开发中

```python
# 未来版本将支持实时交易模拟
from stockbench.apps.live_trading import LiveTradingEngine

engine = LiveTradingEngine(config)
engine.start()
```

---

## 📞 支持与联系

- **GitHub Issues**: https://github.com/ChenYXxxx/stockbench/issues
- **文档**: https://stockbench.github.io/
- **邮件**: support@stockbench.io

---

## 📄 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../LICENSE) 文件。

---

## 🙏 致谢

感谢以下服务和项目：
- Polygon.io - 高质量股票市场数据
- Finnhub - 金融新闻和市场数据
- OpenAI - 强大的LLM能力
- 智谱AI - 中文LLM支持

---

**最后更新**: 2026-01-12

**版本**: 1.0.0
