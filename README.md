# AI交易代理框架

一个模块化的AI驱动交易代理框架，支持多种数据源、经纪人和LLM的组合。

## 📋 功能特性

- **模块化架构**: 三大独立模块（数据源、经纪人、LLM）可灵活替换
- **多种交易行为**: 买入、卖出、观望、获取信息、获取新闻
- **风险管理**: 内置风险评估和限制机制
- **实时决策**: 基于市场数据和新闻的AI驱动决策
- **性能分析**: 完整的交易历史和绩效指标

## 🏗️ 架构设计

```
trading_agent/
├── actions/          # 交易行为定义
├── agents/           # 交易代理实现
├── brokers/          # 经纪人模块（模拟盘）
├── data_sources/     # 数据源模块
├── llm/              # 大语言模型模块
├── config/           # 配置管理
├── utils/            # 工具函数
└── main.py           # 主程序入口
```

### 三大核心模块

1. **数据源模块** (`data_sources/`)
   - 基础抽象类: `BaseDataSource`
   - Tiingo实现: `TiingoDataSource`
   - Finnhub实现: `FinnhubDataSource`
   - 支持历史数据、实时价格、市场信息、新闻数据

2. **经纪人模块** (`brokers/`)
   - 基础抽象类: `BaseBroker`
   - Backtrader实现: `BacktraderBroker`（简化实现，主要用于资金管理和交易执行）
   - 支持交易执行、投资组合管理、绩效分析

3. **LLM模块** (`llm/`)
   - 基础抽象类: `BaseLLM`
   - GPT-4o实现: `GPT4oLLM`
   - 支持交易决策、情绪分析、风险评估

## 🚀 快速开始

### 1. 安装依赖

推荐使用Conda环境：

```bash
# 创建并激活conda环境
conda create -n trading_agent python=3.9
conda activate trading_agent

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

创建 `.env` 文件在项目根目录：

```env
# OpenAI API密钥（必需）
OPENAI_API_KEY=your_openai_api_key_here

# 数据源API密钥（选择一个）
FINNHUB_API_KEY=your_finnhub_api_key_here
# 或
TIINGO_API_KEY=your_tiingo_api_key_here
```

### 3. 运行示例

**重要**: 所有命令必须在项目根目录 (`trading_agent/`) 下运行，使用模块方式运行：

```bash
# 演示模式
python -m trading_agent.main --mode demo

# 单个交易周期
python -m trading_agent.main --mode single

# 连续交易2小时
python -m trading_agent.main --mode continuous --duration 2
```

## 📊 使用示例

### 基础使用

```python
from trading_agent.utils.helpers import create_agent

# 创建交易代理
agent = create_agent()

# 初始化
await agent.initialize()
await agent.start_trading()

# 运行单个交易周期
result = await agent.run_trading_cycle()

# 停止交易
await agent.stop_trading()
```

### 自定义配置

```python
from trading_agent.config.config import TradingConfig
from trading_agent.utils.helpers import create_agent

# 自定义配置
config = TradingConfig(
    initial_cash=50000.0,
    trading_symbols=["AAPL", "GOOGL", "MSFT"],
    max_position_size=0.15,
    risk_tolerance=0.03,
    data_source_type="finnhub"  # 可选: "finnhub" 或 "tiingo"
)

# 创建代理
agent = create_agent(config)
```

### 替换组件

```python
from trading_agent.agents.trading_agent import TradingAgent
from trading_agent.brokers.backtrader_broker import BacktraderBroker
from trading_agent.data_sources.finnhub_data_source import FinnhubDataSource
from trading_agent.llm.gpt4o_llm import GPT4oLLM

# 创建自定义组件组合
broker = BacktraderBroker(config.to_dict())
data_source = FinnhubDataSource(config.to_dict())
llm = GPT4oLLM(config.to_dict())

agent = TradingAgent(broker, data_source, llm, config.to_dict())
```

## 🎯 Action Space

框架支持5种交易行为：

1. **buy** - 买入股票
2. **sell** - 卖出股票
3. **hold** - 观望/持有
4. **get_info** - 获取市场信息
5. **get_news** - 获取相关新闻

每个行为都包含：
- 行为类型
- 股票代码（可选）
- 数量（买卖时需要）
- 价格（可选）
- 决策理由
- 额外参数

## ⚙️ 配置选项

### 通用配置
- `initial_cash`: 初始资金（默认：100,000）
- `trading_symbols`: 交易股票列表
- `data_source_type`: 数据源类型（"finnhub"或"tiingo"，默认："finnhub"）
- `max_position_size`: 最大仓位比例（默认：20%）
- `risk_tolerance`: 风险容忍度（默认：2%）

### LLM配置
- `openai_model`: OpenAI模型名称（默认：gpt-4o）
- `max_tokens`: 最大生成长度（默认：1000）
- `temperature`: 生成温度（默认：0.1）

### 交易配置
- `trading_interval`: 交易周期间隔（默认：300秒）
- `max_trades_per_day`: 每日最大交易次数（默认：10）
- `commission`: 手续费率（默认：0.1%）

## 🔧 扩展框架

### 添加新的数据源

```python
from trading_agent.data_sources.base_data_source import BaseDataSource

class YourDataSource(BaseDataSource):
    async def get_historical_data(self, symbol, start_date, end_date, interval="1D"):
        # 实现您的数据获取逻辑
        pass
    
    async def get_real_time_price(self, symbol):
        # 实现实时价格获取
        pass
    
    # ... 实现其他必需方法
```

### 添加新的经纪人

```python
from trading_agent.brokers.base_broker import BaseBroker

class YourBroker(BaseBroker):
    async def execute_action(self, action):
        # 实现交易执行逻辑
        pass
    
    async def get_portfolio_status(self):
        # 返回投资组合状态
        pass
    
    # ... 实现其他必需方法
```

### 添加新的LLM

```python
from trading_agent.llm.base_llm import BaseLLM

class YourLLM(BaseLLM):
    async def generate_trading_decision(self, market_data, portfolio_status, news_data, historical_context=None):
        # 实现决策生成逻辑
        pass
    
    async def analyze_market_sentiment(self, news_data, symbol=None):
        # 实现情绪分析
        pass
    
    # ... 实现其他必需方法
```

## 📈 性能监控

框架提供完整的性能监控：

- 总收益和收益率
- 交易成功率
- 风险指标
- 市场情绪分析
- 详细的交易历史

## 📝 日志记录

系统会在 `trading_agent/logs/` 目录下记录交易执行的详细日志，包括：
- 交易决策过程
- LLM响应内容
- 交易执行结果
- 投资组合更新

## 🔍 关于Backtrader的使用说明

在当前实现中，Backtrader主要用于：
- 管理资金（跟踪现金余额）
- 执行交易操作（买入/卖出）
- 管理持仓记录
- 计算投资组合价值

这是一个简化的实现，主要用于模拟交易执行，而不是作为完整的回测引擎。真正的交易决策由LLM基于从数据源获取的信息来做出。

## ⚠️ 风险提示

1. 这是一个模拟交易框架，请勿直接用于真实交易
2. AI决策存在不确定性，请谨慎评估风险
3. 建议在充分测试后再考虑实盘应用
4. 注意API调用频率限制和成本

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

MIT License

## 🙋‍♂️ 支持

如有问题，请提交Issue或联系维护者。 