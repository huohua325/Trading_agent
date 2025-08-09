# AI交易代理框架

一个模块化的AI驱动交易代理框架，支持多种数据源、经纪人和大语言模型的灵活组合。

## 📋 功能特性

- **模块化架构**: 三大独立模块（数据源、经纪人、LLM）可自由替换
- **多数据源支持**: 集成 Finnhub、Polygon.io、Alpha Vantage、Tiingo、Quandl 等多个数据源，确保数据完整性
- **历史回测** : 内置本地 CSV / 在线 API（Finnhub、yfinance）两种模式，一键评估策略
- **全面交易行为**: 买入、卖出、观望、信息检索、新闻分析
- **风险管理**: AI 风控 + 多指标绩效评估（CAGR、Max DD、Sharpe、Sortino、Profit Factor、Expectancy）
- **实时决策**: 基于最新行情与新闻的 AI 驱动决策
- **性能分析**: 交易历史 & 图表 & 指标一站式输出
- **数据缓存**: 智能缓存减少 API 调用
- **API限制管理**: 自动限流，避免触发配额
- **数据源优先级**: 智能故障转移，确保数据获取的可靠性

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

### 核心模块

1. **数据源模块** (`data_sources/`)
   - 基础抽象类: `BaseDataSource`
   - Tiingo实现: `TiingoDataSource`
   - Finnhub实现: `FinnhubDataSource`
   - 支持历史数据、实时价格、市场信息和新闻数据

2. **经纪人模块** (`brokers/`)
   - 基础抽象类: `BaseBroker`
   - Backtrader实现: `BacktraderBroker`（简化实现，主要用于资金管理和交易执行）
   - 支持交易执行、投资组合管理和绩效分析

3. **LLM模块** (`llm/`)
   - 基础抽象类: `BaseLLM`
   - GPT-4o实现: `GPT4oLLM`
   - 支持交易决策、情绪分析和风险评估

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

复制 `config_example.env` 为 `.env` 文件在项目根目录，并填入你的API密钥：

```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here

# 数据源API密钥配置（可选择性配置）
# Finnhub (免费额度: 60次/分钟)
FINNHUB_API_KEY=your_finnhub_api_key_here

# Polygon.io (免费额度: 5次/分钟)
POLYGON_API_KEY=your_polygon_api_key_here

# Alpha Vantage (免费额度: 5次/分钟, 500次/天)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Tiingo (付费, 但便宜)
TIINGO_API_KEY=your_tiingo_api_key_here

# Quandl (部分免费)
QUANDL_API_KEY=your_quandl_api_key_here

# NewsAPI (免费额度: 100次/天)
NEWS_API_KEY=your_news_api_key_here
```

**注意**: 系统会按优先级自动选择可用的数据源，配置的API密钥越多，数据获取的成功率越高。

### 3. 运行示例

**重要**: 所有命令必须在项目根目录 (`trading_agent/`) 下运行，使用模块方式运行：

```bash
# 演示模式
python -m trading_agent.main --mode demo

# 单个交易周期
python -m trading_agent.main --mode single

# 连续交易2小时
python -m trading_agent.main --mode continuous --duration 2

### 4. 历史回测（新增）

```bash
# 4.1 使用本地 CSV 数据（需先准备 `backtest_data/*.csv`）
python trading_agent/run_backtest.py \
  --start_date 2025-03-01 --end_date 2025-07-31 \
  --symbols AAPL,MSFT,GOOGL    # 逗号分隔

# 4.2 使用 yfinance 在线拉取历史行情
python trading_agent/run_backtest.py \
  --start_date 2025-03-01 --end_date 2025-07-31 \
  --symbols AAPL,MSFT,GOOGL \
  --yfinance_backtest

# 4.3 使用 Finnhub API 回测（需 FINNHUB_API_KEY，免费档仅近三月日线）
python trading_agent/run_backtest.py \
  --start_date 2025-03-01 --end_date 2025-07-31 \
  --symbols AAPL,MSFT,GOOGL \
  --api_backtest
```

### 5. 多数据源数据下载（新增）

```bash
# 下载历史数据用于回测
python trading_agent/examples/multi_source_data_download_example.py

# 测试所有API并对比数据质量
python trading_agent/test_all_apis.py

# 或者直接使用DataDownloader类
python -c "
import asyncio
from trading_agent.data_sources.data_downloader import DataDownloader

async def download():
    downloader = DataDownloader()
    await downloader.download_all_data(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2025-03-01',
        end_date='2025-07-31'
    )

asyncio.run(download())
"
```

#### 测试所有API模式

测试所有API模式会为每个数据源创建单独的目录，方便对比数据质量：

```bash
# 运行测试所有API模式
python trading_agent/test_all_apis.py
```

测试完成后，会在 `api_test_results/` 目录下创建以下结构：
```
api_test_results/
├── test_yfinance/          # YFinance数据
│   ├── AAPL_prices.csv
│   ├── AAPL_info.json
│   ├── AAPL_financials.json
│   └── test_results.json
├── test_finnhub/           # Finnhub数据
│   ├── AAPL_prices.csv
│   ├── AAPL_info.json
│   ├── AAPL_financials.json
│   └── test_results.json
├── test_polygon/           # Polygon.io数据
│   └── ...
└── ...
```

每个API目录下的 `test_results.json` 包含该API的测试结果和成功率统计。

回测结束后将输出核心指标并在 `logs/` 生成：

* `backtest_result.json` 完整指标与交易明细
* `charts/portfolio_value.png` 资产曲线
* `charts/cumulative_return.png` 收益率曲线
* `charts/drawdown.png` 回撤曲线

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
    data_source_type="finnhub",  # 可选: "finnhub" 或 "tiingo"
    
    # Finnhub数据限制配置
    finnhub_historical_days=365,  # 历史数据天数
    finnhub_price_resolution="D",  # 价格分辨率(1, 5, 15, 30, 60, D, W, M)
    finnhub_api_calls_per_minute=45,  # API调用频率限制
    finnhub_data_cache_enabled=True,  # 启用数据缓存
    finnhub_cache_duration=3600  # 缓存时间(秒)
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

## 🎯 交易行为空间

框架支持五种交易行为：

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
- `news_limit`: 获取新闻条数（默认：10）
- `news_days_back`: 获取多少天前的新闻（默认：7）

### Finnhub数据限制配置
- `finnhub_historical_days`: 历史价格数据获取天数（默认：365）
- `finnhub_price_resolution`: 价格数据分辨率（默认："D"，可选："1", "5", "15", "30", "60", "D", "W", "M"）
- `finnhub_api_calls_per_minute`: 每分钟API调用次数限制（默认：45）
- `finnhub_financial_quarters`: 获取财务数据的季度数（默认：4）
- `finnhub_earnings_limit`: 盈利惊喜数据的限制（默认：4）
- `finnhub_cache_duration`: 数据缓存时间，单位秒（默认：3600）
- `finnhub_data_cache_enabled`: 是否启用数据缓存（默认：True）

## 🔧 扩展框架

### 多数据源系统

系统支持多个数据源的智能切换和故障转移：

```python
# 数据源优先级配置
data_sources = {
    "price": ["yfinance", "finnhub", "polygon", "alpha_vantage", "tiingo"],
    "news": ["finnhub", "newsapi", "yfinance"],
    "financials": ["yfinance", "finnhub", "alpha_vantage"],
    "market_info": ["yfinance", "finnhub", "polygon"]
}

# 系统会自动按优先级尝试数据源，直到成功获取数据
```

### 添加新的数据源

```python
from trading_agent.data_sources.base_data_source import BaseDataSource

class CustomDataSource(BaseDataSource):
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

class CustomBroker(BaseBroker):
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

class CustomLLM(BaseLLM):
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

## 🔄 数据缓存和API限制管理

为了优化性能并避免超出API限制，系统实现了：

1. **智能数据缓存**：
   - 市场情绪、财务数据等会被缓存指定时间
   - 可通过`finnhub_cache_duration`配置缓存时长
   - 可通过`finnhub_data_cache_enabled`开关缓存功能

2. **API调用频率管理**：
   - 自动跟踪每分钟API调用次数
   - 当接近限制时自动暂停并等待
   - 可通过`finnhub_api_calls_per_minute`调整限制值

3. **数据获取优化**：
   - 批量获取数据减少API调用
   - 智能合并多个股票的新闻数据
   - 限制历史数据范围避免过度请求

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