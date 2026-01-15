# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**StockBench** is an LLM-powered stock trading benchmark platform that evaluates Large Language Models in trading decision-making using historical market data. The system simulates real-world trading with a dual-agent architecture and produces comprehensive performance metrics.

## Core Commands

### Development Setup
```bash
# Create environment and install dependencies
conda create -n stockbench python=3.11
conda activate stockbench
pip install -r requirements.txt
```

### Running Backtests
```bash
# Basic run with defaults
bash scripts/run_benchmark.sh

# Custom date range and LLM model
bash scripts/run_benchmark.sh \
    --start-date 2025-04-01 \
    --end-date 2025-05-31 \
    --llm-profile deepseek-v3.1

# Available LLM profiles: openai, deepseek-v3.1, kimi-k2-0711-preview,
# qwen3-235b-a22b-instruct-2507, gpt-oss-20b, gpt-oss-120b, zhipuai
```

### Running via Python (Alternative)
```bash
# Direct Python execution
python -m stockbench.apps.run_backtest --start-date 2025-04-01 --end-date 2025-05-31

# With custom LLM profile
python -m stockbench.apps.run_backtest --llm-profile deepseek-v3.1 --offline
```

### Running Tests
```bash
# Note: Test suite is under development. Placeholder for future tests:
pytest tests/
```

### Data Management
```bash
# Pre-cache data to avoid API calls during backtests
python -m stockbench.apps.pre_cache --symbols AAPL MSFT --start 2025-01-01 --end 2025-06-30
```

## Architecture

### Dual-Agent System
The platform uses a two-stage decision-making process:

1. **Fundamental Filter Agent** ([stockbench/agents/fundamental_filter_agent.py](stockbench/agents/fundamental_filter_agent.py))
   - Analyzes which stocks require deep fundamental analysis
   - Filters stocks based on price movements, position state, and market conditions
   - Outputs: `stocks_need_fundamental` list and reasoning for each stock

2. **Decision Agent** ([stockbench/agents/dual_agent_llm.py](stockbench/agents/dual_agent_llm.py))
   - Makes final trading decisions using filtered/enhanced features
   - Stocks flagged by filter receive full fundamental data (financials, indicators)
   - Other stocks get technical/news data only to reduce noise
   - Validates decisions through unified retry loop with logic checks

3. **Backtest Report Agent** ([stockbench/agents/backtest_report_llm.py](stockbench/agents/backtest_report_llm.py))
   - Generates natural language summaries of backtest results
   - Analyzes trading performance, key metrics, and market conditions
   - Produces actionable insights and strategy recommendations

### Pipeline Context System
The platform includes a unified context management system for multi-agent pipelines:

**PipelineContext** ([stockbench/core/pipeline_context.py](stockbench/core/pipeline_context.py))
- **AgentStep**: Records individual agent execution (timing, tokens, status)
- **AgentTrace**: Tracks entire pipeline execution with success/failure stats
- **Data Bus**: Shared data storage for inter-agent communication via `put()`/`get()`
- Automatic logging and performance metrics for each agent step

### Data Flow
```
Market Data (Polygon/Finnhub APIs)
    ↓
DataHub (stockbench/core/data_hub.py)
    ↓ [Caching Layer: storage/cache/]
Feature Construction (stockbench/core/features.py)
    ↓
┌─────────────────────────────────┐
│     PipelineContext (ctx)       │
│  ┌─────────────────────────┐    │
│  │  Fundamental Filter     │    │
│  │  Agent                  │    │
│  └───────────┬─────────────┘    │
│              ↓                  │
│  ┌─────────────────────────┐    │
│  │  Enhanced Feature       │    │
│  │  Rebuild                │    │
│  └───────────┬─────────────┘    │
│              ↓                  │
│  ┌─────────────────────────┐    │
│  │  Decision Agent         │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
    ↓
Backtest Engine (stockbench/backtest/engine.py)
    ↓
Reports & Metrics (storage/reports/)
```

### Key Components

**DataHub** ([stockbench/core/data_hub.py](stockbench/core/data_hub.py))
- Unified data access layer with cache-first strategy
- Supports two modes: `auto` (cache + API fallback) and `offline_only` (cache only)
- Handles: price bars, news, financials, corporate actions, stock indicators
- Multi-level caching: Parquet partitions, JSON caches, per-day news cache

**Backtest Engine** ([stockbench/backtest/engine.py](stockbench/backtest/engine.py))
- Day-by-day simulation with realistic slippage and commission
- Portfolio management with cash constraints and position limits
- Order validation and comprehensive retry mechanism
- Supports rebalancing at open/close

**LLM Client** ([stockbench/llm/llm_client.py](stockbench/llm/llm_client.py))
- Unified interface for OpenAI-compatible APIs and ZhipuAI SDK
- Content-based caching with run_id and date organization
- Enhanced JSON parsing with repair tools (json_repair, demjson3)
- Token budget tracking and retry logic

### Tool System
The platform provides a modular tool system for LLM function calling:

**Tool Base** ([stockbench/tools/base.py](stockbench/tools/base.py))
- `Tool`: Abstract base class with parameter validation and OpenAI schema conversion
- `ToolParameter`: Parameter definition with type, description, enum support
- `ToolResult`: Standardized result format with success/error states and metadata
- `safe_run()`: Exception-safe execution with automatic timing

**Tool Registry** ([stockbench/tools/registry.py](stockbench/tools/registry.py))
- Singleton registry for tool management
- `register()`/`unregister()`: Dynamic tool registration
- `execute()`: Execute tools by name with automatic validation
- `to_openai_tools()`: Batch conversion to OpenAI function calling format
- `get_tools_by_tag()`: Filter tools by category tags

**Built-in Data Tools** ([stockbench/tools/data_tools.py](stockbench/tools/data_tools.py))
- `PriceDataTool`: Historical OHLCV price data
- `NewsDataTool`: Stock-related news and sentiment
- `FinancialsTool`: Financial statements (income, balance sheet, cash flow)
- `SnapshotTool`: Real-time market snapshots
- `DividendsTool`: Dividend history
- `SplitsTool`: Stock split history
- `TickerDetailsTool`: Company information and fundamentals

### Data Adapters

**Polygon Adapter** ([stockbench/adapters/polygon_client.py](stockbench/adapters/polygon_client.py))
- Primary market data source for price bars, ticker details, and corporate actions
- Handles pagination, rate limiting, and data normalization
- Supports adjusted/unadjusted price data

**Finnhub Adapter** ([stockbench/adapters/finnhub_client.py](stockbench/adapters/finnhub_client.py))
- News and sentiment data provider
- Financial statements (income, balance sheet, cash flow)
- Stock indicators and fundamental metrics

### Configuration System

**Primary Config** ([config.yaml](config.yaml))
- `symbols_universe`: Top 20 DJIA stocks for trading
- `data.mode`: Data fetching strategy (`auto` or `offline_only`)
- `features`: Historical window settings (7-day price series)
- `news`: Lookback days and top-K event selection
- `portfolio`: Initial cash, min cash ratio
- `agents.dual_agent`: Separate configs for filter and decision agents
- `llm_profiles`: Multiple provider configurations (see below)
- `cache.mode`: Cache behavior (`off`, `llm_write_only`, `full`)
- `backtest`: Warmup days, benchmark settings, per-symbol analysis, visualization

**Supported LLM Profiles**:
| Profile | Provider | Model | Use Case |
|---------|----------|-------|----------|
| `openai` | OpenAI | oss-120b | Default production |
| `zhipuai` | ZhipuAI | glm-4.5 | Chinese market support |
| `deepseek-v3.1` | OpenAI-compatible | deepseek-v3.1-250821 | Cost-effective |
| `kimi-k2-0711-preview` | OpenAI-compatible | kimi-k2-0711-preview | Long context |
| `qwen3-235b-a22b-instruct-2507` | OpenAI-compatible | qwen3-235b | High capability |
| `gpt-oss-20b` | OpenAI-compatible | gpt-oss-20b | Fast inference |
| `gpt-oss-120b` | OpenAI-compatible | gpt-oss-120b | Balanced |
| `vllm` | vLLM | Qwen2.5-7B-Instruct | Local deployment |
| `ollama` | Ollama | llama3 | Lightweight local |
| `auto` | Auto-detect | - | Environment-based |
| `none` | Mock | - | Testing only |

**Environment Variables**
```bash
# API Keys (required for new data)
POLYGON_API_KEY=your_key
FINNHUB_API_KEY=your_key
OPENAI_API_KEY=your_key
ZHIPUAI_API_KEY=your_key

# Optional overrides
TA_DATA_MODE=offline_only  # Force offline mode
TA_RUN_ID=custom_run      # Custom run identifier
```

## Data Storage Structure

```
storage/
├── cache/
│   ├── llm/by_run/{run_id}/        # LLM response cache by run and date
│   ├── news/                        # News query cache (hash-based)
│   ├── news_by_day/{symbol}/        # News by date (normalized)
│   ├── financials/                  # Financial statements cache
│   ├── corporate_actions/           # Dividends and splits
│   └── stock_indicators/            # Fundamental metrics cache
├── parquet/{symbol}/day/            # Price data partitioned by date
├── reports/backtest/{run_id}/       # Backtest results and analysis
└── logs/                            # Execution logs
```

## Important Implementation Details

### Feature Construction
- Features are built conditionally based on filter agent results
- Stocks needing fundamental analysis get: financials, indicators, valuation metrics
- Other stocks get: price history, technical data, news only
- Historical decision records included for context (max 7 per symbol, 30 days)

### Decision Validation
The system validates all decisions through multiple checks:
1. **Logic validation**: Actions must match amounts (increase→higher, decrease→lower, close→zero)
2. **Cash constraints**: Total purchases cannot exceed available cash
3. **Cash ratio**: Minimum cash reserve must be maintained
4. **Hallucination filter**: Removes decisions for non-existent symbols

### Retry Mechanism
Unified retry loop with two levels:
- **Engine retries**: For order rejections (insufficient cash, validation failures)
- **LLM retries**: For parse errors, logic errors, constraint violations
- Global limit: `agents.retry.max_attempts` (default: 3) covers both levels

### Caching Strategy
**Cache Modes**:
- `off`: No caching (always fetch fresh)
- `llm_write_only`: Write LLM cache only, read market data from cache
- `full`: Read and write all caches

**Cache Priority** (for market data):
1. Local Parquet partitions (storage/parquet/)
2. Local CSV files (backtest_data/)
3. API calls (Polygon/Finnhub)
4. Fallback to incomplete cached data if API unavailable

### Prompt Engineering
Prompts are stored in `stockbench/agents/prompts/` and referenced in config:
- `fundamental_filter_v1.txt`: Filter agent system prompt
- `decision_agent_v1.txt`: Decision agent system prompt  
- `backtest_report_v1.txt`: Report generation prompt for performance analysis
- Prompts receive JSON-formatted features and must return structured JSON

## Testing and Development

### Adding New LLM Providers
1. Add provider config to `config.yaml` under `llm_profiles`
2. For OpenAI-compatible APIs: set `provider: "openai"`, `base_url`, and `model`
3. For custom SDKs: extend `LLMClient` in [stockbench/llm/llm_client.py](stockbench/llm/llm_client.py)
4. Test with: `bash scripts/run_benchmark.sh --llm-profile your_profile`

### Extending Trading Strategies
1. Create new agent in `stockbench/agents/`
2. Implement decision function returning `{symbol: decision_dict}`
3. Register in `stockbench/backtest/strategies/`
4. Update `config.yaml` with strategy settings

### Custom Metrics
Add metrics in [stockbench/backtest/metrics.py](stockbench/backtest/metrics.py):
- Implement calculation function
- Register in `evaluate()` function
- Add to config `backtest.benchmark.daily_metrics.metrics`

## Common Patterns

### Accessing Historical Data
```python
from stockbench.core.data_hub import get_bars, get_news, get_financials

# Get price bars with caching
bars = get_bars("AAPL", "2025-01-01", "2025-06-30",
                multiplier=1, timespan="day", adjusted=True, cfg=config)

# Get news with lookahead bias prevention
news, cursor = get_news("AAPL", gte="2025-06-01", lte="2025-06-30",
                        limit=100, cfg=config)

# Get financials (annual/quarterly)
financials = get_financials("AAPL", timeframe="quarterly", limit=4, cfg=config)
```

### Feature Building
```python
from stockbench.core.features import build_features_for_prompt

features = build_features_for_prompt(
    bars_day=bars_df,
    snapshot=snapshot_data,
    news_items=news_list,
    position_state=position_info,
    details=ticker_details,
    config=config,
    include_price=True,        # Include current price
    exclude_fundamental=False  # Include fundamental data
)
```

### LLM Calls with Caching
```python
from stockbench.llm.llm_client import LLMClient, LLMConfig

client = LLMClient()
llm_cfg = LLMConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    cache_enabled=True
)

data, meta = client.generate_json(
    role="decision_agent",
    cfg=llm_cfg,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    trade_date="2025-06-30",
    run_id="my_backtest"
)
```

### Using Tool System
```python
from stockbench.tools.registry import ToolRegistry, execute_tool
from stockbench.tools.base import Tool, ToolParameter, ToolParameterType, ToolResult

# Execute built-in tools
result = execute_tool("get_price_data", symbol="AAPL", start_date="2025-01-01", end_date="2025-06-30")
if result.success:
    df = result.data
    print(f"Got {result.metadata['rows']} rows")

# Get OpenAI function calling format
registry = ToolRegistry.default()
tools_schema = registry.to_openai_tools()  # For LLM function calling

# Custom tool example
class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(name="my_tool", description="Custom analysis tool")
    
    def get_parameters(self):
        return [ToolParameter("symbol", ToolParameterType.STRING, "Stock symbol")]
    
    def run(self, symbol: str, **kwargs) -> ToolResult:
        return ToolResult.ok({"analysis": f"Analyzed {symbol}"})

registry.register(MyCustomTool())
```

### Using Pipeline Context
```python
from stockbench.core.pipeline_context import PipelineContext

# Create context for a backtest day
ctx = PipelineContext(
    run_id="backtest_2025_01",
    date="2025-01-15",
    llm_client=llm_client,
    llm_config=llm_config,
    config=config
)

# Store data for inter-agent communication
ctx.put("previous_decisions", prev_decisions, agent_name="loader")
ctx.put("market_features", features_list)

# Track agent execution
step = ctx.start_agent("FundamentalFilter", input_summary="20 stocks")
try:
    result = filter_stocks_needing_fundamental(features_list, ctx=ctx)
    ctx.finish_agent(step, "success", output_summary=f"{len(result)} filtered")
except Exception as e:
    ctx.finish_agent(step, "failed", error=str(e))

# Get execution summary
print(ctx.get_summary())  # {"total_agents": 2, "success": 2, "failed": 0, ...}
```

## Performance Optimization

- **Pre-cache data** before running multiple backtests to avoid repeated API calls
- **Use offline_only mode** for reproducibility and speed once data is cached
- **Adjust feature history windows** in config to reduce prompt size if hitting token limits
- **Enable LLM caching** (`cache.mode: full`) for faster iteration during development
- **Use per-symbol benchmark** for detailed stock-level analysis vs portfolio comparison

## Output and Reports

After a backtest run, find results in `storage/reports/backtest/{run_id}/`:
- `summary.txt`: Overall performance metrics (returns, Sortino, max drawdown)
- `nl_summary.txt`: LLM-generated natural language summary
- `trades.json`: Detailed trade history
- `portfolio_snapshots.json`: Daily portfolio states
- `per_symbol_benchmark/`: Individual stock comparisons vs buy-and-hold
- `aggregated_analysis/`: Cross-stock performance plots
- `conclusion.md`: Key insights and recommendations

## Development Notes

- Platform is Windows-compatible (uses `win32` platform detection)
- Python 3.11+ required for best compatibility
- All dates use ISO format (YYYY-MM-DD) without timezone for consistency
- Logging uses `loguru` with file and console outputs
- Market data adjusted for splits and dividends by default

## Project Structure

```
stockbench/
├── adapters/              # External API clients
│   ├── polygon_client.py  # Polygon.io: prices, corporate actions
│   └── finnhub_client.py  # Finnhub: news, financials, indicators
├── agents/                # LLM-based decision agents
│   ├── prompts/           # System prompt templates
│   │   ├── fundamental_filter_v1.txt
│   │   ├── decision_agent_v1.txt
│   │   └── backtest_report_v1.txt
│   ├── fundamental_filter_agent.py
│   ├── dual_agent_llm.py
│   └── backtest_report_llm.py
├── apps/                  # CLI entry points
│   ├── run_backtest.py    # Main backtest runner
│   └── pre_cache.py       # Data pre-caching utility
├── backtest/              # Backtesting engine
│   ├── engine.py          # Core simulation engine
│   ├── metrics.py         # Performance metrics (Sortino, drawdown, etc.)
│   ├── reports.py         # Report generation
│   ├── visualization.py   # Chart generation and plotting
│   ├── pipeline.py        # Backtest pipeline orchestration
│   ├── slippage.py        # Slippage and commission models
│   ├── summarize.py       # Summary generation
│   └── strategies/        # Trading strategy implementations
│       └── llm_decision.py
├── core/                  # Core data and feature modules
│   ├── data_hub.py        # Unified data access layer
│   ├── features.py        # Feature construction for prompts
│   ├── pipeline_context.py # Agent pipeline context management
│   ├── decorators.py      # Function decorators (caching, retry)
│   ├── executor.py        # Async execution utilities
│   ├── price_utils.py     # Price calculation helpers
│   ├── schemas.py         # Data schemas and validation
│   └── types.py           # Type definitions
├── llm/                   # LLM client layer
│   ├── llm_client.py      # Unified LLM interface with caching
│   ├── providers/         # Provider-specific implementations
│   └── tests/             # LLM-related tests
├── tools/                 # Tool system for LLM function calling
│   ├── base.py            # Tool base class and types
│   ├── registry.py        # Tool registration and execution
│   ├── data_tools.py      # Built-in data retrieval tools
│   └── tests/             # Tool system tests
├── utils/                 # Utility modules
│   ├── logging_setup.py   # Loguru configuration
│   ├── logging_helper.py  # Logging utilities
│   ├── formatting.py      # Output formatting
│   └── io.py              # File I/O helpers
└── examples/              # Usage examples
    └── pipeline_example.py
```

## Troubleshooting

**API Rate Limits**: Use `--offline` mode or pre-cache data to avoid hitting API limits during development.

**JSON Parse Errors**: The system has built-in JSON repair. If persistent, check prompt output format or increase `max_tokens`.

**Missing Data**: Ensure `POLYGON_API_KEY` and `FINNHUB_API_KEY` are set, or use pre-cached data in `storage/` directory.

**Memory Issues**: Reduce `features.history.price_series_days` or process fewer symbols at once.

**Tool Execution Failures**: Check `result.error` and `result.metadata` for debugging. Use `tool.safe_run()` for automatic exception handling.

**Pipeline Context Issues**: Use `ctx.get_summary()` to inspect agent execution status and identify failures.

## Testing

The project includes test modules in multiple locations:

```bash
# Run all tests
pytest stockbench/

# Run specific test modules
pytest stockbench/tools/tests/test_tools.py         # Tool system tests
pytest stockbench/core/tests/test_pipeline_context.py  # Pipeline context tests
pytest stockbench/llm/tests/test_auto_detect.py     # LLM auto-detection tests
```

## Key Design Decisions

1. **Tool Abstraction**: All data operations are wrapped as `Tool` objects for LLM function calling compatibility
2. **Pipeline Context**: Centralized context management enables clean data flow between agents
3. **Dual Retry Levels**: Engine-level (order rejections) and LLM-level (parse/logic errors) retries are unified
4. **Cache-First Strategy**: Market data is cached locally to enable reproducible backtests
5. **Conditional Features**: Fundamental data is only fetched for stocks flagged by the filter agent
