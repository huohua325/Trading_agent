# AI Trading Agent Framework

A modular, AI-driven trading agent framework supporting multiple data sources, brokers, and LLM integrations.

## 📋 Features

- **Modular Architecture**: Three independent modules (data sources, brokers, LLMs) that can be flexibly interchanged
- **Comprehensive Trading Actions**: Buy, sell, hold, information retrieval, and news analysis
- **Risk Management**: Built-in risk assessment and limitation mechanisms
- **Real-time Decision Making**: AI-driven decisions based on market data and news
- **Performance Analytics**: Complete trading history and performance metrics
- **Data Caching**: Intelligent caching mechanism to reduce API calls
- **API Rate Limiting**: Automatic management of API call frequency to avoid exceeding limits

## 🏗️ Architecture

```
trading_agent/
├── actions/          # Trading action definitions
├── agents/           # Trading agent implementations
├── brokers/          # Broker modules (simulation)
├── data_sources/     # Data source modules
├── llm/              # Large language model modules
├── config/           # Configuration management
├── utils/            # Utility functions
└── main.py           # Main program entry
```

### Core Modules

1. **Data Source Module** (`data_sources/`)
   - Base abstract class: `BaseDataSource`
   - Tiingo implementation: `TiingoDataSource`
   - Finnhub implementation: `FinnhubDataSource`
   - Supports historical data, real-time prices, market information, and news

2. **Broker Module** (`brokers/`)
   - Base abstract class: `BaseBroker`
   - Backtrader implementation: `BacktraderBroker` (simplified implementation for fund management and trade execution)
   - Supports trade execution, portfolio management, and performance analysis

3. **LLM Module** (`llm/`)
   - Base abstract class: `BaseLLM`
   - GPT-4o implementation: `GPT4oLLM`
   - Supports trading decisions, sentiment analysis, and risk assessment

## 🚀 Getting Started

### 1. Installation

Recommended to use Conda environment:

```bash
# Create and activate conda environment
conda create -n trading_agent python=3.9
conda activate trading_agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root directory:

```env
# OpenAI API key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Data source API key (choose one)
FINNHUB_API_KEY=your_finnhub_api_key_here
# or
TIINGO_API_KEY=your_tiingo_api_key_here
```

### 3. Running Examples

**Important**: All commands must be run from the project root directory (`trading_agent/`) using the module approach:

```bash
# Demo mode
python -m trading_agent.main --mode demo

# Single trading cycle
python -m trading_agent.main --mode single

# Continuous trading for 2 hours
python -m trading_agent.main --mode continuous --duration 2

# Resume interrupted continuous trading session
python -m trading_agent.main --mode continuous --duration 2 --resume
```

## 📊 Usage Examples

### Basic Usage

```python
from trading_agent.utils.helpers import create_agent

# Create trading agent
agent = create_agent()

# Initialize
await agent.initialize()
await agent.start_trading()

# Run a single trading cycle
result = await agent.run_trading_cycle()

# Stop trading
await agent.stop_trading()
```

### Custom Configuration

```python
from trading_agent.config.config import TradingConfig
from trading_agent.utils.helpers import create_agent

# Custom configuration
config = TradingConfig(
    initial_cash=50000.0,
    trading_symbols=["AAPL", "GOOGL", "MSFT"],
    max_position_size=0.15,
    risk_tolerance=0.03,
    data_source_type="finnhub",  # Options: "finnhub" or "tiingo"
    
    # Finnhub data limit configuration
    finnhub_historical_days=365,  # Historical data days
    finnhub_price_resolution="D",  # Price resolution (1, 5, 15, 30, 60, D, W, M)
    finnhub_api_calls_per_minute=45,  # API call frequency limit
    finnhub_data_cache_enabled=True,  # Enable data caching
    finnhub_cache_duration=3600  # Cache duration (seconds)
)

# Create agent
agent = create_agent(config)
```

### Component Replacement

```python
from trading_agent.agents.trading_agent import TradingAgent
from trading_agent.brokers.backtrader_broker import BacktraderBroker
from trading_agent.data_sources.finnhub_data_source import FinnhubDataSource
from trading_agent.llm.gpt4o_llm import GPT4oLLM

# Create custom component combination
broker = BacktraderBroker(config.to_dict())
data_source = FinnhubDataSource(config.to_dict())
llm = GPT4oLLM(config.to_dict())

agent = TradingAgent(broker, data_source, llm, config.to_dict())
```

## 🎯 Action Space

The framework supports five trading actions:

1. **buy** - Purchase stocks
2. **sell** - Sell stocks
3. **hold** - Observe/hold position
4. **get_info** - Retrieve market information
5. **get_news** - Retrieve relevant news

Each action contains:
- Action type
- Stock symbol (optional)
- Quantity (required for buy/sell)
- Price (optional)
- Decision rationale
- Additional parameters

## ⚙️ Configuration Options

### General Configuration
- `initial_cash`: Initial capital (default: 100,000)
- `trading_symbols`: List of trading stocks
- `data_source_type`: Data source type ("finnhub" or "tiingo", default: "finnhub")
- `max_position_size`: Maximum position size ratio (default: 20%)
- `risk_tolerance`: Risk tolerance level (default: 2%)

### LLM Configuration
- `openai_model`: OpenAI model name (default: gpt-4o)
- `max_tokens`: Maximum generation length (default: 1000)
- `temperature`: Generation temperature (default: 0.1)

### Trading Configuration
- `trading_interval`: Trading cycle interval (default: 300 seconds)
- `max_trades_per_day`: Maximum trades per day (default: 10)
- `commission`: Commission rate (default: 0.1%)
- `news_limit`: Number of news items to retrieve (default: 10)
- `news_days_back`: How many days back to retrieve news (default: 7)

### Finnhub Data Limit Configuration
- `finnhub_historical_days`: Historical price data retrieval days (default: 365)
- `finnhub_price_resolution`: Price data resolution (default: "D", options: "1", "5", "15", "30", "60", "D", "W", "M")
- `finnhub_api_calls_per_minute`: API call limit per minute (default: 45)
- `finnhub_financial_quarters`: Number of quarters for financial data (default: 4)
- `finnhub_earnings_limit`: Earnings surprise data limit (default: 4)
- `finnhub_cache_duration`: Data cache duration in seconds (default: 3600)
- `finnhub_data_cache_enabled`: Whether to enable data caching (default: True)

## 🔧 Framework Extension

### Adding a New Data Source

```python
from trading_agent.data_sources.base_data_source import BaseDataSource

class CustomDataSource(BaseDataSource):
    async def get_historical_data(self, symbol, start_date, end_date, interval="1D"):
        # Implement your data retrieval logic
        pass
    
    async def get_real_time_price(self, symbol):
        # Implement real-time price retrieval
        pass
    
    # ... implement other required methods
```

### Adding a New Broker

```python
from trading_agent.brokers.base_broker import BaseBroker

class CustomBroker(BaseBroker):
    async def execute_action(self, action):
        # Implement trade execution logic
        pass
    
    async def get_portfolio_status(self):
        # Return portfolio status
        pass
    
    # ... implement other required methods
```

### Adding a New LLM

```python
from trading_agent.llm.base_llm import BaseLLM

class CustomLLM(BaseLLM):
    async def generate_trading_decision(self, market_data, portfolio_status, news_data, historical_context=None):
        # Implement decision generation logic
        pass
    
    async def analyze_market_sentiment(self, news_data, symbol=None):
        # Implement sentiment analysis
        pass
    
    # ... implement other required methods
```

## 📈 Performance Monitoring

The framework provides comprehensive performance monitoring:

- Total returns and return rates
- Trade success rate
- Risk metrics
- Market sentiment analysis
- Detailed trading history

## 📝 Logging

The system records detailed logs of trade execution in the `trading_agent/logs/` directory, including:
- Trading decision process
- LLM response content
- Trade execution results
- Portfolio updates

## 🔍 Notes on Backtrader Usage

In the current implementation, Backtrader is primarily used for:
- Fund management (tracking cash balance)
- Executing trading operations (buy/sell)
- Managing position records
- Calculating portfolio value

This is a simplified implementation, mainly used for simulating trade execution rather than as a complete backtesting engine. The actual trading decisions are made by the LLM based on information obtained from data sources.

## 🔄 Data Caching and API Limit Management

To optimize performance and avoid exceeding API limits, the system implements:

1. **Intelligent Data Caching**:
   - Market sentiment, financial data, etc. are cached for a specified time
   - Cache duration configurable via `finnhub_cache_duration`
   - Caching can be toggled via `finnhub_data_cache_enabled`

2. **API Call Frequency Management**:
   - Automatically tracks API calls per minute
   - Automatically pauses and waits when approaching limits
   - Limit value adjustable via `finnhub_api_calls_per_minute`

3. **Data Retrieval Optimization**:
   - Batch data retrieval to reduce API calls
   - Intelligent merging of news data for multiple stocks
   - Limited historical data range to avoid excessive requests

## ⚠️ Risk Disclaimer

1. This is a simulated trading framework; do not use directly for real trading
2. AI decisions involve uncertainty; please carefully evaluate risks
3. Extensive testing is recommended before considering live application
4. Be mindful of API call frequency limits and costs

## 🤝 Contribution Guidelines

Issues and Pull Requests are welcome!

1. Fork the project
2. Create a feature branch
3. Submit changes
4. Initiate a Pull Request

## 📄 License

MIT License

## 🙋‍♂️ Support

For questions, please submit an Issue or contact the maintainer. 