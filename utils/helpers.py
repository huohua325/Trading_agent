import os
from typing import Dict, Any, Optional
from ..config.config import TradingConfig
from ..agents.trading_agent import TradingAgent
from ..brokers.backtrader_broker import BacktraderBroker
from ..data_sources.tiingo_data_source import TiingoDataSource
from ..data_sources.finnhub_data_source import FinnhubDataSource
from ..llm.gpt4o_llm import GPT4oLLM


def load_config_from_env() -> TradingConfig:
    """从环境变量加载配置"""
    return TradingConfig()


def create_agent(config: Optional[TradingConfig] = None) -> TradingAgent:
    """创建交易代理实例"""
    
    if config is None:
        config = load_config_from_env()
    
    # 验证配置
    config.validate_config()
    
    # 创建组件
    broker = BacktraderBroker(config.to_dict())
    
    # 根据配置选择数据源
    if config.data_source_type == "finnhub":
        data_source = FinnhubDataSource(config.to_dict())
    else:
        data_source = TiingoDataSource(config.to_dict())
        
    llm = GPT4oLLM(config.to_dict())
    
    # 创建代理
    agent = TradingAgent(
        broker=broker,
        data_source=data_source,
        llm=llm,
        config=config.to_dict()
    )
    
    return agent


def format_currency(amount: float) -> str:
    """格式化货币金额"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """格式化百分比"""
    return f"{value:.2f}%"


def calculate_position_size(
    total_value: float,
    max_position_size: float,
    price: float
) -> int:
    """计算仓位大小"""
    max_investment = total_value * max_position_size
    shares = int(max_investment / price)
    return max(1, shares)  # 至少买1股


def validate_trading_hours() -> bool:
    """验证是否在交易时间内（简化版本）"""
    # 这里可以实现更复杂的交易时间验证
    from datetime import datetime
    now = datetime.now()
    
    # 简单检查：周一到周五，9点到16点
    if now.weekday() >= 5:  # 周末
        return False
    
    if now.hour < 9 or now.hour >= 16:  # 非交易时间
        return False
    
    return True


def print_portfolio_summary(portfolio: Dict[str, Any]):
    """打印投资组合摘要"""
    print("\n" + "="*50)
    print("投资组合摘要")
    print("="*50)
    
    cash = portfolio.get('cash', 0)
    total_value = portfolio.get('total_value', 0)
    initial_cash = portfolio.get('initial_cash', 100000)
    positions = portfolio.get('positions', {})
    
    print(f"现金余额: {format_currency(cash)}")
    print(f"投资组合价值: {format_currency(total_value)}")
    print(f"总收益: {format_currency(total_value - initial_cash)}")
    print(f"收益率: {format_percentage(((total_value - initial_cash) / initial_cash) * 100)}")
    
    if positions:
        print(f"\n持仓详情 ({len(positions)} 只股票):")
        for symbol, position in positions.items():
            print(f"  {symbol}: {position.get('quantity', 0)} 股 @ {format_currency(position.get('avg_price', 0))}")
            print(f"    当前价值: {format_currency(position.get('value', 0))}")
    else:
        print("\n无持仓")
    
    print("="*50)


def print_trade_result(result: Dict[str, Any]):
    """打印交易结果"""
    success = result.get('success', False)
    message = result.get('message', '')
    
    status = "✅ 成功" if success else "❌ 失败"
    print(f"\n交易结果: {status}")
    print(f"详情: {message}")
    
    if 'explanation' in result:
        print(f"解释: {result['explanation']}")
    
    if 'cost' in result:
        print(f"成本: {format_currency(result['cost'])}")
    
    if 'proceeds' in result:
        print(f"收益: {format_currency(result['proceeds'])}")


def check_environment() -> bool:
    """检查环境配置"""
    required_vars = ['OPENAI_API_KEY', 'TIINGO_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"缺少必需的环境变量: {', '.join(missing_vars)}")
        print("请在 .env 文件中设置这些变量")
        return False
    
    return True 