"""
数据获取工具集

将 data_hub 中的数据获取函数包装为 Tool，提供：
- PriceDataTool: 价格数据获取
- NewsDataTool: 新闻数据获取
- FinancialsTool: 财务数据获取
- SnapshotTool: 实时快照数据获取
- DividendsTool: 分红数据获取

所有工具都是对 stockbench.core.data_hub 函数的封装，
保持向后兼容，不修改原有函数。
"""

from typing import List, Optional, Dict, Any
from .base import Tool, ToolParameter, ToolParameterType, ToolResult


class PriceDataTool(Tool):
    """
    价格数据获取工具
    
    获取股票的历史价格数据，包括 OHLCV（开盘价、最高价、最低价、收盘价、成交量）。
    """
    
    def __init__(self):
        super().__init__(
            name="get_price_data",
            description="获取股票历史价格数据（日线），包括开盘价、最高价、最低价、收盘价、成交量",
            version="1.0.0",
            tags=["data", "price", "market"]
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type=ToolParameterType.STRING,
                description="股票代码，如 AAPL、GOOGL",
                required=True
            ),
            ToolParameter(
                name="start_date",
                type=ToolParameterType.STRING,
                description="开始日期，格式 YYYY-MM-DD",
                required=True
            ),
            ToolParameter(
                name="end_date",
                type=ToolParameterType.STRING,
                description="结束日期，格式 YYYY-MM-DD",
                required=True
            ),
            ToolParameter(
                name="adjusted",
                type=ToolParameterType.BOOLEAN,
                description="是否使用复权价格",
                required=False,
                default=True
            ),
        ]
    
    def run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjusted: bool = True,
        cfg: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """获取价格数据"""
        try:
            # 延迟导入避免循环依赖
            from stockbench.core.data_hub import get_bars
            
            df = get_bars(
                ticker=symbol,
                start=start_date,
                end=end_date,
                multiplier=1,
                timespan="day",
                adjusted=adjusted,
                cfg=cfg
            )
            
            if df is None or df.empty:
                return ToolResult.fail(f"No price data found for {symbol}")
            
            return ToolResult.ok(
                data=df,
                rows=len(df),
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
        except Exception as e:
            return ToolResult.fail(str(e))


class NewsDataTool(Tool):
    """
    新闻数据获取工具
    
    获取股票相关的新闻数据。
    """
    
    def __init__(self):
        super().__init__(
            name="get_news",
            description="获取股票相关新闻数据，包括标题、摘要、发布时间",
            version="1.0.0",
            tags=["data", "news", "sentiment"]
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type=ToolParameterType.STRING,
                description="股票代码",
                required=True
            ),
            ToolParameter(
                name="start_date",
                type=ToolParameterType.STRING,
                description="开始日期，格式 YYYY-MM-DD",
                required=True
            ),
            ToolParameter(
                name="end_date",
                type=ToolParameterType.STRING,
                description="结束日期，格式 YYYY-MM-DD",
                required=True
            ),
            ToolParameter(
                name="limit",
                type=ToolParameterType.INTEGER,
                description="最大返回条数",
                required=False,
                default=10
            ),
        ]
    
    def run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        limit: int = 10,
        cfg: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """获取新闻数据"""
        try:
            from stockbench.core.data_hub import get_news
            
            # get_news returns Tuple[List[Dict], Optional[str]]
            # First element is the news list, second is page_token
            result = get_news(
                ticker=symbol,
                gte=start_date,
                lte=end_date,
                limit=limit,
                cfg=cfg
            )
            
            # Unpack the tuple - get_news returns (news_list, page_token)
            if isinstance(result, tuple):
                news_items = result[0] if result else []
            else:
                # Fallback for backwards compatibility
                news_items = result
            
            if not news_items:
                return ToolResult.ok(
                    data=[],
                    count=0,
                    symbol=symbol
                )
            
            return ToolResult.ok(
                data=news_items,
                count=len(news_items),
                symbol=symbol
            )
            
        except Exception as e:
            return ToolResult.fail(str(e))


class FinancialsTool(Tool):
    """
    财务数据获取工具
    
    获取公司的财务报表数据。
    """
    
    def __init__(self):
        super().__init__(
            name="get_financials",
            description="获取公司财务报表数据，包括收入、利润、资产等",
            version="1.0.0",
            tags=["data", "fundamental", "financials"]
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type=ToolParameterType.STRING,
                description="股票代码",
                required=True
            ),
            ToolParameter(
                name="limit",
                type=ToolParameterType.INTEGER,
                description="返回的报表期数",
                required=False,
                default=4
            ),
        ]
    
    def run(
        self,
        symbol: str,
        limit: int = 4,
        cfg: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """获取财务数据"""
        try:
            from stockbench.core.data_hub import get_financials
            
            financials = get_financials(
                ticker=symbol,
                limit=limit,
                cfg=cfg
            )
            
            if not financials:
                return ToolResult.ok(
                    data=[],
                    count=0,
                    symbol=symbol
                )
            
            return ToolResult.ok(
                data=financials,
                count=len(financials) if isinstance(financials, list) else 1,
                symbol=symbol
            )
            
        except Exception as e:
            return ToolResult.fail(str(e))


class SnapshotTool(Tool):
    """
    实时快照数据获取工具
    
    获取股票的实时行情快照。
    """
    
    def __init__(self):
        super().__init__(
            name="get_snapshot",
            description="获取股票实时行情快照，包括当前价格、涨跌幅、成交量等",
            version="1.0.0",
            tags=["data", "realtime", "market"]
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbols",
                type=ToolParameterType.ARRAY,
                description="股票代码列表",
                required=True
            ),
        ]
    
    def run(
        self,
        symbols: List[str],
        cfg: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """获取快照数据"""
        try:
            from stockbench.core.data_hub import get_universal_snapshots
            
            # 确保 symbols 是列表
            if isinstance(symbols, str):
                symbols = [symbols]
            
            snapshots = get_universal_snapshots(
                symbols=symbols,
                cfg=cfg
            )
            
            return ToolResult.ok(
                data=snapshots,
                count=len(snapshots),
                symbols=symbols
            )
            
        except Exception as e:
            return ToolResult.fail(str(e))


class DividendsTool(Tool):
    """
    分红数据获取工具
    
    获取股票的分红历史数据。
    """
    
    def __init__(self):
        super().__init__(
            name="get_dividends",
            description="获取股票分红历史数据，包括分红金额、除息日等",
            version="1.0.0",
            tags=["data", "fundamental", "dividends"]
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type=ToolParameterType.STRING,
                description="股票代码",
                required=True
            ),
        ]
    
    def run(
        self,
        symbol: str,
        cfg: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """获取分红数据"""
        try:
            from stockbench.core.data_hub import get_dividends
            
            dividends = get_dividends(
                ticker=symbol,
                cfg=cfg
            )
            
            if dividends is None or (hasattr(dividends, 'empty') and dividends.empty):
                return ToolResult.ok(
                    data=[],
                    count=0,
                    symbol=symbol
                )
            
            # 如果是 DataFrame，转换为记录列表
            if hasattr(dividends, 'to_dict'):
                data = dividends.to_dict('records')
            else:
                data = dividends
            
            return ToolResult.ok(
                data=data,
                count=len(data) if isinstance(data, list) else 1,
                symbol=symbol
            )
            
        except Exception as e:
            return ToolResult.fail(str(e))


class TickerDetailsTool(Tool):
    """
    股票详情获取工具
    
    获取股票的基本信息，如公司名称、行业、市值等。
    """
    
    def __init__(self):
        super().__init__(
            name="get_ticker_details",
            description="获取股票基本信息，包括公司名称、行业、市值、描述等",
            version="1.0.0",
            tags=["data", "fundamental", "info"]
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type=ToolParameterType.STRING,
                description="股票代码",
                required=True
            ),
        ]
    
    def run(
        self,
        symbol: str,
        cfg: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """获取股票详情"""
        try:
            from stockbench.core.data_hub import get_ticker_details
            
            details = get_ticker_details(
                ticker=symbol,
                cfg=cfg
            )
            
            if not details:
                return ToolResult.fail(f"No details found for {symbol}")
            
            return ToolResult.ok(
                data=details,
                symbol=symbol
            )
            
        except Exception as e:
            return ToolResult.fail(str(e))


class SplitsTool(Tool):
    """
    股票拆分数据获取工具
    
    获取股票的拆分历史数据。
    """
    
    def __init__(self):
        super().__init__(
            name="get_splits",
            description="获取股票拆分历史数据，包括拆分比例、生效日期等",
            version="1.0.0",
            tags=["data", "fundamental", "splits"]
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type=ToolParameterType.STRING,
                description="股票代码",
                required=True
            ),
        ]
    
    def run(
        self,
        symbol: str,
        cfg: Optional[Dict] = None,
        **kwargs
    ) -> ToolResult:
        """获取拆分数据"""
        try:
            from stockbench.core.data_hub import get_splits
            
            splits = get_splits(
                ticker=symbol,
                cfg=cfg
            )
            
            if splits is None or (hasattr(splits, 'empty') and splits.empty):
                return ToolResult.ok(
                    data=[],
                    count=0,
                    symbol=symbol
                )
            
            # 如果是 DataFrame，转换为记录列表
            if hasattr(splits, 'to_dict'):
                data = splits.to_dict('records')
            else:
                data = splits
            
            return ToolResult.ok(
                data=data,
                count=len(data) if isinstance(data, list) else 1,
                symbol=symbol
            )
            
        except Exception as e:
            return ToolResult.fail(str(e))


# 导出
__all__ = [
    "PriceDataTool",
    "NewsDataTool",
    "FinancialsTool",
    "SnapshotTool",
    "DividendsTool",
    "TickerDetailsTool",
    "SplitsTool",
]
