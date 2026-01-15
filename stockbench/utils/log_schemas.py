"""
Structured logging schemas for standardized log output

This module defines Pydantic schemas for different types of log events,
enabling structured, queryable, and analyzable logs.

Usage:
    from stockbench.utils.log_schemas import DecisionLog, OrderLog, AgentLog
    from loguru import logger
    
    # Log a decision
    decision_log = DecisionLog(
        symbol="AAPL",
        action="increase",
        target_cash_amount=10000.0,
        reasoning="Strong fundamentals"
    )
    logger.info("[AGENT_DECISION] Decision made", **decision_log.to_log_dict())
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class LogSchema(BaseModel):
    """Base schema for all structured logs"""
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging with extra parameters"""
        return self.model_dump(exclude_none=True, exclude_defaults=True)


class DecisionLog(LogSchema):
    """Schema for agent decision logs"""
    
    symbol: str = Field(..., description="Stock symbol")
    action: str = Field(..., description="Decision action: hold/increase/decrease/close")
    target_cash_amount: float = Field(..., description="Target position value in USD")
    reasoning: Optional[str] = Field(None, description="Decision reasoning summary")
    confidence: Optional[float] = Field(None, description="Decision confidence score 0-1")
    
    # Context
    current_position_value: Optional[float] = Field(None, description="Current position value")
    holding_days: Optional[int] = Field(None, description="Days holding this position")
    
    # Metadata
    agent_name: Optional[str] = Field(None, description="Agent that made the decision")
    decision_time_ms: Optional[float] = Field(None, description="Decision duration in ms")


class OrderLog(LogSchema):
    """Schema for order execution logs"""
    
    symbol: str = Field(..., description="Stock symbol")
    side: str = Field(..., description="Order side: buy/sell")
    qty: float = Field(..., description="Order quantity in shares")
    
    # Prices
    order_price: Optional[float] = Field(None, description="Order limit price")
    exec_price: Optional[float] = Field(None, description="Actual execution price")
    
    # Costs
    gross_amount: Optional[float] = Field(None, description="Gross trade amount")
    commission: Optional[float] = Field(None, description="Commission cost")
    net_cost: Optional[float] = Field(None, description="Net cost including commission")
    
    # Status
    status: str = Field(default="pending", description="Order status: pending/filled/rejected")
    filled_qty: Optional[float] = Field(None, description="Filled quantity")
    reject_reason: Optional[str] = Field(None, description="Rejection reason if rejected")
    
    # Metadata
    order_id: Optional[str] = Field(None, description="Order ID")
    slice: Optional[int] = Field(None, description="TWAP slice number")
    twap_slices: Optional[int] = Field(None, description="Total TWAP slices")


class AgentLog(LogSchema):
    """Schema for agent execution logs"""
    
    agent_name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Execution status: started/success/failed")
    
    # Timing
    start_time: Optional[str] = Field(None, description="Start timestamp ISO format")
    end_time: Optional[str] = Field(None, description="End timestamp ISO format")
    duration_ms: Optional[float] = Field(None, description="Execution duration in ms")
    
    # Input/Output
    input_count: Optional[int] = Field(None, description="Number of input items")
    output_count: Optional[int] = Field(None, description="Number of output items")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error type/class")
    
    # Metadata
    run_id: Optional[str] = Field(None, description="Run ID for tracing")
    date: Optional[str] = Field(None, description="Processing date")


class BacktestLog(LogSchema):
    """Schema for backtesting engine logs"""
    
    event_type: str = Field(..., description="Event type: cash/order/position/dividend/split")
    
    # Cash events
    cash_before: Optional[float] = Field(None, description="Cash before operation")
    cash_after: Optional[float] = Field(None, description="Cash after operation")
    cash_change: Optional[float] = Field(None, description="Cash change amount")
    
    # Position events
    symbol: Optional[str] = Field(None, description="Stock symbol")
    shares: Optional[float] = Field(None, description="Position shares")
    avg_price: Optional[float] = Field(None, description="Average position price")
    position_value: Optional[float] = Field(None, description="Position market value")
    
    # Validation
    validation_status: Optional[str] = Field(None, description="Validation status: pass/fail")
    inconsistencies: Optional[int] = Field(None, description="Number of inconsistencies found")
    issues: Optional[List[Dict]] = Field(None, description="List of issues found")
    
    # Corporate actions
    action_type: Optional[str] = Field(None, description="Corporate action type: dividend/split")
    ratio: Optional[float] = Field(None, description="Split ratio")
    dividend_per_share: Optional[float] = Field(None, description="Dividend per share")
    total_dividend: Optional[float] = Field(None, description="Total dividend amount")


class FeatureLog(LogSchema):
    """Schema for feature construction logs"""
    
    symbol: str = Field(..., description="Stock symbol")
    feature_type: str = Field(..., description="Feature type: technical/fundamental/news")
    
    # Data quality
    data_points: Optional[int] = Field(None, description="Number of data points used")
    missing_data: Optional[int] = Field(None, description="Number of missing data points")
    quality_score: Optional[float] = Field(None, description="Data quality score 0-1")
    
    # Specific features
    price_series_days: Optional[int] = Field(None, description="Price series length in days")
    news_count: Optional[int] = Field(None, description="News items count")
    fundamental_fields: Optional[int] = Field(None, description="Fundamental fields count")
    
    # Processing
    construction_time_ms: Optional[float] = Field(None, description="Construction time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class DataLog(LogSchema):
    """Schema for data fetching/processing logs"""
    
    data_type: str = Field(..., description="Data type: bars/news/fundamentals/snapshot")
    source: str = Field(..., description="Data source: polygon/finnhub/cache")
    
    # Request
    symbol: Optional[str] = Field(None, description="Stock symbol")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    
    # Response
    records_fetched: Optional[int] = Field(None, description="Number of records fetched")
    cache_hit: Optional[bool] = Field(None, description="Whether cache was hit")
    api_call: Optional[bool] = Field(None, description="Whether API was called")
    
    # Performance
    fetch_time_ms: Optional[float] = Field(None, description="Fetch duration in ms")
    
    # Error handling
    status: str = Field(default="success", description="Status: success/failed/partial")
    error: Optional[str] = Field(None, description="Error message if failed")
    retry_count: Optional[int] = Field(None, description="Number of retries")


class MemoryLog(LogSchema):
    """Schema for memory system logs"""
    
    operation: str = Field(..., description="Operation: save/load/query/backfill")
    memory_type: str = Field(..., description="Memory type: episodic/semantic/working")
    
    # Content
    symbol: Optional[str] = Field(None, description="Stock symbol")
    episode_count: Optional[int] = Field(None, description="Number of episodes")
    
    # Performance
    operation_time_ms: Optional[float] = Field(None, description="Operation duration in ms")
    
    # Status
    status: str = Field(default="success", description="Status: success/failed")
    records_affected: Optional[int] = Field(None, description="Number of records affected")
    error: Optional[str] = Field(None, description="Error message if failed")


class LLMLog(LogSchema):
    """Schema for LLM API call logs"""
    
    model: str = Field(..., description="LLM model name")
    operation: str = Field(..., description="Operation type: decision/analysis/report")
    
    # Tokens
    prompt_tokens: Optional[int] = Field(None, description="Input prompt tokens")
    completion_tokens: Optional[int] = Field(None, description="Output completion tokens")
    total_tokens: Optional[int] = Field(None, description="Total tokens")
    
    # Performance
    latency_ms: Optional[float] = Field(None, description="API call latency in ms")
    
    # Cache
    cache_hit: Optional[bool] = Field(None, description="Whether cache was hit")
    cache_key: Optional[str] = Field(None, description="Cache key")
    
    # Status
    status: str = Field(default="success", description="Status: success/failed/timeout")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Cost (optional)
    estimated_cost: Optional[float] = Field(None, description="Estimated API cost in USD")


# Helper function to create log with timestamp
def create_log_entry(schema: LogSchema, tag: str) -> Dict[str, Any]:
    """
    Create a complete log entry with timestamp and tag
    
    Args:
        schema: Log schema instance
        tag: Log tag (e.g., "[AGENT_DECISION]")
    
    Returns:
        Dictionary ready for logging
    """
    log_dict = schema.to_log_dict()
    log_dict["timestamp"] = datetime.utcnow().isoformat()
    log_dict["tag"] = tag
    return log_dict
