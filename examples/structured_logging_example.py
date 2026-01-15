"""
Structured Logging Usage Examples

This file demonstrates how to use the structured logging schemas
for consistent, queryable, and analyzable logs.
"""

from loguru import logger
from stockbench.utils.log_schemas import (
    DecisionLog,
    OrderLog,
    AgentLog,
    BacktestLog,
    FeatureLog,
    DataLog,
    MemoryLog,
    LLMLog,
)

# Example 1: Agent Decision Logging
def example_decision_logging():
    """Example of logging agent decisions with structured schema"""
    
    decision_log = DecisionLog(
        symbol="AAPL",
        action="increase",
        target_cash_amount=15000.0,
        reasoning="Strong quarterly earnings beat expectations. Revenue up 12% YoY.",
        confidence=0.85,
        current_position_value=10000.0,
        holding_days=5,
        agent_name="decision_agent",
        decision_time_ms=234.5
    )
    
    # Log with structured data
    logger.info("[AGENT_DECISION] Decision made", **decision_log.to_log_dict())
    
    # The log output (JSON format in file) will include all structured fields:
    # {
    #   "time": "2025-12-15T15:30:00Z",
    #   "level": "INFO",
    #   "message": "[AGENT_DECISION] Decision made",
    #   "symbol": "AAPL",
    #   "action": "increase",
    #   "target_cash_amount": 15000.0,
    #   "reasoning": "Strong quarterly earnings...",
    #   "confidence": 0.85,
    #   ...
    # }


# Example 2: Order Execution Logging
def example_order_logging():
    """Example of logging order executions"""
    
    # Pending order
    order_log = OrderLog(
        symbol="GOOGL",
        side="buy",
        qty=50.0,
        order_price=145.32,
        status="pending",
        order_id="ORD_20251215_001",
        slice=1,
        twap_slices=3
    )
    logger.info("[BT_ORDER] Order submitted", **order_log.to_log_dict())
    
    # Filled order
    filled_log = OrderLog(
        symbol="GOOGL",
        side="buy",
        qty=50.0,
        order_price=145.32,
        exec_price=145.35,
        gross_amount=7267.50,
        commission=7.27,
        net_cost=7274.77,
        status="filled",
        filled_qty=50.0,
        order_id="ORD_20251215_001"
    )
    logger.info("[BT_ORDER] Order filled", **filled_log.to_log_dict())


# Example 3: Agent Execution Logging
def example_agent_logging():
    """Example of logging agent execution lifecycle"""
    
    import time
    
    # Agent started
    start_log = AgentLog(
        agent_name="fundamental_filter",
        status="started",
        start_time="2025-12-15T15:30:00Z",
        input_count=150,
        run_id="backtest_20251215_001",
        date="2025-12-15"
    )
    logger.info("[AGENT_START] Agent execution started", **start_log.to_log_dict())
    
    # Simulate work
    start_time = time.time()
    time.sleep(0.5)  # Simulated work
    duration_ms = (time.time() - start_time) * 1000
    
    # Agent completed successfully
    success_log = AgentLog(
        agent_name="fundamental_filter",
        status="success",
        start_time="2025-12-15T15:30:00Z",
        end_time="2025-12-15T15:30:00.5Z",
        duration_ms=duration_ms,
        input_count=150,
        output_count=45,
        run_id="backtest_20251215_001",
        date="2025-12-15"
    )
    logger.info("[AGENT_DONE] Agent execution completed", **success_log.to_log_dict())


# Example 4: Backtest Event Logging
def example_backtest_logging():
    """Example of logging backtest events"""
    
    # Cash update
    cash_log = BacktestLog(
        event_type="cash",
        cash_before=100000.0,
        cash_after=92500.0,
        cash_change=-7500.0
    )
    logger.info("[BT_CASH] Cash updated", **cash_log.to_log_dict())
    
    # Position validation failure
    validation_log = BacktestLog(
        event_type="position",
        validation_status="fail",
        inconsistencies=2,
        issues=[
            {"symbol": "AAPL", "issue": "negative_shares", "value": -10},
            {"symbol": "MSFT", "issue": "invalid_avg_price", "shares": 100, "avg_price": 0}
        ]
    )
    logger.error("[BT_VALIDATE] Position validation failed", **validation_log.to_log_dict())
    
    # Dividend payment
    dividend_log = BacktestLog(
        event_type="dividend",
        action_type="dividend",
        symbol="AAPL",
        shares=100.0,
        dividend_per_share=0.24,
        total_dividend=24.0,
        cash_before=92500.0,
        cash_after=92524.0
    )
    logger.info("[BT_DIVIDEND] Dividend received", **dividend_log.to_log_dict())


# Example 5: Feature Construction Logging
def example_feature_logging():
    """Example of logging feature construction"""
    
    feature_log = FeatureLog(
        symbol="TSLA",
        feature_type="technical",
        data_points=252,
        missing_data=3,
        quality_score=0.988,
        price_series_days=7,
        construction_time_ms=45.2
    )
    logger.debug("[FEATURE_BUILD] Technical features built", **feature_log.to_log_dict())
    
    # Feature construction with fundamental data
    fundamental_log = FeatureLog(
        symbol="TSLA",
        feature_type="fundamental",
        data_points=5,
        missing_data=0,
        quality_score=1.0,
        fundamental_fields=6,
        construction_time_ms=120.5
    )
    logger.debug("[FEATURE_FUND] Fundamental features built", **fundamental_log.to_log_dict())


# Example 6: Data Fetching Logging
def example_data_logging():
    """Example of logging data fetching operations"""
    
    # Successful API call
    data_log = DataLog(
        data_type="news",
        source="polygon",
        symbol="AAPL",
        start_date="2025-12-01",
        end_date="2025-12-15",
        records_fetched=25,
        cache_hit=False,
        api_call=True,
        fetch_time_ms=850.3,
        status="success"
    )
    logger.info("[DATA_FETCH] News data fetched", **data_log.to_log_dict())
    
    # Cache hit
    cache_log = DataLog(
        data_type="bars",
        source="cache",
        symbol="GOOGL",
        start_date="2025-12-01",
        end_date="2025-12-15",
        records_fetched=10,
        cache_hit=True,
        api_call=False,
        fetch_time_ms=2.1,
        status="success"
    )
    logger.debug("[DATA_CACHE] Cache hit", **cache_log.to_log_dict())


# Example 7: Memory Operations Logging
def example_memory_logging():
    """Example of logging memory system operations"""
    
    # Save episodes
    save_log = MemoryLog(
        operation="save",
        memory_type="episodic",
        symbol="AAPL",
        episode_count=1,
        operation_time_ms=15.3,
        status="success",
        records_affected=1
    )
    logger.debug("[MEM_SAVE] Episode saved", **save_log.to_log_dict())
    
    # Load episodes
    load_log = MemoryLog(
        operation="load",
        memory_type="episodic",
        symbol="AAPL",
        episode_count=5,
        operation_time_ms=8.7,
        status="success"
    )
    logger.debug("[MEM_LOAD] Episodes loaded", **load_log.to_log_dict())


# Example 8: LLM API Call Logging
def example_llm_logging():
    """Example of logging LLM API calls"""
    
    # Successful LLM call
    llm_log = LLMLog(
        model="gpt-4",
        operation="decision",
        prompt_tokens=1500,
        completion_tokens=350,
        total_tokens=1850,
        latency_ms=2340.5,
        cache_hit=False,
        status="success",
        estimated_cost=0.055
    )
    logger.info("[LLM_CALL] LLM decision completed", **llm_log.to_log_dict())
    
    # Cache hit
    cache_llm_log = LLMLog(
        model="gpt-4",
        operation="decision",
        prompt_tokens=1500,
        completion_tokens=350,
        total_tokens=1850,
        latency_ms=12.3,
        cache_hit=True,
        cache_key="hash_abc123",
        status="success",
        estimated_cost=0.0
    )
    logger.info("[LLM_CACHE] LLM cache hit", **cache_llm_log.to_log_dict())


# Example 9: Querying Structured Logs
def example_log_querying():
    """
    Example of how structured logs can be queried
    
    Since logs are in JSON format, they can be easily queried using tools like jq:
    
    # Find all decisions for AAPL
    cat logs/stockbench/2025-12-15.log | jq 'select(.symbol == "AAPL" and .message | contains("AGENT_DECISION"))'
    
    # Find all failed orders
    cat logs/stockbench/2025-12-15.log | jq 'select(.status == "rejected" and .message | contains("BT_ORDER"))'
    
    # Calculate average decision confidence
    cat logs/stockbench/2025-12-15.log | jq 'select(.confidence != null) | .confidence' | jq -s 'add/length'
    
    # Find all LLM calls with high latency (>3 seconds)
    cat logs/stockbench/2025-12-15.log | jq 'select(.latency_ms > 3000 and .message | contains("LLM_CALL"))'
    
    # Find all cache hits
    cat logs/stockbench/2025-12-15.log | jq 'select(.cache_hit == true)'
    
    # Track agent execution timeline
    cat logs/stockbench/2025-12-15.log | jq 'select(.agent_name != null) | {time, agent_name, status, duration_ms}'
    """
    pass


if __name__ == "__main__":
    # Set up basic logging for demo
    from stockbench.utils.logging_setup import setup_json_logging
    
    config = {
        "logging": {
            "console_level": "INFO",
            "file_level": "DEBUG"
        }
    }
    
    setup_json_logging(config)
    
    print("Running structured logging examples...")
    print("=" * 60)
    
    example_decision_logging()
    example_order_logging()
    example_agent_logging()
    example_backtest_logging()
    example_feature_logging()
    example_data_logging()
    example_memory_logging()
    example_llm_logging()
    
    print("=" * 60)
    print("Examples complete. Check logs/stockbench/ for JSON output.")
    print("\nStructured logs enable powerful queries like:")
    print("  - Filter by symbol, action, status")
    print("  - Calculate averages (confidence, latency, cost)")
    print("  - Track execution timelines")
    print("  - Identify bottlenecks and errors")
