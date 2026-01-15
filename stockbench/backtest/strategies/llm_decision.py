"""
LLM Decision Strategy (for backtesting phase)

This module provides an LLM-based trading strategy `Strategy` that is called by the backtesting engine 
on a daily basis during backtesting:
Build features from factors/news/financials, call LLM for analysis and decision-making, generate buy/sell orders

Design objectives:
- Interact with the backtesting engine through a unified `on_bar(ctx)` interface, returning daily order list
- Control through configuration the news lookback window, feature window, etc.
"""
from __future__ import annotations

from typing import Dict, List
import pandas as pd
from loguru import logger

from stockbench.core.executor import decide_batch as unified_decide_batch
from stockbench.core import data_hub
from stockbench.core.pipeline_context import PipelineContext
from stockbench.tools import ToolRegistry


class Strategy:
    """LLM-based backtesting strategy.

    Usage: Called by the backtesting engine on each backtesting day via `on_bar(ctx)` to get order list.

    Attribute descriptions:
    - cfg: Configuration dictionary related to strategy and backtesting
    - news_lookback_days: News lookback window days for feature construction
    - page_limit: News item retrieval limit
    - warmup_days: Historical lookback days needed for feature construction (e.g., moving averages, financials)
    - agent_mode: Agent mode, "dual" (dual-agent) or "single" (single-agent)
    - pending_decisions: Temporary storage for decisions waiting to be recorded after order execution
    - pending_meta: Metadata for pending decisions
    - pending_date: Date for pending decisions
    
    Note: Historical decision records are now managed by the new Memory system (ctx.memory.episodes) 
    in the Agent layer, not in this Strategy class.
    """
    def __init__(self, cfg: Dict) -> None:
        """Initialize strategy.

        Parameters:
        - cfg: Configuration dictionary containing risk/news/backtest/llm sub-configurations
        """
        self.cfg = cfg
        self.news_lookback_days = int((cfg or {}).get("news", {}).get("lookback_days", 7))
        self.page_limit = int((cfg or {}).get("news", {}).get("page_limit", 50))
        self.warmup_days = int((cfg or {}).get("backtest", {}).get("warmup_days", 60))
        # Agent mode: "dual" (dual-agent) or "single" (single-agent), default single-agent
        agents_mode = (cfg or {}).get("agents", {}).get("mode")
        self.agent_mode = str(agents_mode or "single").lower()
        
        # Debug: Detailed configuration parsing process
        logger.debug(
            "[SYS_CONFIG] Agent mode configuration",
            agents_mode=agents_mode,
            final_agent_mode=self.agent_mode
        )
        
        # Temporary storage for pending decisions (to be recorded after execution)
        # These will be saved to ctx.memory.episodes after order execution
        self.pending_decisions: Dict[str, Dict] = {}
        self.pending_meta: Dict = {}
        self.pending_date: str = ""
        
        logger.debug("[MEM_OP] Strategy initialization: Using new Memory system (ctx.memory.episodes)")
    
    def _build_features_for_day(self, ctx) -> List[Dict]:
        """
        Build daily features: Get historical data, news, financials, etc., and build feature list.
        Optimization: Directly build new format features to avoid subsequent repeated conversions
        """
        features_list = []
        open_map = ctx["open_map"]
        if not open_map:
            return []
        
        # 1) Feature construction
        datasets = ctx["datasets"]
        portfolio = ctx["portfolio"]
        
        # Get configuration parameters
        news_lookback_days = int(self.cfg.get("news", {}).get("lookback_days", 7))
        page_limit = int(self.cfg.get("news", {}).get("page_limit", 100))
        warmup_days = int(self.cfg.get("backtest", {}).get("warmup_days", 7))
        
        for symbol in open_map.keys():
            # Get historical data (for feature construction)
            start_date = ctx["date"] - pd.Timedelta(days=warmup_days + 5)  # Get 5 extra days as buffer
            end_date = ctx["date"]
            
            bars_day = datasets.get_day_bars(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            # Get news data (using ToolRegistry)
            news_items = []
            try:
                # News fetching logic: let data_hub.py handle lookahead bias prevention
                # If making decisions on May 1st with lookback_days=3, should fetch news from April 28-30
                news_end_date = end_date  # Pass decision date directly, let get_news() handle bias prevention
                news_start_date = end_date - pd.Timedelta(days=self.news_lookback_days)  # Go back lookback days
                
                logger.debug(
                    "[DATA_FETCH] News fetching parameter",
                    decision_date=end_date.strftime('%Y-%m-%d'),
                    start=news_start_date.strftime('%Y-%m-%d'),
                    end=news_end_date.strftime('%Y-%m-%d')
                )
                
                # Use ToolRegistry to get news data
                tool_registry = ToolRegistry.default()
                news_tool_result = tool_registry.execute(
                    "get_news",
                    symbol=symbol,
                    start_date=news_start_date.strftime("%Y-%m-%d"),
                    end_date=news_end_date.strftime("%Y-%m-%d"),
                    limit=page_limit
                )
                
                if news_tool_result.success:
                    news_raw = news_tool_result.data
                else:
                    logger.debug(
                        "[DATA_FETCH] News tool failed",
                        error=news_tool_result.error
                    )
                    news_raw = []
                
                # Handle different news data formats
                if isinstance(news_raw, dict):
                    if "results" in news_raw and isinstance(news_raw["results"], list):
                        news_items = news_raw["results"]
                    elif "data" in news_raw and isinstance(news_raw["data"], list):
                        news_items = news_raw["data"]
                    else:
                        news_items = news_raw
                elif isinstance(news_raw, list):
                    news_items = news_raw
                else:
                    news_items = []
                
                # Time filtering logic (consistent with fetching logic)
                if news_items:
                    valid_news = []
                    
                    logger.debug(
                        "[DATA_VALIDATE] Start time filtering",
                        news_count=len(news_items),
                        range_start=news_start_date.strftime('%Y-%m-%d'),
                        range_end=news_end_date.strftime('%Y-%m-%d')
                    )
                    
                    for i, news in enumerate(news_items):
                        if not isinstance(news, dict):
                            logger.debug(f"[DATA_VALIDATE] News #{i}: skip - not dictionary type")
                            continue
                            
                        news_time_str = news.get("published_utc") or news.get("published_date")
                        if not news_time_str:
                            logger.debug(f"[DATA_VALIDATE] News #{i}: skip - no time field")
                            continue
                            
                        try:
                            news_time = pd.to_datetime(news_time_str, utc=True, errors="coerce")
                            if pd.isna(news_time):
                                logger.debug(f"[DATA_VALIDATE] News #{i}: skip - time parsing failed: {news_time_str}")
                                continue
                                
                            from stockbench.core.data_hub import _normalize_timestamp_for_comparison
                            news_time_naive = _normalize_timestamp_for_comparison(news_time)
                            filter_start_naive = _normalize_timestamp_for_comparison(news_start_date)
                            # Fix: Let news_end_date include the entire day, not just midnight
                            # Set end date to 23:59:59 of that day
                            news_end_date_eod = news_end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                            filter_end_naive = _normalize_timestamp_for_comparison(news_end_date_eod)
                            
                            logger.debug(f"[DATA_VALIDATE] News #{i}: time comparison - news:{news_time_naive.strftime('%Y-%m-%d %H:%M')}, range:{filter_start_naive.strftime('%Y-%m-%d')} to {news_end_date.strftime('%Y-%m-%d')} 23:59")
                            
                            if filter_start_naive <= news_time_naive <= filter_end_naive:
                                valid_news.append(news)
                                logger.debug(f"[DATA_VALIDATE] News #{i}: Passed time filtering")
                            else:
                                logger.debug(f"[DATA_VALIDATE] News #{i}: Time out of range")
                        except Exception as e:
                            logger.debug(f"[DATA_VALIDATE] News #{i}: Time processing exception: {e}")
                            continue
                    
                    news_items = valid_news
                    logger.debug(f"[DATA_VALIDATE] Time filtering completed - remaining news count: {len(news_items)}")
                        
            except Exception as e:
                # Failed to get news
                import traceback
                traceback.print_exc()
            
            # Get financial data (using ToolRegistry)
            financials = []
            try:
                tool_registry = ToolRegistry.default()
                financials_result = tool_registry.execute("get_financials", symbol=symbol)
                if financials_result.success:
                    financials = financials_result.data
                else:
                    logger.debug(
                        "[DATA_FETCH] Financials tool failed",
                        error=financials_result.error
                    )
            except Exception as e:
                # Failed to get financial data
                pass
            
            # Get dividend and split data (using ToolRegistry)
            dividends = pd.DataFrame()
            splits = pd.DataFrame()
            try:
                tool_registry = ToolRegistry.default()
                
                # Get dividends via ToolRegistry
                dividends_result = tool_registry.execute("get_dividends", symbol=symbol)
                if dividends_result.success and dividends_result.data:
                    dividends = pd.DataFrame(dividends_result.data) if isinstance(dividends_result.data, list) else dividends_result.data
                
                # Get splits via ToolRegistry
                splits_result = tool_registry.execute("get_splits", symbol=symbol)
                if splits_result.success and splits_result.data:
                    splits = pd.DataFrame(splits_result.data) if isinstance(splits_result.data, list) else splits_result.data
            except Exception as e:
                # Failed to get dividend/split data
                pass
            
            # Build market snapshot
            ref_price = open_map.get(symbol, 0.0)
            snapshot = {
                "symbol": symbol,
                "price": ref_price,
                "ts_utc": ctx["date"].strftime("%Y-%m-%dT00:00:00Z")
            }
            
            # Build symbol details
            details = {"ticker": symbol}
            
            # Build position state
            position = portfolio.positions.get(symbol)
            # Use unified price tools to calculate position value
            current_position_value = 0.0
            if position and hasattr(position, "shares") and position.shares:
                from stockbench.core.price_utils import calculate_position_value
                
                # Prepare fallback price
                fallback_price = ref_price or (bars_day["close"].iloc[-1] if not bars_day.empty and "close" in bars_day.columns else 100.0)
                
                try:
                    # Print debug info: check price data in ctx
                    if ctx:
                        open_map_keys = list(ctx.get("open_map", {}).keys())
                        open_price_map_keys = list(ctx.get("open_price_map", {}).keys())
                        logger.debug(
                            "[BT_POSITION] Price data availability",
                            symbol=symbol,
                            open_map_count=len(open_map_keys),
                            open_price_map_count=len(open_price_map_keys),
                            in_open_map=symbol in ctx.get("open_map", {}),
                            in_open_price_map=symbol in ctx.get("open_price_map", {})
                        )
                    
                    current_position_value = calculate_position_value(
                        symbol=symbol,
                        shares=position.shares,
                        ctx=ctx,
                        portfolio=None,  # No portfolio object here
                        position_avg_price=getattr(position, 'avg_price', None)
                    )
                    
                    # If unified tool also fails, use original fallback logic
                    if current_position_value == 0.0 and fallback_price and fallback_price > 0:
                        current_position_value = float(position.shares * fallback_price)
                        logger.debug(
                            "[BT_POSITION] Position value calculated",
                            symbol=symbol,
                            shares=round(position.shares, 2),
                            price=round(fallback_price, 4),
                            value=round(current_position_value, 2),
                            method="final_fallback"
                        )
                        
                except Exception as e:
                    current_position_value = 0.0
                    logger.warning(
                        "[BT_POSITION] Failed to calculate position value",
                        symbol=symbol,
                        error=str(e)
                    )
                    # Print more detailed error information
                    logger.warning(
                        "[BT_POSITION_DEBUG] Detailed error",
                        symbol=symbol,
                        error=str(e)
                    )
            
            # If no position object, create a default position state
            if position is None:
                # Create default position state: 0 shares, 0 avg price, 0 holding days
                position = type('Position', (), {
                    'shares': 0,
                    'avg_price': 0.0,
                    'holding_days': 0
                })()
            

            holding_days = int(getattr(position, "holding_days", 0) or 0) if position else 0
            position_state = {
                "current_position_value": current_position_value,  # Use amount instead of percentage
                "holding_days": holding_days,
                "shares": round(float(getattr(position, "shares", 0) or 0), 2)
            }
            
            # Convert news_items to simple title+description format
            simple_news_list = []
            if news_items:
                for news_item in news_items:
                    if isinstance(news_item, dict):
                        title = news_item.get("title", "")
                        description = news_item.get("description", "")
                        if title:
                            # Format: "title - description" if both exist, otherwise just title
                            if description and description.strip():
                                news_text = f"{title} - {description}"
                            else:
                                news_text = title
                            simple_news_list.append(news_text)
                    elif isinstance(news_item, str) and news_item.strip():
                        simple_news_list.append(news_item)
            
            if not simple_news_list:
                simple_news_list = ["No news available"]
            
            # Build historical close_7d price series correctly
            close_7d = []
            try:
                if not bars_day.empty and "close" in bars_day.columns:
                    # Sort data by date and remove duplicates
                    if "date" in bars_day.columns:
                        bars_clean = bars_day.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                    else:
                        bars_clean = bars_day.drop_duplicates(keep="last")
                    
                    # Get closing prices from the past 7 days (excluding current day)
                    if len(bars_clean) > 1:
                        # Exclude current day and take previous 7 days
                        available_historical_data = len(bars_clean) - 1  # Exclude current day
                        if available_historical_data > 0:
                            start_idx = max(0, available_historical_data - 7)
                            end_idx = available_historical_data  # Exclude current day
                            close_data = bars_clean["close"].iloc[start_idx:end_idx]
                            
                            # Convert to float list
                            for val in close_data:
                                if val is not None and not pd.isna(val):
                                    close_7d.append(float(val))
                                else:
                                    close_7d.append(0.0)
                    
                    # Pad with 0s if insufficient data
                    if len(close_7d) < 7:
                        close_7d = [0.0] * (7 - len(close_7d)) + close_7d
                        
                    # Ensure exactly 7 elements
                    close_7d = close_7d[-7:]  # Take last 7 elements
                        
                else:
                    # No historical data available
                    close_7d = [0.0] * 7
                    
            except Exception as e:
                logger.warning(f"Error building close_7d for {symbol}: {e}")
                close_7d = [0.0] * 7

            # Build minimal features structure
            fi = {
                "symbol": symbol,
                "features": {
                    "market_data": {"ticker": symbol, "open": ref_price, "close_7d": close_7d},
                    "news_events": {"top_k_events": simple_news_list},
                    "position_state": position_state
                },
                "market_ctx": {"daily_drawdown_pct": float(ctx.get("daily_drawdown_pct") or 0.0)}
            }
            
            features_list.append(fi)
        
        return features_list

    def on_bar(self, ctx) -> List[Dict]:
        """
        Generate daily orders: First construct features, then have LLM generate target positions, finally convert to buy/sell orders.
        ctx: {date, symbols, open_map/open_price_map, ref_price_map, portfolio, cfg, datasets, rejected_orders, ...}
        """
        # Call LLM to generate decisions
        open_map = ctx["open_map"]
        if not open_map:
            return []
        
        # 1) Feature construction
        features_list = self._build_features_for_day(ctx)
        
        # 2) Get current date for logging and context
        current_date = ctx["date"].strftime("%Y-%m-%d")
        logger.debug(f"[AGENT_EXECUTOR] Current date: {current_date}")
        
        # 3) Use unified executor for decision-making (automatically route to single or dual Agent mode based on configuration)
        logger.info(f"\n=== Unified Executor Decision Call Started ===")
        logger.info(f"[AGENT_EXECUTOR] Using unified executor for decision-making, Agent mode: {self.agent_mode}")
        logger.info(f"[AGENT_EXECUTOR] Agent mode in configuration: {(self.cfg or {}).get('agents', {}).get('mode', 'single')}")
        logger.info(f"[AGENT_EXECUTOR] Memory system: Agents will use ctx.memory.episodes for history")
        
        # Build bars_data for feature conversion (complete version, including all data needed for dual-agent mode)
        bars_data = {}
        for fi in features_list:
            symbol = fi["symbol"]
            # Get historical data from ctx for feature conversion
            start_date = ctx["date"] - pd.Timedelta(days=self.warmup_days)  # Look back 60 days for feature construction
            end_date = ctx["date"]
            
            # Get market snapshot data
            ref_price = open_map.get(symbol, 0.0)
            snapshot = {
                "symbol": symbol,
                "price": ref_price,
                "ts_utc": ctx["date"].strftime("%Y-%m-%dT00:00:00Z")
            }
            
            # Get details data
            details = {"ticker": symbol}
            
            # Get news data (extract from original features to avoid duplicate API calls)
            news_items = []
            try:
                # Extract news data from already built features
                if "features" in fi and "news_events" in fi["features"]:
                    top_k_events = fi["features"]["news_events"].get("top_k_events", [])
                    # Convert news events to simple news_items format for decision agent
                    if isinstance(top_k_events, list) and top_k_events and top_k_events[0] != "No news data available":
                        for event in top_k_events:
                            if isinstance(event, str) and event.strip():
                                # Since top_k_events is already in title+description format from analysis,
                                # we can use it directly as title for the decision agent
                                news_items.append({
                                    "title": event,  # This already contains "title - description"
                                    "description": "",  # Keep empty since title already has full info
                                    "published_utc": ctx["date"].strftime("%Y-%m-%dT00:00:00Z")
                                })
            except Exception as e:
                logger.warning(f"Failed to extract news data from features {symbol}: {e}")
            
            # Get position state (extract from already built features)
            position_state = {}
            try:
                if "features" in fi and "position_state" in fi["features"]:
                    position_state = fi["features"]["position_state"].copy()
                else:
                    # Fallback plan: build default position state
                    position = ctx["portfolio"].positions.get(symbol)
                    current_position_value = 0.0
                    if position and hasattr(position, "shares") and position.shares:
                        from stockbench.core.price_utils import calculate_position_value
                        
                        # Prepare fallback price
                        fallback_price = ref_price or (bars_day["close"].iloc[-1] if not bars_day.empty and "close" in bars_day.columns else 100.0)
                        
                        try:
                            # Print debug info: check price data in ctx
                            if ctx:
                                open_map_keys = list(ctx.get("open_map", {}).keys())
                                open_price_map_keys = list(ctx.get("open_price_map", {}).keys())
                                logger.debug(
                                    "[BT_POSITION] Price data availability",
                                    symbol=symbol,
                                    open_map_count=len(open_map_keys),
                                    open_price_map_count=len(open_price_map_keys),
                                    in_open_map=symbol in ctx.get("open_map", {}),
                                    in_open_price_map=symbol in ctx.get("open_price_map", {})
                                )
                            
                            current_position_value = calculate_position_value(
                                symbol=symbol,
                                shares=position.shares,
                                ctx=ctx,
                                portfolio=None,  # No portfolio object here
                                position_avg_price=getattr(position, 'avg_price', None)
                            )
                            
                            # If unified tool also fails, use original fallback logic
                            if current_position_value == 0.0 and fallback_price and fallback_price > 0:
                                current_position_value = float(position.shares * fallback_price)
                                logger.debug(
                                    "[BT_POSITION] Position value calculated",
                                    symbol=symbol,
                                    shares=round(position.shares, 2),
                                    price=round(fallback_price, 4),
                                    value=round(current_position_value, 2),
                                    method="final_fallback"
                                )
                                
                        except Exception as e:
                            current_position_value = 0.0
                            logger.warning(
                                "[BT_POSITION] Failed to calculate position value",
                                symbol=symbol,
                                error=str(e)
                            )
                            # Print more detailed error information
                            logger.warning(
                                "[BT_POSITION_DEBUG] Detailed error",
                                symbol=symbol,
                                error=str(e)
                            )
                    
                    holding_days = int(getattr(position, "holding_days", 0) or 0) if position else 0
                    position_state = {
                        "current_position_value": current_position_value,
                        "holding_days": holding_days,
                        "shares": round(float(getattr(position, "shares", 0) or 0), 2)
                    }
            except Exception as e:
                logger.warning(f"Failed to build position state {symbol}: {e}")
                position_state = {
                    "current_position_value": 0.0,
                    "holding_days": 0,
                    "shares": 0.0
                }
            
            bars_data[symbol] = {
                "bars_day": ctx["datasets"].get_day_bars(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
                "snapshot": snapshot,
                "details": details,
                "news_items": news_items,
                "position_state": position_state
            }
        # Get run_id from ctx for organizing LLM cache directory
        run_id = ctx.get("run_id")
        
        # Extract rejected_orders from ctx for retry mechanism
        rejected_orders = ctx.get("rejected_orders", None)
        
        # === Part 2: Create PipelineContext for unified data flow and tracing ===
        pipeline_ctx = PipelineContext(
            run_id=run_id or "unknown",
            date=ctx["date"],
            llm_client=None,  # Will be created by agents as needed
            llm_config=None,
            config=self.cfg
        )
        
        # Store data in pipeline context for agents to access
        pipeline_ctx.put("portfolio", ctx["portfolio"], agent_name="strategy")
        pipeline_ctx.put("bars_data", bars_data, agent_name="strategy")
        pipeline_ctx.put("rejected_orders", rejected_orders, agent_name="strategy")
        pipeline_ctx.put("features_list", features_list, agent_name="strategy")
        
        logger.info(
            "[AGENT_EXECUTOR] Calling unified executor",
            agent_mode=self.agent_mode,
            features_count=len(features_list),
            run_id=run_id,
            rejected_orders=len(rejected_orders) if rejected_orders else 0
        )
        
        # Call unified executor to get decisions
        # Note: The new Memory system (ctx.memory.episodes) will handle history automatically in agents
        decisions_map = unified_decide_batch(
            features_list, 
            cfg=self.cfg, 
            enable_llm=True, 
            bars_data=bars_data, 
            run_id=run_id, 
            rejected_orders=rejected_orders, 
            ctx=pipeline_ctx  # Use PipelineContext with integrated Memory system
        )
        
        logger.info(
            "[AGENT_EXECUTOR] Unified executor completed",
            result_type=type(decisions_map).__name__,
            decision_count=len(decisions_map) if decisions_map else 0
        )
        
        # === Part 2: Output execution trace summary ===
        trace_summary = pipeline_ctx.trace.to_summary()
        logger.info(
            "[AGENT_EXEC] Execution summary",
            total_agents=trace_summary['total_agents'],
            success=trace_summary['success'],
            failed=trace_summary['failed'],
            duration_ms=round(trace_summary['total_duration_ms'], 1)
        )
        
        # Log failed agents if any
        failed_agents = pipeline_ctx.get_failed_agents()
        if failed_agents:
            logger.warning(
                "[AGENT_ERROR] Failed agents detected",
                failed_agents=failed_agents
            )
        
        # Store trace in ctx for later retrieval (e.g., for reports)
        ctx["pipeline_trace"] = trace_summary
        
        logger.info(f"=== Unified Executor Decision Call Ended ===\n")
        
        # Note: Order rejection retry logic has been removed, unified retry mechanism will handle automatically

        # 4) Generate orders
        orders: List[Dict] = []
        pf = ctx["portfolio"]
        ref_price_map = ctx.get("ref_price_map", {}) or {}
        equity_for_sizing = float(ctx.get("equity_for_sizing") or 0.0)
        
        # Add debug logs
        logger.debug(
            "[BT_ENGINE] Portfolio state",
            equity_for_sizing=equity_for_sizing,
            portfolio_equity=pf.equity,
            feature_count=len(features_list),
            decision_count=len(decisions_map)
        )
        
        for fi in features_list:
            s = fi["symbol"]
            decision = decisions_map.get(s, {})
            action = decision.get("action", "hold")
            
            ref_px =  open_map.get(s) # ref_price_map.get(s)
            
            if ref_px is None or ref_px <= 0:
                logger.debug(f"[DEBUG] LLM Strategy: {s} - skip, invalid reference price")
                continue
                
            pos = pf.positions.get(s)
            current_value = (pos.shares * float(ref_px)) if pos else 0.0
            
            # Fix target_cash_amount handling logic for hold operations
            if action == "hold" and "target_cash_amount" not in decision:
                # For hold operations without target_cash_amount, use current position value
                target_cash = current_value
                logger.debug(f"[DEBUG] LLM Strategy: {s} - hold operation auto-set target_cash={target_cash} (current position value)")
            else:
                target_cash = float(decision.get("target_cash_amount", 0.0))
            
            # Add debug information for each symbol
            logger.debug(f"[DEBUG] LLM Strategy: {s} - action={action}, target_cash={target_cash}, ref_px={ref_px}")
            # Directly use LLM output target cash amount, no need to convert to percentage
            target_value = max(0.0, target_cash)  # Ensure target amount is non-negative
            delta_value = target_value - current_value
            
            logger.debug(f"[DEBUG] LLM Strategy: {s} - current_value={current_value}, target_value={target_value}, delta_value={delta_value}")
            
            # CRITICAL FIX: Detect and correct LLM decision logic errors
            # When target_cash_amount equals current_position_value, it should be a hold operation
            if action == "increase" and abs(delta_value) < 0.01:
                logger.warning(f"[DECISION_LOGIC_FIX] {s}: LLM marked as 'increase' but delta_value={delta_value:.4f} ≈ 0, treating as 'hold'")
                action = "hold"  # Override incorrect LLM decision
            elif action == "decrease" and abs(delta_value) < 0.01:
                logger.warning(f"[DECISION_LOGIC_FIX] {s}: LLM marked as 'decrease' but delta_value={delta_value:.4f} ≈ 0, treating as 'hold'")
                action = "hold"  # Override incorrect LLM decision
            
            # Only trigger trades under explicit actions
            if action == "increase" and delta_value > 0:
                qty = round(delta_value / float(ref_px), 2)
                if qty > 0:
                    orders.append({"symbol": s, "side": "buy", "qty": qty})
                    logger.debug(f"[DEBUG] LLM Strategy: {s} - generated buy order: qty={qty}")
                else:
                    logger.debug(f"[DEBUG] LLM Strategy: {s} - skip, calculated quantity is 0")
            elif action in ("decrease", "close") and delta_value < 0:
                qty = round(abs(delta_value) / float(ref_px), 2)
                if pos and pos.shares > 0:
                    qty = min(qty, pos.shares)
                if qty > 0:
                    orders.append({"symbol": s, "side": "sell", "qty": -qty})
                    logger.debug(f"[DEBUG] LLM Strategy: {s} - generated sell order: qty={qty}")
                else:
                    logger.debug(f"[DEBUG] LLM Strategy: {s} - skip, calculated quantity is 0")
            else:
                logger.debug(f"[DEBUG] LLM Strategy: {s} - skip, action={action}, delta_value={delta_value}")
        
        logger.debug(f"[DEBUG] LLM Strategy: final generated order count={len(orders)}")
        
        # 5) Store pending decisions (to be recorded after execution)
        logger.info(f"\n=== Store Pending Decisions Started ===")
        logger.info(f"[PENDING_SAVE] Preparing to store pending decisions")
        logger.debug(f"[PENDING_SAVE] Current date: {current_date}")
        logger.debug(f"[PENDING_SAVE] Decision result type: {type(decisions_map)}")
        logger.debug(f"[PENDING_SAVE] Decision result keys: {list(decisions_map.keys()) if decisions_map else 'None'}")
        
        # Store pending decisions for later recording
        self.pending_decisions = decisions_map.copy() if decisions_map else {}
        
        # Store meta information
        meta = {"date": current_date, "calls": 1}
        if decisions_map and "__meta__" in decisions_map:
            decisions_map["__meta__"]["date"] = current_date
            meta.update(decisions_map["__meta__"])
            logger.debug(f"[PENDING_SAVE] Update meta info - date={current_date}")
            logger.debug(f"[PENDING_SAVE] Complete meta info: {meta}")
        else:
            logger.debug(f"[PENDING_SAVE] Using basic meta info: {meta}")
        
        self.pending_meta = meta.copy()
        
        # Print current decision result summary
        if decisions_map:
            logger.debug(f"\n[PENDING_SAVE] Current decision result summary:")
            for symbol, decision in decisions_map.items():
                if symbol != "__meta__":
                    action = decision.get("action", "unknown")
                    target_cash = decision.get("target_cash_amount", 0.0)
                    cash_change = decision.get("cash_change", 0.0)
                    confidence = decision.get("confidence", 0.0)
                    logger.debug(f"[PENDING_SAVE]   {symbol} - action={action}, target_cash_amount={target_cash}, cash_change={cash_change}, confidence={confidence}")
        
        logger.info(f"[PENDING_SAVE] Pending decisions stored, waiting for recording after trade execution")
        logger.info(f"=== Store Pending Decisions Completed ===\n")
        
        return orders 

    def record_executed_decisions(self, executed_symbols: List[str], portfolio=None) -> None:
        """Clear pending decisions after execution.
        
        Note: Decision recording is now handled automatically by the new Memory system (ctx.memory.episodes)
        in the Agent layer. This method only clears the pending state.
        
        Args:
            executed_symbols: List of symbols that were successfully executed
            portfolio: Portfolio object (kept for backward compatibility but not used)
        """
        if not self.pending_decisions:
            logger.debug(f"[RECORD] No pending decisions, skipping")
            return
            
        logger.info(f"\n=== Decision Recording (New Memory System) ===")
        logger.info(f"[RECORD] Decisions have been saved to ctx.memory.episodes by Agents automatically")
        logger.info(f"[RECORD] Clearing pending decisions state")
        logger.debug(f"[RECORD] Executed symbols: {executed_symbols}")
        logger.debug(f"[RECORD] Pending symbols: {list(self.pending_decisions.keys())}")
        
        # Clear pending decisions
        self.pending_decisions = {}
        self.pending_meta = {}
        self.pending_date = ""
        
        logger.info(f"=== Decision Recording Completed ===\n")