from __future__ import annotations

import json
import os
from typing import Dict, List

from loguru import logger
from stockbench.core.schemas import Order

# Generate orders based on decisions and price information (refactored: consistent with backtesting engine logic)
def plan_orders(decision: Dict, snapshot_price: float, cfg: Dict, portfolio: Dict | None = None) -> List[Dict]:
    twap_slices: int = int(cfg.get("execution", {}).get("twap_slices", 1))
    price_guard_bps: float = float(cfg.get("execution", {}).get("price_guard_bps", 0))
    # Get backtesting fund configuration, prioritize using portfolio.total_cash
    portfolio_cash = float(cfg.get("portfolio", {}).get("total_cash", 1_000_000))
    backtest_cash_default: float = float(cfg.get("backtest", {}).get("cash", portfolio_cash))

    symbol = decision.get("symbol", "UNKNOWN")
    target_cash_amount = float(decision.get("target_cash_amount", 0))
    action = decision.get("action", "hold")

    # Fund/position information
    equity = float((portfolio or {}).get("equity", backtest_cash_default))
    
    # Get trade execution price: prioritize using open price, consistent with backtesting engine logic
    from stockbench.core.price_utils import get_unified_price
    
    # Get open price as trading reference price (consistent with backtesting engine)
    ref_price = get_unified_price(symbol, {}, portfolio, "open", snapshot_price)
    if not ref_price or ref_price <= 0:
        ref_price = snapshot_price  # Fallback to snapshot price
    
    logger.debug(
        "[BT_EXECUTOR] Price reference",
        symbol=symbol,
        ref_price=round(ref_price, 4),
        snapshot_price=round(snapshot_price, 4)
    )
    
    # Calculate current position value (using same price reference)
    position_info = (portfolio or {}).get("positions", {}).get(symbol, {})
    shares = float(position_info.get("shares", 0.0))
    
    if shares > 0:
        current_position_value = shares * ref_price
        logger.debug(
            "[BT_EXECUTOR] Current position",
            symbol=symbol,
            shares=shares,
            ref_price=round(ref_price, 4),
            value=round(current_position_value, 2)
        )
    else:
        current_position_value = 0.0
    
    # Calculate cash change (consistent with backtesting engine logic)
    cash_change = target_cash_amount - current_position_value
    logger.debug(
        "[BT_EXECUTOR] Cash change",
        symbol=symbol,
        target_cash_amount=round(target_cash_amount, 2),
        current_position_value=round(current_position_value, 2)
    )
    
    if action == "increase":
        cash_change = max(0.0, cash_change)  # Cash change cannot be negative when increasing position
        logger.debug(
            "[BT_EXECUTOR] Adjusted cash change",
            symbol=symbol,
            action=action,
            cash_change=round(cash_change, 2)
        )
    elif action == "decrease" or action == "close":
        cash_change = min(0.0, cash_change)  # Cash change cannot be positive when decreasing position
        logger.debug(
            "[BT_EXECUTOR] Adjusted cash change",
            symbol=symbol,
            action=action,
            cash_change=round(cash_change, 2)
        )
    
    if abs(cash_change) <= 0 or ref_price <= 0:
        logger.debug(
            "[BT_EXECUTOR] Skip trade",
            symbol=symbol,
            cash_change=round(cash_change, 2),
            ref_price=round(ref_price, 4),
            reason="invalid_params"
        )
        return []

    # Calculate current total position value
    total_current_position_value = sum(
        float(pos.get("position_value", 0.0)) 
        for pos in (portfolio or {}).get("positions", {}).values()
    )
    
    target_value = abs(cash_change)
    
    # Key fix: use ref_price to calculate shares, consistent with backtesting engine logic
    qty_total = round(target_value / ref_price, 2)
    
    logger.debug(
        "[BT_EXECUTOR] Calculate shares",
        symbol=symbol,
        target_value=round(target_value, 2),
        ref_price=round(ref_price, 4),
        qty_total=qty_total
    )
    
    if qty_total <= 0:
        logger.debug(
            "[BT_EXECUTOR] Skip trade - zero shares",
            symbol=symbol,
            target_value=round(target_value, 2),
            ref_price=round(ref_price, 4)
        )
        return []

    qty_per_slice = max(round(qty_total / max(twap_slices, 1), 2), 0)
    if qty_per_slice == 0:
        qty_per_slice = qty_total  # Merge small orders into one slice
        twap_slices = 1

    # Final validation: ensure actual trade amount does not exceed target amount
    actual_trade_amount = qty_total * ref_price
    expected_amount = target_value
    
    if action == "increase" and actual_trade_amount > expected_amount * 1.01:  # Allow 1% error
        # If calculated trade amount significantly exceeds expected, readjust share count
        qty_total = round(expected_amount / ref_price, 2)
        actual_trade_amount = qty_total * ref_price
        logger.warning(f"[EXECUTOR] {symbol}: Adjust shares to prevent overspending - new shares={qty_total}, actual amount={actual_trade_amount:.2f}")
    
    logger.info(f"[EXECUTOR] {symbol}: Final trade - shares={qty_total}, unit price={ref_price:.4f}, total amount={actual_trade_amount:.2f}")
    logger.debug(f"[EXECUTOR] {symbol}: Expected vs Actual - target={expected_amount:.2f}, actual={actual_trade_amount:.2f}, difference={actual_trade_amount-expected_amount:.2f}")

    # Set trading price protection (use ref_price instead of snapshot_price)
    px_guard = ref_price * price_guard_bps / 10_000.0
    limit = ref_price + px_guard

    # Determine trade direction
    side = "buy" if cash_change > 0 else "sell"
    
    orders: List[Dict] = []
    for i in range(max(twap_slices, 1)):
        ord_obj = Order(symbol=symbol, side=side, qty=qty_per_slice, limit=round(limit, 4), slice=i + 1, twap_slices=twap_slices)
        orders.append(ord_obj.model_dump())
    
    logger.info(f"[EXECUTOR] {symbol}: Generated {len(orders)} {side} orders, each {qty_per_slice} shares, limit price {limit:.4f}")
    return orders



def decide_batch(features_list: List[Dict], cfg: Dict | None = None, **kwargs) -> Dict[str, Dict]:
    """
    Unified decision entry point that supports dual agent mode only
    
    This function routes all decision requests to the dual agent architecture.
    Single agent mode has been removed to simplify the codebase.
    
    Args:
        features_list: Input features list
        cfg: Configuration dictionary containing agent mode settings
        **kwargs: Additional keyword arguments passed to the dual agent
        
    Returns:
        Dictionary {symbol: decision_dict, "__meta__": meta_dict}
    """
    
    # Check agent mode from configuration for logging/warning purposes
    agent_mode = (cfg or {}).get("agents", {}).get("mode", "dual")
    
    if agent_mode != "dual":
        logger.warning(f"[executor] agents.mode is '{agent_mode}', but only 'dual' mode is supported. Using dual agent.")
    
    # Use dual agent architecture only
    from stockbench.agents.dual_agent_llm import decide_batch_dual_agent
    return decide_batch_dual_agent(features_list, cfg, **kwargs) 