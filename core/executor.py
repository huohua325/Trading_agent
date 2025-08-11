from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List

from trading_agent_v2.core.schemas import Order
from trading_agent_v2.utils.io import atomic_append_jsonl, ensure_dir

# 根据决策和快照价格，生成订单
def plan_orders(decision: Dict, snapshot_price: float, cfg: Dict, portfolio: Dict | None = None) -> List[Dict]:
    twap_slices: int = int(cfg.get("execution", {}).get("twap_slices", 1))
    price_guard_bps: float = float(cfg.get("execution", {}).get("price_guard_bps", 0))
    backtest_cash_default: float = float(cfg.get("backtest", {}).get("cash", 1_000_000))

    symbol = decision.get("symbol", "UNKNOWN")
    target_pos_pct = float(decision.get("target_pos_pct", 0))
    action = decision.get("action", "hold")

    # 资金/持仓
    equity = float((portfolio or {}).get("equity", backtest_cash_default))
    current_position_pct = float((portfolio or {}).get("positions", {}).get(symbol, {}).get("position_pct", 0.0))
    delta_pct = max(0.0, target_pos_pct - current_position_pct) if action == "increase" else 0.0
    if delta_pct <= 0 or snapshot_price <= 0:
        return []

    target_value = delta_pct * equity
    qty_total = int(target_value / snapshot_price)
    if qty_total <= 0:
        return []

    qty_per_slice = max(qty_total // max(twap_slices, 1), 0)
    if qty_per_slice == 0:
        qty_per_slice = qty_total  # 小额下单合并成一片
        twap_slices = 1

    px_guard = snapshot_price * price_guard_bps / 10_000.0
    limit = snapshot_price + px_guard

    orders: List[Dict] = []
    for i in range(max(twap_slices, 1)):
        ord_obj = Order(symbol=symbol, side="buy", qty=qty_per_slice, limit=round(limit, 4), slice=i + 1, twap_slices=twap_slices)
        orders.append(ord_obj.model_dump())
    return orders

# 记录订单（JSONL 原子追加）
def record_orders(orders: List[Dict], audit_payload: Dict) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = ts[:10]
    base_dir = os.path.join(os.getcwd(), "trading_agent_v2", "storage", "audit", date_str)
    ensure_dir(base_dir)
    symbol = audit_payload.get("symbol", "UNKNOWN")
    file_path = os.path.join(base_dir, f"{symbol}.jsonl")

    record = dict(audit_payload)
    record["ts_utc"] = ts
    record.setdefault("meta", {})
    record.setdefault("schema_version", audit_payload.get("schema_version", "v1"))
    # 版本元信息（若在 config 中提供则附加）
    cfg = record.get("config", {}) or {}
    pv = cfg.get("prompt_version") or record.get("prompt_version")
    rv = cfg.get("risk_version") or record.get("risk_version")
    if pv:
        record["prompt_version"] = pv
    if rv:
        record["risk_version"] = rv

    record["meta"].setdefault("price_guard_bps", cfg.get("execution", {}).get("price_guard_bps", 0))
    record["orders"] = orders

    atomic_append_jsonl(file_path, record) 