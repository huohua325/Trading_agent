from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, conlist, confloat
from pydantic import ConfigDict


class _ConfigBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class TechSnapshot(_ConfigBase):
    ret: Dict[str, confloat(ge=-1, le=1)] = Field(default_factory=dict)
    atr_pct: confloat(ge=0, le=1) = 0
    trend: Literal["up", "down", "sideways"] = "sideways"
    mom: confloat(ge=0, le=1) = 0.5
    vwap_dev: float = 0.0


class NewsSnapshot(_ConfigBase):
    sentiment: confloat(ge=-1, le=1) = 0.0
    top_k_events: List[str] = Field(default_factory=list, max_length=10)
    src_count: int = 0
    freshness_min: int = 0


class PositionState(_ConfigBase):
    current_position_pct: confloat(ge=0, le=1) = 0
    avg_price: Optional[float] = None
    pnl_pct: float = 0.0
    holding_days: int = 0


class FeatureInput(_ConfigBase):
    schema_version: str = "v1"
    symbol: str
    ts_utc: str
    tech: TechSnapshot
    news: NewsSnapshot
    fund: Dict[str, float] = Field(default_factory=dict)
    market_ctx: Dict[str, Optional[float]] = Field(default_factory=dict)
    position_state: PositionState


class AnalyzerOutput(_ConfigBase):
    schema_version: str = "v1"
    tech_score: confloat(ge=0, le=1)
    sent_score: confloat(ge=-1, le=1)
    event_risk: Literal["low", "normal", "elevated", "high"]
    summary: List[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1)


class RiskLimits(_ConfigBase):
    allowed: List[Literal["increase", "hold", "decrease", "close"]]
    max_pos_pct: confloat(ge=0, le=1)
    cooldown: bool = False


class DecisionOutput(_ConfigBase):
    schema_version: str = "v1"
    action: Literal["increase", "hold", "decrease", "close"]
    target_pos_pct: confloat(ge=0, le=1)
    reasons: List[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) = 0.5


class Order(_ConfigBase):
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    limit: Optional[float] = None
    slice: int = 1
    twap_slices: int = 1


class AuditRecord(_ConfigBase):
    schema_version: str = "v1"
    ts_utc: str
    symbol: str
    config: Dict[str, object] = Field(default_factory=dict)
    features: Dict[str, object] = Field(default_factory=dict)
    analyzer: Optional[Dict[str, object]] = None
    limits: Optional[Dict[str, object]] = None
    decision: Optional[Dict[str, object]] = None
    orders: List[Dict[str, object]] = Field(default_factory=list)
    snapshot: Dict[str, object] = Field(default_factory=dict)
    meta: Dict[str, object] = Field(default_factory=dict)
    # 版本元信息（可选）
    prompt_version: Optional[str] = None
    risk_version: Optional[str] = None 