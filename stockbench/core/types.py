"""
类型定义

为 StockBench Agent 流水线提供类型定义，增强类型安全和 IDE 支持。

包含:
- Decision: 交易决策类型
- FilterResult: 基本面过滤结果类型
- AgentResult: 通用 Agent 结果类型
"""

from typing import TypedDict, List, Optional, Dict, Any


class Decision(TypedDict, total=False):
    """
    交易决策类型
    
    Attributes:
        action: 决策动作 ("increase" | "decrease" | "hold" | "close")
        target_cash_amount: 目标现金金额
        cash_change: 现金变动金额
        reasons: 决策理由列表
        confidence: 置信度 (0.0-1.0)
    """
    action: str  # "increase" | "decrease" | "hold" | "close"
    target_cash_amount: float
    cash_change: float
    reasons: List[str]
    confidence: float


class FilterResult(TypedDict, total=False):
    """
    基本面过滤结果类型
    
    Attributes:
        needs_fundamental: 需要基本面分析的股票列表
        skip_fundamental: 跳过基本面分析的股票列表
        filter_reasons: 每个股票的过滤理由
    """
    needs_fundamental: List[str]
    skip_fundamental: List[str]
    filter_reasons: Dict[str, str]


class AgentResult(TypedDict, total=False):
    """
    通用 Agent 结果类型
    
    Attributes:
        success: 是否成功
        data: 结果数据
        error: 错误信息
        agent_name: Agent 名称
        duration_ms: 执行耗时
    """
    success: bool
    data: Any
    error: Optional[str]
    agent_name: str
    duration_ms: float


class PipelineSummary(TypedDict):
    """
    Pipeline 执行摘要类型
    
    Attributes:
        run_id: 运行 ID
        total_agents: 总 Agent 数
        success: 成功数
        failed: 失败数
        total_duration_ms: 总耗时
        total_tokens: 总 token 数
        steps: 执行步骤列表
    """
    run_id: str
    total_agents: int
    success: int
    failed: int
    total_duration_ms: float
    total_tokens: int
    steps: List[Dict[str, Any]]


# 导出
__all__ = [
    "Decision",
    "FilterResult", 
    "AgentResult",
    "PipelineSummary",
]
