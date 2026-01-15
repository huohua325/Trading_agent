"""
Pipeline Context - Agent 流水线上下文管理

提供统一的上下文传递和执行追踪机制，用于 StockBench 多 Agent 流水线架构。

主要组件:
- AgentStep: 单个 Agent 执行步骤的记录
- AgentTrace: Agent 执行追踪器，记录整个 Pipeline 的执行过程
- PipelineContext: Agent 流水线上下文，统一管理数据流动和追踪

使用示例:
    ctx = PipelineContext(
        run_id="backtest_2025_01_01",
        date="2025-01-01",
        llm_client=llm,
        llm_config=cfg,
        config=config
    )
    
    # 存入初始数据
    ctx.put("previous_decisions", prev_decisions)
    
    # Agent 执行时自动追踪
    filter_result = filter_stocks_needing_fundamental(features_list, ctx=ctx)
    decisions = generate_dual_decisions(features_list, ctx=ctx)
    
    # 获取执行摘要
    print(ctx.trace.to_summary())
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, TYPE_CHECKING
from datetime import datetime
from loguru import logger

if TYPE_CHECKING:
    from stockbench.memory import MemoryStore
    from stockbench.core.message import Message


@dataclass
class AgentStep:
    """
    单个 Agent 执行步骤的记录
    
    Attributes:
        agent_name: Agent 名称
        started_at: 开始时间
        finished_at: 结束时间
        input_summary: 输入摘要（避免存大数据）
        output_summary: 输出摘要
        tokens_used: 使用的 token 数
        status: 执行状态 (running | success | failed)
        error: 错误信息
        duration_ms: 执行耗时（毫秒）
    """
    agent_name: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    tokens_used: int = 0
    status: str = "running"  # running | success | failed
    error: Optional[str] = None
    duration_ms: float = 0
    
    def finish(self, status: str, output_summary: str = None, error: str = None):
        """完成执行步骤"""
        self.finished_at = datetime.now()
        self.status = status
        self.output_summary = output_summary
        self.error = error
        self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "tokens_used": self.tokens_used,
            "status": self.status,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass  
class AgentTrace:
    """
    Agent 执行追踪器
    
    记录整个 Pipeline 的执行过程，包括每个 Agent 的输入/输出/耗时/错误。
    
    Attributes:
        run_id: 运行 ID
        steps: 执行步骤列表
    """
    run_id: str
    steps: List[AgentStep] = field(default_factory=list)
    
    def start_agent(self, agent_name: str, input_summary: str = None) -> AgentStep:
        """
        开始追踪一个 Agent
        
        Args:
            agent_name: Agent 名称
            input_summary: 输入摘要
            
        Returns:
            AgentStep: 新创建的执行步骤
        """
        step = AgentStep(
            agent_name=agent_name,
            started_at=datetime.now(),
            input_summary=input_summary
        )
        self.steps.append(step)
        logger.info(
            f"[AGENT_START] {agent_name} started",
            agent=agent_name,
            input_summary=input_summary or "N/A"
        )
        return step
    
    def finish_agent(self, step: AgentStep, status: str, output_summary: str = None, error: str = None):
        """
        完成追踪一个 Agent
        
        Args:
            step: 执行步骤
            status: 执行状态 (success | failed)
            output_summary: 输出摘要
            error: 错误信息
        """
        step.finish(status, output_summary, error)
        if status == "success":
            logger.info(
                f"[AGENT_DONE] {step.agent_name} completed",
                agent=step.agent_name,
                duration_ms=round(step.duration_ms, 2),
                output_summary=output_summary or "N/A"
            )
        else:
            logger.error(
                f"[AGENT_ERROR] {step.agent_name} failed",
                agent=step.agent_name,
                duration_ms=round(step.duration_ms, 2),
                error=error
            )
    
    def get_failed_agents(self) -> List[str]:
        """获取失败的 Agent 列表"""
        return [s.agent_name for s in self.steps if s.status == "failed"]
    
    def get_successful_agents(self) -> List[str]:
        """获取成功的 Agent 列表"""
        return [s.agent_name for s in self.steps if s.status == "success"]
    
    def get_total_duration_ms(self) -> float:
        """获取总执行时间（毫秒）"""
        return sum(s.duration_ms for s in self.steps)
    
    def get_total_tokens(self) -> int:
        """获取总 token 使用量"""
        return sum(s.tokens_used for s in self.steps)
    
    def to_summary(self) -> Dict[str, Any]:
        """
        输出执行摘要
        
        Returns:
            包含执行统计信息的字典
        """
        return {
            "run_id": self.run_id,
            "total_agents": len(self.steps),
            "success": len([s for s in self.steps if s.status == "success"]),
            "failed": len([s for s in self.steps if s.status == "failed"]),
            "total_duration_ms": self.get_total_duration_ms(),
            "total_tokens": self.get_total_tokens(),
            "steps": [
                {
                    "agent": s.agent_name,
                    "status": s.status,
                    "duration_ms": round(s.duration_ms, 2),
                    "tokens": s.tokens_used,
                    "error": s.error
                } 
                for s in self.steps
            ]
        }


@dataclass
class PipelineContext:
    """
    Agent 流水线上下文
    
    统一管理 Agent 间的数据流动和执行追踪。
    
    Attributes:
        run_id: 运行 ID
        date: 日期
        llm_client: LLM 客户端实例
        llm_config: LLM 配置
        config: 全局配置字典
        trace: 执行追踪器
        logger: 绑定了上下文的 logger 实例
        
    数据总线:
        使用 put()/get() 方法在 Agent 间传递数据
    """
    run_id: str
    date: str
    llm_client: Any  # LLMClient - 使用 Any 避免循环导入
    llm_config: Any  # LLMConfig
    config: Dict[str, Any] = field(default_factory=dict)
    _data_bus: Dict[str, Any] = field(default_factory=dict)
    trace: AgentTrace = field(default=None)
    _memory_store: Optional['MemoryStore'] = field(default=None, repr=False)
    _conversation_history: List['Message'] = field(default_factory=list, repr=False)
    logger: Any = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """初始化后创建追踪器和 logger"""
        if self.trace is None:
            self.trace = AgentTrace(run_id=self.run_id)
        
        # 创建绑定了上下文的 logger
        self.logger = logger.bind(
            run_id=self.run_id,
            date=self.date,
            component="pipeline"
        )
    
    # ==================== 数据总线操作 ====================
    
    def put(self, key: str, value: Any, agent_name: str = None):
        """
        存入数据到数据总线
        
        Args:
            key: 数据键
            value: 数据值
            agent_name: 产生数据的 Agent 名称（可选，用于追踪数据来源）
        """
        self._data_bus[key] = value
        if agent_name:
            self._data_bus[f"_source_{key}"] = agent_name
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        从数据总线获取数据
        
        Args:
            key: 数据键
            default: 默认值
            
        Returns:
            数据值或默认值
        """
        return self._data_bus.get(key, default)
    
    def get_source(self, key: str) -> Optional[str]:
        """
        获取数据来源 Agent
        
        Args:
            key: 数据键
            
        Returns:
            产生该数据的 Agent 名称，如果未记录则返回 None
        """
        return self._data_bus.get(f"_source_{key}")
    
    def keys(self) -> List[str]:
        """获取所有数据键（不含元数据）"""
        return [k for k in self._data_bus.keys() if not k.startswith("_")]
    
    def has(self, key: str) -> bool:
        """检查数据是否存在"""
        return key in self._data_bus
    
    # ==================== Agent 执行追踪 ====================
    
    def start_agent(self, agent_name: str, input_summary: str = None) -> AgentStep:
        """
        开始追踪一个 Agent
        
        Args:
            agent_name: Agent 名称
            input_summary: 输入摘要
            
        Returns:
            AgentStep: 执行步骤对象
        """
        return self.trace.start_agent(agent_name, input_summary)
    
    def finish_agent(self, step: AgentStep, status: str, output_summary: str = None, error: str = None):
        """
        完成追踪一个 Agent
        
        Args:
            step: 执行步骤
            status: 执行状态
            output_summary: 输出摘要
            error: 错误信息
        """
        self.trace.finish_agent(step, status, output_summary, error)
    
    def update_tokens(self, step: AgentStep, tokens: int):
        """更新 Agent 的 token 使用量"""
        step.tokens_used = tokens
    
    def get_agent_logger(self, agent_name: str):
        """为特定 Agent 创建绑定了上下文的 logger"""
        return self.logger.bind(agent=agent_name)
    
    # ==================== 便捷方法 ====================
    
    def get_failed_agents(self) -> List[str]:
        """获取失败的 Agent 列表"""
        return self.trace.get_failed_agents()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return self.trace.to_summary()
    
    # ==================== 记忆系统 ====================
    
    @property
    def memory(self) -> 'MemoryStore':
        """记忆系统入口（延迟初始化）"""
        if self._memory_store is None:
            from stockbench.memory import MemoryStore
            mem_cfg = self.config.get("memory", {})
            self._memory_store = MemoryStore(
                base_path=mem_cfg.get("storage_path", "storage"),
                working_memory_capacity=mem_cfg.get("working_memory", {}).get("capacity", 50),
                working_memory_ttl_minutes=mem_cfg.get("working_memory", {}).get("ttl_minutes", 60),
                episode_max_days=mem_cfg.get("episodic_memory", {}).get("max_days", 30),
            )
        return self._memory_store
    
    @property
    def memory_enabled(self) -> bool:
        """检查记忆系统是否启用"""
        return self.config.get("memory", {}).get("enabled", True)
    
    # ==================== 对话历史 ====================
    
    @property
    def conversation_history(self) -> List['Message']:
        """获取对话历史"""
        return self._conversation_history
    
    def add_to_history(self, message: 'Message'):
        """添加消息到对话历史"""
        from stockbench.core.message import truncate_history
        self._conversation_history.append(message)
        max_msgs = self.config.get("conversation_history", {}).get("max_messages", 20)
        if len(self._conversation_history) > max_msgs:
            self._conversation_history = truncate_history(self._conversation_history, max_msgs)
    
    def clear_history(self):
        """清空对话历史"""
        self._conversation_history.clear()
    
    # ==================== 生命周期钩子 ====================
    
    def on_run_complete(self, portfolio_history: List[Dict] = None):
        """
        运行结束时调用
        
        Args:
            portfolio_history: 持仓历史记录，用于计算实际收益回填
        """
        if self.memory_enabled and self._memory_store:
            # 回填决策结果
            if portfolio_history:
                self._backfill_results(portfolio_history)
            
            # 提交工作记忆到情景记忆
            self._memory_store.commit_working_to_episodes()
            self._memory_store.clear_working()
    
    def _backfill_results(self, portfolio_history: List[Dict]):
        """
        回填决策的实际收益 - 形成闭环学习
        
        Args:
            portfolio_history: 持仓历史记录
                格式: [{"date": "2025-01-01", "positions": {"AAPL": {"value": 1000, "pnl": 50}}, ...}, ...]
        """
        if not self._memory_store:
            return
        
        # 获取最近未回填的决策
        unfilled_episodes = self._memory_store.episodes.query_unfilled(days=7)
        if not unfilled_episodes:
            self.logger.debug("[MEM_BACKFILL] No unfilled episodes to backfill")
            return
        
        # 构建持仓历史索引: {(date, symbol): position_info}
        position_index = {}
        for record in portfolio_history:
            record_date = record.get("date", "")
            positions = record.get("positions", {})
            for symbol, pos_info in positions.items():
                position_index[(record_date, symbol)] = pos_info
        
        filled_count = 0
        for episode in unfilled_episodes:
            # 计算实际收益
            actual_result = self._calculate_episode_result(episode, position_index)
            
            if actual_result is not None:
                self._memory_store.episodes.update_result(
                    episode_id=episode.id,
                    result=actual_result,
                    note=f"Backfilled on {self.date}"
                )
                filled_count += 1
        
        if filled_count > 0:
            self.logger.info(
                "[MEM_BACKFILL] Backfilled episode results",
                filled_count=filled_count
            )
    
    def _calculate_episode_result(self, episode, position_index: Dict) -> Optional[float]:
        """
        计算单个决策的实际收益率
        
        Args:
            episode: 决策记录
            position_index: 持仓历史索引
            
        Returns:
            收益率百分比，如 5.0 表示 +5%，无法计算则返回 None
        """
        from datetime import datetime, timedelta
        
        decision_date = episode.date
        symbol = episode.symbol
        action = episode.action
        
        # 查找决策日的持仓
        entry_pos = position_index.get((decision_date, symbol))
        if not entry_pos:
            return None
        
        entry_value = entry_pos.get("value", 0)
        if entry_value <= 0:
            return None
        
        # 查找 T+3 的持仓（简化评估周期）
        try:
            dt = datetime.strptime(decision_date, "%Y-%m-%d")
            eval_date = (dt + timedelta(days=3)).strftime("%Y-%m-%d")
        except:
            return None
        
        exit_pos = position_index.get((eval_date, symbol))
        if not exit_pos:
            # 尝试查找最近的日期
            for days_offset in range(4, 8):
                try:
                    alt_date = (dt + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                    exit_pos = position_index.get((alt_date, symbol))
                    if exit_pos:
                        break
                except:
                    pass
        
        if not exit_pos:
            return None
        
        exit_value = exit_pos.get("value", 0)
        
        # 计算收益率
        if entry_value > 0:
            result_pct = ((exit_value - entry_value) / entry_value) * 100
            return round(result_pct, 2)
        
        return None
    
    def __repr__(self) -> str:
        return f"PipelineContext(run_id={self.run_id}, date={self.date}, agents={len(self.trace.steps)})"
