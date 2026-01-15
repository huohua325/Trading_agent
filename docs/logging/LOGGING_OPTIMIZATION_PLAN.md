# StockBench 日志系统优化计划

> **文档版本**: v1.0  
> **创建日期**: 2025-12-15  
> **优先级**: 高  
> **预计完成**: Phase 9

---

## 📋 目录

1. [当前状态分析](#1-当前状态分析)
2. [核心问题诊断](#2-核心问题诊断)
3. [优化目标](#3-优化目标)
4. [详细优化方案](#4-详细优化方案)
5. [实施路线图](#5-实施路线图)
6. [最佳实践](#6-最佳实践)

---

## 1. 当前状态分析

### 1.1 日志库使用情况

| 日志库 | 使用文件数 | 主要模块 | 说明 |
|--------|-----------|---------|------|
| **logging** | 11 个文件 | adapters/, agents/, backtest/, core/ | 标准库 logging |
| **loguru** | 12 个文件 | memory/, llm/, tools/, utils/ | 第三方 loguru |
| **混用** | ✅ 已桥接 | logging_setup.py 使用 InterceptHandler | 桥接有效 |

#### 使用 `logging` 的文件：

```
stockbench/
├── adapters/
│   ├── polygon_client.py          # logging.getLogger(__name__)
│   └── finnhub_client.py          # logging.getLogger(__name__)
├── agents/
│   ├── dual_agent_llm.py          # logging.getLogger(__name__)
│   ├── fundamental_filter_agent.py # logging.getLogger(__name__)
│   └── backtest_report_llm.py     # logging.getLogger(__name__)
├── backtest/
│   ├── engine.py                  # logging.getLogger(__name__)
│   └── strategies/llm_decision.py # logging.getLogger(__name__)
├── core/
│   ├── features.py                # logging.getLogger(__name__)
│   ├── executor.py                # logging.getLogger(__name__)
│   └── price_utils.py             # logging.getLogger(__name__)
└── utils/
    └── logging_setup.py           # logging + loguru (桥接)
```

#### 使用 `loguru` 的文件：

```
stockbench/
├── adapters/
│   └── finnhub_client.py          # from loguru import logger
├── core/
│   ├── data_hub.py                # from loguru import logger
│   └── pipeline_context.py        # from loguru import logger
├── llm/
│   └── llm_client.py              # from loguru import logger
├── memory/
│   ├── store.py                   # from loguru import logger
│   └── layers/
│       ├── episodic.py            # from loguru import logger
│       ├── cache.py               # from loguru import logger
│       └── cache_tools.py         # from loguru import logger
├── tools/
│   └── registry.py                # from loguru import logger
└── utils/
    ├── logging_helper.py          # from loguru import logger
    └── logging_setup.py           # from loguru import logger
```

### 1.2 日志数量统计

| 日志级别 | 调用次数 | 主要分布 | 占比 |
|---------|---------|---------|------|
| `logger.debug` | **301 次** | features.py (64), llm_decision.py (52), llm_client.py (48) | 37% |
| `logger.info` | **344 次** | engine.py (89), data_hub.py (68), finnhub_client.py (44) | 43% |
| `logger.warning` | **160 次** | data_hub.py (36), dual_agent_llm.py (21), finnhub_client.py (22) | 20% |
| `logger.error` | ~50 次 | 分散在各模块 | <1% |

**总计**: ~855 条日志语句

### 1.3 日志格式现状

#### 问题 1: 标签不统一

```python
# ✅ 规范示例 (少数)
logger.info(f"[DUAL_AGENT] Starting dual-agent decision process")
logger.info(f"[CASH_FLOW] Initial params: symbol={symbol}, qty={qty}")

# ⚠️ 不规范示例 (大多数)
logger.info(f"🚀 [DUAL_AGENT] Starting...")  # 混用 emoji
logger.info(f"📊 [DUAL_AGENT] Step 1: Calling...")  # emoji 不一致
logger.info("=== Cash Update Operation ===")  # 无标签
logger.debug(f"[DEBUG] LLM Strategy: ...")  # 冗余 DEBUG 标签
```

#### 问题 2: 缺少结构化信息

```python
# ❌ 当前：纯字符串拼接
logger.info(f"Processing {symbol} with {len(features)} features")

# ✅ 理想：结构化日志
logger.info("Processing symbol", extra={
    "symbol": symbol,
    "feature_count": len(features),
    "run_id": run_id,
    "agent": "decision_agent"
})
```

#### 问题 3: 过度详细的日志

```python
# engine.py 中的现金流计算 (每笔交易 10+ 条日志)
logger.info("=== Cash flow calculation started [AAPL] ===")
logger.info(f"[CASH_FLOW] Initial params: symbol=AAPL, qty=100")
logger.info(f"[CASH_FLOW] Trade side: BUY (side=1)")
logger.debug(f"[CASH_FLOW] Price after slippage: 150.05")
logger.debug(f"[CASH_FLOW] Gross notional: 15005.00")
logger.debug(f"[CASH_FLOW] Commission: 15.01")
logger.debug(f"[CASH_FLOW] Net cost: 15020.01")
logger.info(f"[CASH_FLOW] Final: filled_qty=100, net_cost=15020.01")
logger.info("=== Cash flow calculation ended [AAPL] ===")
```

**影响**: 
- 单次回测产生 5000+ 条日志
- 日志文件过大（100MB+）
- 难以定位关键信息

---

## 2. 核心问题诊断

### 2.1 问题清单

| 问题类型 | 严重程度 | 影响范围 | 优先级 |
|---------|---------|---------|--------|
| **日志库混用** | 🟡 中等 | 全局 | P2 |
| **格式不统一** | 🔴 高 | 全局 | P1 |
| **过度日志** | 🔴 高 | engine.py, features.py | P1 |
| **缺少追踪 ID** | 🔴 高 | 全局 | P1 |
| **日志级别滥用** | 🟡 中等 | 全局 | P2 |
| **缺少结构化** | 🟡 中等 | 全局 | P2 |

### 2.2 问题 1: 日志库混用（已部分解决）

#### 现状
- 11 个文件使用 `logging`
- 12 个文件使用 `loguru`
- 已通过 `InterceptHandler` 桥接

#### 问题
```python
# 文件 A
import logging
logger = logging.getLogger(__name__)

# 文件 B
from loguru import logger

# 结果：虽然能工作，但代码风格不统一
```

#### 影响
- **维护成本**: 新开发者不知道该用哪个
- **功能缺失**: `logging` 无法享受 `loguru` 的高级特性（如 bind、contextualize）
- **性能**: 多一层桥接转换

---

### 2.3 问题 2: 格式不统一（严重）

#### 现状分析

**标签命名混乱**:

```python
# 发现的标签样式（30+ 种）
"[DUAL_AGENT]"              # Agent 层
"[CASH_FLOW]"               # 计算层
"[POSITION_VALIDATION]"     # 验证层
"[DEBUG]"                   # 级别标签（冗余）
"[SHARES_CALCULATION]"      # 细节标签
"[NEXT_DAY_PRICE]"         # 功能标签
"[FILTER_STATS]"           # 统计标签
"[PENDING_SAVE]"           # 状态标签
"=== XXX ==="               # 分隔符风格
"📊", "🚀", "✅", "⚠️"       # Emoji 风格
```

**缺少命名规范**:
- 无统一前缀约定
- 标签长度不一（5-25 字符）
- 大小写混用（DUAL_AGENT vs Cash_Flow）

#### 影响
- **日志分析困难**: 无法通过标签快速过滤
- **追踪链路断裂**: 同一流程使用不同标签
- **自动化解析失败**: 日志分析工具无法识别

---

### 2.4 问题 3: 过度日志（严重）

#### 重灾区模块

**1. engine.py (89 条 logger.info)**

```python
# 现金流计算每笔交易 10+ 条日志
def _fill_at_open(...):
    logger.info("=== Cash flow calculation started ===")
    logger.info(f"[CASH_FLOW] Initial params: ...")
    logger.debug(f"[CASH_FLOW] Trade side: ...")
    logger.debug(f"[CASH_FLOW] Price after slippage: ...")
    logger.debug(f"[CASH_FLOW] Gross notional: ...")
    logger.debug(f"[CASH_FLOW] Commission: ...")
    logger.debug(f"[CASH_FLOW] Net cost: ...")
    logger.info(f"[CASH_FLOW] Final: ...")
    logger.info("=== Cash flow calculation ended ===")

# 结果：100 只股票 × 60 天 × 10 条 = 60,000 条日志
```

**2. features.py (64 条 logger.debug)**

```python
# 特征构建每只股票 5+ 条调试日志
logger.debug(f"Building features for {symbol}")
logger.debug(f"Historical data: {len(bars)} bars")
logger.debug(f"News count: {len(news)}")
logger.debug(f"Fundamentals: {fundamentals}")
logger.debug(f"Features built: {features}")
```

**3. llm_decision.py (52 条 logger.debug)**

```python
# 决策流程详细日志
logger.debug(f"[DEBUG] LLM Strategy: current_date={date}")
logger.debug(f"[DEBUG] Agent mode: {mode}")
logger.debug(f"[DEBUG] Feature count: {count}")
logger.debug(f"[DEBUG] Decision count: {count}")
# ... 每次决策 20+ 条
```

#### 性能影响

| 场景 | 日志数量 | 文件大小 | 写入耗时 | 影响 |
|------|---------|---------|---------|------|
| 单天回测 (20 股票) | ~500 条 | ~200 KB | <1ms | ✅ 可接受 |
| 1 月回测 (20 股票) | ~10,000 条 | ~4 MB | ~10ms | 🟡 轻微 |
| 3 月回测 (20 股票) | ~30,000 条 | ~12 MB | ~30ms | 🟡 轻微 |
| 1 年回测 (20 股票) | ~120,000 条 | ~50 MB | ~120ms | 🔴 明显 |

---

### 2.5 问题 4: 缺少追踪 ID（严重）

#### 现状

```python
# 当前：无法追踪单次回测的完整流程
logger.info("[DUAL_AGENT] Starting...")
logger.info("[FUNDAMENTAL_FILTER] Filtering...")
logger.info("[DECISION_AGENT] Deciding...")

# 问题：如果并发运行多个回测，日志混在一起
```

#### 需要的效果

```python
# 理想：每个日志都带 run_id
logger.info("[DUAL_AGENT] Starting...", extra={"run_id": "backtest_20250115_001"})
logger.info("[FUNDAMENTAL_FILTER] Filtering...", extra={"run_id": "backtest_20250115_001"})

# 或使用 loguru 的 contextualize
with logger.contextualize(run_id="backtest_20250115_001"):
    logger.info("[DUAL_AGENT] Starting...")
```

#### 影响
- **并发场景混乱**: 多个回测同时运行时无法区分
- **链路追踪失败**: 无法从头到尾追踪单次回测
- **问题定位困难**: 出错时找不到完整上下文

---

### 2.6 问题 5: 日志级别滥用

#### 现状

| 级别 | 当前用途 | 推荐用途 | 是否合理 |
|------|---------|---------|---------|
| DEBUG | 301 次 | 开发调试、详细计算过程 | ⚠️ 过多 |
| INFO | 344 次 | 关键步骤、业务事件 | ⚠️ 部分合理 |
| WARNING | 160 次 | 预期内的异常、降级 | ✅ 合理 |
| ERROR | ~50 次 | 严重错误、异常 | ✅ 合理 |

#### 问题案例

```python
# ❌ 不合理：INFO 用于详细计算
logger.info(f"[CASH_FLOW] Gross notional: {value}")  # 应为 DEBUG
logger.info(f"[CASH_FLOW] Commission: {comm}")       # 应为 DEBUG

# ❌ 不合理：DEBUG 用于关键步骤
logger.debug(f"Starting backtest for {date}")        # 应为 INFO
logger.debug(f"Portfolio initialized: {value}")      # 应为 INFO

# ✅ 合理
logger.info(f"Backtest started: {start} to {end}")
logger.debug(f"Detailed calculation: {steps}")
logger.warning(f"Degraded to fallback: {reason}")
logger.error(f"Critical error: {error}")
```

---

## 3. 优化目标

### 3.1 核心目标

| 目标 | 描述 | 成功标准 |
|------|------|---------|
| **统一性** | 统一日志库、格式、命名 | 100% 使用 loguru，标签规范化 |
| **可追踪** | 完整的链路追踪 | 每条日志都有 run_id/trace_id |
| **高效性** | 减少冗余日志 | 日志数量减少 60%+ |
| **结构化** | 支持自动化分析 | 所有日志可 JSON 解析 |
| **可观测** | 清晰的调试信息 | 关键步骤 100% 可追踪 |

### 3.2 量化指标

| 指标 | 当前值 | 目标值 | 改进幅度 |
|------|-------|-------|---------|
| **日志总量** | ~855 条语句 | ~350 条 | -60% |
| **INFO 日志** | 344 条 | ~150 条 | -56% |
| **DEBUG 日志** | 301 条 | ~100 条 | -67% |
| **日志文件大小** (3月回测) | ~12 MB | ~4 MB | -67% |
| **标签种类** | 30+ 种 | ~10 种 | -67% |
| **缺少 run_id 的日志** | 100% | 0% | -100% |

---

## 4. 详细优化方案

### 4.1 Phase 1: 统一日志库（优先级：P2）

#### 目标
全面迁移到 `loguru`，移除 `logging` 的直接使用。

#### 实施步骤

**Step 1: 创建统一的 Logger 工厂**

```python
# stockbench/utils/logger.py (新建)

from loguru import logger
from typing import Optional
import os

def get_logger(module_name: Optional[str] = None):
    """
    获取统一的 Logger 实例
    
    Args:
        module_name: 模块名（用于日志过滤），可选
    
    Returns:
        logger: 配置好的 logger 实例
    """
    if module_name:
        return logger.bind(module=module_name)
    return logger

# 便捷函数
def get_module_logger(file_path: str):
    """
    根据文件路径自动生成模块名
    
    Usage:
        logger = get_module_logger(__file__)
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    return get_logger(module_name)
```

**Step 2: 批量替换**

```python
# ❌ 旧代码
import logging
logger = logging.getLogger(__name__)

# ✅ 新代码
from stockbench.utils.logger import get_logger
logger = get_logger(__name__)

# 或更简洁
from loguru import logger
```

**Step 3: 迁移清单**

| 文件 | 当前 | 改为 | 优先级 |
|------|------|------|--------|
| `adapters/polygon_client.py` | logging | loguru | P2 |
| `adapters/finnhub_client.py` | logging | loguru | P2 |
| `agents/dual_agent_llm.py` | logging | loguru | P2 |
| `agents/fundamental_filter_agent.py` | logging | loguru | P2 |
| `agents/backtest_report_llm.py` | logging | loguru | P3 |
| `backtest/engine.py` | logging | loguru | P1 |
| `backtest/strategies/llm_decision.py` | logging | loguru | P1 |
| `core/features.py` | logging | loguru | P2 |
| `core/executor.py` | logging | loguru | P2 |
| `core/price_utils.py` | logging | loguru | P3 |

---

### 4.2 Phase 2: 标准化日志格式（优先级：P1）

#### 目标
建立统一的日志命名规范和格式标准。

#### 标签命名规范

**1. 标签分类体系**

| 层级 | 标签前缀 | 示例 | 用途 |
|------|---------|------|------|
| **系统层** | `SYS_` | `SYS_INIT`, `SYS_CONFIG` | 系统初始化、配置加载 |
| **数据层** | `DATA_` | `DATA_FETCH`, `DATA_CACHE` | 数据获取、缓存操作 |
| **Agent 层** | `AGENT_` | `AGENT_FILTER`, `AGENT_DECISION` | Agent 执行 |
| **回测层** | `BT_` | `BT_ENGINE`, `BT_ORDER` | 回测引擎、订单执行 |
| **LLM 层** | `LLM_` | `LLM_CALL`, `LLM_PARSE` | LLM 调用、解析 |
| **Memory 层** | `MEM_` | `MEM_SAVE`, `MEM_LOAD` | Memory 读写 |
| **工具层** | `TOOL_` | `TOOL_EXEC`, `TOOL_FAIL` | Tool 执行 |

**2. 标准化模板**

```python
# ✅ 推荐格式
logger.info(
    f"[{TAG}] {动作} {对象}",
    extra={
        "run_id": run_id,
        "symbol": symbol,
        "action": action,
        ...
    }
)

# 示例
logger.info(
    "[AGENT_DECISION] Making decision for AAPL",
    extra={
        "run_id": "backtest_20250115_001",
        "symbol": "AAPL",
        "agent": "decision_agent",
        "feature_count": 10
    }
)
```

**3. 禁止使用的格式**

```python
# ❌ 禁止：无标签
logger.info("Processing data")

# ❌ 禁止：冗余分隔符
logger.info("=== Starting Process ===")

# ❌ 禁止：混用 emoji（除非统一规范）
logger.info("🚀 [AGENT] Starting...")

# ❌ 禁止：DEBUG 标签冗余
logger.debug("[DEBUG] Some info")  # debug 级别已明确
```

#### 标签映射表

**当前标签 → 标准标签**

| 当前标签 | 标准标签 | 说明 |
|---------|---------|------|
| `[DUAL_AGENT]` | `[AGENT_DECISION]` | 决策 Agent |
| `[FUNDAMENTAL_FILTER]` | `[AGENT_FILTER]` | 过滤 Agent |
| `[CASH_FLOW]` | `[BT_CASH]` | 现金流计算 |
| `[POSITION_VALIDATION]` | `[BT_VALIDATE]` | 持仓验证 |
| `[SHARES_CALCULATION]` | `[BT_SHARES]` | 份额计算 |
| `[NEXT_DAY_PRICE]` | `[BT_PRICE]` | 价格获取 |
| `[UNIFIED_EXECUTOR]` | `[AGENT_EXECUTOR]` | 执行器 |
| `[PENDING_SAVE]` | `[MEM_SAVE]` | Memory 保存 |
| `[LLM_CLIENT]` | `[LLM_CALL]` | LLM 调用 |
| `[MEMORY]` | `[MEM_OP]` | Memory 操作 |

---

### 4.3 Phase 3: 添加追踪 ID（优先级：P1）

#### 目标
为每条日志添加 `run_id` 和 `trace_id`，支持完整链路追踪。

#### 方案 1: 使用 loguru 的 contextualize

```python
# 在回测入口设置全局上下文
from loguru import logger

def run_backtest(config, start_date, end_date, run_id):
    # 设置全局上下文
    with logger.contextualize(
        run_id=run_id,
        start_date=start_date,
        end_date=end_date
    ):
        logger.info("[SYS_INIT] Backtest started")
        # 所有后续日志自动带 run_id
        strategy = Strategy(config)
        results = engine.run(strategy, ...)
        logger.info("[SYS_COMPLETE] Backtest completed")
```

#### 方案 2: 使用 PipelineContext 传递

```python
# 在 PipelineContext 中自动绑定
class PipelineContext:
    def __init__(self, run_id: str, ...):
        self.run_id = run_id
        self.logger = logger.bind(run_id=run_id)
    
    def log_info(self, tag: str, message: str, **kwargs):
        """统一日志接口"""
        self.logger.info(f"[{tag}] {message}", **kwargs)

# 在 Agent 中使用
@traced_agent("decision_agent")
def decide_agent(features, ctx: PipelineContext):
    ctx.log_info("AGENT_DECISION", "Starting decision", symbol_count=len(features))
    # 自动带 run_id
```

#### 方案 3: 混合方案（推荐）

```python
# 1. 回测入口设置全局 contextualize
with logger.contextualize(run_id=run_id, backtest_type="daily"):
    
    # 2. Agent 内部使用 PipelineContext 的 logger
    ctx = PipelineContext(run_id=run_id, ...)
    ctx.logger.info("[AGENT_START] Agent started", agent="decision_agent")
    
    # 3. 关键步骤添加额外上下文
    with logger.contextualize(symbol="AAPL", date="2025-01-15"):
        ctx.logger.info("[AGENT_DECISION] Making decision")
```

#### 实施步骤

**Step 1: 修改 PipelineContext**

```python
# stockbench/core/pipeline_context.py

from loguru import logger

class PipelineContext:
    def __init__(self, run_id: str, date: str, ...):
        self.run_id = run_id
        self.date = date
        # 创建绑定了上下文的 logger
        self.logger = logger.bind(
            run_id=run_id,
            date=date,
            component="pipeline"
        )
    
    def get_agent_logger(self, agent_name: str):
        """为特定 Agent 创建 logger"""
        return self.logger.bind(agent=agent_name)
```

**Step 2: 修改 traced_agent 装饰器**

```python
# stockbench/core/decorators.py

def traced_agent(agent_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            ctx = kwargs.get("ctx")
            if ctx and hasattr(ctx, "logger"):
                agent_logger = ctx.get_agent_logger(agent_name)
                agent_logger.info(f"[AGENT_START] {agent_name} started")
                # 执行 Agent
                result = func(*args, **kwargs)
                agent_logger.info(f"[AGENT_DONE] {agent_name} completed")
                return result
            else:
                # Fallback
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

**Step 3: 修改回测入口**

```python
# stockbench/apps/run_backtest.py

from loguru import logger

def main():
    run_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 设置全局上下文
    with logger.contextualize(run_id=run_id):
        logger.info("[SYS_INIT] Initializing backtest", 
                   start=args.start, end=args.end, symbols=args.symbols)
        
        # 运行回测
        results = run_backtest(...)
        
        logger.info("[SYS_COMPLETE] Backtest completed",
                   total_return=results["return"],
                   duration_sec=results["duration"])
```

---

### 4.4 Phase 4: 减少冗余日志（优先级：P1）

#### 目标
减少不必要的日志输出，提升性能。

#### 策略 1: 合并重复日志

**Before**:

```python
# engine.py - 现金流计算 (9 条日志)
logger.info("=== Cash flow calculation started ===")
logger.info(f"[CASH_FLOW] Initial params: symbol={symbol}, qty={qty}")
logger.info(f"[CASH_FLOW] Trade side: {'BUY' if side > 0 else 'SELL'}")
logger.debug(f"[CASH_FLOW] Price after slippage: {px}")
logger.debug(f"[CASH_FLOW] Gross notional: {gross}")
logger.debug(f"[CASH_FLOW] Commission: {commission}")
logger.debug(f"[CASH_FLOW] Net cost: {net_cost}")
logger.info(f"[CASH_FLOW] Final: filled_qty={filled_qty}, net_cost={net_cost}")
logger.info("=== Cash flow calculation ended ===")
```

**After**:

```python
# 合并为 1 条 INFO + 1 条 DEBUG (可选)
logger.info(
    "[BT_ORDER] Order filled",
    symbol=symbol,
    side="buy" if qty > 0 else "sell",
    filled_qty=filled_qty,
    price=open_price,
    net_cost=net_cost,
    commission=commission
)

# 仅在 DEBUG 级别输出详细计算
if logger.level("DEBUG").no >= logger._core.min_level:
    logger.debug(
        "[BT_ORDER_DETAIL] Order calculation details",
        symbol=symbol,
        slippage_price=px,
        gross=gross,
        commission=commission,
        net=net_cost
    )
```

**减少**: 9 条 → 1-2 条 (减少 78%-89%)

#### 策略 2: 按日志级别分层

| 级别 | 内容 | 条件 |
|------|------|------|
| **INFO** | 关键业务事件 | 始终输出 |
| **DEBUG** | 详细计算过程 | 仅开发调试时 |
| **TRACE** | 极详细的调试信息 | 仅追踪特定问题时 |

```python
# INFO: 关键步骤
logger.info("[BT_DAY] Processing trading day", date=date, symbol_count=len(symbols))

# DEBUG: 中等详细
logger.debug("[BT_ORDER] Placing order", symbol=symbol, side=side, qty=qty)

# TRACE: 极详细（新增级别）
logger.trace("[BT_CALC] Slippage calculation", 
             base_price=base, slippage=slip, final=final)
```

#### 策略 3: 使用采样日志

```python
# 对于高频日志，使用采样
class SamplingLogger:
    def __init__(self, logger, sample_rate=0.1):
        self.logger = logger
        self.sample_rate = sample_rate
        self.counter = 0
    
    def maybe_log(self, level, message, **kwargs):
        self.counter += 1
        if random.random() < self.sample_rate or self.counter % 100 == 0:
            getattr(self.logger, level)(message, **kwargs)

# 使用
sampler = SamplingLogger(logger, sample_rate=0.1)

for symbol in symbols:
    sampler.maybe_log("debug", "[BT_PROCESS] Processing symbol", symbol=symbol)
    # 只有 10% 的调用会实际输出日志
```

#### 策略 4: 聚合日志

```python
# ❌ Before: 每只股票 1 条日志 (20 条)
for symbol in symbols:
    logger.info(f"[AGENT_DECISION] Decision made for {symbol}: {action}")

# ✅ After: 聚合为 1 条日志
decisions_summary = {
    "increase": [s for s, d in decisions.items() if d["action"] == "increase"],
    "decrease": [s for s, d in decisions.items() if d["action"] == "decrease"],
    "hold": [s for s, d in decisions.items() if d["action"] == "hold"]
}

logger.info(
    "[AGENT_DECISION] Batch decisions completed",
    total=len(decisions),
    increase=len(decisions_summary["increase"]),
    decrease=len(decisions_summary["decrease"]),
    hold=len(decisions_summary["hold"]),
    symbols_increase=decisions_summary["increase"][:5]  # 仅显示前 5 个
)
```

#### 重点优化模块

| 模块 | 当前日志数 | 目标日志数 | 优化策略 |
|------|-----------|-----------|---------|
| `engine.py` | 89 条 INFO | ~30 条 | 合并 + 分层 |
| `features.py` | 64 条 DEBUG | ~20 条 | 采样 + 聚合 |
| `llm_decision.py` | 52 条 DEBUG | ~15 条 | 聚合 + 条件输出 |
| `data_hub.py` | 68 条 INFO | ~25 条 | 聚合 + 缓存命中率 |

---

### 4.5 Phase 5: 结构化日志（优先级：P2）

#### 目标
所有日志支持 JSON 格式，便于自动化分析。

#### 实现方式

**1. 使用 extra 参数**

```python
# ✅ 推荐：结构化字段
logger.info(
    "[AGENT_DECISION] Decision made",
    extra={
        "symbol": "AAPL",
        "action": "increase",
        "target_amount": 5000.0,
        "confidence": 0.85,
        "reasons": ["Strong earnings", "Positive sentiment"]
    }
)

# JSON 输出：
# {
#   "time": "2025-01-15T10:30:00Z",
#   "level": "INFO",
#   "message": "[AGENT_DECISION] Decision made",
#   "symbol": "AAPL",
#   "action": "increase",
#   "target_amount": 5000.0,
#   "confidence": 0.85,
#   "reasons": ["Strong earnings", "Positive sentiment"]
# }
```

**2. 定义标准 Schema**

```python
# stockbench/utils/log_schemas.py (新建)

from typing import TypedDict, List, Optional

class DecisionLogSchema(TypedDict):
    """决策日志 Schema"""
    symbol: str
    action: str  # increase/decrease/hold/close
    target_amount: float
    confidence: float
    reasons: List[str]
    run_id: str
    date: str

class OrderLogSchema(TypedDict):
    """订单日志 Schema"""
    symbol: str
    side: str  # buy/sell
    qty: float
    price: float
    net_cost: float
    commission: float
    run_id: str
    date: str

class AgentLogSchema(TypedDict):
    """Agent 日志 Schema"""
    agent_name: str
    status: str  # started/completed/failed
    duration_ms: float
    input_count: int
    output_count: int
    run_id: str

# 使用
from stockbench.utils.log_schemas import DecisionLogSchema

decision_log = DecisionLogSchema(
    symbol="AAPL",
    action="increase",
    target_amount=5000.0,
    confidence=0.85,
    reasons=["Reason 1", "Reason 2"],
    run_id=run_id,
    date=date
)

logger.info("[AGENT_DECISION] Decision made", **decision_log)
```

**3. 配置 JSON 输出**

```python
# stockbench/utils/logging_setup.py

def setup_json_logging(config: dict):
    # JSON 文件：结构化日志
    logger.add(
        "logs/{time:YYYY-MM-DD}_structured.json",
        serialize=True,  # JSON 格式
        format="{message}",
        level="INFO"
    )
    
    # 文本文件：人类可读
    logger.add(
        "logs/{time:YYYY-MM-DD}_readable.log",
        serialize=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="DEBUG"
    )
```

---

### 4.6 Phase 6: 日志分析工具（优先级：P3）

#### 目标
提供日志分析和可视化工具。

#### 工具 1: 日志查询 CLI

```python
# scripts/log_query.py (新建)

import json
import argparse
from datetime import datetime

def query_logs(log_file, filters):
    """查询日志"""
    results = []
    with open(log_file) as f:
        for line in f:
            try:
                log = json.loads(line)
                # 应用过滤器
                if all(log.get(k) == v for k, v in filters.items()):
                    results.append(log)
            except:
                continue
    return results

# 使用
# python scripts/log_query.py \
#     --log logs/2025-01-15_structured.json \
#     --filter run_id=backtest_20250115_001 \
#     --filter symbol=AAPL \
#     --tag AGENT_DECISION
```

#### 工具 2: 性能分析

```python
# scripts/log_performance.py (新建)

def analyze_performance(log_file):
    """分析性能瓶颈"""
    agent_stats = {}
    
    with open(log_file) as f:
        for line in f:
            log = json.loads(line)
            if "agent" in log and "duration_ms" in log:
                agent = log["agent"]
                duration = log["duration_ms"]
                
                if agent not in agent_stats:
                    agent_stats[agent] = []
                agent_stats[agent].append(duration)
    
    # 输出统计
    for agent, durations in agent_stats.items():
        print(f"{agent}:")
        print(f"  Count: {len(durations)}")
        print(f"  Avg: {sum(durations)/len(durations):.2f}ms")
        print(f"  Max: {max(durations):.2f}ms")
```

#### 工具 3: 链路追踪可视化

```python
# scripts/log_trace.py (新建)

def visualize_trace(log_file, run_id):
    """可视化追踪链路"""
    events = []
    
    with open(log_file) as f:
        for line in f:
            log = json.loads(line)
            if log.get("run_id") == run_id:
                events.append({
                    "time": log["time"],
                    "level": log["level"],
                    "message": log["message"],
                    "agent": log.get("agent", "system")
                })
    
    # 生成时间线图
    # (使用 matplotlib 或导出为 HTML)
```

---

## 5. 实施路线图

### 5.1 Phase 时间表

| Phase | 内容 | 工作量 | 优先级 | 预计完成 |
|-------|------|-------|--------|---------|
| **Phase 1** | 统一日志库 (logging → loguru) | 2 天 | P2 | Week 1 |
| **Phase 2** | 标准化日志格式和标签 | 3 天 | P1 | Week 1-2 |
| **Phase 3** | 添加追踪 ID (run_id/trace_id) | 2 天 | P1 | Week 2 |
| **Phase 4** | 减少冗余日志 | 3 天 | P1 | Week 2-3 |
| **Phase 5** | 结构化日志和 Schema | 2 天 | P2 | Week 3 |
| **Phase 6** | 日志分析工具 | 2 天 | P3 | Week 4 |

**总计**: 14 工作日 (~3 周)

### 5.2 详细实施步骤

#### Week 1: Phase 1-2

**Day 1-2: 统一日志库**
- [ ] 创建 `stockbench/utils/logger.py`
- [ ] 迁移 `backtest/engine.py` (最复杂)
- [ ] 迁移 `backtest/strategies/llm_decision.py`
- [ ] 迁移 `agents/dual_agent_llm.py`
- [ ] 迁移 `agents/fundamental_filter_agent.py`

**Day 3-5: 标准化格式**
- [ ] 定义标签命名规范文档
- [ ] 创建标签映射表
- [ ] 批量替换标签（使用脚本）
- [ ] 移除 emoji 和分隔符
- [ ] 代码审查

#### Week 2: Phase 3-4

**Day 6-7: 添加追踪 ID**
- [ ] 修改 `PipelineContext` 添加 logger 支持
- [ ] 修改 `@traced_agent` 装饰器
- [ ] 修改回测入口添加 contextualize
- [ ] 测试追踪链路完整性

**Day 8-10: 减少冗余日志**
- [ ] 优化 `engine.py` (89 → 30 条)
- [ ] 优化 `features.py` (64 → 20 条)
- [ ] 优化 `llm_decision.py` (52 → 15 条)
- [ ] 优化 `data_hub.py` (68 → 25 条)
- [ ] 性能测试

#### Week 3: Phase 5

**Day 11-12: 结构化日志**
- [ ] 创建 `stockbench/utils/log_schemas.py`
- [ ] 定义标准 Schema (Decision/Order/Agent)
- [ ] 更新关键日志点使用 Schema
- [ ] 配置双输出（JSON + 文本）

#### Week 4: Phase 6

**Day 13-14: 分析工具**
- [ ] 开发 `scripts/log_query.py`
- [ ] 开发 `scripts/log_performance.py`
- [ ] 开发 `scripts/log_trace.py`
- [ ] 编写使用文档

---

## 6. 最佳实践

### 6.1 日志编写规范

#### DO ✅

```python
# 1. 使用标准标签
logger.info("[AGENT_DECISION] Making decision", symbol="AAPL")

# 2. 添加结构化字段
logger.info(
    "[BT_ORDER] Order filled",
    symbol=symbol,
    side=side,
    qty=qty,
    price=price
)

# 3. 使用上下文 logger
ctx.logger.info("[AGENT_START] Agent started")

# 4. 合理使用日志级别
logger.info("Key milestone")      # 关键步骤
logger.debug("Detailed info")      # 详细信息
logger.warning("Degraded mode")    # 降级/警告
logger.error("Critical error")     # 严重错误

# 5. 聚合批量操作
logger.info(
    "[DATA_FETCH] Batch data fetched",
    success=20,
    failed=0,
    duration_sec=1.5
)
```

#### DON'T ❌

```python
# 1. 避免无标签日志
logger.info("Something happened")

# 2. 避免冗余分隔符
logger.info("=== Starting Process ===")

# 3. 避免字符串拼接
logger.info(f"Symbol {symbol} price {price}")  # 应使用 extra

# 4. 避免过度日志
for i in range(1000):
    logger.debug(f"Processing item {i}")  # 应使用采样

# 5. 避免敏感信息
logger.info(f"API Key: {api_key}")  # 危险！
```

### 6.2 日志级别指南

| 级别 | 使用场景 | 示例 | 是否默认输出 |
|------|---------|------|------------|
| **INFO** | 关键业务事件、重要里程碑 | 回测开始/结束、决策完成 | ✅ 是 |
| **DEBUG** | 详细调试信息、中间结果 | 特征构建、价格计算 | ❌ 否 (仅开发) |
| **WARNING** | 预期内的异常、降级处理 | 缓存未命中、API 降级 | ✅ 是 |
| **ERROR** | 严重错误、需要关注 | 数据加载失败、Agent 崩溃 | ✅ 是 |
| **CRITICAL** | 致命错误、系统崩溃 | 配置错误、内存耗尽 | ✅ 是 |

### 6.3 性能优化建议

```python
# 1. 延迟字符串格式化
# ❌ Bad: 总是格式化
logger.debug(f"Expensive calculation: {expensive_func()}")

# ✅ Good: 仅在需要时格式化
if logger.level("DEBUG").no >= logger._core.min_level:
    logger.debug(f"Expensive calculation: {expensive_func()}")

# 2. 使用异步日志
logger.add("file.log", enqueue=True)  # 异步写入

# 3. 控制日志文件大小
logger.add(
    "file.log",
    rotation="100 MB",  # 100MB 轮转
    retention="10 days",  # 保留 10 天
    compression="zip"  # 压缩
)
```

---

## 7. 验收标准

### 7.1 Phase 完成标准

| Phase | 验收标准 | 检查方法 |
|-------|---------|---------|
| **Phase 1** | 100% 迁移到 loguru | `grep -r "import logging" stockbench/` 无结果 |
| **Phase 2** | 标签规范化 | 手动审查关键模块 |
| **Phase 3** | 100% 日志带 run_id | 随机抽查 100 条日志 |
| **Phase 4** | 日志减少 60%+ | 对比前后日志文件大小 |
| **Phase 5** | 关键日志结构化 | JSON 解析成功率 100% |
| **Phase 6** | 工具可用 | 运行查询和分析脚本 |

### 7.2 回归测试

```bash
# 1. 功能测试：确保日志不影响功能
python -m pytest tests/ -v

# 2. 性能测试：对比优化前后性能
time python -m stockbench.apps.run_backtest \
    --start 2025-01-01 --end 2025-03-31 \
    --symbols AAPL,GOOGL

# 3. 日志完整性测试
python scripts/validate_logs.py \
    --log logs/2025-01-15_structured.json \
    --check-run-id \
    --check-schema
```

---

## 8. 风险与缓解

### 8.1 风险清单

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| **日志丢失** | 高 | 低 | 分阶段实施，每阶段测试 |
| **性能下降** | 中 | 低 | 异步日志，性能测试 |
| **兼容性问题** | 中 | 中 | 保留 InterceptHandler 桥接 |
| **开发混乱** | 低 | 中 | 提供迁移指南和示例 |

### 8.2 回滚计划

如果出现严重问题：
1. 保留旧代码分支 `backup/logging_before_optimization`
2. 每个 Phase 独立提交，可单独回滚
3. 关键模块优先迁移，验证无误后再继续

---

## 9. 总结

### 9.1 预期收益

| 指标 | 改进 |
|------|------|
| **日志数量** | ↓ 60% |
| **日志文件大小** | ↓ 67% |
| **写入性能** | ↑ 50%+ |
| **可追踪性** | 100% 链路可追踪 |
| **分析效率** | ↑ 10x (结构化 + 工具) |

### 9.2 长期价值

- **可维护性**: 统一的日志规范，降低维护成本
- **可观测性**: 完整的链路追踪，快速定位问题
- **可扩展性**: 结构化日志，支持自动化分析
- **性能优化**: 减少 IO，提升回测速度

---

## 附录

### A. 标签速查表

| 标签 | 用途 | 示例 |
|------|------|------|
| `[SYS_INIT]` | 系统初始化 | 配置加载、环境检查 |
| `[DATA_FETCH]` | 数据获取 | API 调用、缓存读取 |
| `[AGENT_FILTER]` | Agent 过滤 | 基本面过滤 |
| `[AGENT_DECISION]` | Agent 决策 | 交易决策 |
| `[BT_ENGINE]` | 回测引擎 | 引擎启动、日期迭代 |
| `[BT_ORDER]` | 订单执行 | 下单、成交 |
| `[BT_CASH]` | 现金管理 | 现金流计算 |
| `[LLM_CALL]` | LLM 调用 | API 请求、响应 |
| `[MEM_SAVE]` | Memory 保存 | 历史记录保存 |

### B. 迁移检查清单

- [ ] Phase 1: 统一日志库
  - [ ] 创建 `utils/logger.py`
  - [ ] 迁移所有 `logging` 到 `loguru`
  - [ ] 删除冗余 `logging.getLogger()`
  
- [ ] Phase 2: 标准化格式
  - [ ] 定义标签规范
  - [ ] 批量替换标签
  - [ ] 移除 emoji 和分隔符
  
- [ ] Phase 3: 添加追踪 ID
  - [ ] 修改 `PipelineContext`
  - [ ] 修改 `@traced_agent`
  - [ ] 回测入口添加 contextualize
  
- [ ] Phase 4: 减少冗余
  - [ ] 优化 4 个重点模块
  - [ ] 合并重复日志
  - [ ] 聚合批量日志
  
- [ ] Phase 5: 结构化
  - [ ] 定义 Schema
  - [ ] 更新关键日志点
  - [ ] 配置 JSON 输出
  
- [ ] Phase 6: 分析工具
  - [ ] 日志查询工具
  - [ ] 性能分析工具
  - [ ] 链路追踪可视化

### C. 参考资源

- [Loguru 官方文档](https://loguru.readthedocs.io/)
- [Python Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [Structured Logging Guide](https://www.structlog.org/)

---

*文档生成时间: 2025-12-15*  
*StockBench Team*
