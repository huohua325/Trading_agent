# Dual Agent LLM API 文档

本文档详细说明 `dual_agent_llm.py` 中所有函数的输入输出结构。

---

## 目录

1. [辅助函数](#辅助函数)
   - [`_prompt_dir()`](#_prompt_dir)
   - [`_load_prompt()`](#_load_prompt)
   - [`_prompt_version()`](#_prompt_version)
   - [`_filter_hallucination_decisions()`](#_filter_hallucination_decisions)
   - [`_validate_decision_logic()`](#_validate_decision_logic)
   - [`_extract_decision_tags()`](#_extract_decision_tags)

2. [核心决策函数](#核心决策函数)
   - [`decide_batch_dual_agent()`](#decide_batch_dual_agent)
   - [`_decide_batch_portfolio_dual_agent()`](#_decide_batch_portfolio_dual_agent)

---

## 辅助函数

### `_prompt_dir()`

获取 prompt 文件所在目录路径。

**输入**：无

**输出**：
- **类型**: `str`
- **说明**: prompt 文件目录的绝对路径
- **示例**: `"/path/to/stockbench/agents/prompts"`

---

### `_load_prompt()`

从文件加载 prompt 模板内容。

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | prompt 文件名（如 `"decision_agent_v1.txt"`） |

**输出**：
- **类型**: `str`
- **说明**: prompt 文件的文本内容
- **失败时**: 返回默认 prompt 字符串

**示例**：
```python
prompt = _load_prompt("decision_agent_v1.txt")
# 返回: "You are a decision agent..."
```

---

### `_prompt_version()`

从 prompt 文件名提取版本号。

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | prompt 文件名 |

**输出**：
- **类型**: `str`
- **说明**: 版本号字符串（将下划线替换为斜杠）

**示例**：
```python
version = _prompt_version("decision_agent_v1.txt")
# 返回: "decision/agent/v1"
```

---

### `_filter_hallucination_decisions()`

过滤 LLM 幻觉决策，只保留实际输入的股票代码。

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `decisions_data` | `dict` | LLM 返回的决策数据字典 |
| `valid_symbols` | `set` | 有效的股票代码集合 |

**输入结构示例**：
```python
decisions_data = {
    "AAPL": {"action": "increase", "target_cash_amount": 5000},
    "TSLA": {"action": "hold", "target_cash_amount": 3000},
    "FAKE": {"action": "increase", "target_cash_amount": 2000}  # 幻觉
}
valid_symbols = {"AAPL", "TSLA"}
```

**输出**：
- **类型**: `dict`
- **说明**: 过滤后的决策字典（移除幻觉代码）

**输出结构示例**：
```python
{
    "AAPL": {"action": "increase", "target_cash_amount": 5000},
    "TSLA": {"action": "hold", "target_cash_amount": 3000}
}
# "FAKE" 被过滤掉
```

---

### `_validate_decision_logic()`

验证决策逻辑是否合理。

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `action` | `str` | 决策动作（"increase", "decrease", "hold", "close"） |
| `target_cash_amount` | `float` | 目标现金金额 |
| `current_position_value` | `float` | 当前持仓价值 |

**输出**：
- **类型**: `bool`
- **说明**: `True` 表示逻辑合理，`False` 表示逻辑不合理

**验证规则**：
| 动作 | 验证条件 |
|------|----------|
| `increase` | `target_cash_amount > current_position_value` |
| `decrease` | `target_cash_amount < current_position_value` |
| `close` | `target_cash_amount ≈ 0` (允许 0.01 误差) |
| `hold` | `target_cash_amount ≈ current_position_value` (允许 1% 或 100 单位误差) |

**示例**：
```python
# 合理的增仓
_validate_decision_logic("increase", 5000, 3000)  # True

# 不合理的增仓（目标金额小于当前持仓）
_validate_decision_logic("increase", 2000, 3000)  # False
```

---

### `_extract_decision_tags()`

从决策和特征中提取标签，用于 EpisodicMemory 索引。基于实际的 features 结构提取可靠的标签。

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `decision` | `Dict` | 决策字典（包含 action, confidence, reasons） |
| `features` | `Dict` (可选) | 特征字典（包含 market_data, fundamental_data, news_events, position_state） |

**输入结构示例**：
```python
decision = {
    "action": "increase",
    "confidence": 0.85,
    "reasons": [
        "Strong momentum with positive 7-day trend",
        "Reasonable valuation with pe_ratio below sector average"
    ]
}
features = {
    "market_data": {
        "close_7d": [180.0, 182.0, 181.5, 183.0, 185.0, 186.5, 188.0]
    },
    "fundamental_data": {
        "pe_ratio": 22.5,
        "market_cap": 2500000000000,  # 2.5万亿美元
        "dividend_yield": 1.5
    },
    "news_events": {
        "top_k_events": ["Strong earnings beat expectations"]
    },
    "position_state": {
        "current_position_value": 5000,
        "holding_days": 45
    }
}
```

**输出**：
- **类型**: `List[str]`
- **说明**: 标签列表（已去重）

**输出示例**：
```python
[
    "increase",              # 来自 action
    "high_confidence",       # 来自 confidence >= 0.8
    "momentum",              # 从 reasons 提取
    "trend",                 # 从 reasons 提取
    "valuation",             # 从 reasons 提取
    "pe_ratio",              # 从 reasons 提取
    "uptrend",               # 从 market_data.close_7d 计算趋势
    "has_fundamental",       # fundamental_data 存在
    "large_cap",             # market_cap > 1000亿美元
    "has_news",              # 有新闻数据
    "positive_news",         # 新闻情感分析
    "has_position",          # 有持仓
    "medium_hold"            # 持仓 30-90 天
]
```

**标签提取规则**：

1. **动作标签**：直接使用 `action` 值
   - `"increase"`, `"decrease"`, `"hold"`, `"close"`

2. **置信度标签**：
   - `confidence >= 0.8` → `"high_confidence"`
   - `confidence <= 0.3` → `"low_confidence"`

3. **关键词标签**（从 `reasons` 提取）：
   - 匹配英文关键词：`"breakout"`, `"support"`, `"resistance"`, `"trend"`, `"momentum"`, `"overbought"`, `"oversold"`, `"risk"`, `"stop_loss"`, `"volatility"`, `"volume"`, `"news"`, `"earnings"`, `"dividend"`, `"valuation"`, `"fundamental"`, `"technical"`, `"pe_ratio"`, `"market_cap"`

4. **市场数据标签**（从 `market_data` 提取）：
   - 分析 `close_7d` 计算趋势：
     - 最后一天 vs 倒数第二天 > +2% → `"uptrend"`
     - 最后一天 vs 倒数第二天 < -2% → `"downtrend"`

5. **基本面标签**（从 `fundamental_data` 提取）：
   - 存在基本面数据 → `"has_fundamental"`
   - 不存在基本面数据 → `"no_fundamental"`
   - PE 估值：
     - `pe_ratio > 30` → `"high_pe"`
     - `pe_ratio < 15` → `"low_pe"`
   - 股息：
     - `dividend_yield > 2.0%` → `"dividend_stock"`
   - 市值：
     - `market_cap > 1000亿美元` → `"large_cap"`
     - `market_cap < 100亿美元` → `"small_cap"`

6. **新闻标签**（从 `news_events` 提取）：
   - 有新闻数据 → `"has_news"`
   - 新闻情感分析：
     - 包含 "positive", "beat", "strong", "growth", "upgrade" → `"positive_news"`
     - 包含 "negative", "miss", "weak", "loss", "downgrade" → `"negative_news"`

7. **持仓标签**（从 `position_state` 提取）：
   - `current_position_value > 0` → `"has_position"`
   - `current_position_value = 0` → `"no_position"`
   - 持仓时间：
     - `holding_days > 90` → `"long_hold"`
     - `30 < holding_days <= 90` → `"medium_hold"`
     - `0 < holding_days <= 30` → `"short_hold"`

**注意事项**：
- 只匹配英文关键词（因为模型输入输出都是英文）
- 基于实际的 features 结构（不假设存在不存在的字段如 `technical_indicators`）
- 所有标签都来自可验证的数据源，避免依赖不可靠的匹配

---

## 核心决策函数

### `decide_batch_dual_agent()`

双代理批量决策主函数（带装饰器 `@traced_agent("decision_agent")`）。

**功能**：实现双代理架构的三步流程：
1. **Fundamental Filter Agent**：判断哪些股票需要基本面分析
2. **Enhanced Feature Construction**：根据筛选结果构建增强特征
3. **Decision Agent**：使用增强特征做最终交易决策

**输入**：
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `features_list` | `List[Dict]` | ✅ | 特征列表 |
| `cfg` | `Dict` | ❌ | 配置字典（包含 llm 子配置） |
| `enable_llm` | `bool` | ❌ | 是否启用 LLM（默认 `True`） |
| `bars_data` | `Dict[str, Dict]` | ❌ | 原始历史数据字典 |
| `run_id` | `str` | ❌ | 回测运行 ID |
| `previous_decisions` | `Dict` | ❌ | 上一次决策结果 |
| `decision_history` | `Dict[str, List[Dict]]` | ❌ | 长期历史决策记录 |
| `ctx` | `PipelineContext` 或 `Dict` | ❌ | 上下文（包含 portfolio 信息） |
| `rejected_orders` | `List[Dict]` | ❌ | 被拒绝的订单列表（用于重试） |

**输入结构示例**：

```python
features_list = [
    {
        "symbol": "AAPL",
        "features": {
            "market_data": {
                "ticker": "AAPL",
                "open": 184.20,
                "close_7d": [180.0, 182.0, 181.5, 183.0, 185.0, 186.5, 188.0],  # 最近7天收盘价
                "date": "2025-06-15",
                "price": 185.50  # 可选字段，只在 include_price=True 时存在
            },
            "fundamental_data": {  # 可选字段，只在 exclude_fundamental=False 时存在
                "market_cap": 2850000000000.0,  # 市值（美元）
                "pe_ratio": 28.5,               # 市盈率
                "dividend_yield": 0.52,         # 股息率（百分比）
                "week_52_high": 199.62,         # 52周最高价
                "week_52_low": 164.08,          # 52周最低价
                "quarterly_dividend": 0.24      # 季度股息（美元/股）
            },
            "news_events": {
                "top_k_events": [
                    "Apple announces new product - Revolutionary AI chip unveiled",
                    "Strong Q2 earnings beat analyst expectations"
                ]
            },
            "position_state": {
                "current_position_value": 5000.0,
                "holding_days": 45,
                "shares": 27.0
            },
            "filter_reasoning": "Technical pattern shows volatility requiring fundamental validation"  # Dual-agent 特有字段
        }
    },
    {
        "symbol": "TSLA",
        "features": {
            "market_data": { ... },
            # 注意：TSLA 可能没有 fundamental_data 字段（如果 filter agent 认为不需要）
            "news_events": { ... },
            "position_state": { ... },
            "filter_reasoning": "Stable technical indicators, no fundamental analysis needed"
        }
    }
]

cfg = {
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 8000
    },
    "agents": {
        "dual_agent": {
            "decision_agent": {
                "prompt": "decision_agent_v1.txt",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 8000
            }
        },
        "retry": {
            "max_attempts": 3
        }
    },
    "portfolio": {
        "total_cash": 100000,
        "min_cash_ratio": 0.1
    },
    "cache": {
        "mode": "full"
    }
}

bars_data = {
    "AAPL": {
        "bars_day": <DataFrame>,      # 日线数据（pandas DataFrame）
        "snapshot": {                 # 快照数据
            "symbol": "AAPL",
            "price": 185.50,
            "ts_utc": "2025-06-15T14:30:00Z"
        },
        "details": {                  # 资产详情
            "ticker": "AAPL",
            "name": "Apple Inc."
        },
        "news_items": [               # 新闻列表
            {
                "title": "Apple announces new product",
                "description": "Revolutionary AI chip unveiled"
            }
        ],
        "position_state": {           # 持仓状态
            "current_position_value": 5000.0,
            "holding_days": 45,
            "shares": 27.0
        }
    }
}

rejected_orders = [
    {
        "symbol": "AAPL",
        "qty": 27,                              # 订单数量（股数）
        "reason": "insufficient_cash",          # 拒绝原因（小写下划线格式）
        "rejection_reason": "insufficient_cash", # 冗余字段（与 reason 相同）
        "retry_count": 1,                       # 重试次数
        "context": {                            # 详细上下文
            "required_cash_this_order": 5000,   # 本订单所需现金
            "available_cash": 10000,            # 可用现金
            "total_cash_required_all_orders": 15000,  # 所有订单总需求
            "cash_shortfall": 5000,             # 现金缺口
            "retry_attempt": 1,                 # 当前重试次数
            "all_orders_count": 5,              # 总订单数
            "portfolio_rebalance_needed": True, # 是否需要组合再平衡
            "suggestion": "Total portfolio cash requirement (15000.00) exceeds available cash (10000.00) by 5000.00. Please reduce all order sizes proportionally or select fewer positions to fit within budget."
        }
    }
]
```

**输出**：
- **类型**: `Dict[str, Dict]`
- **说明**: 决策结果字典，包含每个股票的决策和元数据

**输出结构示例**：
```python
{
    "AAPL": {
        "action": "increase",
        "target_cash_amount": 7500.0,
        "cash_change": 2500.0,
        "reasons": [
            "Strong momentum with positive 7-day trend from market_data.close_7d",
            "Reasonable valuation with pe_ratio below sector average",
            "Positive earnings catalyst from recent news events"
        ],
        "confidence": 0.75,
        "timestamp": "2025-06-15T14:30:00.123456"
    },
    "TSLA": {
        "action": "hold",
        "target_cash_amount": 3000.0,
        "cash_change": 0.0,
        "reasons": [
            "Stable price trend from market_data.close_7d with normal volatility",
            "No significant news catalysts requiring position adjustments"
        ],
        "confidence": 0.6,
        "timestamp": "2025-06-15T14:30:00.123456"
    },
    "__meta__": {
        "calls": 2,
        "cache_hits": 1,
        "parse_errors": 0,
        "latency_ms_sum": 1250,
        "tokens_prompt": 3500,
        "tokens_completion": 850,
        "prompt_version": "decision/agent/v1"
    }
}
```

**字段说明**：
- `action`: 决策动作（"increase", "decrease", "hold", "close"）
- `target_cash_amount`: 目标持仓金额（总金额，非增量）
- `cash_change`: 现金变化量（正数表示买入，负数表示卖出）
- `reasons`: 决策理由列表（英文，来自 LLM）
- `confidence`: 置信度（0-1 之间的浮点数）
- `timestamp`: 决策时间戳（ISO 格式）

**注意**：
- 正常决策**不包含** `analysis_excerpt`, `tech_score`, `sent_score`, `event_risk` 字段
- 这些字段只在 **fallback hold 决策**中存在（当 `enable_llm=False` 或发生错误时）
- `reasons` 字段是英文（因为模型输入输出都是英文）

**特殊行为**：
- 如果 `enable_llm=False`，所有股票返回 `hold` 决策
- 如果处理出错，返回 `hold` 决策作为 fallback
- 如果使用 `PipelineContext`，会自动从 `EpisodicMemory` 加载历史决策
- 决策结果会存入 `PipelineContext` 的数据总线

---

### `_decide_batch_portfolio_dual_agent()`

双代理批量组合决策的内部实现函数（包含完整的重试机制）。

**功能**：
- 构建符合 prompt 模板的输入格式
- 调用 LLM 生成决策
- 验证决策逻辑（动作合理性、资金约束、现金比例）
- 处理重试逻辑（包括引擎级重试和 LLM 级重试）
- 存储决策到 EpisodicMemory

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `features_list` | `List[Dict]` | 增强后的特征列表 |
| `llm_cfg` | `LLMConfig` | LLM 配置对象 |
| `system_prompt` | `str` | 系统提示词 |
| `client` | `LLMClient` | LLM 客户端实例 |
| `meta_agg` | `Dict` | 元数据聚合字典 |
| `cfg` | `Dict` | 全局配置字典 |
| `bars_data` | `Dict` | 原始数据字典 |
| `run_id` | `str` | 运行 ID |
| `previous_decisions` | `Dict` | 上一次决策 |
| `decision_history` | `Dict[str, List[Dict]]` | 历史决策记录 |
| `ctx` | `PipelineContext` 或 `Dict` | 上下文 |
| `rejected_orders` | `List[Dict]` | 被拒绝的订单 |
| `pipeline_ctx` | `PipelineContext` | Pipeline 上下文（用于 Memory） |

**LLM 输入格式**（发送给 LLM 的 JSON）：
```python
{
    "portfolio_info": {
        "total_assets": 100000.0,
        "available_cash": 50000.0,
        "position_value": 50000.0
    },
    "symbols": {
        "AAPL": {
            "features": {
                "market_data": {
                    "ticker": "AAPL",
                    "open": 184.20,
                    "close_7d": [180.0, 182.0, 181.5, 183.0, 185.0, 186.5, 188.0],
                    "date": "2025-06-15",
                    "price": 185.50
                },
                "fundamental_data": {  # 可选字段
                    "market_cap": 2850000000000.0,
                    "pe_ratio": 28.5,
                    "dividend_yield": 0.52,
                    "week_52_high": 199.62,
                    "week_52_low": 164.08,
                    "quarterly_dividend": 0.24
                },
                "news_events": {
                    "top_k_events": [
                        "Apple announces new product - Revolutionary AI chip unveiled",
                        "Strong Q2 earnings beat analyst expectations"
                    ]
                },
                "position_state": {
                    "current_position_value": 5000.0,
                    "holding_days": 45,
                    "shares": 27.0
                },
                "filter_reasoning": "Technical pattern shows volatility requiring fundamental validation"
            }
        },
        "TSLA": {
            "features": {
                "market_data": {...},
                "news_events": {...},
                "position_state": {...},
                "filter_reasoning": "Stable technical indicators, no fundamental analysis needed"
                # 注意：TSLA 没有 fundamental_data 字段
            }
        }
    },
    "history": {
        "AAPL": "Previous decisions:\n2025-06-10: increase to $5000 (confidence: 0.8)\n2025-06-11: hold at $5000 (confidence: 0.7)\n...",
        "TSLA": "Previous decisions:\n2025-06-10: hold at $3000 (confidence: 0.6)\n..."
    }
}
```

**LLM 输出格式**（期望的 JSON 结构）：
```python
{
    "decisions": {
        "AAPL": {
            "action": "increase",
            "target_cash_amount": "7500.0",
            "confidence": "0.75",
            "reasons": [
                "Strong momentum with positive 7-day trend from market_data.close_7d",
                "Reasonable valuation with pe_ratio below sector average",
                "Positive earnings catalyst from recent news events"
            ]
        },
        "TSLA": {
            "action": "hold",
            "target_cash_amount": "3000.0",
            "confidence": "0.6",
            "reasons": [
                "Stable price trend with normal volatility",
                "No significant catalysts requiring position adjustments"
            ]
        }
    }
}
```

**输出**：
- **类型**: `Dict[str, Dict]`
- **说明**: 与 `decide_batch_dual_agent()` 相同的输出格式

**重试机制**：

1. **统一重试限制**：
   - `engine_retry_count`（引擎级重试）+ `llm_retry_count`（LLM 级重试）≤ `max_attempts`
   - 默认 `max_attempts = 3`

2. **重试触发条件**：
   - LLM 返回无效数据格式
   - 决策逻辑验证失败（动作与金额不匹配）
   - 资金不足（预计现金使用超过可用现金）
   - 现金比例违规（低于最小现金比例要求）
   - 响应被截断（token 限制）

3. **重试提示增强**：
   - 逻辑错误：详细说明每个股票的逻辑问题
   - 资金不足：提示减少购买金额或选择更少股票
   - 现金比例违规：提示保持更高现金储备
   - 订单被拒：包含拒绝原因和建议

4. **Fallback 策略**：
   - 所有重试失败后，返回 `hold` 决策（保持当前持仓）

**验证流程**：
```
1. 解析 LLM 响应
   ↓
2. 过滤幻觉决策
   ↓
3. 验证决策逻辑（action vs target_cash_amount）
   ↓
4. 计算预计现金使用
   ↓
5. 检查资金约束（available_cash - predicted_usage >= 0）
   ↓
6. 检查现金比例（remaining_ratio >= min_cash_ratio）
   ↓
7. 全部通过 → 返回结果
   ↓
8. 任一失败 → 重试（如果未达到限制）
```

**Memory 存储**：
- 只存储非 `hold` 决策到 EpisodicMemory
- 每个决策存储为 `DecisionEpisode` 对象
- 包含标签（通过 `_extract_decision_tags()` 提取）

---

## 数据流架构

```
┌─────────────────────────────────────────────────────────────┐
│                  decide_batch_dual_agent()                  │
│                     (主入口函数)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Step 1: Fundamental Filter Agent       │
        │  (filter_stocks_needing_fundamental)    │
        │  输出: stocks_need_fundamental          │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Step 2: Enhanced Feature Construction  │
        │  - 需要基本面: 包含 fundamental_data    │
        │  - 不需要基本面: 排除 fundamental_data  │
        │  输出: enhanced_features_list           │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Step 3: Decision Agent                 │
        │  (_decide_batch_portfolio_dual_agent)   │
        │  输出: decisions                        │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  验证 & 重试循环                         │
        │  - 逻辑验证                              │
        │  - 资金约束验证                          │
        │  - 现金比例验证                          │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  存储到 EpisodicMemory (Phase 7)        │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    返回最终决策结果
```

---

## 配置参数说明

### LLM 配置 (`cfg.llm`)

```yaml
llm:
  provider: "openai"              # LLM 提供商
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"            # 默认模型
  temperature: 0.7                # 温度参数
  max_tokens: 8000                # 最大 token 数
  seed: 42                        # 随机种子
  timeout_sec: 60                 # 超时时间
  retry:
    max_retries: 3                # API 重试次数
    backoff_factor: 0.5           # 退避因子
  cache:
    enabled: true                 # 是否启用缓存
    ttl_hours: 24                 # 缓存过期时间
  budget:
    max_prompt_tokens: 200000     # Prompt token 预算
    max_completion_tokens: 200000 # Completion token 预算
```

### 代理配置 (`cfg.agents`)

```yaml
agents:
  dual_agent:
    decision_agent:
      prompt: "decision_agent_v1.txt"  # Prompt 文件名
      model: "gpt-4o-mini"             # 决策代理专用模型
      temperature: 0.7
      max_tokens: 8000
  retry:
    max_attempts: 3                    # 统一重试上限
```

### 组合配置 (`cfg.portfolio`)

```yaml
portfolio:
  total_cash: 100000                   # 总资金
  min_cash_ratio: 0.1                  # 最小现金比例（10%）
```

### 缓存模式 (`cfg.cache.mode`)

- `"full"`: 读写缓存（默认）
- `"llm_write_only"`: 只写缓存，不读缓存
- `"off"`: 完全禁用缓存

---

## 错误处理

### Fallback 决策结构

当发生错误时，返回 `hold` 决策：

```python
{
    "action": "hold",
    "target_cash_amount": <current_position_value>,
    "cash_change": 0.0,
    "reasons": ["错误原因描述"],
    "confidence": 0.5,
    "timestamp": "2025-06-15T14:30:00"
}
```

### 常见错误场景

| 错误类型 | 触发条件 | Fallback 行为 |
|---------|---------|--------------|
| LLM 未启用 | `enable_llm=False` | 所有股票返回 `hold` |
| 配置缺失 | 无 `cfg.llm` 配置 | 抛出 `ValueError` |
| LLM 响应无效 | 无法解析 JSON | 重试或返回 `hold` |
| 逻辑验证失败 | 动作与金额不匹配 | 重试或返回 `hold` |
| 资金不足 | 预计使用超过可用现金 | 重试或返回 `hold` |
| 重试次数耗尽 | 达到 `max_attempts` | 返回 `hold` |
| 异常错误 | 未预期的异常 | 返回 `hold` |

---

## Phase 7 集成：Memory 系统

### EpisodicMemory 加载

在 `decide_batch_dual_agent()` 中：

```python
if pipeline_ctx and pipeline_ctx.memory_enabled and not decision_history:
    decision_history = {}
    for item in features_list:
        symbol = item.get("symbol")
        # 从 EpisodicMemory 获取历史（最近 5 条）
        history_text = pipeline_ctx.memory.episodes.get_for_prompt(symbol, n=5)
        if history_text:
            decision_history[symbol] = history_text
```

### EpisodicMemory 存储

在 `_decide_batch_portfolio_dual_agent()` 中：

```python
if pipeline_ctx and pipeline_ctx.memory_enabled:
    for symbol, decision in results.items():
        if decision.get("action") != "hold":  # 只存储非 hold 决策
            episode = DecisionEpisode(
                symbol=symbol,
                action=decision["action"],
                target_amount=decision["target_cash_amount"],
                reasoning="; ".join(decision["reasons"]),
                confidence=decision["confidence"],
                market_context=features.get("market_data", {}),
                signals=features.get("technical_indicators", {}),
                tags=_extract_decision_tags(decision, features)
            )
            pipeline_ctx.memory.episodes.add(episode)
```

---

## 使用示例

### 基本用法

```python
from stockbench.agents.dual_agent_llm import decide_batch_dual_agent

# 准备输入数据
features_list = [...]  # 特征列表
cfg = {...}            # 配置字典
bars_data = {...}      # 原始数据

# 调用决策函数
results = decide_batch_dual_agent(
    features_list=features_list,
    cfg=cfg,
    enable_llm=True,
    bars_data=bars_data,
    run_id="backtest_20250615"
)

# 处理结果
for symbol, decision in results.items():
    if symbol == "__meta__":
        continue
    print(f"{symbol}: {decision['action']} -> ${decision['target_cash_amount']}")
```

### 使用 PipelineContext

```python
from stockbench.core.pipeline_context import PipelineContext

# 创建 PipelineContext
ctx = PipelineContext(
    config=cfg,
    run_id="backtest_20250615",
    memory_enabled=True
)

# 设置 portfolio 信息
ctx.put("portfolio", portfolio_obj)

# 调用决策函数
results = decide_batch_dual_agent(
    features_list=features_list,
    ctx=ctx
)

# 从数据总线获取决策
decisions = ctx.get("decisions")
```

### 处理订单拒绝重试

```python
# 第一次调用
results = decide_batch_dual_agent(features_list=features_list, cfg=cfg)

# 模拟订单被拒绝
rejected_orders = [
    {
        "symbol": "AAPL",
        "reason": "Insufficient cash",
        "retry_count": 1,
        "context": {
            "portfolio_rebalance_needed": True,
            "available_cash": 10000,
            "total_cash_required_all_orders": 15000,
            "cash_shortfall": 5000
        }
    }
]

# 重试调用
results = decide_batch_dual_agent(
    features_list=features_list,
    cfg=cfg,
    rejected_orders=rejected_orders
)
```

---

## 性能优化建议

1. **缓存策略**：
   - 生产环境使用 `cache.mode: "full"`
   - 测试环境使用 `cache.mode: "llm_write_only"`

2. **Token 优化**：
   - 调整 `max_tokens` 避免响应截断
   - 使用更小的模型（如 `gpt-4o-mini`）降低成本

3. **重试配置**：
   - 根据 LLM 稳定性调整 `max_attempts`
   - 设置合理的 `timeout_sec`

4. **Memory 管理**：
   - 控制历史记录数量（`n=5`）
   - 只存储重要决策（非 `hold`）

---

## 版本历史

- **Phase 7**: 集成 Memory 和 Message 系统
- **Phase 9**: 简化历史加载逻辑，优先使用 Memory 系统
- **当前版本**: 支持统一重试机制、订单拒绝处理、资金约束验证

---

## 相关文档

- [`fundamental_filter_agent.py`](./fundamental_filter_agent.py) - 基本面筛选代理
- [`core/features.py`](../core/features.py) - 特征构建
- [`core/pipeline_context.py`](../core/pipeline_context.py) - Pipeline 上下文
- [`memory/`](../memory/) - Memory 系统
