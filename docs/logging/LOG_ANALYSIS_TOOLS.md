# 日志分析工具使用指南

> **工具集**: 3 个强大的日志分析工具  
> **目标**: 快速查询、性能分析、执行追踪  

---

## 📚 工具概览

| 工具 | 用途 | 输出格式 |
|------|------|---------|
| `log_query.py` | 查询结构化日志 | text / json / csv |
| `log_performance.py` | 性能指标分析 | 统计报告 |
| `log_trace.py` | 执行链路追踪 | text / html |

---

## 🔍 工具 1: log_query.py - 日志查询

### **功能**
快速查询和过滤结构化 JSON 日志。

### **基本用法**

```bash
# 查找特定股票的决策
python scripts/log_query.py --symbol AAPL --tag AGENT_DECISION

# 查找失败的订单
python scripts/log_query.py --status rejected --tag BT_ORDER

# 查找高延迟的 LLM 调用
python scripts/log_query.py --tag LLM_CALL --min-latency 3000

# 查找低置信度决策
python scripts/log_query.py --tag AGENT_DECISION --max-confidence 0.6
```

### **所有参数**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--log-dir` | 日志目录 | `logs/stockbench` |
| `--date` | 日期 (YYYY-MM-DD) | `2025-12-15` |
| `--symbol` | 股票代码 | `AAPL` |
| `--tag` | 日志标签 | `AGENT_DECISION` |
| `--status` | 状态 | `success`, `failed`, `rejected` |
| `--agent-name` | Agent 名称 | `decision_agent` |
| `--action` | 决策动作 | `hold`, `increase`, `decrease` |
| `--min-confidence` | 最小置信度 | `0.8` |
| `--max-confidence` | 最大置信度 | `0.6` |
| `--min-latency` | 最小延迟 (ms) | `1000` |
| `--max-latency` | 最大延迟 (ms) | `5000` |
| `--cache-hit` | 缓存命中 | `true` / `false` |
| `--level` | 日志级别 | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--limit` | 结果数量限制 | `100` |
| `--output` | 输出文件 | `results.csv` |
| `--format` | 输出格式 | `text`, `json`, `csv` |

### **实用查询示例**

**1. 查找所有增仓决策**
```bash
python scripts/log_query.py --action increase --tag AGENT_DECISION
```

**2. 查找缓存未命中的数据获取**
```bash
python scripts/log_query.py --cache-hit false --tag DATA_FETCH
```

**3. 导出 AAPL 的所有决策到 CSV**
```bash
python scripts/log_query.py --symbol AAPL --output aapl_decisions.csv
```

**4. 查找错误日志**
```bash
python scripts/log_query.py --level ERROR
```

**5. 查找特定 Agent 的执行记录**
```bash
python scripts/log_query.py --agent-name fundamental_filter
```

### **输出格式**

**Text 格式** (默认):
```
Found 15 matching log entries:
================================================================================

[1] 2025-12-15T10:30:00Z | INFO | [AGENT_DECISION] Decision made
    Symbol: AAPL
    Action: increase
    Target: $15,000.00
    Confidence: 85.00%

[2] 2025-12-15T10:30:05Z | INFO | [AGENT_DECISION] Decision made
    Symbol: GOOGL
    Action: hold
    Target: $10,000.00
    Confidence: 60.00%
...
```

**JSON 格式**:
```json
[
  {
    "time": "2025-12-15T10:30:00Z",
    "level": "INFO",
    "message": "[AGENT_DECISION] Decision made",
    "symbol": "AAPL",
    "action": "increase",
    "target_cash_amount": 15000.0,
    "confidence": 0.85
  }
]
```

**CSV 格式**:
```
time,level,message,symbol,action,target_cash_amount,confidence
2025-12-15T10:30:00Z,INFO,[AGENT_DECISION] Decision made,AAPL,increase,15000.0,0.85
```

---

## 📊 工具 2: log_performance.py - 性能分析

### **功能**
分析日志中的性能指标，生成统计报告。

### **基本用法**

```bash
# 分析今天的日志
python scripts/log_performance.py

# 分析特定日期
python scripts/log_performance.py --date 2025-12-15

# 生成详细报告
python scripts/log_performance.py --detailed

# 保存到文件
python scripts/log_performance.py --output performance_report.txt
```

### **所有参数**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--log-dir` | 日志目录 | `logs/stockbench` |
| `--date` | 日期 (YYYY-MM-DD) | `2025-12-15` |
| `--detailed` | 详细报告 | flag |
| `--output` | 输出文件 | `report.txt` |
| `--focus` | 关注特定指标 | `agents`, `llm`, `data`, `decisions` |

### **报告示例**

```
================================================================================
📊 LOG PERFORMANCE ANALYSIS REPORT
================================================================================

🤖 AGENT PERFORMANCE
--------------------------------------------------------------------------------

[decision_agent]
  Executions: 50 (✅ 48 / ❌ 2)
  Success Rate: 96.0%
  Duration: avg=234.5ms, median=220.0ms
  Range: 180.0ms - 450.0ms
  Total Time: 11,725.0ms

[fundamental_filter]
  Executions: 50 (✅ 50 / ❌ 0)
  Success Rate: 100.0%
  Duration: avg=45.2ms, median=42.0ms
  Range: 30.0ms - 85.0ms
  Total Time: 2,260.0ms

🧠 LLM PERFORMANCE
--------------------------------------------------------------------------------

[gpt-4]
  Total Calls: 100
  Cache Hits: 35 (35.0%)
  Latency: avg=2340.5ms, median=2200.0ms
  Tokens: total=185,000, avg=1850
  Cost: total=$5.5500, avg=$0.0555

📦 DATA FETCH PERFORMANCE
--------------------------------------------------------------------------------

[news]
  Total Fetches: 150
  Cache Hits: 120 (80.0%)
  Avg Fetch Time: 85.3ms
  Total Records: 3,750

[bars]
  Total Fetches: 200
  Cache Hits: 180 (90.0%)
  Avg Fetch Time: 12.5ms
  Total Records: 50,000

📈 DECISION STATISTICS
--------------------------------------------------------------------------------
  Total Decisions: 150
  Avg Confidence: 72.50%
  Action Distribution:
    - hold: 90 (60.0%)
    - increase: 30 (20.0%)
    - decrease: 20 (13.3%)
    - close: 10 (6.7%)

================================================================================
```

### **性能洞察**

通过性能报告，你可以：
1. **识别瓶颈**: 找出执行时间最长的 Agent
2. **优化缓存**: 提高缓存命中率，减少 API 调用
3. **成本控制**: 追踪 LLM 使用成本
4. **质量监控**: 监控 Agent 成功率和决策置信度

---

## 🔗 工具 3: log_trace.py - 执行链路追踪

### **功能**
可视化展示执行链路和依赖关系。

### **基本用法**

```bash
# 追踪特定 run_id
python scripts/log_trace.py --run-id backtest_20251215_001

# 追踪特定日期的所有执行
python scripts/log_trace.py --date 2025-12-15

# 生成 HTML 可视化
python scripts/log_trace.py --run-id backtest_20251215_001 --html trace.html
```

### **所有参数**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--log-dir` | 日志目录 | `logs/stockbench` |
| `--run-id` | 运行 ID | `backtest_20251215_001` |
| `--date` | 日期 (YYYY-MM-DD) | `2025-12-15` |
| `--html` | 输出 HTML 文件 | `trace.html` |

### **文本追踪示例**

```
================================================================================
🔍 EXECUTION TRACE: backtest_20251215_001
📅 Date: 2025-12-15
================================================================================

🤖 AGENT EXECUTION TIMELINE
--------------------------------------------------------------------------------
2025-12-15T10:30:00Z | ✅ fundamental_filter
  Duration: 45.2ms
  Input: 150 items
  Output: 45 items

2025-12-15T10:30:01Z | ✅ decision_agent
  Duration: 234.5ms
  Input: 45 items
  Output: 45 items

2025-12-15T10:30:02Z | ❌ backtest_report
  Duration: 120.3ms
  ❌ Error: Connection timeout

📈 DECISIONS SUMMARY
--------------------------------------------------------------------------------
Total Decisions: 45
  - hold: 30
  - increase: 10
  - decrease: 5

High Confidence Decisions (12):
  - AAPL: increase (confidence=85.0%)
  - GOOGL: increase (confidence=82.5%)
  - MSFT: hold (confidence=90.0%)

🧠 LLM CALLS SUMMARY
--------------------------------------------------------------------------------
Total Calls: 50
Cache Hits: 18 (36.0%)
Total Tokens: 92,500
Total Latency: 117,025.0ms
Avg Latency: 2340.5ms

📦 DATA FETCHES SUMMARY
--------------------------------------------------------------------------------
Total Fetches: 75
Cache Hits: 60 (80.0%)

⚠️  ERRORS & WARNINGS
--------------------------------------------------------------------------------
2025-12-15T10:30:02Z | ERROR
  Message: [AGENT_ERROR] Failed
  Error: Connection timeout after 60s

================================================================================
```

### **HTML 可视化**

HTML 输出提供：
- 📊 统计卡片（Agent 数量、决策数、LLM 调用）
- 📈 时间线可视化（Agent 执行顺序）
- 🎨 颜色编码（成功=绿色，失败=红色）
- 📋 交互式界面（可在浏览器中查看）

---

## 🎯 实战场景

### **场景 1: 调试失败的回测**

```bash
# 1. 查找失败的 Agent 执行
python scripts/log_query.py --status failed --level ERROR

# 2. 追踪完整执行链路
python scripts/log_trace.py --run-id backtest_20251215_001

# 3. 分析性能瓶颈
python scripts/log_performance.py --date 2025-12-15
```

### **场景 2: 优化 LLM 成本**

```bash
# 1. 查找所有 LLM 调用
python scripts/log_query.py --tag LLM_CALL --output llm_calls.csv

# 2. 分析缓存命中率
python scripts/log_performance.py --focus llm

# 3. 找出缓存未命中的调用
python scripts/log_query.py --cache-hit false --tag LLM_CALL
```

### **场景 3: 分析决策质量**

```bash
# 1. 查找低置信度决策
python scripts/log_query.py --max-confidence 0.6 --tag AGENT_DECISION

# 2. 导出所有决策到 CSV 进行分析
python scripts/log_query.py --tag AGENT_DECISION --output all_decisions.csv

# 3. 查看决策统计
python scripts/log_performance.py --focus decisions
```

### **场景 4: 监控系统健康**

```bash
# 1. 查找所有错误和警告
python scripts/log_query.py --level ERROR
python scripts/log_query.py --level WARNING

# 2. 查看 Agent 成功率
python scripts/log_performance.py

# 3. 生成完整报告
python scripts/log_performance.py --detailed --output daily_report.txt
```

---

## 💡 高级技巧

### **技巧 1: 组合使用工具**

```bash
# 先查询，再分析
python scripts/log_query.py --symbol AAPL --output aapl.csv
# 然后在 Excel 中分析 aapl.csv

# 追踪后生成报告
python scripts/log_trace.py --run-id xxx --html trace.html
# 在浏览器中查看 trace.html
```

### **技巧 2: 定时报告**

创建定时任务每天生成报告：

```bash
# Linux/Mac (crontab)
0 8 * * * python /path/to/scripts/log_performance.py --output /path/to/reports/daily_$(date +\%Y\%m\%d).txt

# Windows (Task Scheduler)
python scripts/log_performance.py --output reports\daily_report.txt
```

### **技巧 3: 快速诊断脚本**

```bash
#!/bin/bash
# quick_diagnose.sh

DATE=$(date +%Y-%m-%d)

echo "=== Errors ==="
python scripts/log_query.py --date $DATE --level ERROR --limit 10

echo "=== Failed Agents ==="
python scripts/log_query.py --date $DATE --status failed --limit 10

echo "=== Performance Summary ==="
python scripts/log_performance.py --date $DATE
```

---

## 📋 常见问题

**Q: 日志文件在哪里？**  
A: 默认在 `logs/stockbench/YYYY-MM-DD.log`，可通过 `--log-dir` 参数修改。

**Q: 查询很慢怎么办？**  
A: 使用 `--date` 指定日期，使用 `--limit` 限制结果数量。

**Q: 如何导出给其他人分析？**  
A: 使用 `--output` 参数导出为 CSV 或 JSON 格式。

**Q: HTML 追踪不显示怎么办？**  
A: 确保日志包含 `run_id` 字段。使用 PipelineContext 自动添加。

**Q: 如何查询多个日期？**  
A: 暂不支持，需要分别查询或直接使用 `jq` 工具处理多个文件。

---

## 🔗 相关文档

- **Schema 定义**: `stockbench/utils/log_schemas.py`
- **使用示例**: `examples/structured_logging_example.py`
- **迁移指南**: `docs/STRUCTURED_LOGGING_MIGRATION.md`
- **实施报告**: `LOGGING_OPTIMIZATION_IMPLEMENTATION.md`

---

## 🚀 快速参考

```bash
# 查询
python scripts/log_query.py --symbol AAPL --tag AGENT_DECISION

# 性能
python scripts/log_performance.py --date 2025-12-15

# 追踪
python scripts/log_trace.py --run-id xxx --html trace.html

# 帮助
python scripts/log_query.py --help
python scripts/log_performance.py --help
python scripts/log_trace.py --help
```
