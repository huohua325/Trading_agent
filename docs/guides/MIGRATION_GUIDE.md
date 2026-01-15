# StockBench Migration Guide

## v0.8.0 → v1.0.0

This guide helps you migrate from the old memory system to the new unified Memory and PipelineContext system.

---

## Overview of Changes

### ✅ What's New
- **Unified Memory System**: All historical data managed through `ctx.memory.episodes`
- **PipelineContext**: Standardized context object for all agents
- **Deprecation Warnings**: Clear migration path with warnings for legacy APIs

### ⚠️ What's Deprecated
- `previous_decisions` parameter in agent functions
- `decision_history` parameter in agent functions  
- Dict-type `ctx` parameter (replaced by PipelineContext)

---

## Migration Steps

### 1. Replace `previous_decisions` and `decision_history` Parameters

**❌ Old Code (Deprecated)**:
```python
from stockbench.agents import decide_batch_dual_agent

decisions = decide_batch_dual_agent(
    features_list,
    previous_decisions=prev_decisions,  # Will be removed in v1.0
    decision_history=history_data,      # Will be removed in v1.0
)
```

**✅ New Code**:
```python
from stockbench.core import PipelineContext
from stockbench.agents import decide_batch_dual_agent

# Create PipelineContext
ctx = PipelineContext(
    config=config,
    run_id=run_id,
    date=current_date
)

# Historical decisions are automatically loaded from ctx.memory.episodes
decisions = decide_batch_dual_agent(
    features_list,
    ctx=ctx  # All data including history comes from ctx
)
```

### 2. Migrate from Dict `ctx` to PipelineContext

**❌ Old Code (Deprecated)**:
```python
# Dict-based context
ctx = {
    "portfolio": portfolio,
    "config": config,
    "previous_decisions": prev_decisions
}

decisions = decide_batch_dual_agent(features_list, ctx=ctx)
```

**✅ New Code**:
```python
from stockbench.core import PipelineContext

# PipelineContext with proper type safety
ctx = PipelineContext(
    config=config,
    run_id=run_id,
    date=current_date
)

# Store additional data in the context
ctx.put("portfolio", portfolio)

# Memory system handles historical decisions automatically
decisions = decide_batch_dual_agent(features_list, ctx=ctx)
```

### 3. Access Historical Decisions

**❌ Old Code**:
```python
# Manual history management
history = strategy.decision_history.get(symbol, [])
```

**✅ New Code**:
```python
# Automatic history from Memory system
# In your agent code, history is loaded automatically:
if ctx.memory_enabled:
    history = ctx.memory.episodes.get_for_prompt(symbol, n=5)
```

---

## Affected Components

### Agent Functions

| Function | Changed Parameters | Migration Required |
|----------|-------------------|-------------------|
| `decide_batch_dual_agent` | `previous_decisions`, `decision_history` | ✅ Yes |
| `filter_stocks_needing_fundamental` | Internal `decision_history` loading | ✅ Automatic |

### Strategy Classes

| Class | Removed Attributes | Migration Required |
|-------|-------------------|-------------------|
| `LLMDecisionStrategy` | `self.previous_decisions`, `self.decision_history` | ✅ Already migrated |

---

## Breaking Changes Timeline

### v0.8.0 (Current)
- ⚠️ Deprecation warnings added
- ✅ Old code still works with warnings
- ✅ New code recommended

### v0.9.0 (Future)
- ⚠️ Stricter warnings
- ✅ Migration tools and validation

### v1.0.0 (Future)
- ❌ Legacy parameters removed
- ❌ Dict ctx no longer supported
- ✅ Only PipelineContext accepted

---

## Testing Your Migration

### Step 1: Run with Warnings
```bash
python -W default::DeprecationWarning scripts/run_backtest.py \
    --symbols AAPL \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

If you see deprecation warnings, update your code following this guide.

### Step 2: Verify Memory System
```python
# Check that memory is enabled
assert ctx.memory_enabled == True

# Verify episodes are being saved
episodes = ctx.memory.episodes.get_for_prompt("AAPL", n=5)
assert len(episodes) > 0  # Should have historical data
```

### Step 3: Check Agent Traces
```python
# View agent execution traces
for trace in ctx.get_traces():
    print(f"Agent: {trace.agent_name}")
    print(f"Status: {trace.status}")
    print(f"Duration: {trace.duration_ms}ms")
```

---

## Common Issues

### Issue 1: "Parameter 'previous_decisions' is deprecated"

**Solution**: Remove the parameter and use PipelineContext instead.

### Issue 2: "Passing Dict as 'ctx' is deprecated"

**Solution**: Replace `ctx = {...}` with `ctx = PipelineContext(...)`.

### Issue 3: No historical data in decisions

**Cause**: Memory system not enabled or no data saved yet.

**Solution**: 
```python
# Ensure memory is enabled in config
config = {
    "memory": {
        "enabled": True,
        "episodes": {
            "backend": "file",
            "ttl_days": 90
        }
    }
}

ctx = PipelineContext(config=config, ...)
```

---

## Support

- **Documentation**: See `UPGRADE_PLAN_PHASE2.md` for detailed architecture
- **Examples**: Check `stockbench/examples/pipeline_example.py`
- **Issues**: Report migration problems in project issues

---

## Summary

The migration to the new Memory system provides:
- ✅ Unified historical data management
- ✅ Better type safety with PipelineContext
- ✅ Automatic history loading
- ✅ Clear data flow and tracing
- ✅ Future-proof architecture

Follow this guide to ensure a smooth migration before v1.0 release.
