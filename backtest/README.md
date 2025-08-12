# trading_agent_v2/backtest 使用说明

本目录说明回测（Backtest）模块的使用方式与关键参数，配合 `apps/run_backtest.py` 一起完成从数据→策略→撮合→指标→报告的闭环。

## 一、概述
- 支持策略：
  - `rule_baseline`：SMA5 > SMA20 时增持至上限
  - `llm_decision`：Analyzer/Decision LLM 生成目标仓位（回测默认 cache-only，可开启真实调用）
- 执行撮合：
  - `timespan=day`：按日开盘价撮合、收盘估值
  - `timespan=minute`：当日净订单等份切片至每分钟，按该分钟 `vwap`（无则 `close`）撮合，最后一分钟收盘估值
- 成本模型：分层覆盖 `commission_bps`、`slippage_bps`、`fill_ratio`（`low|med|high`）
- 新闻聚合：`mean|median|trimmed_mean`，`trimmed_mean` 支持 `trim_alpha`（0.0~0.49）
- 结果输出：`trading_agent_v2/storage/reports/backtest/<run_id>/`

## 二、快速开始

### 1) 环境准备
- 依赖：见项目根 `trading_agent/requirements.txt`
- 配置：`trading_agent_v2/config/config.yaml`（设置 `symbols_universe` 等）
- 可选：如需真实 LLM 调用
  - `export OPENAI_API_KEY=你的key`
- 必需：访问行情与参照数据
  - `export POLYGON_API_KEY=你的key`

### 2) 日级最小冒烟（规则基线，低成本）
```bash
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-06-01 --end 2025-06-08 \
  --symbols AAPL \
  --strategy rule_baseline \
  --timespan day --cost-tier low \
  --run-id quick_day_low
```
预期：控制台打印单条结果；输出目录包含 `trades.parquet`、`daily_nav.parquet`、`metrics.json`、`summary.txt`。

### 3) 分钟级冒烟（规则基线，中等成本，稳健聚合）
```bash
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-06-03 --end 2025-06-04 \
  --symbols AAPL \
  --strategy rule_baseline \
  --timespan minute --cost-tier med \
  --news-agg trimmed_mean --news-trim-alpha 0.1 \
  --run-id quick_min_med
```
预期：`trades.parquet` 中可见多笔分钟级等份撮合，`exec_price` 接近分钟 `vwap`。

### 4) LLM 决策（真实调用，单次运行，单智能体）
```bash
export OPENAI_API_KEY=你的key
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-03-01 --end 2025-03-10 \
  --symbols AAPL,MSFT,NVDA \
  --strategy llm_decision --no-llm-cache-only \
  --agent-mode single \
  --timespan day --cost-tier med \
  --news-agg trimmed_mean --news-trim-alpha 0.1 \
  --run-id llm_single_day_single
```
预期：若 LLM 与数据正常，将产生非零决策与成交；产物同上。

## 三、CLI 参数详解

- 基础
  - `--cfg PATH`（必填）：YAML 配置文件
  - `--start YYYY-MM-DD` / `--end YYYY-MM-DD`：回测区间（含端点）
  - `--symbols "AAPL,MSFT"`：标的清单（逗号或空格分隔）；为空回退 `cfg.symbols_universe`
  - `--strategy rule_baseline|llm_decision`：选择策略
  - `--replay`：回放模式（从审计 `jsonl` 读取订单），配合 `--audit-dir PATH`
  - `--run-id NAME`：输出目录名（扫描时会自动拼接后缀）
- 风控与 LLM
  - `--max-positions INT`：最大同时持仓数
  - `--cooldown-days INT`：清仓后的冷却天数
  - `--min-holding-days INT`：最小持有天数（未达阈值禁止卖出）
  - `--llm-cache-only / --no-llm-cache-only`：是否仅用缓存（默认 True）
- 执行与成本
  - `--timespan day|minute`：撮合粒度（未显式传入时将使用 `config.backtest.timespan`）
  - `--cost-tier low|med|high`：覆盖 `commission_bps`、`slippage_bps`、`fill_ratio`
    - low: `0.0 / 5.0 / 1.0`
    - med: `2.0 / 10.0 / 0.8`
    - high: `5.0 / 20.0 / 0.6`
- 新闻情绪聚合（稳健性）
  - `--news-agg mean|median|trimmed_mean`
  - `--news-trim-alpha FLOAT (0.0~0.49)`：仅在 `trimmed_mean` 下生效
- 敏感性扫描（多组合）
  - `--sweep-news-agg ...`、`--sweep-news-trim-alpha ...`、`--sweep-cost-tier ...`
  - 三者做笛卡尔积，多次连续运行（看起来像“一直在输出”属正常）
  - 不传任何 `--sweep-*` 即单次运行

> 布尔开关用法：启用 `--flag`，关闭 `--no-flag`（不要写 True/False）。

## 四、风控与财报窗（重要）
- 已接入统一风控：`llm_decision` 回测调用 `risk_guard.make_limits`，并读取：
  - `market_ctx.atr_pct`（技术波动）、`market_ctx.daily_drawdown_pct`（当日回撤近似）
  - `market_ctx.in_earnings_window`（财报窗标记）
- 财报窗来源：Polygon 事件日历 `/vX/reference/tickers/{ticker}/events` 全量拉取后本地过滤 `earn/eps` 相关；如接口缺失数据，支持回退新闻关键词近似用于“就近日期查询”。
- 配置项建议：`risk.max_pos_pct_earnings_window`（财报窗降杠杆上限）、`risk.min_holding_days`（最小持有期）。

### 查找就近财报日期（辅助测试）
```bash
python -m trading_agent_v2.apps.find_earnings --symbol AAPL --asof 2025-01-15T00:00:00Z
```
输出：`nearest_earnings_date=YYYY-MM-DD delta_days=N type=events_type|news_keyword`。

## 五、输出与查看
- 目录：`trading_agent_v2/storage/reports/backtest/<run_id>/`
  - `trades.parquet`：逐笔成交（分钟级会有多笔 TWAP 切片）
  - `daily_nav.parquet`：每日净值
  - `metrics.json`：主要指标
  - `summary.txt`：run_id、区间、关键参数（成本/聚合/策略/timespan）、指标摘要、成交条数
  - `config.json`：本次运行的配置快照
  - `meta.json`：环境（Python、pandas/pyarrow/httpx 版本）

## 六、指标口径
- `cum_return`：累计收益 = `NAV_end / NAV_start - 1`
- `max_drawdown`：最大回撤，按净值相对历史峰值的最小比值
- `volatility`：日频收益的标准差年化（×√252）
- `sharpe`：年化收益 / 年化波动（未扣无风险利率）

## 七、常见问题
- “一直在输出”：启用了 `--sweep-*` 参数，扫描多组组合。去掉 `--sweep-*` 即单次运行。
- 指标全为 0：多为 `--llm-cache-only` 命中缓存为空导致不下单，可改用 `--no-llm-cache-only`（需 API Key），或先用 `run_intraday` 生成缓存，再回测。
- 分钟级运行慢：缩短日期范围、减少标的数，或先用 `timespan=day` 冒烟。
- timespan 联动：CLI 未传 `--timespan` 时，系统将使用 `config.yaml` 的 `backtest.timespan`。

## 八、更多示例

- 扫描（LLM 决策，缓存模式，多组合，默认单智能体）
```bash
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-03-01 --end 2025-03-10 \
  --symbols AAPL,MSFT,NVDA \
  --strategy llm_decision --llm-cache-only \
  --agent-mode single \
  --timespan day \
  --sweep-news-agg mean,median,trimmed_mean \
  --sweep-news-trim-alpha 0.0,0.1,0.2 \
  --sweep-cost-tier low,med,high \
  --run-id sweep_llm_day_single
```

- 回放模式（读取审计）
```bash
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-08-10 --end 2025-08-10 \
  --symbols AAPL \
  --strategy llm_decision --replay \
  --audit-dir trading_agent_v2/storage/audit/2025-08-10 \
  --run-id replay_demo
``` 