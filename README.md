# trading_agent_v2 使用指南

本目录提供了基于 Polygon 与 LLM 的最小交易闭环：数据→特征→分析→风控→决策→审计。

## 一、准备
- 环境变量：
  - `POLYGON_API_KEY`: Polygon API Key
  - `OPENAI_API_KEY` 或 `LLM_API_KEY`: LLM API Key（可选；若仅用缓存或禁用 LLM 可不配）
- 依赖：参见项目根 `trading_agent/requirements.txt`
- 配置文件：`trading_agent_v2/config/config.yaml`
  - 关键项：`symbols_universe`、`news.lookback_days`、`news.page_limit`、`llm.*`
  - 回测与执行关键项（新增/对齐）：
    - `backtest.timespan`: day|minute（CLI 未显式传入时使用此默认值）
    - `backtest.warmup_days`: 特征历史回看窗口
    - `backtest.max_positions`、`backtest.cooldown_days`
    - `backtest.slippage_buy_bps`、`backtest.slippage_sell_bps`（优先于 `slippage_bps`）
    - `bars.minute.open_price_source`、`bars.minute.mark_price_source`（参考价策略，当前引擎默认 open=vwap、mark=close）
    - `news.agg`、`news.trim_alpha`
    - `api.polygon.api_key`（可用环境变量注入）
    - 新增：`agents.mode: single|multi`（默认我们推荐 `single`，也可在 CLI 传 `--agent-mode single` 覆盖）

- 环境变量设置示例（Linux/macOS，bash/zsh）：
  ```bash
  # 临时设置（当前终端有效）
  export POLYGON_API_KEY="你的_polygon_key"
  export OPENAI_API_KEY="你的_openai_key"   # 若使用其他兼容 LLM 服务，也可用 LLM_API_KEY
  
  # 验证是否生效
  echo $POLYGON_API_KEY | sed 's/.\{6\}$/******/'
  echo $OPENAI_API_KEY | sed 's/.\{6\}$/******/'
  
  # 若需长期生效，可追加到 shell 配置文件
  echo 'export POLYGON_API_KEY="你的_polygon_key"' >> ~/.bashrc
  echo 'export OPENAI_API_KEY="你的_openai_key"'  >> ~/.bashrc
  # 或将 OPENAI_API_KEY 改为 LLM_API_KEY：
  # echo 'export LLM_API_KEY="你的_llm_key"'       >> ~/.bashrc
  source ~/.bashrc
  ```

- 在 `config.yaml` 中设置/修改 LLM 的 `base_url`（以及模型名）：
  ```yaml
  # 文件：trading_agent_v2/config/config.yaml
  llm:
    base_url: "https://api.openai.com/v1"   # 如使用自建/代理服务，请改为你的兼容 OpenAI 接口地址
    analyzer_model: "gpt-4o-mini"           # 可改为你的可用模型/部署名
    decision_model: "gpt-4o-mini"           # 可改为你的可用模型/部署名
    timeout_sec: 60
    retry:
      max_retries: 3
      backoff_factor: 0.5
  ```
  - 提示：若你的服务不使用 `OPENAI_API_KEY`，可改用 `LLM_API_KEY` 注入；`llm.base_url` 需指向 OpenAI 兼容的 REST 接口根路径。

## 二、核心命令
使用盘中最小闭环：

```
python -m trading_agent_v2.apps.run_intraday \
  --cfg trading_agent_v2/config/config.yaml \
  --since 2025-01-01 --until 2025-01-02 \
  --max-symbols 1 \
  --enable-llm --llm-news-sent \
  --print-features --print-analysis --print-llm-prompts \
  --gray-percent 100 --dry-run
```

查找就近财报日期（便于构造财报窗测试）：

```
python -m trading_agent_v2.apps.find_earnings --symbol AAPL --asof 2025-01-15T00:00:00Z
```

## 回测使用说明（Backtest）

为保持 README 简洁，回测的命令与详尽用法不再在此展开。请直接查看：
- 回测产出结构与查看指引：`trading_agent_v2/storage/README.md`
- 回测用法与参数总览（可选）：`trading_agent_v2/backtest/README.md`

你可以在上述文档中找到：如何运行回测、常见参数、产物目录结构（trades.parquet/daily_nav.parquet/metrics.json/summary.txt/nl_summary.txt）以及如何解读指标与结果。

## 三、参数说明（run_intraday）
- `--cfg` 必填：配置文件路径。
- `--since`：历史与新闻开始时间（ISO 日期）。用于分钟/日线与新闻窗口下界。
- `--until`：历史与新闻结束时间（ISO 日期）。用于分钟/日线与新闻窗口上界及特征时间戳回退。
- `--max-symbols`：最多处理的标的数。0 表示不限制。
- `--dry-run`：干跑，不写审计日志/Parquet 落盘。
- `--gray-percent`：灰度比例（按列表截断）。
- `--enable-llm`：启用 LLM（Analyzer/Decision/新闻情绪打分均可用）。
- `--llm-cache-only`：仅使用缓存结果，不实际调用 LLM。
- `--llm-news-sent`：启用 LLM 对新闻逐条打分并汇总到 `features.news.sentiment`（需 `--enable-llm`）。
- `--print-features`：打印 `features_list`（FeatureInput 列表）。
- `--print-analysis`：打印 Analyzer 输出。
- `--print-llm-prompts`：按角色分段打印三类 LLM 请求（analyzer/decision/news_sentiment）的 `system`/`user`/`raw` 与用量、时间戳、cache_key。

## 四、数据来源与落盘
- 历史价格（日/分钟）：Polygon Aggregates，按日分片落 `trading_agent_v2/storage/parquet/<symbol>/<granularity>/`。
- 快照价格：Polygon 通用快照，多票批量接口。
- 新闻：Polygon `v2/reference/news`，带 24h 文件缓存与去重。
- 分红/拆股/详情：Polygon `v3/reference`（失败时回退本地样例）。
- 财报：Polygon `/vX/reference/financials`，带 24h 文件缓存，摘要并入 `features.fund.*`。
- 事件日历：Polygon `/vX/reference/tickers/{ticker}/events`（本地过滤 earnings 相关）用于 `in_earnings_window`。

更多关于产出目录结构、文件含义与查看方式，请见：`trading_agent_v2/storage/README.md`。

## 六、特征结构（FeatureInput）
- `symbol`、`ts_utc`
- `tech`: `{ret.{1d,5d,20d}, atr_pct, trend, mom, vwap_dev}`
- `news`: `{sentiment, top_k_events, src_count, freshness_min}`
  - `sentiment`：未启用 `--llm-news-sent` 时为来源均值；启用后由 LLM 逐条打分再聚合（mean/median/trimmed_mean）。
- `fund`: 分红/拆股最小快照 + 财报摘要：`fin_rev_yoy`、`fin_eps_yoy`、`fin_gross_margin`、`fin_op_margin`、`fin_net_margin`、`fin_fcf_margin`、`fin_debt_to_equity`、`fin_shares_dilution_1y`
- `market_ctx`: `{last_price, vol_zscore_20d, realized_vol_20d, gap_1d, atr_pct, daily_drawdown_pct, in_earnings_window}`
- `position_state`
  - 包含：`current_position_pct`、`avg_price`、`pnl_pct`、`holding_days`

## 七、LLM 调用与调试（概览）
- Analyzer 与 Decision：提示词位于 `agents/prompts/*.txt`，缓存位于 `storage/cache/llm/*.json`。
- News Sentiment：`agents/news_sentiment_llm.py`，提示词 `agents/prompts/news_sentiment_v1.txt`。
- 回顾性报告（自然语言）：`agents/report_llm.py`（盘中单票）、回测总结在回测产出目录的 `nl_summary.txt`（细节见 `storage/README.md`）。

## 八、常见问题
- `news.sentiment = 0`：新闻缺失数值情绪或 LLM 打分正负抵消。可调大 `news.page_limit`、采用时效加权/中位数聚合。
- `freshness_min` 极大或负数：来自时间对齐差异。可在特征层做截断或归零。
- 无法访问 Polygon/LLM：检查 API Key，或使用 `--llm-cache-only` 与本地回退数据。
- timespan 联动：CLI 未传 `--timespan` 时，`pipeline.run_backtest` 将使用 `config.yaml` 的 `backtest.timespan`。

## 九、扩展建议
- 为新闻情绪聚合提供加权与稳健统计选项（中位数/截尾均值）。已支持：`--news-agg`、`--news-trim-alpha`；并可用 `--sweep-*` 做敏感性扫描。
- 更细粒度的风控与执行策略（滑点、成交量约束、风控事件处理）。
- 完善 EOD/回测流程、报告与可视化。

## 十、推荐运行示例（充分发挥能力）
- 盘中联调示例仍在本页（见上方“核心命令”）。
- 回测命令与结果查看：请移步 `trading_agent_v2/storage/README.md`（以及可选的 `trading_agent_v2/backtest/README.md`）以获取最新详解。

提示：若想减少 `news.sentiment` 被正负抵消，可在后续版本使用“稳健聚合”（median/trimmed_mean）。当前默认 `mean` 已适配大多数场景。 

## 十一、能力展示命令（精选）

- 盘中全链路（含 LLM、新闻情绪打分、打印特征/分析/Prompt）
```bash
python -m trading_agent_v2.apps.run_intraday \
  --cfg trading_agent_v2/config/config.yaml \
  --since 2025-01-15 --until 2025-01-16 \
  --max-symbols 1 \
  --enable-llm --llm-news-sent \
  --print-features --print-analysis --print-llm-prompts \
  --gray-percent 100
```

- 分钟级回测（规则基线，TWAP 切片，配置驱动 timespan）
```bash
# 确保 config.yaml 中 backtest.timespan: minute
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-06-03 --end 2025-06-03 \
  --symbols AAPL \
  --strategy rule_baseline \
  --cost-tier med \
  --run-id demo_minute_twap_cfg
```

- LLM 决策（分钟级，单智能体默认，显式传入 timespan）
```bash
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-01-14 --end 2025-01-16 \
  --symbols AAPL \
  --strategy llm_decision \
  --agent-mode single \
  --timespan minute \
  --cost-tier med \
  --run-id demo_llm_minute_single
```

- 财报窗定点测试（先找就近财报，再回测该日以触发风控，单智能体）
```bash
python -m trading_agent_v2.apps.find_earnings --symbol AAPL --asof 2025-01-15T00:00:00Z
# 假设输出 nearest_earnings_date=2025-01-15
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-01-15 --end 2025-01-15 \
  --symbols AAPL \
  --strategy llm_decision \
  --agent-mode single \
  --timespan day \
  --run-id demo_earnings_window_single
```

- 敏感性扫描（LLM 决策，缓存模式，聚合/成本/代理模式多组合）
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
  --run-id sweep_showcase_single
```

- 审计回放（复现某日实盘/盘中生成的订单）
```bash
python -m trading_agent_v2.apps.run_backtest \
  --cfg trading_agent_v2/config/config.yaml \
  --start 2025-08-10 --end 2025-08-10 \
  --symbols AAPL \
  --strategy llm_decision --replay \
  --audit-dir trading_agent_v2/storage/audit/2025-08-10 \
  --run-id replay_showcase
``` 