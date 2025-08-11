# trading_agent_v2 使用指南

本目录提供了基于 Polygon 与 LLM 的最小交易闭环：数据→特征→分析→风控→决策→审计。

## 一、准备
- 环境变量：
  - `POLYGON_API_KEY`: Polygon API Key
  - `OPENAI_API_KEY` 或 `LLM_API_KEY`: LLM API Key（可选；若仅用缓存或禁用 LLM 可不配）
- 依赖：参见项目根 `trading_agent/requirements.txt`
- 配置文件：`trading_agent_v2/config/config.yaml`
  - 关键项：`symbols_universe`、`news.lookback_days`、`news.page_limit`、`llm.*`

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

更多关于产出目录结构、文件含义与查看方式，请见：`trading_agent_v2/storage/README.md`。

## 六、特征结构（FeatureInput）
- `symbol`、`ts_utc`
- `tech`: `{ret.{1d,5d,20d}, atr_pct, trend, mom, vwap_dev}`
- `news`: `{sentiment, top_k_events, src_count, freshness_min}`
  - `sentiment`：未启用 `--llm-news-sent` 时为来源均值；启用后由 LLM 逐条打分再聚合（mean/median/trimmed_mean）。
- `fund`: 分红/拆股最小快照 + 财报摘要：`fin_rev_yoy`、`fin_eps_yoy`、`fin_gross_margin`、`fin_op_margin`、`fin_net_margin`、`fin_fcf_margin`、`fin_debt_to_equity`、`fin_shares_dilution_1y`
- `market_ctx`: `{last_price, vol_zscore_20d, realized_vol_20d, gap_1d}`
- `position_state`

## 七、LLM 调用与调试（概览）
- Analyzer 与 Decision：提示词位于 `agents/prompts/*.txt`，缓存位于 `storage/cache/llm/*.json`。
- News Sentiment：`agents/news_sentiment_llm.py`，提示词 `agents/prompts/news_sentiment_v1.txt`。
- 回顾性报告（自然语言）：`agents/report_llm.py`（盘中单票）、回测总结在回测产出目录的 `nl_summary.txt`（细节见 `storage/README.md`）。

## 八、常见问题
- `news.sentiment = 0`：新闻缺失数值情绪或 LLM 打分正负抵消。可调大 `news.page_limit`、采用时效加权/中位数聚合。
- `freshness_min` 极大或负数：来自时间对齐差异。可在特征层做截断或归零。
- 无法访问 Polygon/LLM：检查 API Key，或使用 `--llm-cache-only` 与本地回退数据。

## 九、扩展建议
- 为新闻情绪聚合提供加权与稳健统计选项（中位数/截尾均值）。已支持：`--news-agg`、`--news-trim-alpha`；并可用 `--sweep-*` 做敏感性扫描。
- 更细粒度的风控与执行策略（滑点、成交量约束、风控事件处理）。
- 完善 EOD/回测流程、报告与可视化。

## 十、推荐运行示例（充分发挥能力）
- 盘中联调示例仍在本页（见上方“核心命令”）。
- 回测命令与结果查看：请移步 `trading_agent_v2/storage/README.md`（以及可选的 `trading_agent_v2/backtest/README.md`）以获取最新详解。

提示：若想减少 `news.sentiment` 被正负抵消，可在后续版本使用“稳健聚合”（median/trimmed_mean）。当前默认 `mean` 已适配大多数场景。 