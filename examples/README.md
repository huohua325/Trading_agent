# Finnhub API 测试

这个目录包含用于测试 Finnhub API 功能的示例代码。

## 使用方法

1. 首先在项目根目录创建 `.env` 文件，并设置 Finnhub API 密钥：

```
FINNHUB_API_KEY=your_finnhub_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # 用于AI决策示例
```

2. 安装所需依赖：

```bash
pip install finnhub-python python-dotenv pandas openai
```

3. 运行测试脚本：

```bash
# 测试基本API功能
python finnhub_api_test.py

# 测试财报相关API功能
python finnhub_financials_test.py

# 测试已集成到FinnhubDataSource的可用API功能
python finnhub_available_api_test.py

# 测试财务数据对AI交易决策的影响
python financial_decision_example.py
```

## 测试的 API 功能

### 基本API测试 (finnhub_api_test.py)

#### 基本功能 (免费API可用)

1. 股票报价 (Quote)
2. 公司基本信息 (Company Profile)
3. 公司新闻 (Company News)
4. 推荐趋势 (Recommendation Trends)
5. 股票代码查询 (Symbol Lookup)
6. K线数据 (Stock Candles)

#### 高级功能 (可能需要付费订阅)

7. 技术指标分析 (Technical Indicator) - RSI
8. 新闻情绪分析 (News Sentiment)

### 财报API测试 (finnhub_financials_test.py)

1. 基本财务数据 (Basic Financials)
2. 财务报表 (Financial Statements)
3. 已报告的财务报表 (Reported Financials)
4. 盈利惊喜 (Earnings Surprises)
5. EPS预测 (EPS Estimates)
6. 收入预测 (Revenue Estimates)
7. EBITDA预测 (EBITDA Estimates)
8. EBIT预测 (EBIT Estimates)
9. 收入细分 (Revenue Breakdown)
10. 盈利质量评分 (Earnings Quality Score)

### 已集成API测试 (finnhub_available_api_test.py)

测试已集成到 `FinnhubDataSource` 类中的可用API功能：

1. 基本财务数据 (Basic Financials)
2. 已报告的财务报表 (Reported Financials)
3. 盈利惊喜数据 (Earnings Surprises)
4. 推荐趋势 (Recommendation Trends)
5. 关键财务指标 (Financial Metrics)
6. 综合财务数据 (Company Financials)

### 财务数据辅助AI决策 (financial_decision_example.py)

测试财务数据对AI交易决策的影响：

1. 使用财务数据生成交易决策
2. 不使用财务数据生成交易决策
3. 比较两种决策的差异
4. 分析财务数据如何影响决策质量

## 注意事项

- 免费 API 密钥有调用频率限制，如果遇到错误可能是达到了限制
- 某些 API 功能需要付费订阅，脚本会自动处理 403 权限错误
- 测试脚本使用 "AAPL" (苹果公司) 作为示例股票代码
- 脚本中添加了 1 秒延时，避免频繁调用 API 导致限制
- 财务数据辅助AI决策示例需要 OpenAI API 密钥 