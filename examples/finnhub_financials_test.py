import os
import finnhub
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

# 获取API密钥
api_key = os.getenv("FINNHUB_API_KEY")
if not api_key:
    raise ValueError("请在.env文件中设置FINNHUB_API_KEY")

# 初始化客户端
client = finnhub.Client(api_key=api_key)

# 设置测试的股票代码
symbol = "AAPL"

print(f"测试Finnhub财报相关API - 使用股票: {symbol}")
print("注意: 某些API可能需要付费订阅")
print("-" * 50)

# 添加延时函数，防止API调用过于频繁
def api_call_with_delay(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        print("API调用成功，等待1秒...")
        time.sleep(1)  # 添加1秒延时，避免达到API调用限制
        return result, None
    except Exception as e:
        print(f"API调用失败: {e}")
        return None, e

# 测试所有财报相关的API
apis_to_test = [
    {
        "name": "基本财务数据 (Basic Financials)",
        "func": lambda: client.company_basic_financials(symbol, 'all'),
        "result_handler": lambda r: f"指标数量: {len(r.get('metric', {}))}"
    },
    {
        "name": "财务报表 (Financial Statements)",
        "func": lambda: client.financials(symbol, 'bs', 'annual'),
        "result_handler": lambda r: f"报表项目数: {len(r.get('financials', []))}"
    },
    {
        "name": "已报告的财务报表 (Reported Financials)",
        "func": lambda: client.financials_reported(symbol=symbol, freq='annual'),
        "result_handler": lambda r: f"报表数量: {len(r.get('data', []))}"
    },
    {
        "name": "盈利惊喜 (Earnings Surprises)",
        "func": lambda: client.company_earnings(symbol, limit=5),
        "result_handler": lambda r: f"盈利数据点数量: {len(r)}"
    },
    {
        "name": "EPS预测 (EPS Estimates)",
        "func": lambda: client.company_eps_estimates(symbol, freq='quarterly'),
        "result_handler": lambda r: f"EPS预测数量: {len(r.get('data', []))}"
    },
    {
        "name": "收入预测 (Revenue Estimates)",
        "func": lambda: client.company_revenue_estimates(symbol, freq='quarterly'),
        "result_handler": lambda r: f"收入预测数量: {len(r.get('data', []))}"
    },
    {
        "name": "EBITDA预测 (EBITDA Estimates)",
        "func": lambda: client.company_ebitda_estimates(symbol, freq='quarterly'),
        "result_handler": lambda r: f"EBITDA预测数量: {len(r.get('data', []))}"
    },
    {
        "name": "EBIT预测 (EBIT Estimates)",
        "func": lambda: client.company_ebit_estimates(symbol, freq='quarterly'),
        "result_handler": lambda r: f"EBIT预测数量: {len(r.get('data', []))}"
    },
    {
        "name": "收入细分 (Revenue Breakdown)",
        "func": lambda: client.stock_revenue_breakdown(symbol),
        "result_handler": lambda r: f"细分数据点数量: {len(r.get('data', []))}"
    },
    {
        "name": "盈利质量评分 (Earnings Quality Score)",
        "func": lambda: client.company_earnings_quality_score(symbol, 'quarterly'),
        "result_handler": lambda r: f"质量评分数据点数量: {len(r.get('data', []))}"
    }
]

# 执行测试
for i, api in enumerate(apis_to_test, 1):
    print(f"\n{i}. {api['name']}:")
    try:
        result, error = api_call_with_delay(api["func"])
        if error:
            print(f"  测试失败: {error}")
        elif result:
            print(f"  {api['result_handler'](result)}")
            # 打印第一个数据点的样例
            if isinstance(result, list) and result:
                print(f"  样例数据: {result[0]}")
            elif isinstance(result, dict):
                if 'data' in result and result['data'] and isinstance(result['data'], list):
                    print(f"  样例数据: {result['data'][0]}")
                elif 'metric' in result:
                    metrics = list(result['metric'].items())[:3]  # 只显示前3个指标
                    print(f"  样例指标: {dict(metrics)}")
    except Exception as e:
        print(f"  处理结果时出错: {e}")

print("\n测试完成!") 