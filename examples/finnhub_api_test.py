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

# 获取当前时间和一个月前的时间戳
end_timestamp = int(datetime.now().timestamp())
start_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())

print(f"测试Finnhub API功能 - 使用股票: {symbol}")
print("注意: 免费API密钥可能无法访问所有功能，测试将跳过返回403错误的API")
print("-" * 50)

# 添加延时函数，防止API调用过于频繁
def api_call_with_delay(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        print("API调用成功，等待1秒...")
        time.sleep(1)  # 添加1秒延时，避免达到API调用限制
        return result, None
    except Exception as e:
        return None, e

# 测试基本功能 - 这些通常在免费版中可用
try:
    # 1. 股票报价 (Quote)
    print("\n1. 股票报价:")
    quote, err = api_call_with_delay(client.quote, symbol)
    if err:
        print(f"获取报价失败: {err}")
    else:
        print(f"当前价格: ${quote.get('c', 'N/A')}")
        print(f"今日变化: {quote.get('dp', 'N/A')}%")
    
    # 2. 公司基本信息
    print("\n2. 公司基本信息:")
    profile, err = api_call_with_delay(client.company_profile2, symbol=symbol)
    if err:
        print(f"获取公司信息失败: {err}")
    else:
        print(f"公司名称: {profile.get('name', 'N/A')}")
        print(f"行业: {profile.get('finnhubIndustry', 'N/A')}")
    
    # 3. 公司新闻
    print("\n3. 公司新闻:")
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    news, err = api_call_with_delay(
        client.company_news,
        symbol, 
        _from=from_date, 
        to=to_date
    )
    if err:
        print(f"获取新闻失败: {err}")
    else:
        print(f"获取到 {len(news)} 条新闻")
        if news:
            print(f"最新新闻: {news[0].get('headline', 'N/A')}")
    
    # 4. 推荐趋势 (Recommendation Trends)
    print("\n4. 推荐趋势:")
    trends, err = api_call_with_delay(client.recommendation_trends, symbol)
    if err:
        print(f"获取推荐趋势失败: {err}")
    else:
        if trends:
            print(f"最新推荐: {trends[0]}")
        else:
            print("无推荐数据")
    
    # 5. 股票代码查询
    print("\n5. 股票代码查询:")
    symbols, err = api_call_with_delay(client.symbol_lookup, "apple")
    if err:
        print(f"查询股票代码失败: {err}")
    else:
        print(f"查询结果数量: {len(symbols.get('result', []))}")
        if symbols.get('result'):
            print(f"第一个结果: {symbols['result'][0]}")
    
    # 6. 尝试获取K线数据
    print("\n6. K线数据:")
    candles, err = api_call_with_delay(
        client.stock_candles,
        symbol, 
        'D', 
        start_timestamp, 
        end_timestamp
    )
    if err:
        print(f"获取K线数据失败: {err}")
    else:
        if candles.get('s') == 'ok':
            print(f"获取到 {len(candles.get('c', []))} 个K线数据点")
            if candles.get('c'):
                print(f"最新收盘价: ${candles['c'][-1]}")
        else:
            print(f"K线数据状态: {candles.get('s', 'N/A')}")
    
    # 以下是高级功能，可能需要付费订阅，仅作尝试
    print("\n--- 以下是高级功能，可能需要付费订阅 ---")
    
    # 7. 技术指标 (可能需要付费)
    print("\n7. 技术指标分析 (RSI):")
    rsi, err = api_call_with_delay(
        client.technical_indicator,
        symbol=symbol, 
        resolution='D', 
        _from=start_timestamp, 
        to=end_timestamp, 
        indicator='rsi', 
        indicator_fields={"timeperiod": 14}
    )
    if err:
        print(f"获取RSI失败: {err}")
    else:
        print(f"RSI数据点数量: {len(rsi.get('rsi', []))}")
        if rsi.get('rsi'):
            print(f"最新RSI值: {rsi['rsi'][-1]}")
    
    # 8. 情绪分析 (可能需要付费)
    print("\n8. 新闻情绪分析:")
    sentiment, err = api_call_with_delay(client.news_sentiment, symbol)
    if err:
        print(f"获取情绪分析失败: {err}")
    else:
        print(f"情绪得分: {sentiment.get('sentiment', {}).get('bullishPercent', 'N/A')}")

except Exception as e:
    print(f"测试过程中出错: {e}")

print("\n测试完成!") 