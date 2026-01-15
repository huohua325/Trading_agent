#!/usr/bin/env python3
"""
OpenAI API Key 测试脚本
用于测试API Key是否有效，以及base_url配置是否正确
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
from datetime import datetime

def test_openai_api(api_key=None, base_url=None):
    """
    测试OpenAI API连接
    
    Args:
        api_key: OpenAI API密钥，如果不提供则从环境变量读取
        base_url: API基础URL，如果不提供则从环境变量读取或使用默认值
    """
    print("=" * 70)
    print("OpenAI API 连接测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 获取API密钥
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ 错误: API Key未设置")
        print("\n请通过以下方式之一设置API Key:")
        print("  方式1 - 环境变量 (PowerShell):")
        print("    $env:OPENAI_API_KEY='sk-your-key-here'")
        print("\n  方式2 - 命令行参数:")
        print("    python test_openai_api.py --api-key sk-your-key-here")
        print("\n  方式3 - 修改脚本中的变量:")
        print("    API_KEY = 'sk-your-key-here'")
        return False
    
    # 掩码显示API Key
    masked_key = api_key[:7] + '*' * (len(api_key) - 11) + api_key[-4:] if len(api_key) > 11 else '***'
    print(f"✓ API Key: {masked_key}")
    
    # 2. 获取Base URL
    if base_url is None:
        base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    
    print(f"✓ Base URL: {base_url}")
    print()
    
    # 3. 测试1: 列出可用模型
    print("-" * 70)
    print("测试 1: 列出可用模型")
    print("-" * 70)
    
    try:
        url = f"{base_url.rstrip('/')}/models"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        print(f"请求URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            print(f"✓ 成功! 找到 {len(models)} 个可用模型")
            
            if models:
                print("\n前10个可用模型:")
                for i, model in enumerate(models[:10], 1):
                    model_id = model.get('id', 'unknown')
                    print(f"  {i}. {model_id}")
                
                if len(models) > 10:
                    print(f"  ... 还有 {len(models) - 10} 个模型")
        else:
            print(f"❌ 请求失败")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        print("可能原因:")
        print("  1. 网络连接问题")
        print("  2. Base URL不正确")
        print("  3. 需要使用代理")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接失败: {e}")
        print("可能原因:")
        print("  1. Base URL不正确")
        print("  2. 网络无法访问该地址")
        print("  3. 需要配置代理")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False
    
    # 4. 测试2: 简单的对话测试
    print("\n" + "-" * 70)
    print("测试 2: 对话完成测试")
    print("-" * 70)
    
    try:
        url = f"{base_url.rstrip('/')}/chat/completions"
        
        payload = {
            "model": "gpt-3.5-turbo",  # 使用最常见的模型
            "messages": [
                {"role": "user", "content": "Say 'API test successful' in Chinese"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        print(f"请求URL: {url}")
        print(f"测试模型: {payload['model']}")
        print(f"测试消息: {payload['messages'][0]['content']}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # 提取响应内容
            if 'choices' in data and len(data['choices']) > 0:
                message = data['choices'][0].get('message', {})
                content = message.get('content', '')
                
                print(f"\n✓ 对话测试成功!")
                print(f"模型响应: {content}")
                
                # 显示使用统计
                usage = data.get('usage', {})
                if usage:
                    print(f"\nToken使用统计:")
                    print(f"  - 提示词: {usage.get('prompt_tokens', 0)} tokens")
                    print(f"  - 生成: {usage.get('completion_tokens', 0)} tokens")
                    print(f"  - 总计: {usage.get('total_tokens', 0)} tokens")
            else:
                print("⚠️ 响应格式异常")
                print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 请求失败")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.text}")
            
            # 常见错误提示
            if response.status_code == 401:
                print("\n💡 提示: API Key无效或已过期")
            elif response.status_code == 429:
                print("\n💡 提示: 请求频率超限或配额用尽")
            elif response.status_code == 404:
                print("\n💡 提示: 模型不存在或Base URL不正确")
            
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时（30秒）")
        print("可能原因: API响应慢或网络问题")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    # 5. 总结
    print("\n" + "=" * 70)
    print("✅ 所有测试通过! API配置正确")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 在 config.yaml 中设置相同的 base_url")
    print("  2. 设置环境变量:")
    print(f"     $env:OPENAI_API_KEY='{masked_key}'")
    print("  3. 运行回测:")
    print("     python -m stockbench.apps.run_backtest --llm-profile openai")
    print()
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试OpenAI API连接')
    parser.add_argument('--api-key', help='OpenAI API Key (或使用环境变量 OPENAI_API_KEY)')
    parser.add_argument('--base-url', help='API Base URL (或使用环境变量 OPENAI_BASE_URL)')
    
    args = parser.parse_args()
    
    # ==========================================
    # 💡 快速配置区域 - 可直接在这里修改
    # ==========================================
    
    # 选项1: 直接在这里设置（不推荐，仅用于快速测试）
    API_KEY = None  # 例如: 'sk-xxxxx'
    BASE_URL = None  # 例如: 'https://api.openai.com/v1'
    
    # 选项2: 使用命令行参数（推荐）
    # python test_openai_api.py --api-key sk-xxx --base-url https://api.xxx.com/v1
    
    # 选项3: 使用环境变量（最推荐）
    # $env:OPENAI_API_KEY='sk-xxx'
    # $env:OPENAI_BASE_URL='https://api.xxx.com/v1'
    
    # ==========================================
    
    # 优先级: 命令行参数 > 脚本变量 > 环境变量
    api_key = args.api_key or API_KEY
    base_url = args.base_url or BASE_URL
    
    success = test_openai_api(api_key=api_key, base_url=base_url)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
