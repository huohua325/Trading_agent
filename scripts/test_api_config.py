#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from loguru import logger

def test_environment_variables():
    """测试环境变量配置"""
    print("=" * 60)
    print("1. 环境变量检查")
    print("=" * 60)
    
    env_vars = {
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'ZHIPUAI_API_KEY': os.environ.get('ZHIPUAI_API_KEY'),
        'POLYGON_API_KEY': os.environ.get('POLYGON_API_KEY'),
        'FINNHUB_API_KEY': os.environ.get('FINNHUB_API_KEY'),
    }
    
    for key, value in env_vars.items():
        if value:
            masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '***'
            print(f"✓ {key}: {masked_value}")
        else:
            print(f"✗ {key}: 未设置")
    
    return env_vars

def test_config_file():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("2. 配置文件检查 (config.yaml)")
    print("=" * 60)
    
    config_path = project_root / 'config.yaml'
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        return None
    
    print(f"✓ 配置文件存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"\n股票池配置:")
        symbols = config.get('symbols_universe', [])
        print(f"  - 股票数量: {len(symbols)}")
        print(f"  - 股票列表: {', '.join(symbols[:5])}...")
        
        print(f"\n数据模式:")
        data_mode = config.get('data', {}).get('mode', 'auto')
        print(f"  - 模式: {data_mode}")
        
        print(f"\nLLM配置:")
        llm_profiles = config.get('llm_profiles', {})
        print(f"  - 可用配置: {', '.join(llm_profiles.keys())}")
        
        print(f"\n智能体配置:")
        agent_mode = config.get('agents', {}).get('mode', 'dual')
        print(f"  - 模式: {agent_mode}")
        
        print(f"\n回测配置:")
        backtest = config.get('backtest', {})
        print(f"  - 初始资金: ${config.get('portfolio', {}).get('total_cash', 0):,.0f}")
        print(f"  - 佣金: {backtest.get('commission_bps', 0)} bps")
        print(f"  - 滑点: {backtest.get('slippage_bps', 0)} bps")
        
        return config
    except Exception as e:
        print(f"✗ 读取配置文件失败: {e}")
        return None

def test_llm_config(config):
    """测试LLM配置"""
    print("\n" + "=" * 60)
    print("3. LLM配置详细检查")
    print("=" * 60)
    
    if not config:
        print("✗ 无法检查（配置文件未加载）")
        return
    
    llm_profiles = config.get('llm_profiles', {})
    
    for profile_name, profile_config in llm_profiles.items():
        print(f"\n[{profile_name}]")
        print(f"  - Provider: {profile_config.get('provider', 'N/A')}")
        print(f"  - Model: {profile_config.get('model', 'N/A')}")
        print(f"  - Base URL: {profile_config.get('base_url', 'N/A')}")
        print(f"  - Auth Required: {profile_config.get('auth_required', 'N/A')}")
        print(f"  - Timeout: {profile_config.get('timeout_sec', 'N/A')}s")

def test_data_sources():
    """测试数据源配置"""
    print("\n" + "=" * 60)
    print("4. 数据源API配置")
    print("=" * 60)
    
    polygon_key = os.environ.get('POLYGON_API_KEY')
    finnhub_key = os.environ.get('FINNHUB_API_KEY')
    
    print(f"\nPolygon.io:")
    if polygon_key:
        print(f"  ✓ API Key已配置")
        print(f"  - 用途: 股票K线数据、基本面数据")
    else:
        print(f"  ✗ API Key未配置")
        print(f"  - 设置方法: export POLYGON_API_KEY='your_key'")
        print(f"  - 免费注册: https://polygon.io/")
    
    print(f"\nFinnhub.io:")
    if finnhub_key:
        print(f"  ✓ API Key已配置")
        print(f"  - 用途: 公司新闻、财务指标")
    else:
        print(f"  ✗ API Key未配置")
        print(f"  - 设置方法: export FINNHUB_API_KEY='your_key'")
        print(f"  - 免费注册: https://finnhub.io/")

def test_cache_directory():
    """测试缓存目录"""
    print("\n" + "=" * 60)
    print("5. 缓存目录检查")
    print("=" * 60)
    
    cache_dir = project_root / 'storage' / 'cache'
    print(f"\n缓存目录: {cache_dir}")
    
    if cache_dir.exists():
        print(f"✓ 目录存在")
        
        subdirs = list(cache_dir.iterdir())
        if subdirs:
            print(f"  - 子目录数量: {len(subdirs)}")
            print(f"  - 包含: {', '.join([d.name for d in subdirs[:5]])}")
        else:
            print(f"  - 目录为空（首次运行正常）")
    else:
        print(f"✗ 目录不存在（将在首次运行时创建）")

def provide_recommendations():
    """提供配置建议"""
    print("\n" + "=" * 60)
    print("6. 配置建议")
    print("=" * 60)
    
    env_vars = {
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'ZHIPUAI_API_KEY': os.environ.get('ZHIPUAI_API_KEY'),
        'POLYGON_API_KEY': os.environ.get('POLYGON_API_KEY'),
        'FINNHUB_API_KEY': os.environ.get('FINNHUB_API_KEY'),
    }
    
    missing_llm = not env_vars['OPENAI_API_KEY'] and not env_vars['ZHIPUAI_API_KEY']
    missing_data = not env_vars['POLYGON_API_KEY'] or not env_vars['FINNHUB_API_KEY']
    
    if missing_llm:
        print("\n⚠️ LLM API配置缺失")
        print("  至少需要配置以下之一:")
        print("  - OPENAI_API_KEY: 用于OpenAI/DeepSeek等兼容API")
        print("  - ZHIPUAI_API_KEY: 用于智谱AI")
        print("\n  配置方法 (Windows PowerShell):")
        print("    $env:OPENAI_API_KEY='your_key_here'")
        print("  或在config.yaml中设置不需要认证的本地模型:")
        print("    llm_profile: vllm 或 ollama")
    else:
        print("\n✓ LLM API配置正常")
    
    if missing_data:
        print("\n⚠️ 数据源API配置不完整")
        print("  建议配置:")
        print("  - POLYGON_API_KEY (必需，用于股票价格数据)")
        print("  - FINNHUB_API_KEY (必需，用于新闻数据)")
        print("\n  配置方法 (Windows PowerShell):")
        print("    $env:POLYGON_API_KEY='your_key_here'")
        print("    $env:FINNHUB_API_KEY='your_key_here'")
        print("\n  ⚡ 快速测试方法（使用缓存数据）:")
        print("    python -m stockbench.apps.run_backtest --offline")
    else:
        print("\n✓ 数据源API配置完整")
    
    print("\n" + "=" * 60)
    print("下一步建议:")
    print("=" * 60)
    
    if not missing_llm and not missing_data:
        print("\n✓ 配置完整，可以开始回测:")
        print("  bash scripts/run_benchmark.sh")
    elif not missing_data:
        print("\n1. 配置LLM API密钥")
        print("2. 运行回测: bash scripts/run_benchmark.sh")
    elif not missing_llm:
        print("\n1. 配置数据源API密钥")
        print("2. 或使用离线模式测试: python -m stockbench.apps.run_backtest --offline")
    else:
        print("\n选项1: 完整配置（推荐）")
        print("  1. 配置所有API密钥")
        print("  2. 运行: bash scripts/run_benchmark.sh")
        print("\n选项2: 快速测试（使用缓存）")
        print("  1. 仅配置LLM API密钥")
        print("  2. 运行: python -m stockbench.apps.run_backtest --offline")
        print("\n选项3: 本地模型（无需API）")
        print("  1. 启动本地vLLM或Ollama")
        print("  2. 运行: python -m stockbench.apps.run_backtest --llm-profile vllm --offline")

def main():
    print("\n" + "=" * 60)
    print("StockBench API 配置诊断工具")
    print("=" * 60)
    
    env_vars = test_environment_variables()
    config = test_config_file()
    test_llm_config(config)
    test_data_sources()
    test_cache_directory()
    provide_recommendations()
    
    print("\n" + "=" * 60)
    print("诊断完成!")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
