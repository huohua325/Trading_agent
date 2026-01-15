"""
TradingAgent LLM Module

Provides large language model client and configuration management functionality.

Part 1 升级内容:
- LLMProvider: 提供商常量 (openai/zhipuai/vllm/ollama/modelscope/local/auto)
- PROVIDER_DEFAULTS: 各提供商默认配置
- _auto_detect_provider: 自动检测提供商函数
"""

from .llm_client import (
    LLMConfig, 
    LLMClient,
    LLMProvider,
    PROVIDER_DEFAULTS,
    _auto_detect_provider,
)

__all__ = [
    "LLMConfig",
    "LLMClient",
    "LLMProvider",
    "PROVIDER_DEFAULTS",
    "_auto_detect_provider",
]
