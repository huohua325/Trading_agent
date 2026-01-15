"""
LLM 提供商扩展模块

通过继承 BaseLLMProvider 扩展新的 LLM 提供商，无需修改核心代码。

使用示例:
    # 使用预定义的提供商
    from stockbench.llm.providers import VLLMProvider, OllamaProvider
    
    vllm_client = VLLMProvider()
    cfg = vllm_client.get_default_config()
    
    # 扩展自定义提供商
    from stockbench.llm.providers import BaseLLMProvider
    
    class MyCustomProvider(BaseLLMProvider):
        PROVIDER_NAME = "my_provider"
        DEFAULT_BASE_URL = "https://api.myprovider.com/v1"
        ENV_KEY_NAME = "MY_PROVIDER_API_KEY"
        DEFAULT_MODEL = "my-model"
        
        def _custom_init(self):
            # 自定义初始化逻辑
            pass
"""

from typing import Optional

from stockbench.llm.llm_client import LLMClient, LLMConfig, LLMProvider, PROVIDER_DEFAULTS


class BaseLLMProvider(LLMClient):
    """
    LLM 提供商基类
    
    继承此类可以轻松添加新的 LLM 提供商支持。
    子类只需定义类属性即可完成配置。
    
    类属性:
        PROVIDER_NAME: 提供商名称 (对应 LLMProvider 常量)
        DEFAULT_BASE_URL: 默认 API 地址
        ENV_KEY_NAME: API Key 环境变量名
        DEFAULT_MODEL: 默认模型名
        AUTH_REQUIRED: 是否需要认证
    """
    
    PROVIDER_NAME: str = "custom"
    DEFAULT_BASE_URL: str = ""
    ENV_KEY_NAME: Optional[str] = None
    DEFAULT_MODEL: str = "default"
    AUTH_REQUIRED: bool = True
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化提供商客户端
        
        Args:
            cache_dir: 缓存目录，默认使用 storage/cache/llm
        """
        super().__init__(
            api_key_env=self.ENV_KEY_NAME or "auto",
            cache_dir=cache_dir
        )
        self._custom_init()
    
    def _custom_init(self):
        """
        子类可重写此方法进行自定义初始化
        
        在 __init__ 完成后调用，可用于:
        - 设置额外的客户端配置
        - 初始化特定提供商的 SDK
        - 验证环境变量
        """
        pass
    
    def get_default_config(self) -> LLMConfig:
        """
        获取此提供商的默认配置
        
        Returns:
            LLMConfig: 预填充的配置对象
        """
        return LLMConfig(
            provider=self.PROVIDER_NAME,
            base_url=self.DEFAULT_BASE_URL,
            model=self.DEFAULT_MODEL,
            auth_required=self.AUTH_REQUIRED,
        )


# ==================== 预定义提供商 ====================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI 提供商"""
    
    PROVIDER_NAME = LLMProvider.OPENAI
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    ENV_KEY_NAME = "OPENAI_API_KEY"
    DEFAULT_MODEL = "gpt-4o-mini"
    AUTH_REQUIRED = True


class ZhipuAIProvider(BaseLLMProvider):
    """智谱AI 提供商"""
    
    PROVIDER_NAME = LLMProvider.ZHIPUAI
    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    ENV_KEY_NAME = "ZHIPUAI_API_KEY"
    DEFAULT_MODEL = "glm-4-flash"
    AUTH_REQUIRED = True


class VLLMProvider(BaseLLMProvider):
    """VLLM 本地模型提供商"""
    
    PROVIDER_NAME = LLMProvider.VLLM
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    ENV_KEY_NAME = None  # 本地服务无需 API Key
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    AUTH_REQUIRED = False


class OllamaProvider(BaseLLMProvider):
    """Ollama 本地模型提供商"""
    
    PROVIDER_NAME = LLMProvider.OLLAMA
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    ENV_KEY_NAME = None
    DEFAULT_MODEL = "llama3"
    AUTH_REQUIRED = False


class ModelScopeProvider(BaseLLMProvider):
    """ModelScope 提供商"""
    
    PROVIDER_NAME = LLMProvider.MODELSCOPE
    DEFAULT_BASE_URL = "https://api-inference.modelscope.cn/v1/"
    ENV_KEY_NAME = "MODELSCOPE_API_KEY"
    DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
    AUTH_REQUIRED = True


# 导出所有提供商
__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "ZhipuAIProvider",
    "VLLMProvider",
    "OllamaProvider",
    "ModelScopeProvider",
]
