"""
LLM 自动检测机制测试

测试 LLMConfig 的自动检测功能和 LLMClient 的动态 API Key 解析。

运行方式:
    pytest stockbench/llm/tests/test_auto_detect.py -v
"""
import os
import pytest
from unittest.mock import patch

from stockbench.llm.llm_client import LLMConfig, LLMClient, LLMProvider, PROVIDER_DEFAULTS


class TestLLMProviderConstants:
    """测试 LLMProvider 常量定义"""
    
    def test_provider_constants_exist(self):
        """验证所有提供商常量已定义"""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ZHIPUAI == "zhipuai"
        assert LLMProvider.VLLM == "vllm"
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.MODELSCOPE == "modelscope"
        assert LLMProvider.LOCAL == "local"
        assert LLMProvider.AUTO == "auto"
    
    def test_provider_defaults_complete(self):
        """验证所有提供商都有默认配置"""
        required_keys = ["base_url", "default_model", "auth_required"]
        for provider in [LLMProvider.OPENAI, LLMProvider.ZHIPUAI, 
                         LLMProvider.VLLM, LLMProvider.OLLAMA, 
                         LLMProvider.MODELSCOPE, LLMProvider.LOCAL]:
            assert provider in PROVIDER_DEFAULTS, f"Missing defaults for {provider}"
            for key in required_keys:
                assert key in PROVIDER_DEFAULTS[provider], f"Missing {key} for {provider}"


class TestAutoDetectProvider:
    """测试自动检测提供商"""
    
    def test_detect_openai_by_env(self):
        """通过 OPENAI_API_KEY 检测"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.OPENAI
    
    def test_detect_zhipuai_by_env(self):
        """通过 ZHIPUAI_API_KEY 检测 (优先于 OPENAI_API_KEY)"""
        with patch.dict(os.environ, {
            "ZHIPUAI_API_KEY": "zhipu-key",
            "OPENAI_API_KEY": "sk-openai-key"
        }, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.ZHIPUAI
    
    def test_detect_modelscope_by_env(self):
        """通过 MODELSCOPE_API_KEY 检测"""
        with patch.dict(os.environ, {"MODELSCOPE_API_KEY": "ms-key"}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.MODELSCOPE
    
    def test_detect_vllm_by_base_url(self):
        """通过 localhost:8000 检测 VLLM"""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig(provider="auto", base_url="http://localhost:8000/v1")
            assert cfg.provider == LLMProvider.VLLM
    
    def test_detect_ollama_by_base_url(self):
        """通过 localhost:11434 检测 Ollama"""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig(provider="auto", base_url="http://localhost:11434/v1")
            assert cfg.provider == LLMProvider.OLLAMA
    
    def test_detect_local_by_localhost(self):
        """通过 localhost (非标准端口) 检测为 local"""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig(provider="auto", base_url="http://localhost:5000/v1")
            assert cfg.provider == LLMProvider.LOCAL
    
    def test_detect_by_llm_base_url_env(self):
        """通过 LLM_BASE_URL 环境变量检测"""
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.VLLM
    
    def test_detect_zhipuai_by_base_url(self):
        """通过 open.bigmodel.cn 域名检测"""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig(provider="auto", base_url="https://open.bigmodel.cn/api/paas/v4")
            assert cfg.provider == LLMProvider.ZHIPUAI
    
    def test_default_to_openai(self):
        """无任何环境变量时默认返回 openai"""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig(provider="auto")
            assert cfg.provider == LLMProvider.OPENAI


class TestExplicitProvider:
    """测试显式指定提供商"""
    
    def test_explicit_provider_overrides_auto(self):
        """显式指定 provider 应覆盖自动检测"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            cfg = LLMConfig(provider="zhipuai")
            assert cfg.provider == "zhipuai"
    
    def test_explicit_vllm_provider(self):
        """显式指定 vllm provider"""
        cfg = LLMConfig(provider="vllm")
        assert cfg.provider == "vllm"
        assert cfg.auth_required == False
    
    def test_explicit_ollama_provider(self):
        """显式指定 ollama provider"""
        cfg = LLMConfig(provider="ollama")
        assert cfg.provider == "ollama"
        assert cfg.auth_required == False


class TestResolveDefaults:
    """测试默认值解析"""
    
    def test_resolve_base_url_for_openai(self):
        """OpenAI provider 应解析默认 base_url"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            cfg = LLMConfig(provider="openai")
            assert "api.openai.com" in cfg.base_url or cfg.base_url != ""
    
    def test_resolve_base_url_for_vllm(self):
        """VLLM provider 应解析本地 base_url"""
        cfg = LLMConfig(provider="vllm")
        assert "localhost:8000" in cfg.base_url
    
    def test_resolve_model_for_openai(self):
        """OpenAI provider 应有默认 model"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            cfg = LLMConfig(provider="openai")
            assert cfg.model != ""
    
    def test_resolve_model_for_ollama(self):
        """Ollama provider 应有默认 model"""
        cfg = LLMConfig(provider="ollama")
        assert cfg.model == "llama3"
    
    def test_env_overrides_defaults(self):
        """环境变量应覆盖默认值"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "LLM_BASE_URL": "https://custom.api.com/v1",
            "LLM_MODEL_ID": "custom-model"
        }, clear=True):
            cfg = LLMConfig(provider="openai")
            assert cfg.base_url == "https://custom.api.com/v1"
            assert cfg.model == "custom-model"
    
    def test_local_no_auth_required(self):
        """本地服务不需要认证"""
        cfg = LLMConfig(provider="vllm")
        assert cfg.auth_required == False
        
        cfg = LLMConfig(provider="ollama")
        assert cfg.auth_required == False


class TestLLMClientAPIKeyResolution:
    """测试 LLMClient API Key 解析"""
    
    def test_get_api_key_for_openai(self):
        """OpenAI 使用 OPENAI_API_KEY"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
            client = LLMClient()
            cfg = LLMConfig(provider="openai")
            api_key = client._get_api_key(cfg)
            assert api_key == "sk-test-key"
    
    def test_get_api_key_for_zhipuai(self):
        """智谱AI 使用 ZHIPUAI_API_KEY"""
        with patch.dict(os.environ, {
            "ZHIPUAI_API_KEY": "zhipu-key",
            "OPENAI_API_KEY": "sk-openai"
        }, clear=True):
            client = LLMClient()
            cfg = LLMConfig(provider="zhipuai")
            api_key = client._get_api_key(cfg)
            assert api_key == "zhipu-key"
    
    def test_get_api_key_for_local_no_auth(self):
        """本地服务无需 API Key"""
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient()
            cfg = LLMConfig(provider="vllm", auth_required=False)
            api_key = client._get_api_key(cfg)
            assert api_key == "local-no-key"
    
    def test_fallback_to_llm_api_key(self):
        """降级到通用 LLM_API_KEY"""
        with patch.dict(os.environ, {"LLM_API_KEY": "generic-key"}, clear=True):
            client = LLMClient()
            cfg = LLMConfig(provider="openai")
            api_key = client._get_api_key(cfg)
            assert api_key == "generic-key"


class TestBackwardCompatibility:
    """测试向后兼容性"""
    
    def test_old_api_key_env_param(self):
        """旧的 api_key_env 参数仍然有效"""
        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-value"}, clear=True):
            client = LLMClient(api_key_env="MY_CUSTOM_KEY")
            assert client.api_key == "custom-value"
    
    def test_explicit_config_values(self):
        """显式配置值应被保留"""
        cfg = LLMConfig(
            provider="openai",
            base_url="https://custom.api.com/v1",
            model="custom-model",
            temperature=0.7
        )
        assert cfg.provider == "openai"
        assert cfg.base_url == "https://custom.api.com/v1"
        assert cfg.model == "custom-model"
        assert cfg.temperature == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
