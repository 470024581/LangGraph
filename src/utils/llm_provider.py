"""
LLM提供商管理器
支持Ollama本地模型和OpenAI云端模型的统一接口
"""
import asyncio
import httpx
from typing import Dict, Any, Optional, AsyncGenerator, List
from abc import ABC, abstractmethod

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from loguru import logger

try:
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama 相关包未安装，将无法使用本地模型")


class BaseLLMProvider(ABC):
    """LLM提供商基类"""
    
    @abstractmethod
    async def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """生成回答"""
        pass
    
    @abstractmethod
    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncGenerator[str, None]:
        """流式生成回答"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama本地模型提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        if not OLLAMA_AVAILABLE:
            raise ImportError("请安装 langchain-community 以使用 Ollama 模型")
        
        self.config = config
        self.base_url = config.get("ollama_base_url", "http://localhost:11434")
        self.model_name = config.get("llm_model", "qwen2.5:7b")
        self.timeout = config.get("ollama_timeout", 300)
        
        # 初始化Ollama聊天模型
        self.chat_model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=config.get("temperature", 0.7),
            num_predict=config.get("max_tokens", 2048),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k", 40),
            keep_alive=config.get("ollama_keep_alive", "5m"),
            timeout=self.timeout,
        )
        
        logger.info(f"初始化 Ollama 模型: {self.model_name} @ {self.base_url}")
    
    async def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """生成回答"""
        try:
            # Ollama 支持异步调用
            response = await self.chat_model.agenerate([messages])
            result = response.generations[0][0].text.strip()
            
            logger.info(f"Ollama 生成完成，输出长度: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Ollama 生成失败: {str(e)}")
            return f"生成失败: {str(e)}"
    
    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncGenerator[str, None]:
        """流式生成回答"""
        try:
            async for chunk in self.chat_model.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Ollama 流式生成失败: {str(e)}")
            yield f"流式生成失败: {str(e)}"
    
    async def health_check(self) -> bool:
        """检查Ollama服务健康状态"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # 检查服务状态
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # 检查模型是否可用
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                if self.model_name not in available_models:
                    logger.warning(f"模型 {self.model_name} 不在可用列表中: {available_models}")
                    return False
                
                logger.info(f"Ollama 健康检查通过，可用模型: {len(available_models)}")
                return True
                
        except Exception as e:
            logger.error(f"Ollama 健康检查失败: {str(e)}")
            return False
    
    async def pull_model(self, model_name: Optional[str] = None) -> bool:
        """拉取模型"""
        target_model = model_name or self.model_name
        
        try:
            async with httpx.AsyncClient(timeout=600) as client:  # 拉取模型可能需要很长时间
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": target_model}
                )
                
                if response.status_code == 200:
                    logger.info(f"模型 {target_model} 拉取成功")
                    return True
                else:
                    logger.error(f"模型 {target_model} 拉取失败: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"拉取模型异常: {str(e)}")
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI云端模型提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("llm_model", "gpt-3.5-turbo")
        
        # 初始化OpenAI聊天模型
        self.chat_model = ChatOpenAI(
            model_name=self.model_name,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
            streaming=True,
            api_key=config.get("openai_api_key"),
            base_url=config.get("openai_base_url")
        )
        
        logger.info(f"初始化 OpenAI 模型: {self.model_name}")
    
    async def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """生成回答"""
        try:
            with get_openai_callback() as cb:
                response = await self.chat_model.agenerate([messages])
                
                logger.info(f"OpenAI 调用统计 - Tokens: {cb.total_tokens}, 成本: ${cb.total_cost:.4f}")
            
            result = response.generations[0][0].text.strip()
            return result
            
        except Exception as e:
            logger.error(f"OpenAI 生成失败: {str(e)}")
            return f"生成失败: {str(e)}"
    
    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncGenerator[str, None]:
        """流式生成回答"""
        try:
            async for chunk in self.chat_model.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI 流式生成失败: {str(e)}")
            yield f"流式生成失败: {str(e)}"
    
    async def health_check(self) -> bool:
        """检查OpenAI API健康状态"""
        try:
            # 尝试一个简单的API调用
            test_messages = [HumanMessage(content="Hello")]
            response = await self.generate(test_messages)
            
            if "失败" in response:
                return False
            
            logger.info("OpenAI API 健康检查通过")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI 健康检查失败: {str(e)}")
            return False


class LLMProviderManager:
    """LLM提供商管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_type = config.get("provider", "ollama")
        self.provider: Optional[BaseLLMProvider] = None
        
        # 初始化提供商
        self._initialize_provider()
    
    def _initialize_provider(self):
        """初始化LLM提供商"""
        try:
            if self.provider_type == "ollama":
                self.provider = OllamaProvider(self.config)
            elif self.provider_type == "openai":
                self.provider = OpenAIProvider(self.config)
            else:
                raise ValueError(f"不支持的提供商类型: {self.provider_type}")
            
            logger.info(f"LLM提供商初始化成功: {self.provider_type}")
            
        except Exception as e:
            logger.error(f"LLM提供商初始化失败: {str(e)}")
            # 回退到OpenAI（如果配置可用）
            if self.provider_type != "openai" and self.config.get("openai_api_key"):
                logger.info("尝试回退到 OpenAI 提供商")
                self.provider_type = "openai"
                self.provider = OpenAIProvider(self.config)
    
    async def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """生成回答"""
        if not self.provider:
            return "LLM提供商未初始化"
        
        return await self.provider.generate(messages, **kwargs)
    
    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncGenerator[str, None]:
        """流式生成回答"""
        if not self.provider:
            yield "LLM提供商未初始化"
            return
        
        async for chunk in self.provider.astream(messages, **kwargs):
            yield chunk
    
    async def health_check(self) -> bool:
        """健康检查"""
        if not self.provider:
            return False
        
        return await self.provider.health_check()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """获取提供商信息"""
        return {
            "type": self.provider_type,
            "model": self.config.get("llm_model"),
            "base_url": self.config.get(f"{self.provider_type}_base_url"),
            "available": self.provider is not None
        }
    
    async def switch_provider(self, provider_type: str, config_updates: Dict[str, Any] = None):
        """切换提供商"""
        if config_updates:
            self.config.update(config_updates)
        
        self.provider_type = provider_type
        self.config["provider"] = provider_type
        
        self._initialize_provider()
        
        # 执行健康检查
        if await self.health_check():
            logger.info(f"成功切换到 {provider_type} 提供商")
        else:
            logger.warning(f"切换到 {provider_type} 提供商后健康检查失败")


def create_llm_provider(config: Dict[str, Any]) -> LLMProviderManager:
    """创建LLM提供商管理器"""
    return LLMProviderManager(config)


# ============== 便捷函数 ==============

async def test_all_providers(config: Dict[str, Any]):
    """测试所有可用的提供商"""
    results = {}
    
    # 测试Ollama
    if OLLAMA_AVAILABLE:
        try:
            ollama_config = config.copy()
            ollama_config["provider"] = "ollama"
            ollama_manager = create_llm_provider(ollama_config)
            
            health = await ollama_manager.health_check()
            results["ollama"] = {
                "available": health,
                "model": config.get("llm_model", "qwen2.5:7b"),
                "base_url": config.get("ollama_base_url", "http://localhost:11434")
            }
            
        except Exception as e:
            results["ollama"] = {"available": False, "error": str(e)}
    
    # 测试OpenAI
    if config.get("openai_api_key"):
        try:
            openai_config = config.copy()
            openai_config["provider"] = "openai"
            openai_manager = create_llm_provider(openai_config)
            
            health = await openai_manager.health_check()
            results["openai"] = {
                "available": health,
                "model": config.get("llm_model", "gpt-3.5-turbo")
            }
            
        except Exception as e:
            results["openai"] = {"available": False, "error": str(e)}
    
    return results


if __name__ == "__main__":
    # 测试脚本
    import asyncio
    from config import get_config
    
    async def main():
        config = get_config("standard")
        
        print("🧪 测试所有LLM提供商...")
        results = await test_all_providers(config)
        
        for provider, info in results.items():
            status = "✅" if info.get("available") else "❌"
            print(f"{status} {provider.upper()}: {info}")
        
        # 测试生成
        if any(info.get("available") for info in results.values()):
            manager = create_llm_provider(config)
            test_message = [HumanMessage(content="你好，请简单介绍一下自己。")]
            
            print(f"\n🤖 测试生成功能...")
            response = await manager.generate(test_message)
            print(f"回答: {response}")
    
    asyncio.run(main()) 