"""
Ollama本地模型提供商
支持Ollama本地模型的统一接口
"""
import asyncio
import httpx
from typing import Dict, Any, Optional, AsyncGenerator, List

from langchain.schema import BaseMessage, HumanMessage
from loguru import logger

try:
    # 优先使用新的 langchain-ollama 包
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        # 回退到旧的包
        from langchain_community.chat_models import ChatOllama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
        logger.warning("Ollama 相关包未安装，将无法使用本地模型")


class OllamaProvider:
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
    
    def get_llm(self) -> ChatOllama:
        """返回初始化的ChatOllama实例"""
        return self.chat_model

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


def create_ollama_provider(config: Dict[str, Any]) -> OllamaProvider:
    """创建Ollama提供商"""
    return OllamaProvider(config)


async def test_ollama_connection(config: Dict[str, Any]) -> Dict[str, Any]:
    """测试Ollama连接"""
    result = {
        "connected": False,
        "models": [],
        "error": None
    }
    
    try:
        base_url = config.get("ollama_base_url", "http://localhost:11434")
        
        async with httpx.AsyncClient(timeout=10) as client:
            # 检查服务状态
            response = await client.get(f"{base_url}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                result["connected"] = True
                result["models"] = [model["name"] for model in data.get("models", [])]
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text}"
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


if __name__ == "__main__":
    # 测试脚本
    import asyncio
    
    async def main():
        from config import get_config
        
        config = get_config("standard")
        
        print("🧪 测试Ollama连接...")
        test_result = await test_ollama_connection(config)
        
        if test_result["connected"]:
            print(f"✅ Ollama连接成功")
            print(f"📦 可用模型: {test_result['models']}")
            
            # 测试模型
            if config.get("llm_model") in test_result["models"]:
                provider = create_ollama_provider(config)
                test_message = [HumanMessage(content="你好，请简单介绍一下自己。")]
                
                print(f"\n🤖 测试模型生成...")
                response = await provider.generate(test_message)
                print(f"回答: {response}")
            else:
                print(f"⚠️ 所需模型 {config.get('llm_model')} 未安装")
        else:
            print(f"❌ Ollama连接失败: {test_result['error']}")
    
    asyncio.run(main()) 