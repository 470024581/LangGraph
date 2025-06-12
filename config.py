"""
LangGraph 文档问答系统配置文件
支持OpenAI和Ollama本地模型配置
"""
import os
from typing import Dict, Any

# ============== Ollama 本地模型配置 ==============

# Ollama 服务配置
OLLAMA_CONFIG = {
    # Ollama 服务器地址
    "base_url": "http://localhost:11434",
    
    # 聊天模型配置
    "chat_model": "qwen2.5:7b",  # 可选: llama3.1:8b, mistral:7b, gemma2:9b
    
    # 嵌入模型配置（Ollama）
    "ollama_embedding_model": "nomic-embed-text",  # 可选: mxbai-embed-large
    
    # 文档处理嵌入模型（sentence-transformers）
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    
    # 生成参数
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 2048,  # 最大输出token数
    
    # 连接配置
    "timeout": 300,  # 请求超时时间（秒）
    "keep_alive": "5m",  # 模型保持活跃时间
}

# ============== OpenAI 模型配置 ==============

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "chat_model": "gpt-3.5-turbo",
    "embedding_model": "text-embedding-ada-002",
    "temperature": 0.7,
    "max_tokens": 2048,
}

# ============== 系统配置 ==============

SYSTEM_CONFIG = {
    # 模型提供商选择: "ollama" 或 "openai"
    "provider": "ollama",  # 默认使用ollama本地模型
    
    # 文档处理配置
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_document_size": 50 * 1024 * 1024,  # 50MB
    
    # 检索配置
    "retrieval_top_k": 5,
    "similarity_threshold": 0.6,
    
    # 质量控制配置
    "quality_threshold": 8,
    "max_revisions": 3,
    
    # 向量存储配置
    "vector_store_type": "faiss",  # 或 "chroma"
    "persist_directory": "./vector_store",
    
    # 日志配置
    "log_level": "INFO",
    "log_file": "logs/document_qa.log",
}

# ============== 路径配置 ==============

PATHS = {
    "document_dir": "./data/document/",
    "database_dir": "./data/database/",
    "output_dir": "./output/",
    "logs_dir": "./logs/",
    "checkpoints_dir": "./checkpoints/",
}

# ============== Ollama 模型预设配置 ==============

OLLAMA_MODEL_PRESETS = {
    # 轻量级配置（使用tinyllama）
    "light": {
        "chat_model": "tinyllama:latest",
        "ollama_embedding_model": "nomic-embed-text",  # 需要单独安装
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.7,
        "num_predict": 1024,
    },
    
    # 标准配置（使用已安装的mistral）
    "standard": {
        "chat_model": "mistral:latest",
        "ollama_embedding_model": "nomic-embed-text",  # 需要单独安装
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.7,
        "num_predict": 2048,
    },
    
    # 高性能配置（使用phi3）
    "performance": {
        "chat_model": "phi3:mini",
        "ollama_embedding_model": "nomic-embed-text",  # 需要单独安装
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.6,
        "num_predict": 4096,
    },
    
    # 推荐配置（需要下载qwen2.5）
    "recommended": {
        "chat_model": "qwen2.5:7b",  # 中文优化
        "ollama_embedding_model": "nomic-embed-text",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.7,
        "num_predict": 2048,
    },
    
    # 代码专用配置
    "code": {
        "chat_model": "codellama:13b",  # 需要单独安装
        "ollama_embedding_model": "nomic-embed-text",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.2,
        "num_predict": 2048,
    },
}

def get_config(preset: str = "standard") -> Dict[str, Any]:
    """
    获取完整的系统配置
    
    Args:
        preset: 配置预设 ("light", "standard", "performance", "code")
    
    Returns:
        完整的配置字典
    """
    
    # 基础配置
    config = {
        **SYSTEM_CONFIG,
        **PATHS,
    }
    
    # 根据provider选择模型配置
    if config["provider"] == "ollama":
        model_config = OLLAMA_CONFIG.copy()
        
        # 应用预设配置
        if preset in OLLAMA_MODEL_PRESETS:
            model_config.update(OLLAMA_MODEL_PRESETS[preset])
        
        config.update({
            "llm_model": model_config["chat_model"],
            "embedding_model": model_config["embedding_model"],
            "temperature": model_config["temperature"],
            "ollama_base_url": model_config["base_url"],
            "ollama_timeout": model_config["timeout"],
            "ollama_keep_alive": model_config["keep_alive"],
            "max_tokens": model_config.get("num_predict", 2048),
        })
        
    elif config["provider"] == "openai":
        config.update({
            "llm_model": OPENAI_CONFIG["chat_model"],
            "embedding_model": OPENAI_CONFIG["embedding_model"],
            "temperature": OPENAI_CONFIG["temperature"],
            "openai_api_key": OPENAI_CONFIG["api_key"],
            "openai_base_url": OPENAI_CONFIG["base_url"],
            "max_tokens": OPENAI_CONFIG["max_tokens"],
        })
    
    return config

def get_ollama_health_check_config() -> Dict[str, Any]:
    """获取Ollama健康检查配置"""
    return {
        "url": f"{OLLAMA_CONFIG['base_url']}/api/tags",
        "timeout": 10,
        "required_models": [
            OLLAMA_CONFIG["chat_model"],
            OLLAMA_CONFIG["ollama_embedding_model"]
        ]
    }

# ============== 环境变量配置示例 ==============

ENV_TEMPLATE = """
# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# OpenAI配置（备用）
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# 系统配置
MODEL_PROVIDER=ollama
QUALITY_THRESHOLD=8
MAX_REVISIONS=3
RETRIEVAL_TOP_K=5

# 路径配置
DOCUMENT_DIR=./data/document/
DATABASE_DIR=./data/database/
LOG_LEVEL=INFO
"""

if __name__ == "__main__":
    # 示例：打印不同预设的配置
    presets = ["light", "standard", "performance", "code"]
    
    for preset in presets:
        print(f"\n{'='*20} {preset.upper()} 配置 {'='*20}")
        config = get_config(preset)
        
        print(f"聊天模型: {config['llm_model']}")
        print(f"嵌入模型: {config['embedding_model']}")
        print(f"温度参数: {config['temperature']}")
        print(f"最大Token: {config['max_tokens']}")
        print(f"质量阈值: {config['quality_threshold']}")
        
    # 健康检查配置
    print(f"\n{'='*20} 健康检查配置 {'='*20}")
    health_config = get_ollama_health_check_config()
    print(f"检查URL: {health_config['url']}")
    print(f"所需模型: {health_config['required_models']}") 