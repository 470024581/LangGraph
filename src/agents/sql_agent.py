from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from loguru import logger
from pathlib import Path

from ..utils.ollama_provider import OllamaProvider

class SQLAgent:
    """
    一个能够将自然语言转换为SQL查询并执行的代理。
    """
    def __init__(self, ollama_provider: OllamaProvider, db_path: str = "data/database/erp.db"):
        self.llm = ollama_provider.get_llm()
        
        # 检查数据库文件是否存在
        if not Path(db_path).exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        logger.info(f"成功连接到数据库: {db_path}")

        # 创建 SQL Agent
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=True,
        )

    async def run(self, query: str) -> str:
        """
        运行SQL Agent来回答问题。
        
        参数:
            query (str): 用户的自然语言问题。
            
        返回:
            str: Agent生成的回答。
        """
        logger.info(f"SQL Agent 开始处理问题: {query}")
        try:
            # LangChain的SQL Agent目前主要使用同步接口，我们在异步函数中通过await asyncio.to_thread运行
            import asyncio
            
            # 由于 agent_executor.invoke 是一个阻塞操作，我们用 to_thread 包装它
            # 以避免阻塞整个异步事件循环
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"input": query}
            )
            
            answer = result.get("output", "未能从SQL Agent获取到明确的回答。")
            
            # 尝试从输出中提取SQL查询结果，如果它是一个结构化的字符串
            if "SQLResult" in answer:
                try:
                    # 这是一个启发式方法，可能需要根据具体模型输出进行调整
                    final_part = answer.split("SQLResult:")[-1].strip()
                    # 去掉结尾的 "Final Answer:" 等部分
                    if "Final Answer:" in final_part:
                        final_part = final_part.split("Final Answer:")[0].strip()
                    answer = final_part
                except Exception:
                    # 如果解析失败，则返回原始答案
                    pass

            logger.info(f"SQL Agent 处理完成，最终回答: {answer}")
            return answer

        except Exception as e:
            logger.error(f"SQL Agent 运行出错: {e}", exc_info=True)
            return f"在查询数据库时遇到错误: {e}"

# 工厂函数，用于创建SQLAgent实例
def create_sql_agent_instance(ollama_provider: OllamaProvider) -> SQLAgent:
    return SQLAgent(ollama_provider=ollama_provider) 