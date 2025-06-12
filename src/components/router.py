from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from loguru import logger

from ..utils.ollama_provider import OllamaProvider

# 定义输出结构
class RouteQuery(BaseModel):
    """根据用户问题选择路由到 'rag' (文档问答) 或 'sql' (数据库查询)。"""
    datasource: str = Field(
        description="给定用户问题，选择 'rag' 或 'sql'。",
        enum=["sql", "rag"],
    )

class Router:
    """
    一个智能路由，用于判断用户问题应该由RAG流程还是SQL Agent处理。
    """
    def __init__(self, ollama_provider: OllamaProvider):
        llm = ollama_provider.get_llm()
        
        # 将LLM与结构化输出工具绑定
        structured_llm = llm.with_structured_output(RouteQuery)
        
        # 创建路由提示模板
        self.prompt = PromptTemplate(
            template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个善于将用户问题路由到向量存储（RAG）或SQL数据库的专家。

向量存储包含了关于销售报告分析、产品介绍和公司通用信息的非结构化文档。
SQL数据库包含了关于产品、库存和销售记录的结构化数据表。

你需要根据用户的问题，决定最合适的数据源。
- 如果问题涉及到具体的、精确的数据查询、聚合或计算（例如"库存最少的产品"、"总销售额"、"上周的订单数量"），请选择 'sql'。
- 如果问题是关于文档内容的理解、总结或开放式提问（例如"总结一下第三季度的销售报告"、"公司未来的发展方向是什么？"），请选择 'rag'。

<|eot_id|><|start_header_id|>user<|end_header_id|>
用户问题: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
            input_variables=["question"],
        )
        
        # 构建处理链
        self.chain = self.prompt | structured_llm

    async def route(self, question: str) -> str:
        """
        根据问题进行路由决策。

        参数:
            question (str): 用户的问题。

        返回:
            str: 'sql' 或 'rag'
        """
        logger.info(f"开始为问题路由: '{question}'")
        result = await self.chain.ainvoke({"question": question})
        logger.info(f"路由决策: '{result.datasource}'")
        return result.datasource 