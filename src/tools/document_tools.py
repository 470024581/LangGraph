"""
文档工具链
提供文档摘要、翻译、分析等工具函数
"""
from typing import List, Dict, Any, Optional
import asyncio

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_community.callbacks.manager import get_openai_callback
from loguru import logger

from ..schemas.agent_state import DocumentChunk, AgentState
from ..utils.ollama_provider import OllamaProvider
from ..components.document_processor import DocumentProcessor


class DocumentTools:
    """
    一个集成了文档处理、向量存储和信息检索的工具集。
    """
    
    def __init__(self, ollama_provider: OllamaProvider):
        """初始化文档工具集"""
        self.document_processor = DocumentProcessor()
        # vector_store 在 document_processor 内部初始化，在其处理文档后可用
        self.vector_store = None 
        
        self.llm = ollama_provider.get_llm()
        logger.info(f"文档工具使用 Ollama 模型: {ollama_provider.model_name}")
    
    async def summarize_document(self, chunks: List[DocumentChunk], max_length: int = 500) -> str:
        """文档摘要生成"""
        if not chunks:
            return "无文档内容可摘要"
        
        # 合并文档内容
        combined_content = "\n\n".join([chunk.content for chunk in chunks[:10]])  # 限制块数量
        
        # 构建摘要prompt
        prompt = f"""请对以下文档内容进行摘要，要求：
1. 提取主要观点和关键信息
2. 保持逻辑清晰，结构完整
3. 控制在{max_length}字以内
4. 使用中文

文档内容：
{combined_content}

摘要："""

        try:
            messages = [HumanMessage(content=prompt)]
            
            summary = await self.llm.generate(messages)
            
            logger.info(f"文档摘要生成完成，长度: {len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            return f"摘要生成失败: {str(e)}"
    
    async def translate_text(self, text: str, target_language: str = "英文") -> str:
        """文本翻译"""
        if not text.strip():
            return "无内容可翻译"
        
        prompt = f"""请将以下文本翻译成{target_language}，保持原意并确保翻译自然流畅：

原文：
{text}

翻译："""

        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            translation = response.generations[0][0].text.strip()
            
            logger.info(f"文本翻译完成，目标语言: {target_language}")
            return translation
            
        except Exception as e:
            logger.error(f"翻译失败: {str(e)}")
            return f"翻译失败: {str(e)}"
    
    async def extract_key_points(self, chunks: List[DocumentChunk]) -> List[str]:
        """提取关键要点"""
        if not chunks:
            return ["无文档内容可分析"]
        
        combined_content = "\n\n".join([chunk.content for chunk in chunks[:5]])
        
        prompt = f"""请从以下文档内容中提取5-10个关键要点，每个要点用一句话概括：

文档内容：
{combined_content}

关键要点：
1."""

        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            result = response.generations[0][0].text.strip()
            
            # 解析要点列表
            points = []
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')) or 
                           line.startswith(('•', '-', '*'))):
                    # 去除序号
                    point = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if point:
                        points.append(point)
            
            logger.info(f"关键要点提取完成，共{len(points)}个要点")
            return points if points else [result]
            
        except Exception as e:
            logger.error(f"关键要点提取失败: {str(e)}")
            return [f"关键要点提取失败: {str(e)}"]
    
    async def analyze_document_structure(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """分析文档结构"""
        if not chunks:
            return {"error": "无文档内容可分析"}
        
        # 统计信息
        total_chunks = len(chunks)
        total_length = sum(len(chunk.content) for chunk in chunks)
        
        # 按文件类型分组
        file_types = {}
        for chunk in chunks:
            file_type = chunk.metadata.get("file_type", "unknown")
            if file_type not in file_types:
                file_types[file_type] = []
            file_types[file_type].append(chunk)
        
        # 按来源文件分组
        source_files = {}
        for chunk in chunks:
            source = chunk.source_file
            if source not in source_files:
                source_files[source] = []
            source_files[source].append(chunk)
        
        structure = {
            "total_chunks": total_chunks,
            "total_length": total_length,
            "average_chunk_length": total_length // total_chunks if total_chunks > 0 else 0,
            "file_types": {ft: len(chunks) for ft, chunks in file_types.items()},
            "source_files": {sf: len(chunks) for sf, chunks in source_files.items()},
            "has_database_content": any(chunk.metadata.get("source_type") == "database" for chunk in chunks)
        }
        
        logger.info(f"文档结构分析完成: {total_chunks}个块，{len(source_files)}个源文件")
        return structure
    
    async def generate_questions(self, chunks: List[DocumentChunk], count: int = 5) -> List[str]:
        """基于文档内容生成相关问题"""
        if not chunks:
            return ["无文档内容可生成问题"]
        
        combined_content = "\n\n".join([chunk.content for chunk in chunks[:8]])
        
        prompt = f"""基于以下文档内容，生成{count}个相关的问题，这些问题应该：
1. 涵盖文档的主要内容
2. 具有实际意义和价值
3. 可以通过文档内容回答
4. 使用中文

文档内容：
{combined_content}

相关问题：
1."""

        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            result = response.generations[0][0].text.strip()
            
            # 解析问题列表
            questions = []
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line and any(line.startswith(f'{i}.') for i in range(1, count + 2)):
                    # 去除序号
                    question = line.split('.', 1)[-1].strip()
                    if question and question.endswith('?'):
                        questions.append(question)
            
            logger.info(f"相关问题生成完成，共{len(questions)}个问题")
            return questions if questions else [result]
            
        except Exception as e:
            logger.error(f"问题生成失败: {str(e)}")
            return [f"问题生成失败: {str(e)}"]
    
    async def compare_documents(self, chunks_a: List[DocumentChunk], chunks_b: List[DocumentChunk]) -> str:
        """比较两个文档的异同"""
        if not chunks_a or not chunks_b:
            return "需要两个文档才能进行比较"
        
        content_a = "\n".join([chunk.content for chunk in chunks_a[:5]])
        content_b = "\n".join([chunk.content for chunk in chunks_b[:5]])
        
        prompt = f"""请比较以下两个文档的异同，分析它们的：
1. 共同点
2. 不同点
3. 各自的特色
4. 整体关系

文档A：
{content_a}

文档B：
{content_b}

比较分析："""

        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            comparison = response.generations[0][0].text.strip()
            
            logger.info("文档比较完成")
            return comparison
            
        except Exception as e:
            logger.error(f"文档比较失败: {str(e)}")
            return f"文档比较失败: {str(e)}"
    
    async def extract_entities(self, chunks: List[DocumentChunk]) -> Dict[str, List[str]]:
        """提取文档中的实体（人名、地名、机构等）"""
        if not chunks:
            return {"error": ["无文档内容可分析"]}
        
        combined_content = "\n\n".join([chunk.content for chunk in chunks[:5]])
        
        prompt = f"""请从以下文档中提取重要的实体信息，包括：
- 人名
- 地名
- 机构名
- 日期时间
- 数字金额
- 专业术语

文档内容：
{combined_content}

请按类别列出提取的实体：

人名：
地名：
机构名：
日期时间：
数字金额：
专业术语："""

        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            result = response.generations[0][0].text.strip()
            
            # 解析实体
            entities = {
                "人名": [],
                "地名": [],
                "机构名": [],
                "日期时间": [],
                "数字金额": [],
                "专业术语": []
            }
            
            current_category = None
            lines = result.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.endswith('：') or line.endswith(':'):
                    category = line.rstrip('：:')
                    if category in entities:
                        current_category = category
                elif current_category and line:
                    # 移除序号和特殊符号
                    entity = line.lstrip('- • * 1234567890. ').strip()
                    if entity:
                        entities[current_category].append(entity)
            
            logger.info(f"实体提取完成: {sum(len(v) for v in entities.values())}个实体")
            return entities
            
        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")
            return {"error": [f"实体提取失败: {str(e)}"]}

    async def process_and_store_documents(self, state: AgentState) -> AgentState:
        """处理并存储文档，然后更新自身的向量存储引用"""
        processed_state = await self.document_processor.process_documents(state)
        # 在处理完文档后，从 document_processor 获取向量存储
        self.vector_store = self.document_processor.vector_store
        logger.info("向量存储已在 DocumentTools 中更新。")
        return processed_state

    async def search_relevant_info(self, state: AgentState) -> AgentState:
        """
        根据用户问题从向量存储和数据库中检索信息。
        """
        logger.info(f"开始检索相关信息，查询: '{state.user_query}'")
        logger.info(f"检索参数 - top_k: {state.retrieval_top_k}")
        
        # 确保向量存储已准备就绪
        if not self.vector_store:
            logger.warning("向量存储尚未就绪，无法进行相似性搜索。可能需要先运行文档处理。")
            state.retrieved_info = "错误：向量存储未就绪。"
            return state

        logger.info(f"向量存储已就绪，文档块总数: {len(state.doc_chunks)}")

        # 1. 向量存储相似性搜索
        logger.info("开始向量相似性搜索...")
        similar_results = self.document_processor.search_similar_chunks(
            state.user_query, 
            top_k=state.retrieval_top_k
        )
        
        logger.info(f"向量搜索返回 {len(similar_results)} 个结果")
        
        # 2. 数据库信息检索 (这里可以添加根据 state.database_schema 的检索逻辑)
        # ... (暂未实现)

        # 3. 组合检索到的信息
        context = "从文档中检索到的信息：\n"
        relevant_chunks = []
        
        if similar_results:
            logger.info("处理向量搜索结果...")
            # 根据索引获取文档块内容
            all_chunks = state.doc_chunks
            for idx, score in similar_results:
                if idx < len(all_chunks):
                    chunk = all_chunks[idx]
                    relevant_chunks.append(chunk)
                    
                    # 详细分析每个检索结果
                    chunk_type = chunk.metadata.get('file_type', 'unknown')
                    logger.info(f"结果 {len(relevant_chunks)}: 索引={idx}, 相似度={score:.4f}")
                    logger.info(f"  来源文件: {chunk.source_file}")
                    logger.info(f"  文件类型: {chunk_type}")
                    logger.info(f"  内容长度: {len(chunk.content)} 字符")
                    logger.debug(f"  内容预览: {chunk.content[:100]}...")
                    
                    # 特别检查PDF内容
                    if chunk_type == 'pdf':
                        logger.info(f"  📄 这是PDF内容块")
                        if 'longliang' in chunk.content.lower() or 'long liang' in chunk.content.lower():
                            logger.info(f"  ✅ 发现包含 LongLiang 的内容！")
                        else:
                            logger.debug(f"  ❌ 不包含 LongLiang")
                    elif chunk_type == 'csv':
                        logger.debug(f"  📊 这是CSV数据块")
                    else:
                        logger.debug(f"  📝 文件类型: {chunk_type}")
                    
                    context += f"- 来源: {chunk.source_file} ({chunk_type})\n  内容: {chunk.content}\n\n"
                else:
                    logger.warning(f"索引 {idx} 超出文档块范围 (总数: {len(all_chunks)})")
        else:
            logger.warning("向量搜索没有返回任何结果")
            context += "未找到相关文档内容。\n"

        # 检查是否找到了相关内容
        found_relevant = any(
            "longliang" in chunk.content.lower() or "long liang" in chunk.content.lower()
            for chunk in relevant_chunks
        )
        
        # 如果向量搜索没有找到相关内容，尝试关键词搜索
        if not found_relevant and ("longliang" in state.user_query.lower() or "long liang" in state.user_query.lower()):
            logger.info("向量搜索未找到LongLiang相关内容，启动关键词搜索...")
            
            # 关键词搜索
            keyword_chunks = []
            search_terms = ["longliang", "long liang", "liang"]
            
            for chunk in state.doc_chunks:
                content_lower = chunk.content.lower()
                if any(term in content_lower for term in search_terms):
                    keyword_chunks.append(chunk)
                    logger.info(f"✅ 关键词搜索找到匹配: {chunk.source_file}")
                    logger.debug(f"   内容预览: {chunk.content[:200]}...")
            
            if keyword_chunks:
                logger.info(f"关键词搜索找到 {len(keyword_chunks)} 个相关文档块")
                # 使用关键词搜索结果替换或补充向量搜索结果
                relevant_chunks = keyword_chunks[:state.retrieval_top_k]
            else:
                logger.warning("关键词搜索也未找到LongLiang相关内容")
        
        logger.info(f"检索完成，找到 {len(relevant_chunks)} 个相关文档块")
        
        # 更新状态
        state.retrieved_info = context
        if relevant_chunks:
            from ..schemas.agent_state import RetrievalResult
            state.retrieval_results = RetrievalResult(
                chunks=relevant_chunks[:state.retrieval_top_k],
                scores=[1.0] * len(relevant_chunks[:state.retrieval_top_k]),
                query=state.user_query
            )
            state.relevant_context = context
        
        return state


# 工具函数
async def batch_process_documents(documents: List[List[DocumentChunk]], tool_function, **kwargs) -> List[Any]:
    """批量处理多个文档"""
    tools = DocumentTools()
    
    tasks = []
    for doc_chunks in documents:
        task = getattr(tools, tool_function)(doc_chunks, **kwargs)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


def create_document_tools(ollama_provider: OllamaProvider) -> DocumentTools:
    """创建文档工具实例"""
    return DocumentTools(ollama_provider) 