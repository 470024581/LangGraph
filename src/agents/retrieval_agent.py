"""
检索Agent
实现RAG（检索增强生成）功能，从文档和数据库中检索相关信息
"""
import sqlite3
from typing import List, Dict, Any
from pathlib import Path

from langchain.schema import Document
from loguru import logger

from ..schemas.agent_state import AgentState, RetrievalResult, DocumentChunk, ProcessingStatus
from ..components.document_processor import DocumentProcessor


class RetrievalAgent:
    """检索Agent"""
    
    def __init__(self, document_processor: DocumentProcessor = None):
        """初始化检索Agent"""
        self.document_processor = document_processor or DocumentProcessor()
        
    async def retrieve_information(self, state: AgentState) -> AgentState:
        """检索相关信息的主入口函数"""
        try:
            state.processing_status = ProcessingStatus.PROCESSING
            state.current_step = "retrieval"
            
            # 执行文档检索
            doc_results = await self._retrieve_from_documents(state)
            
            # 执行数据库检索
            db_results = await self._retrieve_from_database(state)
            
            # 合并检索结果
            all_chunks = doc_results + db_results
            
            # 创建检索结果
            retrieval_result = RetrievalResult(
                chunks=all_chunks[:state.retrieval_top_k],
                scores=[1.0] * len(all_chunks[:state.retrieval_top_k]),
                query=state.user_query
            )
            
            state.retrieval_results = retrieval_result
            
            # 构建相关上下文
            state.relevant_context = self._build_context(all_chunks[:state.retrieval_top_k])
            
            state.processing_status = ProcessingStatus.COMPLETED
            state.current_step = "answer_generation"
            
            logger.info(f"检索完成，共找到 {len(all_chunks)} 个相关块")
            return state
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"检索失败: {str(e)}"
            return state
    
    async def _retrieve_from_documents(self, state: AgentState) -> List[DocumentChunk]:
        """从文档中检索相关信息"""
        if not state.documents_loaded or not state.doc_chunks:
            logger.warning("文档未加载或为空")
            return []
        
        logger.info(f"开始从 {len(state.doc_chunks)} 个文档块中检索相关信息")
        logger.info(f"检索查询: '{state.user_query}'")
        
        try:
            # 使用向量相似度搜索
            similar_indices = self.document_processor.search_similar_chunks(
                state.user_query, 
                top_k=state.retrieval_top_k
            )
            
            logger.info(f"向量搜索返回 {len(similar_indices)} 个结果")
            
            relevant_chunks = []
            for idx, score in similar_indices:
                if idx < len(state.doc_chunks):
                    chunk = state.doc_chunks[idx]
                    logger.debug(f"检索结果 - 索引: {idx}, 分数: {score:.4f}, 来源: {chunk.source_file}")
                    logger.debug(f"内容预览: {chunk.content[:100]}...")
                    
                    # 可以根据分数进行过滤
                    if score > 0.3:  # 相似度阈值
                        relevant_chunks.append(chunk)
                        logger.debug(f"块被接受 (分数 {score:.4f} > 0.3)")
                    else:
                        logger.debug(f"块被拒绝 (分数 {score:.4f} <= 0.3)")
                else:
                    logger.warning(f"索引 {idx} 超出文档块范围 ({len(state.doc_chunks)})")
            
            logger.info(f"向量搜索筛选后得到 {len(relevant_chunks)} 个相关块")
            
            # 如果向量搜索结果不够，使用关键词搜索作为补充
            if len(relevant_chunks) < state.retrieval_top_k:
                logger.info(f"向量搜索结果不足，启用关键词搜索补充")
                keyword_chunks = self._keyword_search(state.doc_chunks, state.user_query)
                logger.info(f"关键词搜索找到 {len(keyword_chunks)} 个额外块")
                relevant_chunks.extend(keyword_chunks)
            
            # 去重并限制数量
            seen_chunks = set()
            unique_chunks = []
            for chunk in relevant_chunks:
                if chunk.chunk_id not in seen_chunks:
                    unique_chunks.append(chunk)
                    seen_chunks.add(chunk.chunk_id)
            
            final_chunks = unique_chunks[:state.retrieval_top_k]
            logger.info(f"最终返回 {len(final_chunks)} 个去重后的文档块")
            
            # 详细记录最终选择的块
            for i, chunk in enumerate(final_chunks):
                logger.info(f"最终块 {i+1}: 来源={chunk.source_file}, 类型={chunk.metadata.get('file_type', 'unknown')}")
                logger.debug(f"最终块 {i+1} 内容: {chunk.content[:200]}...")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}", exc_info=True)
            return []
    
    async def _retrieve_from_database(self, state: AgentState) -> List[DocumentChunk]:
        """从数据库中检索相关信息"""
        if not state.database_loaded or not state.database_schema:
            logger.warning("数据库未加载或无架构信息")
            return []
        
        db_chunks = []
        
        try:
            db_path = Path("data/database/erp.db")
            if not db_path.exists():
                return []
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 根据查询内容，智能选择相关表
            relevant_tables = self._identify_relevant_tables(state.user_query, state.database_schema)
            
            for table_name in relevant_tables:
                # 执行表查询
                table_data = self._query_table(cursor, table_name, state.user_query)
                
                if table_data:
                    # 创建数据库结果块
                    chunk_content = self._format_database_result(table_name, table_data, state.database_schema[table_name])
                    
                    db_chunk = DocumentChunk(
                        content=chunk_content,
                        metadata={
                            "source_type": "database",
                            "table_name": table_name,
                            "record_count": len(table_data)
                        },
                        chunk_id=f"db_{table_name}_{hash(chunk_content)}",
                        source_file=str(db_path)
                    )
                    db_chunks.append(db_chunk)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"数据库检索失败: {str(e)}")
        
        return db_chunks
    
    def _keyword_search(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """关键词搜索"""
        keywords = query.lower().split()
        logger.info(f"关键词搜索，提取关键词: {keywords}")
        
        relevant_chunks = []
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            if matches > 0:
                relevant_chunks.append(chunk)
                logger.debug(f"关键词匹配 - 来源: {chunk.source_file}, 匹配数: {matches}, 内容: {chunk.content[:100]}...")
        
        logger.info(f"关键词搜索找到 {len(relevant_chunks)} 个匹配的文档块")
        
        # 按匹配度排序
        relevant_chunks.sort(
            key=lambda x: sum(1 for keyword in keywords if keyword in x.content.lower()),
            reverse=True
        )
        
        # 记录排序后的前几个结果
        for i, chunk in enumerate(relevant_chunks[:5]):  # 只记录前5个
            match_count = sum(1 for keyword in keywords if keyword in chunk.content.lower())
            logger.debug(f"关键词排序结果 {i+1}: 匹配数={match_count}, 来源={chunk.source_file}")
        
        return relevant_chunks
    
    def _identify_relevant_tables(self, query: str, database_schema: Dict[str, Any]) -> List[str]:
        """识别与查询相关的数据库表"""
        query_lower = query.lower()
        relevant_tables = []
        
        # 定义一些常见的业务领域关键词映射
        keyword_table_mapping = {
            "用户": ["users", "customers", "user", "customer"],
            "订单": ["orders", "order", "purchase"],
            "产品": ["products", "items", "product", "item"],
            "销售": ["sales", "sell", "revenue"],
            "库存": ["inventory", "stock"],
            "员工": ["employees", "staff", "employee"],
            "部门": ["departments", "department"],
            "财务": ["finance", "financial", "money", "payment"],
        }
        
        for table_name, table_info in database_schema.items():
            table_score = 0
            
            # 检查表名匹配
            if any(keyword in query_lower for keyword in [table_name.lower()]):
                table_score += 3
            
            # 检查列名匹配
            for column_info in table_info["columns"]:
                if column_info["name"].lower() in query_lower:
                    table_score += 2
            
            # 检查业务领域匹配
            for domain, keywords in keyword_table_mapping.items():
                if domain in query_lower:
                    for keyword in keywords:
                        if keyword in table_name.lower():
                            table_score += 1
            
            if table_score > 0:
                relevant_tables.append((table_name, table_score))
        
        # 按评分排序并返回表名
        relevant_tables.sort(key=lambda x: x[1], reverse=True)
        return [table_name for table_name, _ in relevant_tables[:3]]  # 最多返回3个表
    
    def _query_table(self, cursor, table_name: str, query: str) -> List[tuple]:
        """查询数据库表"""
        try:
            # 简单的查询策略：返回表的前几行数据
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"查询表 {table_name} 失败: {str(e)}")
            return []
    
    def _format_database_result(self, table_name: str, data: List[tuple], table_schema: Dict[str, Any]) -> str:
        """格式化数据库查询结果"""
        if not data:
            return f"表 {table_name} 无相关数据"
        
        # 获取列名
        column_names = table_schema["column_names"]
        
        result = f"数据表: {table_name}\n"
        result += f"列名: {', '.join(column_names)}\n\n"
        result += "查询结果:\n"
        
        for i, row in enumerate(data, 1):
            result += f"记录 {i}:\n"
            for col_name, value in zip(column_names, row):
                result += f"  {col_name}: {value}\n"
            result += "\n"
        
        return result
    
    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """构建相关上下文"""
        if not chunks:
            return "未找到相关信息"
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = f"来源: {Path(chunk.source_file).name}"
            if chunk.page_number:
                source_info += f" (第{chunk.page_number}页)"
            
            context_part = f"相关信息 {i}:\n{source_info}\n{chunk.content}\n"
            context_parts.append(context_part)
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def get_retrieval_summary(self, state: AgentState) -> str:
        """获取检索摘要信息"""
        if not state.retrieval_results:
            return "未执行检索"
        
        chunks = state.retrieval_results.chunks
        doc_chunks = [c for c in chunks if c.metadata.get("source_type") != "database"]
        db_chunks = [c for c in chunks if c.metadata.get("source_type") == "database"]
        
        summary = f"检索摘要:\n"
        summary += f"- 文档相关块: {len(doc_chunks)}个\n"
        summary += f"- 数据库相关块: {len(db_chunks)}个\n"
        summary += f"- 总计相关信息: {len(chunks)}个\n"
        
        if doc_chunks:
            file_sources = set(Path(c.source_file).name for c in doc_chunks)
            summary += f"- 涉及文档: {', '.join(file_sources)}\n"
        
        if db_chunks:
            table_sources = set(c.metadata.get("table_name", "unknown") for c in db_chunks)
            summary += f"- 涉及数据表: {', '.join(table_sources)}\n"
        
        return summary 