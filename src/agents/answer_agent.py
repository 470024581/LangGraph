"""
回答Agent
基于检索到的相关信息生成高质量的回答
"""
from typing import List, Dict, Any, Optional
import os
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_community.callbacks.manager import get_openai_callback
from loguru import logger

from ..schemas.agent_state import AgentState, ProcessingStatus
from ..utils.ollama_provider import OllamaProvider


class AnswerAgent:
    """回答生成Agent"""
    
    def __init__(self, ollama_provider: OllamaProvider):
        """初始化回答Agent"""
        self.llm = ollama_provider.get_llm()
        self.is_ollama = True # 假设这个Agent总是使用Ollama
        logger.info(f"使用 Ollama 模型: {ollama_provider.model_name}")
        
        self.max_context_length = 4000  # 最大上下文长度
        
    async def generate_answer(self, state: AgentState) -> AgentState:
        """生成回答的主入口函数"""
        try:
            state.processing_status = ProcessingStatus.PROCESSING
            state.current_step = "answer_generation"
            
            # 构建prompt
            prompt = self._build_prompt(state)
            
            # 生成回答
            answer = await self._generate_response(prompt)
            
            # 提取答案来源
            sources = self._extract_sources(state)
            
            state.generated_answer = answer
            state.answer_sources = sources
            
            state.processing_status = ProcessingStatus.COMPLETED
            state.current_step = "review"
            
            logger.info(f"回答生成完成，字数: {len(answer)}")
            return state
            
        except Exception as e:
            logger.error(f"回答生成失败: {str(e)}")
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"回答生成失败: {str(e)}"
            return state
    
    def _build_prompt(self, state: AgentState) -> str:
        """构建LLM提示词"""
        
        # 系统角色定义
        system_prompt = """你是一个专业的文档问答助手。你的任务是基于提供的相关信息，准确、详细、有用地回答用户的问题。

回答要求：
1. 基于提供的相关信息进行回答，不要编造不存在的信息
2. 如果信息不足以完全回答问题，请明确说明
3. 回答要结构清晰，逻辑性强
4. 适当引用具体的信息来源
5. 使用中文回答
6. 保持专业和客观的语调

如果用户问题涉及数据分析，请提供具体的数据和分析结果。
如果用户问题涉及文档内容，请引用相关段落并说明出处。"""

        # 用户问题
        user_query = state.user_query
        
        # 相关上下文信息
        context = state.relevant_context if state.relevant_context else "未找到相关信息"
        
        # 截断过长的上下文
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "...\n(上下文已截断)"
        
        # 构建完整prompt
        full_prompt = f"""{system_prompt}

相关信息:
{context}

用户问题: {user_query}

请基于上述相关信息回答用户问题："""

        return full_prompt
    
    async def _generate_response(self, prompt: str) -> str:
        """使用LLM生成回答"""
        try:
            messages = [HumanMessage(content=prompt)]
            
            # 使用 Ollama 生成
            response_text = await self.llm.ainvoke(prompt)
            if isinstance(response_text, BaseMessage):
                 response_text = response_text.content
            
            logger.info(f"Ollama生成完成，输出长度: {len(response_text)}")
            return response_text
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}", exc_info=True)
            return f"抱歉，我无法生成回答。错误信息: {e}"
    
    def _extract_sources(self, state: AgentState) -> List[str]:
        """提取回答来源"""
        sources = []
        
        if state.retrieval_results and state.retrieval_results.chunks:
            for chunk in state.retrieval_results.chunks:
                source_info = f"{chunk.source_file}"
                
                if hasattr(chunk, 'page_number') and chunk.page_number:
                    source_info += f" (第{chunk.page_number}页)"
                
                if "source_type" in chunk.metadata and chunk.metadata.get("source_type") == "database":
                    table_name = chunk.metadata.get("table_name", "未知表")
                    source_info += f" - 数据表: {table_name}"
                
                if source_info not in sources:
                    sources.append(source_info)
        
        return sources
    
    async def regenerate_answer(self, state: AgentState, feedback: str = "") -> AgentState:
        """基于反馈重新生成回答"""
        try:
            state.revision_count += 1
            state.processing_status = ProcessingStatus.PROCESSING
            state.current_step = "answer_regeneration"
            
            # 构建包含反馈的改进prompt
            improved_prompt = self._build_improved_prompt(state, feedback)
            
            # 重新生成回答
            new_answer = await self._generate_response(improved_prompt)
            
            state.generated_answer = new_answer
            state.processing_status = ProcessingStatus.COMPLETED
            state.current_step = "review"
            
            logger.info(f"回答重新生成完成 (第{state.revision_count}次修订)")
            return state
            
        except Exception as e:
            logger.error(f"重新生成回答失败: {str(e)}")
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"重新生成回答失败: {str(e)}"
            return state
    
    def _build_improved_prompt(self, state: AgentState, feedback: str) -> str:
        """构建包含改进建议的prompt"""
        
        system_prompt = """你是一个专业的文档问答助手。你需要基于之前的回答和审阅反馈，生成一个改进的回答。

改进要求：
1. 仔细考虑审阅反馈中的建议
2. 保持回答的准确性和专业性
3. 改进回答的结构和表达
4. 补充遗漏的重要信息
5. 确保回答更加全面和有用
6. 使用中文回答"""
        
        previous_answer = state.generated_answer
        context = state.relevant_context if state.relevant_context else "未找到相关信息"
        user_query = state.user_query
        
        # 截断过长的上下文
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "...\n(上下文已截断)"
        
        improved_prompt = f"""{system_prompt}

原始问题: {user_query}

相关信息:
{context}

之前的回答:
{previous_answer}

审阅反馈:
{feedback}

请基于审阅反馈改进你的回答："""

        return improved_prompt
    
    def get_answer_statistics(self, state: AgentState) -> Dict[str, Any]:
        """获取回答统计信息"""
        stats = {
            "answer_length": len(state.generated_answer) if state.generated_answer else 0,
            "revision_count": state.revision_count,
            "sources_count": len(state.answer_sources),
            "has_database_sources": any("数据表" in source for source in state.answer_sources),
            "has_document_sources": any("数据表" not in source for source in state.answer_sources)
        }
        
        if state.retrieval_results:
            stats["retrieved_chunks"] = len(state.retrieval_results.chunks)
        
        return stats
    
    def format_final_answer(self, state: AgentState) -> str:
        """格式化最终回答"""
        if not state.generated_answer:
            return "抱歉，无法生成回答。"
        
        answer = state.generated_answer
        
        # 添加来源信息
        if state.answer_sources:
            answer += "\n\n📚 **信息来源:**\n"
            for i, source in enumerate(state.answer_sources, 1):
                answer += f"{i}. {source}\n"
        
        # 添加统计信息
        stats = self.get_answer_statistics(state)
        if stats["revision_count"] > 0:
            answer += f"\n*（此回答经过 {stats['revision_count']} 次修订优化）*"
        
        return answer
    
    async def stream_answer(self, state: AgentState):
        """流式生成回答（用于实时显示）"""
        prompt = self._build_prompt(state)
        
        try:
            messages = [HumanMessage(content=prompt)]
            
            # Ollama 流式生成
            async for chunk in self.llm.astream(messages):
                yield chunk
            
        except Exception as e:
            logger.error(f"流式生成失败: {str(e)}")
            yield f"生成回答时出现错误: {str(e)}" 