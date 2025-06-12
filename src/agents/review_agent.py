"""
审阅Agent
评估回答质量，提供评分和改进建议
"""
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from loguru import logger
from pydantic import BaseModel, Field

from ..schemas.agent_state import AgentState, ProcessingStatus
from ..schemas.review_result import ReviewResult
from ..utils.ollama_provider import OllamaProvider


class ReviewAgent:
    """审阅Agent"""
    
    def __init__(self, ollama_provider: OllamaProvider):
        """初始化审阅Agent"""
        self.llm = ollama_provider.get_llm()
        self.structured_llm = self.llm.with_structured_output(ReviewResult)
        logger.info(f"审阅Agent使用 Ollama 模型: {ollama_provider.model_name}")

    async def review_answer(self, state: AgentState) -> AgentState:
        """审阅回答的主入口函数"""
        try:
            state.processing_status = ProcessingStatus.PROCESSING
            state.current_step = "review"
            
            # 执行质量评估
            review_result_dict = await self._evaluate_answer_quality(state)
            # 将字典转换为 ReviewResult 对象
            state.review_result = ReviewResult(**review_result_dict)
            
            # 根据评分决定下一步
            if state.review_result.approved and state.review_result.score >= state.quality_threshold:
                state.current_step = "completed"
                state.processing_status = ProcessingStatus.COMPLETED
                logger.info(f"回答通过审核，评分: {state.review_result.score}/10")
            else:
                if state.revision_count >= state.max_revisions:
                    state.current_step = "max_revisions_reached"
                    state.processing_status = ProcessingStatus.COMPLETED
                    logger.warning(f"达到最大修订次数，当前评分: {state.review_result.score}/10")
                else:
                    state.current_step = "needs_revision"
                    logger.info(f"回答需要修订，评分: {state.review_result.score}/10")
            
            return state
            
        except Exception as e:
            logger.error(f"审阅失败: {str(e)}")
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"审阅失败: {str(e)}"
            return state
    
    async def _evaluate_answer_quality(self, state: AgentState) -> Dict[str, Any]:
        """使用 structured_llm 直接评估回答质量并返回 ReviewResult 对象。"""
        
        # 构建评估prompt
        evaluation_prompt = self._build_evaluation_prompt(state)
        
        try:
            # structured_llm 会自动调用 LLM 并将输出解析为 ReviewResult 对象
            review_result = await self.structured_llm.ainvoke(evaluation_prompt)
            logger.info("结构化审阅完成。")
            # 将 ReviewResult 对象转换为字典
            return review_result.dict()
        except Exception as e:
            logger.error(f"结构化审阅LLM调用失败: {e}", exc_info=True)
            # 在失败时返回一个默认的、标记为未通过的 ReviewResult 字典
            return {
                "score": 1,
                "comment": f"审阅时发生严重错误: {e}",
                "approved": False
            }
    
    def _build_evaluation_prompt(self, state: AgentState) -> str:
        """构建评估prompt"""
        
        system_prompt = """你是一个专业的回答质量审阅员。你需要从多个维度评估回答的质量，并给出1-10的评分。

评估维度：
1. **准确性** (25%)：回答是否基于提供的信息，是否有事实错误
2. **完整性** (25%)：回答是否全面，是否遗漏重要信息
3. **相关性** (20%)：回答是否直接回应了用户问题
4. **清晰性** (15%)：回答是否结构清晰，表达明确
5. **有用性** (15%)：回答是否对用户有实际帮助

你需要根据用户的原始问题和生成的回答，输出一个JSON对象，该对象包含以下字段： "score" (int), "comment" (str), "approved" (bool)。
"""

        user_query = state.user_query
        generated_answer = state.generated_answer
        
        evaluation_prompt = f"""{system_prompt}

[原始问题]:
{user_query}

[待审阅的回答]:
{generated_answer}

请输出审阅结果的JSON对象。
"""
        return evaluation_prompt
    
    def generate_feedback_summary(self, state: AgentState) -> str:
        """生成反馈摘要"""
        if not state.review_result:
            return "暂无审阅结果"
        
        review = state.review_result
        
        summary = f"""
📊 **回答质量评估报告**

📈 **评分：{review.score}/10** {'✅ 通过' if review.approved else '❌ 需要改进'}

🔍 **评估意见：**
{review.comment}"""

        if state.revision_count > 0:
            summary += f"\n\n🔄 **修订历史：** 当前为第 {state.revision_count} 次修订"
        
        return summary
    
    def get_quality_metrics(self, state: AgentState) -> Dict[str, Any]:
        """获取质量指标"""
        if not state.review_result:
            return {}
        
        review = state.review_result
        
        metrics = {
            "quality_score": review.score,
            "approved": review.approved,
            "revision_count": state.revision_count,
            "needs_improvement": not review.approved and state.revision_count < state.max_revisions,
            "max_revisions_reached": state.revision_count >= state.max_revisions
        }
        
        # 质量等级
        if review.score >= 9:
            metrics["quality_level"] = "优秀"
        elif review.score >= 7:
            metrics["quality_level"] = "良好"
        elif review.score >= 5:
            metrics["quality_level"] = "一般"
        elif review.score >= 3:
            metrics["quality_level"] = "较差"
        else:
            metrics["quality_level"] = "很差"
        
        return metrics
    
    async def batch_review(self, answers: List[str], contexts: List[str], queries: List[str]) -> List[ReviewResult]:
        """批量审阅多个回答"""
        results = []
        
        for answer, context, query in zip(answers, contexts, queries):
            # 创建临时状态进行评估
            temp_state = AgentState(
                user_query=query,
                generated_answer=answer,
                relevant_context=context
            )
            
            result = await self._evaluate_answer_quality(temp_state)
            results.append(result)
        
        return results
    
    def create_improvement_prompt(self, state: AgentState) -> str:
        """根据审阅结果创建用于改进的prompt"""
        if not state.review_result:
            return ""
            
        feedback = f"之前的回答存在一些问题，审阅分数为 {state.review_result.score}/10。\n"
        feedback += f"审阅意见: {state.review_result.comment}\n"

        prompt = f"""原始问题: {state.user_query}
相关信息: {state.retrieved_info}
之前的回答: {state.generated_answer}

审阅反馈:
{feedback}

请根据以上反馈，生成一个更高质量、更完善的新回答。
"""
        return prompt 