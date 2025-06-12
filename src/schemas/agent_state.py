"""
LangGraph Agent State 定义
定义系统中所有状态字段和数据结构
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from .document_chunk import DocumentChunk
from .review_result import ReviewResult
from .processing_status import ProcessingStatus


class RetrievalResult(BaseModel):
    """检索结果数据结构"""
    chunks: List[DocumentChunk] = Field(default_factory=list, description="相关文档块")
    scores: List[float] = Field(default_factory=list, description="相似度分数")
    query: str = Field(..., description="检索查询")


class AgentState(BaseModel):
    """
    智能体状态管理
    """
    user_query: str                 # 用户的原始问题
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    route: str = ""

    # 文档和数据库处理状态
    doc_chunks: List[DocumentChunk] = Field(default_factory=list)
    database_schema: Dict[str, Any] = Field(default_factory=dict)
    documents_loaded: bool = False
    vector_store_ready: bool = False
    database_loaded: bool = False
    
    # RAG流程状态
    retrieved_info: str = ""
    retrieval_results: Optional[RetrievalResult] = None
    relevant_context: str = ""
    retrieval_top_k: int = 5  # 检索时返回的最相关文档块数量

    # 回答生成状态
    generated_answer: str = ""
    final_answer: str = ""
    answer_sources: List[str] = Field(default_factory=list)
    
    # 审阅状态
    review_result: Optional[ReviewResult] = None
    revision_count: int = 0
    max_revisions: int = 3
    
    current_step: str = "start"
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: str = ""
    
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    quality_threshold: int = 8

    class Config:
        """Pydantic配置"""
        arbitrary_types_allowed = True
        json_encoders = {
            # 自定义编码器
        }


def get_initial_state(user_query: str, session_id: Optional[str] = None) -> AgentState:
    """获取初始状态"""
    state_data = {"user_query": user_query}
    if session_id:
        state_data["session_id"] = session_id
    return AgentState(**state_data)


def save_state_history(state: AgentState) -> AgentState:
    """保存状态历史"""
    current_state_snapshot = {
        "timestamp": datetime.now().isoformat(),
        "step": state.current_step,
        "status": state.processing_status.value,
        "query": state.user_query,
        "answer": state.generated_answer,
        "review_score": state.review_result.score if state.review_result else None
    }
    
    state.state_history.append(current_state_snapshot)
    return state 