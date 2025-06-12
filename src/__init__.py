"""
LangGraph 文档问答系统
一个基于LangGraph的智能文档问答系统，支持多种文档格式和质量审核机制
"""

__version__ = "1.0.0"
__author__ = "LangGraph Team"
__description__ = "基于LangGraph的智能文档问答系统"

# 导出主要类和函数
from .graphs.document_qa_graph import DocumentQAGraph, create_document_qa_graph
from .schemas.agent_state import AgentState, get_initial_state
from .tools.document_tools import DocumentTools, create_document_tools

__all__ = [
    "DocumentQAGraph",
    "create_document_qa_graph", 
    "AgentState",
    "get_initial_state",
    "DocumentTools",
    "create_document_tools"
] 