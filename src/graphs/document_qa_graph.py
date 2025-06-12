"""
LangGraph 文档问答系统核心流程编排
定义整个系统的状态图、节点和边
"""
from typing import Dict, Any
import asyncio
from loguru import logger
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..schemas.agent_state import AgentState, ProcessingStatus, save_state_history, get_initial_state
from ..components.document_processor import DocumentProcessor
from ..agents.retrieval_agent import RetrievalAgent
from ..agents.answer_agent import AnswerAgent
from ..agents.review_agent import ReviewAgent
from ..tools.document_tools import DocumentTools
from ..components.router import Router
from ..agents.sql_agent import SQLAgent
from ..schemas.review_result import ReviewResult
from ..utils.ollama_provider import OllamaProvider


class DocumentQAGraph:
    """文档问答系统LangGraph编排器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化文档问答图"""
        self.config = config or {}
        
        # 准备 OllamaProvider 的配置字典
        ollama_config = {
            "llm_model": self.config.get("model", "mistral:latest"),
            "ollama_base_url": self.config.get("ollama_host"),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 2048),
            "top_p": self.config.get("top_p", 0.9),
            "top_k": self.config.get("top_k", 40),
            "ollama_keep_alive": self.config.get("ollama_keep_alive", "5m"),
        }
        self.ollama_provider = OllamaProvider(config=ollama_config)

        # 初始化所有组件
        self.answer_agent = AnswerAgent(self.ollama_provider)
        self.review_agent = ReviewAgent(self.ollama_provider)
        self.document_tools = DocumentTools(self.ollama_provider)
        self.router = Router(self.ollama_provider)
        try:
            self.sql_agent = SQLAgent(self.ollama_provider)
            logger.info("SQL Agent 初始化成功。")
        except FileNotFoundError as e:
            self.sql_agent = None
            logger.warning(f"SQL Agent 初始化失败: {e}。SQL路由将不可用。")
        
        # 设置检查点存储（用于状态持久化）
        self.checkpointer = None
        self._checkpointer_path = None
        
        if AsyncSqliteSaver:
            try:
                # 创建检查点数据库目录
                from pathlib import Path
                checkpoint_dir = Path("data/checkpoints")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                db_path = checkpoint_dir / "langgraph_checkpoints.db"
                self._checkpointer_path = str(db_path)
                
                # 异步检查点存储将在运行时初始化
                logger.info(f"✅ 异步SQLite检查点存储配置完成，数据库路径: {db_path}")
                
            except Exception as e:
                logger.warning(f"异步SQLite检查点存储配置失败: {e}，将禁用检查点功能")
                logger.warning("💡 请确保已安装: pip install aiosqlite")
                self._checkpointer_path = None
        else:
            logger.warning("AsyncSqliteSaver 不可用，将禁用状态持久化")
            logger.info("💡 要启用状态持久化，请安装: pip install aiosqlite langgraph-checkpoint-sqlite")
        
        # 构建状态图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图"""
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加所有节点
        workflow.add_node("route_question", self._route_question_node)
        workflow.add_node("document_processing", self._document_processing_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("answer_revision", self._answer_revision_node)
        workflow.add_node("completed", self._completed_node)
        workflow.add_node("sql_agent", self._sql_agent_node)
        workflow.add_node("error_handling", self._error_handling_node)
        
        # 设置图的入口点
        workflow.set_entry_point("route_question")
        
        # 设置图的边
        workflow.add_conditional_edges(
            "route_question",
            self._decide_route,
            {
                "sql": "sql_agent",
                "rag": "document_processing"
            }
        )
        
        workflow.add_edge("sql_agent", "completed")
        workflow.add_edge("document_processing", "retrieval")
        workflow.add_edge("retrieval", "answer_generation")
        workflow.add_edge("answer_generation", "review")
        
        workflow.add_conditional_edges(
            "review",
            self._should_revise_answer,
            {
                "completed": "completed",
                "needs_revision": "answer_revision",
                "max_revisions_reached": "completed",
                "error": "error_handling"
            }
        )
        
        workflow.add_edge("answer_revision", "review")
        workflow.add_edge("error_handling", END)
        
        # 编译图 - 暂时禁用检查点以避免异步复杂性
        # 注意：AsyncSqliteSaver需要在异步上下文中正确初始化，这很复杂
        # 暂时使用无状态模式，系统仍然可以正常工作
        logger.info("使用无状态模式编译图形（暂时禁用检查点存储）")
        return workflow.compile()
    
    async def _route_question_node(self, state: AgentState) -> AgentState:
        """路由节点，决定问题流向"""
        logger.info("开始路由问题...")
        if not self.sql_agent:
            logger.warning("SQL Agent 未初始化，强制路由到 RAG 流程。")
            state.route = "rag"
            return state

        try:
            route = await self.router.route(state.user_query)
            state.route = route
            return state
        except Exception as e:
            logger.error(f"路由决策时出错: {e}", exc_info=True)
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"路由决策失败: {e}"
            # 默认回退到RAG流程
            state.route = "rag"
            return state

    def _decide_route(self, state: AgentState) -> str:
        """根据路由节点的决策返回下一跳"""
        return state.route

    async def _sql_agent_node(self, state: AgentState) -> AgentState:
        """执行 SQL Agent"""
        logger.info("问题流向 SQL Agent...")
        state = save_state_history(state)
        state.current_step = "sql_query"
        
        try:
            answer = await self.sql_agent.run(state.user_query)
            state.generated_answer = answer
            state.final_answer = answer
            state.answer_sources = ["erp.db"]

            # SQL Agent 的结果直接采纳，不经过审阅
            state.review_result = ReviewResult(
                approved=True,
                score=10,
                comment="SQL Agent直接查询结果，自动采纳。"
            )
            return state
        except Exception as e:
            logger.error(f"SQL Agent 节点执行出错: {e}", exc_info=True)
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"SQL Agent 执行失败: {e}"
            return state

    async def _document_processing_node(self, state: AgentState) -> AgentState:
        """文档处理节点：加载并处理所有文档，构建向量存储。"""
        logger.info("开始文档处理...")
        state = save_state_history(state)
        state.current_step = "document_processing"
        
        try:
            # 使用 document_tools 处理文档
            processed_state = await self.document_tools.process_and_store_documents(state)
            return processed_state
            
        except Exception as e:
            logger.error(f"文档处理节点异常: {e}", exc_info=True)
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"文档处理失败: {e}"
            return state
    
    async def _retrieval_node(self, state: AgentState) -> AgentState:
        """检索节点：根据用户问题从向量存储和数据库中检索信息。"""
        logger.info("开始信息检索...")
        state = save_state_history(state)
        state.current_step = "retrieval"
        
        try:
            # 使用 document_tools 检索信息
            retrieved_state = await self.document_tools.search_relevant_info(state)
            return retrieved_state
            
        except Exception as e:
            logger.error(f"检索节点异常: {e}", exc_info=True)
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"信息检索失败: {e}"
            return state
    
    async def _answer_generation_node(self, state: AgentState) -> AgentState:
        """回答生成节点"""
        logger.info("开始生成回答...")
        state = save_state_history(state)
        
        try:
            state = await self.answer_agent.generate_answer(state)
            return state
            
        except Exception as e:
            logger.error(f"回答生成节点异常: {str(e)}")
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"回答生成失败: {str(e)}"
            return state
    
    async def _review_node(self, state: AgentState) -> AgentState:
        """审阅节点"""
        logger.info("开始审阅回答...")
        state = save_state_history(state)
        
        try:
            state = await self.review_agent.review_answer(state)
            return state
            
        except Exception as e:
            logger.error(f"审阅节点异常: {str(e)}")
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"回答审阅失败: {str(e)}"
            return state
    
    async def _answer_revision_node(self, state: AgentState) -> AgentState:
        """回答修订节点"""
        logger.info(f"开始修订回答 (第{state.revision_count + 1}次)...")
        state = save_state_history(state)
        
        try:
            # 生成改进反馈
            feedback = self.review_agent.create_improvement_prompt(state)
            
            # 重新生成回答
            state = await self.answer_agent.regenerate_answer(state, feedback)
            
            return state
            
        except Exception as e:
            logger.error(f"回答修订节点异常: {str(e)}")
            state.processing_status = ProcessingStatus.FAILED
            state.error_message = f"回答修订失败: {str(e)}"
            return state
    
    async def _completed_node(self, state: AgentState) -> AgentState:
        """完成节点"""
        logger.info("问答流程完成")
        state = save_state_history(state)
        state.processing_status = ProcessingStatus.COMPLETED
        state.current_step = "completed"
        # 如果是SQL流程，final_answer已在sql_agent_node中设置
        # 如果是RAG流程，需要在这里设置
        if state.route == 'rag':
            state.final_answer = state.generated_answer
        return state
    
    async def _error_handling_node(self, state: AgentState) -> AgentState:
        """错误处理节点"""
        logger.error(f"流程出现错误: {state.error_message}")
        state = save_state_history(state)
        state.processing_status = ProcessingStatus.FAILED
        return state
    
    def _should_revise_answer(self, state: AgentState) -> str:
        """判断是否需要修订回答"""
        
        # 如果出现错误
        if state.processing_status == ProcessingStatus.FAILED:
            return "error"
        
        # 如果没有审阅结果
        if not state.review_result:
            return "error"
        
        # 如果已达到最大修订次数
        if state.revision_count >= state.max_revisions:
            return "max_revisions_reached"
        
        # 如果回答质量达标
        if state.review_result.approved and state.review_result.score >= state.quality_threshold:
            return "completed"
        
        # 需要修订
        return "needs_revision"
    
    async def run(self, user_query: str, session_id: str = None, config: Dict[str, Any] = None) -> AgentState:
        """运行文档问答流程"""
        
        # 创建初始状态
        initial_state_data = get_initial_state(user_query, session_id)
        
        # 应用运行时配置
        if config:
            initial_state_data_dict = initial_state_data.dict()
            initial_state_data_dict.update(config)
            initial_state = AgentState(**initial_state_data_dict)
        else:
            initial_state = initial_state_data
        
        logger.info(f"开始处理问题: {initial_state.user_query} (Session: {initial_state.session_id})")
        
        try:
            # 运行状态图
            thread = {"configurable": {"thread_id": initial_state.session_id}}
            final_state_dict = await self.graph.ainvoke(initial_state.dict(), thread)

            # 确保返回的是AgentState对象
            final_state = AgentState(**final_state_dict)
            
            logger.info(f"问答流程完成，最终状态: {final_state.current_step}")
            return final_state
            
        except Exception as e:
            logger.error(f"运行流程异常: {e}", exc_info=True)
            initial_state.processing_status = ProcessingStatus.FAILED
            initial_state.error_message = f"流程运行失败: {e}"
            return initial_state
    
    async def stream_run(self, user_query: str, session_id: str = None, config: Dict[str, Any] = None):
        """流式运行文档问答流程"""
        
        # 创建初始状态
        initial_state_data = get_initial_state(user_query, session_id)
        
        # 应用运行时配置
        if config:
            initial_state_data_dict = initial_state_data.dict()
            initial_state_data_dict.update(config)
            initial_state = AgentState(**initial_state_data_dict)
        else:
            initial_state = initial_state_data
        
        logger.info(f"开始流式处理问题: {initial_state.user_query} (Session: {initial_state.session_id})")
        
        try:
            # 流式运行
            if self._checkpointer_path:
                thread = {"configurable": {"thread_id": initial_state.session_id}}
                async for state in self.graph.astream(initial_state, thread):
                    yield state
            else:
                async for state in self.graph.astream(initial_state):
                    yield state
                
        except Exception as e:
            logger.error(f"流式运行异常: {str(e)}")
            yield {
                "error": True,
                "message": f"流程运行失败: {str(e)}"
            }
    
    def get_graph_visualization(self) -> str:
        """获取图的可视化描述"""
        return """
文档问答系统流程图:

[开始] 
    ↓
[文档处理] - 加载PDF/DOCX/TXT/CSV文件，构建向量存储
    ↓
[信息检索] - 基于用户问题检索相关文档和数据库信息
    ↓
[回答生成] - 使用LLM基于检索信息生成回答
    ↓
[质量审阅] - 评估回答质量，给出评分和建议
    ↓
[条件判断] - 评分是否达标？
    ├─ 是 → [完成]
    └─ 否 → [回答修订] → 返回[质量审阅]
           (最多重试3次)
"""
    
    def get_current_status(self, session_id: str) -> Dict[str, Any]:
        """获取当前会话状态"""
        if not self._checkpointer_path:
            return {"status": "检查点存储不可用，无法获取状态"}
        
        try:
            thread = {"configurable": {"thread_id": session_id}}
            state = self.graph.get_state(thread)
            
            if state and state.values:
                return {
                    "current_step": state.values.get("current_step", "未知"),
                    "processing_status": state.values.get("processing_status", "未知"),
                    "revision_count": state.values.get("revision_count", 0),
                    "error_message": state.values.get("error_message", ""),
                    "has_answer": bool(state.values.get("generated_answer"))
                }
            else:
                return {"status": "会话不存在"}
                
        except Exception as e:
            logger.error(f"获取状态失败: {str(e)}")
            return {"error": f"获取状态失败: {str(e)}"}


# 工厂函数
def create_document_qa_graph(config: Dict[str, Any] = None) -> DocumentQAGraph:
    """创建文档问答图实例"""
    return DocumentQAGraph(config) 