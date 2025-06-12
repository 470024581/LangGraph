"""
LangGraph 文档问答系统主程序
提供命令行界面来测试和使用系统
"""
import os
import asyncio
import argparse
import signal
from typing import Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.graphs.document_qa_graph import create_document_qa_graph
from src.schemas.agent_state import get_initial_state
from src.tools.document_tools import create_document_tools
from config import get_config


# 加载环境变量
load_dotenv()

# 配置日志
logger.add("logs/document_qa.log", rotation="1 day", retention="7 days")


class DocumentQAApp:
    """文档问答应用主类"""
    
    def __init__(self, config: Dict[str, Any] = None, preset: str = "standard"):
        """初始化应用"""
        if config is None:
            # 使用配置文件的预设
            self.config = get_config(preset)
        else:
            self.config = config
        
        # DocumentQAGraph 内部会自行创建和管理所有需要的组件，包括 DocumentTools
        self.qa_graph = create_document_qa_graph(self.config)
        

    
    async def interactive_mode(self):
        """交互模式"""
        print("🚀 LangGraph 文档问答系统")
        print("=" * 50)
        print(self.qa_graph.get_graph_visualization())
        print("=" * 50)
        print("💡 使用说明：")
        print("- 输入问题开始对话")
        print("- 输入 'tools' 查看可用工具")
        print("- 输入 'status' 查看系统状态")
        print("- 输入 'quit' 退出")
        print("=" * 50)
        
        session_id = None
        
        while True:
            try:
                try:
                    user_input = input("\n❓ 请输入您的问题: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 程序被用户中断，再见！")
                    break
                    
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("👋 再见！")
                    break
                
                elif user_input.lower() == 'tools':
                    # 工具菜单的逻辑需要调整，因为 document_tools 现在由 qa_graph 管理
                    # 为简化，暂时禁用部分交互
                    print("\n💡 工具调用功能正在集成到主流程中。")
                    print("   请直接通过提问来使用工具，例如：'总结一下[文件名]'")
                    # await self._show_tools_menu()
                    continue
                
                elif user_input.lower() == 'status':
                    if session_id:
                        status = self.qa_graph.get_current_status(session_id)
                        print(f"📊 当前状态: {status}")
                    else:
                        print("📊 暂无活动会话")
                    continue
                
                # 处理问题
                print("\n🔄 正在处理您的问题...")
                
                result = await self.qa_graph.run(
                    user_query=user_input,
                    session_id=session_id,
                    config=self.config
                )
                
                session_id = result.session_id
                
                # 显示结果
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                break
            except asyncio.CancelledError:
                print("\n\n⏹️ 任务被取消，程序退出")
                break
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"交互模式异常: {str(e)}\n{error_trace}")
                print(f"❌ 处理过程中出现错误: {str(e)}")
                
    
    def _display_result(self, result):
        """显示处理结果"""
        print("\n" + "=" * 60)
        
        if result.processing_status.value == "failed":
            print(f"❌ 处理失败: {result.error_message}")
            return
        
        # 显示回答
        if result.generated_answer:
            formatted_answer = self.qa_graph.answer_agent.format_final_answer(result)
            print("💬 回答:")
            print(formatted_answer)
        
        # 显示审阅结果
        if result.review_result:
            print("\n📊 **回答质量评估报告**")
            print(f"📈 **评分：{result.review_result.score}/10** {'✅ 通过' if result.review_result.approved else '❌ 需要改进'}")
            print(f"\n🔍 **评估意见：**\n{result.review_result.comment}")
            if result.revision_count > 0:
                print(f"\n🔄 **修订历史：** 当前为第 {result.revision_count} 次修订")
        
        print("=" * 60)
    
    async def batch_mode(self, questions_file: str):
        """批处理模式"""
        if not os.path.exists(questions_file):
            print(f"❌ 问题文件不存在: {questions_file}")
            return
        
        print(f"📝 开始批处理模式，读取问题文件: {questions_file}")
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n🔄 处理问题 {i}/{len(questions)}: {question}")
            
            result = await self.qa_graph.run(question)
            results.append(result)
            
            # 简要显示结果
            if result.generated_answer:
                print(f"✅ 回答: {result.generated_answer[:100]}...")
            if result.review_result:
                print(f"📊 评分: {result.review_result.score}/10")
        
        # 保存结果
        output_file = f"batch_results_{len(questions)}_questions.txt"
        self._save_batch_results(results, output_file)
        print(f"\n💾 批处理结果已保存到: {output_file}")
    
    def _save_batch_results(self, results, output_file: str):
        """保存批处理结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                f.write(f"问题 {i}: {result.user_query}\n")
                f.write(f"回答: {result.generated_answer}\n")
                if result.review_result:
                    f.write(f"评分: {result.review_result.score}/10\n")
                    f.write(f"评语: {result.review_result.comment}\n")
                f.write("-" * 80 + "\n\n")
    
    async def demo_mode(self):
        """演示模式"""
        demo_questions = [
            "这些文档主要讲了什么内容？",
            "数据库中有哪些表？",
            "请总结一下关键信息",
            "有什么重要的数据统计？"
        ]
        
        print("🎬 演示模式 - 自动测试预设问题")
        print("=" * 50)
        
        for question in demo_questions:
            print(f"\n🤖 自动问题: {question}")
            print("🔄 处理中...")
            
            result = await self.qa_graph.run(question)
            self._display_result(result)
            
            # 等待用户确认继续
            input("\n按 Enter 继续下一个问题...")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LangGraph 文档问答系统")
    parser.add_argument("--mode", choices=["interactive", "batch", "demo"], 
                       default="interactive", help="运行模式")
    parser.add_argument("--questions", help="批处理模式的问题文件路径")
    parser.add_argument("--preset", choices=["light", "standard", "performance", "recommended", "code"],
                       default="standard", help="模型配置预设")
    parser.add_argument("--provider", choices=["ollama", "openai"],
                       help="LLM提供商（覆盖配置文件设置）")
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.preset)
    
    # 如果指定了提供商，则覆盖配置
    if args.provider:
        config["provider"] = args.provider
    
    # 检查提供商配置
    if config["provider"] == "openai" and not config.get("openai_api_key"):
        print("❌ 使用OpenAI需要设置 OPENAI_API_KEY 环境变量")
        return
    elif config["provider"] == "ollama":
        # 检查Ollama连接
        from src.utils.ollama_provider import test_ollama_connection
        test_result = await test_ollama_connection(config)
        if not test_result["connected"]:
            print(f"❌ Ollama连接失败: {test_result.get('error', '未知错误')}")
            print("💡 请确保Ollama服务正在运行：ollama serve")
            return
        print(f"✅ Ollama连接成功，可用模型: {test_result['models']}")
    
    # 创建必要的目录
    os.makedirs("logs", exist_ok=True)
    
    try:
        # 创建应用实例
        app = DocumentQAApp(config=config)
        
        print(f"🤖 使用 {config['provider'].upper()} 模型: {config['llm_model']}")
        print(f"📊 配置预设: {args.preset}")
        
        # 根据模式运行
        if args.mode == "interactive":
            await app.interactive_mode()
        elif args.mode == "batch":
            if not args.questions:
                print("❌ 批处理模式需要指定 --questions 参数")
                return
            await app.batch_mode(args.questions)
        elif args.mode == "demo":
            await app.demo_mode()
            
    except Exception as e:
        logger.error(f"主程序异常: {str(e)}")
        print(f"❌ 程序运行异常: {str(e)}")


if __name__ == "__main__":
    # 运行主程序
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断，正常退出")
    except Exception as e:
        logger.error(f"程序异常退出: {str(e)}")
        print(f"❌ 程序异常退出: {str(e)}") 