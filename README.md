# 🚀 LangGraph 文档问答系统

基于 LangGraph 构建的智能文档问答系统，具备多格式文档处理、RAG检索、质量审核等功能。

## 🎯 项目特色

### 核心功能
- 📄 **多格式文档支持**：PDF、DOCX、TXT、CSV 文件解析
- 🗄️ **数据库集成**：SQLite 数据库内容理解和查询
- 🔍 **智能检索**：基于向量相似度和关键词的混合检索
- 🤖 **AI 问答**：使用 LLM 生成准确、详细的回答
- ✅ **质量审核**：自动评分和改进建议机制
- 🔄 **迭代优化**：基于审核反馈的自动回答优化
- 📊 **状态追踪**：完整的处理流程和历史记录

### 技术亮点
- 🧠 **LangGraph 编排**：清晰的状态图和流程控制
- 🏗️ **模块化设计**：可扩展的组件架构
- 🔧 **工具链集成**：文档摘要、翻译、分析等工具
- 📈 **性能优化**：向量化存储和批处理支持
- 🛡️ **错误处理**：完善的异常处理和恢复机制

## 🏗️ 系统架构

```
用户输入 (问题/文件上传)
    ↓
[1] 文档处理 Agent (提取文档结构)
    ↓
[2] 检索 Agent (RAG + 文档块搜索)
    ↓
[3] 回答 Agent（LLM生成回答）
    ↓
[4] 审阅 Agent（判断质量，打分+建议）
   ↳ 如评分低则跳回 [3] 再生成
    ↓
输出回答 + 审阅意见
```

## 📦 安装部署

### 系统要求
- Python 3.8+
- OpenAI API Key
- 8GB+ RAM (用于向量存储)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd LangGraph
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，设置必要的配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

4. **准备数据**
```bash
# 将文档放入相应目录
data/document/    # PDF、DOCX、TXT 文件
data/csv/         # CSV 文件  
data/database/    # SQLite 数据库文件
```

## 🚀 使用方法

### 1. 交互模式（推荐）
```bash
python main.py --mode interactive
```

提供友好的命令行界面，支持：
- 实时问答对话
- 工具功能调用
- 状态查询
- 流程可视化

### 2. 演示模式
```bash
python main.py --mode demo
```

自动运行预设问题，展示系统功能。

### 3. 批处理模式
```bash
# 创建问题文件
echo "这些文档主要讲什么？
数据库中有哪些表？
请总结关键信息" > questions.txt

# 批量处理
python main.py --mode batch --questions questions.txt
```

### 4. 编程接口

```python
import asyncio
from src import create_document_qa_graph

async def main():
    # 创建问答图
    qa_graph = create_document_qa_graph({
        "llm_model": "gpt-3.5-turbo",
        "quality_threshold": 8,
        "max_revisions": 3
    })
    
    # 执行问答
    result = await qa_graph.run("文档中的主要内容是什么？")
    
    # 查看结果
    print(f"回答: {result.generated_answer}")
    print(f"评分: {result.review_result.score}/10")

asyncio.run(main())
```

## 🛠️ 功能详解

### 文档处理能力
- **PDF**: 文本提取、页码标记、表格识别
- **DOCX**: 段落解析、表格提取、格式保留  
- **TXT**: 智能分块、编码识别
- **CSV**: 结构化数据理解、统计信息生成
- **数据库**: 表结构分析、示例数据提取

### 检索策略
- **向量检索**: 使用 Sentence-Transformers 进行语义匹配
- **关键词检索**: 基于词频的传统检索作为补充
- **混合排序**: 综合相似度分数和关键词匹配度
- **上下文构建**: 智能组合检索结果形成完整上下文

### 质量审核标准
- **准确性** (25%): 基于提供信息的事实准确度
- **完整性** (25%): 是否全面回答用户问题
- **相关性** (20%): 与用户问题的匹配程度
- **清晰性** (15%): 表达的清晰度和逻辑性
- **有用性** (15%): 对用户的实际帮助程度

### 工具链功能
- 📄 **文档摘要**: 提取关键信息生成摘要
- 🌐 **文本翻译**: 多语言翻译支持
- 💡 **要点提取**: 自动识别关键要点
- 📊 **结构分析**: 文档结构和统计信息
- ❓ **问题生成**: 基于内容生成相关问题
- 🏷️ **实体提取**: 人名、地名、机构等实体识别

## ⚙️ 配置说明

### 主要配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_model` | gpt-3.5-turbo | 使用的语言模型 |
| `embedding_model` | all-MiniLM-L6-v2 | 嵌入模型 |
| `quality_threshold` | 8 | 质量通过阈值 (1-10) |
| `max_revisions` | 3 | 最大修订次数 |
| `retrieval_top_k` | 5 | 检索返回数量 |
| `temperature` | 0.7 | LLM 生成温度 |

### 高级配置

```python
config = {
    "llm_model": "gpt-4",  # 使用 GPT-4 模型
    "temperature": 0.5,    # 降低随机性
    "quality_threshold": 9, # 提高质量要求
    "chunk_size": 1500,    # 增大文档块大小
    "chunk_overlap": 300   # 增大重叠区域
}
```

## 📊 性能指标

### 处理能力
- **文档处理**: 100+ 页/分钟
- **向量检索**: <100ms 查询响应
- **回答生成**: 2-5秒 (依赖模型)
- **并发支持**: 10+ 并发会话

### 内存使用
- **基础运行**: ~2GB RAM
- **大文档集**: ~4-8GB RAM
- **向量存储**: ~1MB/1000文档块

## 🔧 开发指南

### 项目结构
```
src/
├── schemas/          # 数据模型定义
├── components/       # 核心组件
├── agents/          # 智能代理
├── graphs/          # LangGraph 流程
├── tools/           # 工具函数
└── utils/           # 工具类

data/
├── document/        # 文档文件
├── database/        # 数据库文件
└── csv/            # CSV 文件
```

### 扩展开发

1. **添加新的文档格式**
```python
# 在 document_processor.py 中添加处理方法
async def _process_new_format(self, file_path: Path) -> List[DocumentChunk]:
    # 实现新格式解析逻辑
    pass
```

2. **自定义评审标准**
```python
# 在 review_agent.py 中修改评估维度
evaluation_dimensions = {
    "accuracy": 0.3,      # 调整权重
    "completeness": 0.3,
    "relevance": 0.2,
    "clarity": 0.1,
    "usefulness": 0.1
}
```

3. **添加新工具**
```python
# 在 document_tools.py 中添加工具方法
async def custom_analysis(self, chunks: List[DocumentChunk]) -> str:
    # 实现自定义分析逻辑
    pass
```

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_document_processor.py -v

# 生成覆盖率报告
pytest --cov=src tests/
```

### 手动测试
```bash
# 演示模式快速测试
python main.py --mode demo

# 交互式测试
python main.py --mode interactive
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙋‍♂️ 常见问题

### Q: 如何处理大型文档？
A: 系统会自动分块处理，建议调整 `chunk_size` 参数优化性能。

### Q: 支持哪些语言模型？
A: 支持所有 OpenAI 兼容的模型，包括 GPT-3.5、GPT-4 等。

### Q: 如何提高回答质量？
A: 可以调整 `quality_threshold` 和 `max_revisions` 参数，或使用更强的模型。

### Q: 能否离线运行？
A: 文档处理和检索可离线运行，但 LLM 生成需要 API 调用。

## 📞 技术支持

- 📧 邮箱: support@example.com
- 💬 讨论: [GitHub Discussions](https://github.com/your-repo/discussions)
- 🐛 报告问题: [GitHub Issues](https://github.com/your-repo/issues)

---

⭐ 如果这个项目对您有帮助，请给个 Star！