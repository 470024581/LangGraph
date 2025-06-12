from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    """文档块数据结构"""
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    chunk_id: str = Field(..., description="块ID")
    source_file: str = Field(..., description="源文件路径")
    page_number: Optional[int] = Field(None, description="页码") 