from pydantic import BaseModel, Field

class ReviewResult(BaseModel):
    """审阅结果数据结构"""
    approved: bool = Field(..., description="是否批准")
    score: int = Field(..., description="评分 (1-10)")
    comment: str = Field(..., description="反馈意见") 