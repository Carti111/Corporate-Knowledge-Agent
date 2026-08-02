from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    message: str
    document_count: int
    chunk_count: int


class DocumentInfo(BaseModel):
    title: str
    source: str
    chunk_count: int
    preview: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    stream: bool = Field(default=False, description="是否启用 SSE 流式输出")


class Source(BaseModel):
    title: str
    source: str
    score: float
    chunk: str


class HistoryEntry(BaseModel):
    question: str
    answer: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    plan: dict
    history: List[HistoryEntry]


class ToolCall(BaseModel):
    """Agent 工具调用记录"""
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    result: Optional[str] = None


class AgentStep(BaseModel):
    """Agent 推理步骤"""
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[str] = None


class AgentPlan(BaseModel):
    """Agent 推理计划"""
    reasoning: str
    steps: List[AgentStep] = Field(default_factory=list)
    needs_retrieval: bool = False
    needs_multistep: bool = False
    tools_used: List[str] = Field(default_factory=list)
