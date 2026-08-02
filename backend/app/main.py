"""
FastAPI 主服务 — 企业知识库 RAG 问答系统。

提供：
- 文档管理 (上传/列表/删除)
- 对话历史管理
- 同步问答 (POST /query)
- 流式问答 SSE (POST /query/stream)
"""

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .models import (
    AgentPlan,
    DocumentInfo,
    DocumentUploadResponse,
    HistoryEntry,
    QueryRequest,
    QueryResponse,
    Source,
)
from .rag import RAGService, extract_text_from_file

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="面向企业知识管理场景的 RAG + Agent 智能问答系统，支持 FAISS 向量检索、混合检索、Cross-Encoder 重排序和 SSE 流式输出。",
)

# 单例 RAG 服务
rag_service = RAGService()


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": "1.0.0",
        "chunks_indexed": len(rag_service.chunks),
        "documents": len(rag_service.documents),
    }


# ---------------------------------------------------------------------------
# 文档管理
# ---------------------------------------------------------------------------

@app.get("/documents", response_model=List[DocumentInfo])
def list_documents() -> List[DocumentInfo]:
    """列出所有已上传的文档及其分块信息。"""
    return [
        DocumentInfo(
            title=item["title"],
            source=item["source"],
            chunk_count=item["chunk_count"],
            preview=item.get("preview", ""),
        )
        for item in rag_service.list_documents()
    ]


@app.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_documents(files: List[UploadFile] = File(...)) -> DocumentUploadResponse:
    """上传 PDF 或 TXT 文档，自动分块、embedding 并构建 FAISS 索引。"""
    saved_files: List[Path] = []
    chunk_count = 0

    for upload in files:
        if not upload.filename:
            continue
        destination = settings.upload_dir / upload.filename
        with destination.open("wb") as buffer:
            buffer.write(upload.file.read())
        saved_files.append(destination)
        logger.info(f"文件已保存: {upload.filename}")

    for file_path in saved_files:
        text = extract_text_from_file(file_path)
        title = file_path.stem
        chunk_count += rag_service.ingest_text(
            title=title, content=text, source=file_path.name
        )

    return DocumentUploadResponse(
        message="文档已成功上传并构建 FAISS 索引",
        document_count=len(saved_files),
        chunk_count=chunk_count,
    )


@app.delete("/documents/{filename}")
def delete_document(filename: str) -> JSONResponse:
    """删除指定文档及其对应的 FAISS 索引。"""
    target_path = settings.upload_dir / filename
    if target_path.exists():
        target_path.unlink(missing_ok=True)
    deleted = rag_service.delete_document(filename)
    return JSONResponse(
        status_code=200 if deleted else 404,
        content={"message": "文档已删除，索引已更新" if deleted else "文档不存在"},
    )


# ---------------------------------------------------------------------------
# 对话历史
# ---------------------------------------------------------------------------

@app.get("/chat/history", response_model=List[HistoryEntry])
def get_history() -> List[HistoryEntry]:
    """获取对话历史。"""
    return [
        HistoryEntry(
            question=item["question"],
            answer=item["answer"],
            source=item.get("source", ""),
        )
        for item in rag_service.get_history()
    ]


@app.delete("/chat/history")
def clear_history() -> JSONResponse:
    """清空对话历史。"""
    rag_service.clear_history()
    return JSONResponse(status_code=200, content={"message": "对话历史已清空"})


# ---------------------------------------------------------------------------
# 同步问答
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """同步问答：提交问题，返回完整回答 + Agent 推理计划 + 引用来源。"""
    logger.info(f"收到问题: {request.question[:80]}...")
    result = rag_service.answer_question(request.question, top_k=request.top_k)
    history = [
        HistoryEntry(
            question=item["question"],
            answer=item["answer"],
            source=item.get("source", ""),
        )
        for item in rag_service.get_history()
    ]
    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        plan=result["plan"],
        history=history,
    )


# ---------------------------------------------------------------------------
# 流式问答 (SSE)
# ---------------------------------------------------------------------------

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式问答 (Server-Sent Events)。

    事件类型：
    - plan: Agent 推理计划
    - sources: 检索到的文档来源
    - chunk: LLM 生成的文本片段
    - done: 生成完成

    前端可通过 EventSource 或 fetch + ReadableStream 消费。
    """

    async def event_generator():
        async for event in rag_service.answer_question_stream(
            request.question, top_k=request.top_k
        ):
            yield {
                "event": event["type"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

    logger.info(f"收到流式问题: {request.question[:80]}...")
    return EventSourceResponse(event_generator())
