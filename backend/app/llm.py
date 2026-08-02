"""
LLM 服务：支持普通生成 + SSE 流式输出。
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .config import settings


class LLMService:
    """LLM 调用服务，封装 DeepSeek API（兼容 OpenAI 格式）。"""

    def __init__(self) -> None:
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_api_url.rstrip("/")
        self.model = settings.deepseek_model

    # ------------------------------------------------------------------
    # 同步生成（兼容旧接口）
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        plan: dict,
    ) -> str:
        """同步生成回答。"""
        if not self.api_key:
            return self._fallback_answer(question, contexts)

        prompt = self._build_prompt(question, contexts, plan)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一名企业知识库问答助手。请基于给定的上下文回答问题，并明确标注答案依据。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_answer(question, contexts)

    # ------------------------------------------------------------------
    # 流式生成 (SSE)
    # ------------------------------------------------------------------

    async def generate_answer_stream(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        plan: dict,
    ) -> AsyncGenerator[str, None]:
        """异步流式生成回答，逐 token yield 文本片段。"""
        if not self.api_key:
            yield self._fallback_answer(question, contexts)
            return

        prompt = self._build_prompt(question, contexts, plan)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一名企业知识库问答助手。请基于给定的上下文回答问题，并明确标注答案依据。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except Exception:
            yield self._fallback_answer(question, contexts)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_prompt(
        self, question: str, contexts: List[Dict[str, Any]], plan: dict
    ) -> str:
        context_text = "\n\n".join(
            [f"来源: {item['title']} | 内容: {item['chunk']}" for item in contexts]
        )
        return (
            f"问题: {question}\n"
            f"Agent 推理计划: {json.dumps(plan, ensure_ascii=False)}\n"
            f"参考上下文:\n{context_text}\n\n"
            "请用简洁的中文回答，并在结尾注明参考来源。"
        )

    def _fallback_answer(
        self, question: str, contexts: List[Dict[str, Any]]
    ) -> str:
        sources = ", ".join([item["title"] for item in contexts])
        return (
            f"根据检索到的文档内容，我给出以下回答：{question}。"
            f"参考来源包括：{sources}。"
        )
