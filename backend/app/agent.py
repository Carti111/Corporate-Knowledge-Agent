"""
ReAct (Reasoning + Acting) Agent  —  真正的 LLM 驱动的 Agent 推理路由。

不再使用关键词匹配的假 Agent，而是基于 DeepSeek function calling
让 LLM 自主决定：是否需要检索、调用什么工具、是否需要多步推理。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx

from .models import AgentPlan, AgentStep, ToolCall

# ---------------------------------------------------------------------------
# Tool 定义 (符合 OpenAI / DeepSeek function-calling 规范)
# ---------------------------------------------------------------------------

TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "从企业知识库中检索与问题相关的文档片段。"
                "当用户询问公司制度、流程、政策、规定等需要查阅文档的问题时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "优化后的检索查询，可以是关键词或自然语言问题",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的文档片段数量，默认 4",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_documents",
            "description": (
                "对比不同文档中的规定或条款。"
                "当用户要求比较、区别、对比多个制度或条款时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_a": {
                        "type": "string",
                        "description": "第一个要对比的主题",
                    },
                    "topic_b": {
                        "type": "string",
                        "description": "第二个要对比的主题",
                    },
                },
                "required": ["topic_a", "topic_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": (
                "汇总某个主题下的所有相关文档内容。"
                "当用户要求总结、汇总、概述某个制度或主题时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "需要汇总的主题名称",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_tools_needed",
            "description": (
                "当问题不需要任何工具或检索即可回答时调用。"
                "例如：问候、感谢、简单闲聊、或者关于系统本身的询问。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# System prompt 指导 LLM 如何进行 ReAct 推理
AGENT_SYSTEM_PROMPT = """\
你是一个企业知识库的智能 Agent。你的任务是分析用户的问题，决定使用什么工具来获取信息。

## 推理规则

1. **需要检索 (retrieve_documents)**: 用户提出的任何需要从知识库中查找信息的问题。
   这包括但不限于：企业制度、流程、政策、规定、表格数据、CSV 文件内容、
   文档中的具体数字/日期/人名、统计信息等。只要不是简单的闲聊问候，
   都应优先考虑检索。

2. **需要对比 (compare_documents)**: 用户要求比较两个或多个制度/条款/数据的异同。

3. **需要汇总 (summarize_topic)**: 用户要求总结、汇总某个主题。

4. **无需工具 (no_tools_needed)**: 仅限于以下情况：
   - 简单的问候（你好、早上好）
   - 感谢（谢谢）
   - 道别（再见）
   - 询问系统本身功能的问题
   **注意：任何涉及具体数据、文档内容、文件查询的问题，都必须先检索！**

## 决策要求

- 先思考再行动：每次只能调用一个工具
- **重要：默认优先使用 retrieve_documents**——宁可多检索，不可漏检索
- 如果问题简单直接，一步就够了
- 如果问题复杂（需要比较、需要同时查多个方面），则分多步
- 最后一个 action 必须是 no_tools_needed，表示推理结束

请用中文进行推理和回答。"""


# ---------------------------------------------------------------------------
# Agent Router
# ---------------------------------------------------------------------------

@dataclass
class ReActAgent:
    """ReAct Agent：让 LLM 自主决策工具调用。

    工作流程：
    1. 将用户问题 + 可用工具列表发给 LLM
    2. LLM 返回 thought + tool_call
    3. 执行 tool_call，将结果作为 observation 追加
    4. 重复步骤 2-3 直到 LLM 认为任务完成
    5. 返回完整的推理链
    """

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_steps: int = 5
    tools: List[dict] = field(default_factory=lambda: TOOLS)

    def plan(self, question: str) -> AgentPlan:
        """执行 ReAct 循环，返回完整的推理计划。"""
        if not self.api_key:
            return self._fallback_plan(question)

        messages: List[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题：{question}\n\n请分析并决定使用什么工具（如果需要的话）。"},
        ]

        plan = AgentPlan(reasoning="", steps=[], needs_retrieval=False, needs_multistep=False, tools_used=[])
        step_count = 0

        try:
            while step_count < self.max_steps:
                step_count += 1

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tools,
                    "temperature": 0.1,
                    "stream": False,
                }

                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                choice = data["choices"][0]
                msg = choice["message"]

                # 记录 LLM 的思考
                thought = msg.get("content", "") or "正在分析..."
                tool_calls = msg.get("tool_calls", [])

                if not tool_calls:
                    # LLM 决定不需要更多工具调用
                    plan.reasoning = thought
                    break

                # 处理工具调用
                for tc in tool_calls:
                    func = tc["function"]
                    tool_name = func["name"]
                    args = json.loads(func.get("arguments", "{}"))

                    # 记录工具调用
                    tool_call = ToolCall(tool_name=tool_name, arguments=args)

                    # 模拟执行工具（实际执行由 RAGService 完成）
                    if tool_name == "retrieve_documents":
                        plan.needs_retrieval = True
                        plan.tools_used.append("retrieve_documents")
                        tool_call.result = f"检索完成：查询 \"{args.get('query', question)}\""
                    elif tool_name == "compare_documents":
                        plan.needs_retrieval = True
                        plan.needs_multistep = True
                        plan.tools_used.append("compare_documents")
                        tool_call.result = f"对比完成：{args.get('topic_a', '')} vs {args.get('topic_b', '')}"
                    elif tool_name == "summarize_topic":
                        plan.needs_retrieval = True
                        plan.tools_used.append("summarize_topic")
                        tool_call.result = f"汇总完成：{args.get('topic', '')}"
                    elif tool_name == "no_tools_needed":
                        plan.tools_used.append("no_tools_needed")
                        tool_call.result = "无需工具，可直接回答"
                    else:
                        tool_call.result = f"未知工具: {tool_name}"

                    step = AgentStep(
                        thought=thought if thought else f"调用 {tool_name}",
                        action=tool_call,
                        observation=tool_call.result,
                    )
                    plan.steps.append(step)

                    # 把 assistant 的 tool_call 和 tool result 追加到消息历史
                    messages.append({
                        "role": "assistant",
                        "content": thought,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tool_name, "arguments": func.get("arguments", "{}")},
                            }
                        ],
                    })
                    messages.append({
                        "role": "tool",
                        "content": tool_call.result,
                        "tool_call_id": tc["id"],
                    })

                if not plan.needs_retrieval and "no_tools_needed" in plan.tools_used:
                    break

                # 如果已经调用了工具，完成循环
                if plan.tools_used and plan.tools_used[-1] != "no_tools_needed":
                    # 让 LLM 做最终总结
                    messages.append({
                        "role": "user",
                        "content": "请基于以上工具执行结果，给出你的最终判断：任务是否完成？是否需要更多工具调用？如已完成，请调用 no_tools_needed。",
                    })
                    continue

                break

        except Exception:
            return self._fallback_plan(question)

        if not plan.reasoning:
            plan.reasoning = "基于 ReAct 推理完成工具调用决策"

        return plan

    def _fallback_plan(self, question: str) -> AgentPlan:
        """当 API 不可用时的降级推理。"""
        lowered = question.lower()
        needs_retrieval = not any(
            token in lowered for token in ["你好", "早上好", "下午好", "晚上好", "谢谢", "再见"]
        )
        needs_multistep = any(
            token in lowered for token in ["区别", "比较", "为什么", "流程", "步骤", "如何", "怎么办", "分析"]
        )
        uses_tools = any(
            token in lowered for token in ["汇总", "列表", "统计", "总结", "最新", "对比"]
        )

        step = AgentStep(
            thought="无需 API 的降级推理：根据关键词判断是否需要检索",
            action=ToolCall(
                tool_name="retrieve_documents" if needs_retrieval else "no_tools_needed",
                arguments={"query": question},
            ),
            observation="降级模式：根据关键词规则判断",
        )

        return AgentPlan(
            reasoning="降级模式：API 不可用，使用关键词规则进行判断",
            steps=[step],
            needs_retrieval=needs_retrieval,
            needs_multistep=needs_multistep,
            tools_used=["retrieve_documents"] if needs_retrieval else [],
        )


# 向后兼容的别名
AgentRouter = ReActAgent
