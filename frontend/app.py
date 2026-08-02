"""
Streamlit 前端 — 企业知识库问答助手。

支持：
- 文档上传与管理
- 同步问答（展示 Agent 推理计划 + 引用来源）
- 流式问答（SSE 逐字输出）
- 对话历史查看
"""

import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="企业知识库问答助手",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 全局样式
st.markdown("""
<style>
    /* 主标题 */
    .main-header {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1a1a1a;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #2b2b2b;
        margin-bottom: 0.5rem;
    }
    .main-header span {
        font-weight: 400;
        color: #666;
        font-size: 0.85rem;
        letter-spacing: 0;
    }
    /* 卡片容器 */
    .card {
        background: #fff;
        border: 1px solid #e8e8e8;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin: 0.5rem 0;
    }
    /* 来源条目 */
    .source-item {
        border-left: 3px solid #2b2b2b;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        background: #fafafa;
    }
    .source-item .score {
        font-size: 0.75rem;
        color: #888;
        font-weight: 500;
    }
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #e8e8e8;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #555;
        margin-top: 1.5rem;
    }
    /* 按钮 */
    .stButton button {
        border-radius: 4px;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    /* 文本输入 */
    textarea {
        border-radius: 4px !important;
        border: 1px solid #ddd !important;
    }
    textarea:focus {
        border-color: #2b2b2b !important;
        box-shadow: 0 0 0 1px #2b2b2b !important;
    }
    /* 状态指示 */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.ok { background: #2ea043; }
    .status-dot.off { background: #cf222e; }
    /* 分割线 */
    hr { margin: 1.25rem 0; border-color: #e8e8e8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State 初始化
# ---------------------------------------------------------------------------
defaults = {
    "uploading": False,
    "last_result": None,
    "streaming": False,
    "stream_answer": "",
    "stream_plan": None,
    "stream_sources": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def fetch_documents():
    try:
        response = requests.get(f"{API_BASE}/documents", timeout=30)
        if response.ok:
            return response.json()
    except Exception:
        pass
    return []


def fetch_history():
    try:
        response = requests.get(f"{API_BASE}/chat/history", timeout=30)
        if response.ok:
            return response.json()
    except Exception:
        pass
    return []


def health_check() -> dict:
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.ok:
            return response.json()
    except Exception:
        pass
    return {"status": "disconnected"}


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    # 系统状态
    health = health_check()
    if health.get("status") == "ok":
        docs_count = health.get("documents", 0)
        chunks_count = health.get("chunks_indexed", 0)
        st.markdown(
            f'<div style="font-size:0.85rem; color:#2ea043; font-weight:500;">'
            f'<span class="status-dot ok"></span> 服务运行中 — {docs_count} 文档, {chunks_count} 分块'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:0.85rem; color:#cf222e; font-weight:500;">'
            f'<span class="status-dot off"></span> 后端未连接'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 文档上传
    st.markdown("**上传文档**")
    uploaded_files = st.file_uploader(
        "选择 PDF 或 TXT 文件",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("建索引", use_container_width=True) and uploaded_files:
        st.session_state.uploading = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            response = requests.post(
                f"{API_BASE}/documents/upload",
                files=[("files", (f.name, f.getvalue())) for f in uploaded_files],
                timeout=120,
            )
            progress_bar.progress(1.0)
            status_text.text("完成")
            if response.ok:
                data = response.json()
                st.success(f"{data['message']}（{data['chunk_count']} 分块）")
                st.rerun()
            else:
                st.error("上传失败")
        except Exception as e:
            st.error(f"连接失败: {e}")
        finally:
            st.session_state.uploading = False

    st.markdown("<hr>", unsafe_allow_html=True)

    # 文档管理
    st.markdown("**文档管理**")
    docs = fetch_documents()
    if docs:
        for doc in docs:
            label = f"{doc['source']} ({doc['chunk_count']} 块)"
            with st.expander(label):
                st.caption(f"标题: {doc['title']}")
                st.text(doc.get("preview", "")[:200] + "..." if len(doc.get("preview", "")) > 200 else doc.get("preview", ""))
                if st.button("删除", key=f"del_{doc['source']}"):
                    try:
                        r = requests.delete(f"{API_BASE}/documents/{doc['source']}", timeout=30)
                        if r.ok:
                            st.success("已删除")
                            st.rerun()
                    except Exception as e:
                        st.error(f"失败: {e}")
    else:
        st.caption("暂无文档")

    st.markdown("<hr>", unsafe_allow_html=True)

    # 对话历史
    st.markdown("**对话历史**")
    history = fetch_history()
    if history:
        for item in reversed(history[-10:]):
            with st.expander(item['question'][:45] + "..."):
                st.write(item["answer"][:300])
                st.caption(f"来源: {item.get('source', '')}")
    else:
        st.caption("暂无历史记录")

    if st.button("清空历史", use_container_width=True):
        try:
            r = requests.delete(f"{API_BASE}/chat/history", timeout=30)
            if r.ok:
                st.success("已清空")
                st.rerun()
        except Exception as e:
            st.error(f"失败: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # 示例问题
    st.markdown("**示例问题**")
    examples = [
        "解释公司新员工入职流程",
        "哪些文档提到审批流程？",
        "比较员工福利制度和考核制度的区别",
        "公司对请假有什么规定？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()


# ---------------------------------------------------------------------------
# 主内容区
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-header">企业知识库问答助手'
    '<span>&nbsp; RAG + Agent · 智能检索与生成</span></div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["问答", "关于系统"])

with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_area(
            "请输入问题",
            height=100,
            placeholder="例如：解释公司的请假审批流程...",
            key="question_input",
            label_visibility="collapsed",
        )
        if "pending_question" in st.session_state and st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = ""

    with col2:
        st.caption("检索设置")
        top_k = st.slider("Top-K", 1, 10, 4)
        use_stream = st.toggle("流式输出", value=True, help="SSE 流式逐字输出")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        submit_clicked = st.button("提交问题", use_container_width=True, type="primary")

    if submit_clicked and question:
        if use_stream:
            # ---------- 流式模式 ----------
            st.session_state.streaming = True
            st.session_state.stream_answer = ""
            st.session_state.stream_plan = None
            st.session_state.stream_sources = []

            plan_placeholder = st.empty()
            answer_placeholder = st.empty()
            sources_placeholder = st.empty()

            try:
                response = requests.post(
                    f"{API_BASE}/query/stream",
                    json={"question": question, "top_k": top_k},
                    stream=True,
                    timeout=120,
                )

                full_answer = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line

                    if line_str.startswith("event: "):
                        event_type = line_str[7:]
                    elif line_str.startswith("data: "):
                        data_str = line_str[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if event_type == "plan":
                            st.session_state.stream_plan = data
                            with plan_placeholder.expander("推理计划", expanded=False):
                                st.json(data)
                        elif event_type == "sources":
                            st.session_state.stream_sources = data
                        elif event_type == "chunk":
                            full_answer += str(data)
                            st.session_state.stream_answer = full_answer
                            answer_placeholder.markdown(
                                f"<div class='card'>{full_answer}<span style='color:#999'>|</span></div>",
                                unsafe_allow_html=True,
                            )
                        elif event_type == "done":
                            break

                answer_placeholder.markdown(
                    f"<div class='card'>{full_answer}</div>",
                    unsafe_allow_html=True,
                )

                if st.session_state.stream_sources:
                    with sources_placeholder.expander("引用来源", expanded=True):
                        for src in st.session_state.stream_sources:
                            st.markdown(
                                f"<div class='source-item'>"
                                f"<strong>{src['title']}</strong>"
                                f"<span class='score'> &nbsp;相似度 {src['score']}</span>"
                                f"<br><span style='font-size:0.85rem;color:#555;'>{src['chunk'][:200]}...</span>"
                                f"<br><span style='font-size:0.75rem;color:#999;'>文件: {src['source']}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                st.success("已保存到对话历史")

            except Exception as e:
                st.error(f"请求失败: {e}")
            finally:
                st.session_state.streaming = False

        else:
            # ---------- 同步模式 ----------
            with st.spinner("检索中..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/query",
                        json={"question": question, "top_k": top_k},
                        timeout=120,
                    )
                    if response.ok:
                        data = response.json()
                        st.session_state.last_result = data
                        st.rerun()
                    else:
                        st.error("请求失败")
                except Exception as e:
                    st.error(f"连接失败: {e}")

    # 显示同步模式结果
    if st.session_state.last_result and not use_stream:
        data = st.session_state.last_result

        st.markdown(f"<div class='card'>{data['answer']}</div>", unsafe_allow_html=True)

        with st.expander("推理计划", expanded=False):
            st.json(data["plan"])

        with st.expander("引用来源", expanded=True):
            for item in data["sources"]:
                st.markdown(
                    f"<div class='source-item'>"
                    f"<strong>{item['title']}</strong>"
                    f"<span class='score'> &nbsp;相似度 {item['score']}</span>"
                    f"<br><span style='font-size:0.85rem;color:#555;'>{item['chunk'][:200]}...</span>"
                    f"<br><span style='font-size:0.75rem;color:#999;'>来源: {item['source']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.success("已保存到对话历史")


with tab2:
    st.markdown("""
    <style>
    .arch-table { width: 100%; border-collapse: collapse; }
    .arch-table td { padding: 8px 16px; border-bottom: 1px solid #eee; }
    .arch-table td:first-child { font-weight: 500; width: 180px; color: #444; }
    </style>

    ### 系统架构

    本系统面向企业知识管理场景，提供基于 **RAG + Agent** 的智能问答能力。

    <table class="arch-table">
    <tr><td>后端框架</td><td>FastAPI (异步)</td></tr>
    <tr><td>前端框架</td><td>Streamlit</td></tr>
    <tr><td>向量索引</td><td>FAISS</td></tr>
    <tr><td>语义检索</td><td>BGE-large-zh-v1.5</td></tr>
    <tr><td>关键词检索</td><td>BM25</td></tr>
    <tr><td>混合融合</td><td>RRF (Reciprocal Rank Fusion)</td></tr>
    <tr><td>重排序</td><td>Cross-Encoder Reranker</td></tr>
    <tr><td>Agent</td><td>ReAct (Reasoning + Acting)</td></tr>
    <tr><td>大语言模型</td><td>DeepSeek Chat API</td></tr>
    <tr><td>流式输出</td><td>Server-Sent Events (SSE)</td></tr>
    </table>

    ### 检索管线

    ```
    用户问题
      -> ReAct Agent 分析
      -> BGE Embedding 向量化
      -> FAISS 语义检索 + BM25 关键词检索 (并行)
      -> RRF 融合排序
      -> Cross-Encoder 重排序
      -> LLM 生成回答 (SSE 流式输出)
      -> 附带来源引用
    ```

    ### Agent 推理

    基于 **ReAct** 范式，Agent 自主完成意图分析、检索决策、多步推理，给出可解释的推理链路。

    ### 版本历史

    - **v1.0** — MVP: 内存检索 + 关键词 Agent
    - **v2.0** — FAISS 索引 + BM25 混合检索 + ReAct Agent + Cross-Encoder Reranker + SSE 流式 + Docker
    """, unsafe_allow_html=True)
