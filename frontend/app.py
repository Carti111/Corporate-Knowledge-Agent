"""
企业知识库问答助手 — 新版前端
三页面布局：问答 | 知识库管理 | 对话管理
"""

import json
import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8001")

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="企业知识库问答助手",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 全局 CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== 基础变量 ===== */
    :root {
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 2px 8px rgba(0,0,0,0.06);
        --shadow-lg: 0 4px 16px rgba(0,0,0,0.08);
        --color-bg: #f7f8fa;
        --color-surface: #ffffff;
        --color-border: #e8ecf1;
        --color-text: #1a1a2e;
        --color-text-secondary: #6b7280;
        --color-accent: #2563eb;
        --color-accent-light: #eff6ff;
        --color-success: #16a34a;
        --color-danger: #dc2626;
        --color-warning: #f59e0b;
    }

    /* ===== 全局重置 ===== */
    .stApp {
        background: var(--color-bg);
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: var(--color-surface);
        border-right: 1px solid var(--color-border);
    }
    [data-testid="stSidebar"] .stButton button {
        border-radius: var(--radius-md) !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.15s ease;
        border: none !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: var(--color-accent-light) !important;
        color: var(--color-accent) !important;
    }

    /* ===== 导航按钮 active 态 ===== */
    .nav-btn-active {
        background: var(--color-accent-light) !important;
        color: var(--color-accent) !important;
        border-left: 3px solid var(--color-accent) !important;
    }

    /* ===== 卡片 ===== */
    .card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    .card-hover:hover {
        box-shadow: var(--shadow-md);
        border-color: #d1d5db;
        transition: all 0.2s ease;
    }

    /* ===== 聊天气泡 ===== */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem 0;
    }
    .chat-bubble {
        display: flex;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .chat-bubble.user {
        justify-content: flex-end;
    }
    .chat-bubble.assistant {
        justify-content: flex-start;
    }
    .chat-bubble .avatar {
        width: 32px; height: 32px;
        border-radius: var(--radius-sm);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: 600;
        flex-shrink: 0;
        margin: 0 8px;
    }
    .chat-bubble.user .avatar {
        background: var(--color-accent);
        color: white;
        order: 2;
    }
    .chat-bubble.assistant .avatar {
        background: #f1f5f9;
        color: var(--color-accent);
        order: 1;
    }
    .chat-bubble .bubble-content {
        max-width: 72%;
        padding: 0.85rem 1.1rem;
        border-radius: var(--radius-lg);
        font-size: 0.92rem;
        line-height: 1.65;
        word-break: break-word;
    }
    .chat-bubble.user .bubble-content {
        background: var(--color-accent);
        color: #fff;
        border-bottom-right-radius: var(--radius-sm);
    }
    .chat-bubble.assistant .bubble-content {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-bottom-left-radius: var(--radius-sm);
        box-shadow: var(--shadow-sm);
    }

    /* ===== 输入区域 ===== */
    .input-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem 0 2rem;
    }
    .input-wrapper {
        display: flex;
        gap: 0.5rem;
        align-items: flex-end;
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-xl);
        padding: 0.5rem 0.5rem 0.5rem 1.2rem;
        box-shadow: var(--shadow-md);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .input-wrapper:focus-within {
        border-color: var(--color-accent);
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
    }
    .input-wrapper textarea {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        resize: none !important;
        font-size: 0.92rem !important;
        padding: 0.4rem 0 !important;
        background: transparent !important;
    }
    .input-wrapper textarea:focus {
        border: none !important;
        box-shadow: none !important;
    }

    /* ===== 按钮 ===== */
    .stButton button {
        border-radius: var(--radius-md) !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stButton button[kind="primary"] {
        background: var(--color-accent) !important;
        border: none !important;
    }
    .stButton button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        box-shadow: var(--shadow-md);
    }

    /* ===== 文档卡片 ===== */
    .doc-card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        transition: all 0.2s ease;
    }
    .doc-card:hover {
        box-shadow: var(--shadow-md);
        border-color: #d1d5db;
    }
    .doc-card .doc-icon {
        width: 40px; height: 40px;
        border-radius: var(--radius-md);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem;
        margin-right: 0.75rem;
    }
    .doc-icon.pdf { background: #fef2f2; color: #dc2626; }
    .doc-icon.txt { background: #eff6ff; color: #2563eb; }
    .doc-icon.csv { background: #f0fdf4; color: #16a34a; }
    .doc-icon.xlsx, .doc-icon.xls { background: #fefce8; color: #ca8a04; }

    /* ===== 对话历史条目 ===== */
    .history-item {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    .history-item:hover {
        box-shadow: var(--shadow-md);
        border-color: #d1d5db;
    }

    /* ===== 状态点 ===== */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.ok { background: var(--color-success); }
    .status-dot.off { background: var(--color-danger); }

    /* ===== 标签 ===== */
    .tag {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .tag-blue { background: #eff6ff; color: #2563eb; }
    .tag-green { background: #f0fdf4; color: #16a34a; }
    .tag-gray { background: #f3f4f6; color: #6b7280; }

    /* ===== 来源引用 ===== */
    .source-card {
        background: #fafbfc;
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .source-card .source-score {
        font-size: 0.7rem;
        color: var(--color-text-secondary);
        background: #f3f4f6;
        padding: 0.1rem 0.5rem;
        border-radius: 100px;
    }

    /* ===== 分割线 ===== */
    hr {
        margin: 1rem 0;
        border-color: var(--color-border);
        opacity: 0.6;
    }

    /* ===== Expander ===== */
    [data-testid="stExpander"] {
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: none !important;
    }

    /* ===== 隐藏默认元素 ===== */
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
DEFAULTS = {
    "page": "问答",
    "messages": [],       # [{"role":"user"|"assistant","content":"...","sources":[...],"plan":{}}]
    "pending_question": "",
    "_clear_input": False,
    "_pending_stream": None,
    "kb_docs": [],
    "kb_health": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def fetch_documents():
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return []


@st.cache_data(ttl=5)
def fetch_history():
    try:
        r = requests.get(f"{API_BASE}/chat/history", timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return []


def health_check():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"status": "disconnected"}


def clear_caches():
    fetch_documents.clear()
    fetch_history.clear()


# ---------------------------------------------------------------------------
# 侧边栏导航
# ---------------------------------------------------------------------------

with st.sidebar:
    # Logo 区域
    st.markdown("""
    <div style="display:flex;align-items:center;padding:0.5rem 0 1rem;gap:0.5rem;">
        <div style="width:36px;height:36px;background:#2563eb;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;
                    color:white;font-size:1.2rem;font-weight:700;">▣</div>
        <div>
            <div style="font-weight:700;font-size:1rem;color:#1a1a2e;">知识库助手</div>
            <div style="font-size:0.72rem;color:#9ca3af;">Enterprise RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 系统状态
    health = health_check()
    if health.get("status") == "ok":
        st.markdown(
            f'<div style="font-size:0.78rem;color:#16a34a;padding:0.3rem 0 0.8rem;">'
            f'<span class="status-dot ok"></span>在线 · {health.get("documents",0)} 文档 · {health.get("chunks_indexed",0)} 分块'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#dc2626;padding:0.3rem 0 0.8rem;">'
            f'<span class="status-dot off"></span>后端未连接'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 导航 — 用按钮模拟
    pages = ["💬  问答", "📚  知识库管理", "📋  对话管理"]
    for p in pages:
        label = p.split("  ")[1]  # 纯文字部分
        is_active = st.session_state.page == label
        btn_style = "primary" if is_active else "secondary"
        if st.button(p, use_container_width=True, type=btn_style, key=f"nav_{label}"):
            st.session_state.page = label
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # 底部设置区
    with st.expander("⚙️ 设置", expanded=False):
        st.session_state["_top_k"] = st.slider("检索条数 Top-K", 1, 10, 6, key="sidebar_topk")
        st.session_state["_use_stream"] = st.toggle("流式输出", value=True, key="sidebar_stream")


# ---------------------------------------------------------------------------
# 页面路由
# ---------------------------------------------------------------------------

page = st.session_state.page

# ===================== 问答页 =====================
if page == "问答":

    top_k = st.session_state.get("_top_k", 6)
    use_stream = st.session_state.get("_use_stream", True)

    # ---- 消息展示区 ----
    chat_area = st.container()

    with chat_area:
        if not st.session_state.messages:
            # 空状态欢迎
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#9ca3af;">
                <div style="font-size:3rem;margin-bottom:1rem;">▣</div>
                <div style="font-size:1.1rem;font-weight:500;color:#6b7280;margin-bottom:0.5rem;">
                    有什么可以帮助你的？
                </div>
                <div style="font-size:0.85rem;">
                    基于 RAG + Agent 的企业知识库智能问答系统
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 示例问题快捷入口
            st.markdown("<div style='max-width:600px;margin:0 auto;'>", unsafe_allow_html=True)
            examples = [
                "解释公司新员工入职流程",
                "公司对请假有什么规定？",
                "比较员工福利制度和考核制度的区别",
                "哪些文档提到审批流程？",
            ]
            cols = st.columns(2)
            for i, ex in enumerate(examples):
                with cols[i % 2]:
                    if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                        # 按钮在 text_area 之前渲染，可直接设置 chat_input
                        st.session_state.chat_input = ex
                        st.session_state.pending_question = ""
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # 渲染聊天记录
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                is_user = role == "user"

                avatar_letter = "U" if is_user else "AI"
                align_class = "user" if is_user else "assistant"

                st.markdown(f"""
                <div class="chat-bubble {align_class}">
                    <div class="avatar">{avatar_letter}</div>
                    <div class="bubble-content">{content}</div>
                </div>
                """, unsafe_allow_html=True)

                # 非用户消息：显示来源和推理计划
                if not is_user:
                    sources = msg.get("sources", [])
                    plan = msg.get("plan")

                    if sources:
                        with st.expander("📎 引用来源", expanded=False):
                            for src in sources:
                                score_pct = f"{src['score']:.0%}" if src['score'] <= 1 else f"{src['score']:.2f}"
                                st.markdown(f"""
                                <div class="source-card">
                                    <div style="display:flex;justify-content:space-between;align-items:center;">
                                        <strong>{src['title']}</strong>
                                        <span class="source-score">{score_pct}</span>
                                    </div>
                                    <div style="color:#6b7280;margin-top:0.3rem;font-size:0.82rem;">
                                        {src['chunk'][:180]}...
                                    </div>
                                    <div style="color:#9ca3af;font-size:0.72rem;margin-top:0.2rem;">
                                        📄 {src['source']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    if plan and plan.get("steps"):
                        with st.expander("🧠 Agent 推理计划", expanded=False):
                            st.caption(f"推理: {plan.get('reasoning', '')}")
                            for step_idx, step in enumerate(plan["steps"]):
                                st.markdown(
                                    f"**Step {step_idx+1}**: {step.get('thought', '')}\n\n"
                                    f"→ 工具: `{step.get('action', {}).get('tool_name', 'none')}`"
                                )

        # ---- 流式生成：主渲染流程中渐进输出 ----
        if st.session_state.get("_pending_stream"):
            q, _top_k = st.session_state._pending_stream
            st.session_state._pending_stream = None

            stream_meta = {"plan": None, "sources": []}

            def sse_generator():
                try:
                    response = requests.post(
                        f"{API_BASE}/query/stream",
                        json={"question": q, "top_k": _top_k},
                        stream=True,
                        timeout=120,
                    )
                    event_type = None
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                        if line_str.startswith("event: "):
                            event_type = line_str[7:]
                        elif line_str.startswith("data: "):
                            try:
                                data = json.loads(line_str[6:])
                            except json.JSONDecodeError:
                                continue
                            if event_type == "plan":
                                stream_meta["plan"] = data
                            elif event_type == "sources":
                                stream_meta["sources"] = data
                            elif event_type == "chunk":
                                yield str(data)
                            elif event_type == "done":
                                break
                except Exception as e:
                    yield f"\n\n> ⚠️ 请求失败: {e}"

            # 流式输出 AI 回答（生成完成后 rerun 会以气泡样式重新渲染）
            with st.chat_message("assistant", avatar="🤖"):
                full_answer = st.write_stream(sse_generator())

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer or "（未获取到回答）",
                "sources": stream_meta["sources"],
                "plan": stream_meta["plan"],
            })
            st.rerun()

    # ---- 输入区（固定底部效果） ----
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)

    # 在 text_area widget 渲染之前修改其 session_state 值
    # （Streamlit 禁止在 widget 渲染后修改，所以必须前置）
    if st.session_state.get("_clear_input"):
        if "chat_input" in st.session_state:
            st.session_state.chat_input = ""
        st.session_state._clear_input = False

    pending = st.session_state.get("pending_question", "")
    if pending:
        if "chat_input" in st.session_state:
            st.session_state.chat_input = pending
        st.session_state.pending_question = ""

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_area(
            "输入问题",
            placeholder="输入你的问题，按 Ctrl+Enter 发送...",
            height=68,
            key="chat_input",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_send, col_settings = st.columns([2, 1])
        with col_send:
            send_clicked = st.button("发送 ↵", use_container_width=True, type="primary")
        with col_settings:
            with st.popover("⚙"):
                st.session_state["_top_k"] = st.slider("Top-K", 1, 10, top_k, key="inline_topk")
                st.session_state["_use_stream"] = st.toggle("流式", use_stream, key="inline_stream")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- 发送逻辑 ----
    if send_clicked and question.strip():
        question = question.strip()
        top_k = st.session_state.get("_top_k", 6)
        use_stream = st.session_state.get("_use_stream", True)

        # 添加用户消息，标记下次渲染前清空输入框
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state._clear_input = True

        if use_stream:
            # 流式：设标志位，让主渲染流程中的 st.write_stream 接管
            st.session_state._pending_stream = (question, top_k)
        else:
            # 同步模式：直接请求
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "top_k": top_k},
                    timeout=120,
                )
                if resp.ok:
                    data = resp.json()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "plan": data.get("plan"),
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "请求失败，请稍后重试。",
                        "sources": [],
                        "plan": None,
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"连接失败: {e}",
                    "sources": [],
                    "plan": None,
                })

        st.rerun()


# ===================== 知识库管理页 =====================
elif page == "知识库管理":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h2 style="margin:0;font-weight:700;">📚 知识库管理</h2>
        <p style="color:#6b7280;margin:0.3rem 0 0;font-size:0.9rem;">
            上传、查看和删除企业文档，构建专属知识库
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 状态概览
    health = health_check()
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("📄 文档数量", health.get("documents", 0) if health.get("status") == "ok" else "—")
    with col_stat2:
        st.metric("🧩 知识分块", health.get("chunks_indexed", 0) if health.get("status") == "ok" else "—")
    with col_stat3:
        st.metric("🔌 服务状态", "🟢 在线" if health.get("status") == "ok" else "🔴 离线")

    st.markdown("<hr>", unsafe_allow_html=True)

    # 上传区域
    st.markdown("### 上传文档")
    upload_col1, upload_col2 = st.columns([3, 1])
    with upload_col1:
        uploaded_files = st.file_uploader(
            "选择 PDF 或 TXT 文件",
            type=["pdf", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="kb_uploader",
        )
    with upload_col2:
        st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
        do_upload = st.button("🚀 上传并建索引", use_container_width=True, type="primary")

    if do_upload and uploaded_files:
        with st.spinner("正在处理文档..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/documents/upload",
                    files=[("files", (f.name, f.getvalue())) for f in uploaded_files],
                    timeout=120,
                )
                if resp.ok:
                    data = resp.json()
                    st.success(f"✅ {data['message']}（新增 {data['chunk_count']} 分块）")
                    clear_caches()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("上传失败，请检查后端服务")
            except Exception as e:
                st.error(f"连接失败: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # 文档列表
    st.markdown("### 已索引文档")
    docs = fetch_documents()

    if not docs:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#9ca3af;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📭</div>
            <div>暂无文档，请上传 PDF 或 TXT 文件</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # 卡片网格
        cols = st.columns(2)
        for i, doc in enumerate(docs):
            with cols[i % 2]:
                ext = doc["source"].split(".")[-1] if "." in doc["source"] else "txt"
                icon_map = {
                    "pdf": ("pdf", "📕"),
                    "csv": ("csv", "📊"),
                    "xlsx": ("xlsx", "📈"),
                    "xls": ("xls", "📈"),
                }
                icon_class, icon = icon_map.get(ext, ("txt", "📄"))

                st.markdown(f"""
                <div class="doc-card card-hover" style="margin-bottom:0.75rem;">
                    <div style="display:flex;align-items:flex-start;">
                        <div class="doc-icon {icon_class}">{icon}</div>
                        <div style="flex:1;min-width:0;">
                            <div style="font-weight:600;font-size:0.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                                {doc['title']}
                            </div>
                            <div style="font-size:0.78rem;color:#9ca3af;margin-top:0.15rem;">
                                {doc['source']} · {doc['chunk_count']} 分块
                            </div>
                            <div style="font-size:0.8rem;color:#6b7280;margin-top:0.5rem;line-height:1.4;
                                        display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">
                                {doc.get('preview', '')[:120]}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 删除按钮
                col_del, _ = st.columns([1, 3])
                with col_del:
                    if st.button("🗑 删除", key=f"del_kb_{i}", use_container_width=True):
                        try:
                            r = requests.delete(f"{API_BASE}/documents/{doc['source']}", timeout=30)
                            if r.ok:
                                st.success("已删除")
                                clear_caches()
                                time.sleep(0.5)
                                st.rerun()
                        except Exception as e:
                            st.error(f"失败: {e}")


# ===================== 对话管理页 =====================
elif page == "对话管理":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h2 style="margin:0;font-weight:700;">📋 对话管理</h2>
        <p style="color:#6b7280;margin:0.3rem 0 0;font-size:0.9rem;">
            浏览、搜索和管理历史对话记录
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = fetch_history()

    # 操作栏
    col_info, col_search, col_clear = st.columns([2, 2, 1])
    with col_info:
        st.markdown(f"共 **{len(history)}** 条对话")
    with col_search:
        search_term = st.text_input("🔍 搜索", placeholder="输入关键词...", label_visibility="collapsed", key="history_search")
    with col_clear:
        if st.button("🗑 清空全部", use_container_width=True):
            try:
                r = requests.delete(f"{API_BASE}/chat/history", timeout=30)
                if r.ok:
                    st.success("已清空")
                    clear_caches()
                    # 同时清空当前会话消息
                    st.session_state.messages = []
                    time.sleep(0.5)
                    st.rerun()
            except Exception as e:
                st.error(f"失败: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#9ca3af;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">💬</div>
            <div>暂无对话记录</div>
            <div style="font-size:0.85rem;margin-top:0.3rem;">去问答页面开始第一次对话吧</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # 搜索过滤
        if search_term:
            history = [h for h in history if search_term.lower() in h.get("question", "").lower()
                       or search_term.lower() in h.get("answer", "").lower()]

        if not history:
            st.caption("未找到匹配的对话")
        else:
            # 列表展示
            for i, item in enumerate(reversed(history)):
                q = item.get("question", "")
                a = item.get("answer", "")
                src = item.get("source", "")
                ts = item.get("timestamp", "")

                with st.expander(f"❓ {q[:60]}{'...' if len(q) > 60 else ''}", expanded=False):
                    st.markdown(f"""
                    <div style="background:#f9fafb;border-radius:12px;padding:1rem;margin-bottom:0.5rem;">
                        <div style="font-size:0.78rem;color:#9ca3af;margin-bottom:0.3rem;">🕐 {ts}</div>
                        <div style="font-weight:500;color:#1a1a2e;margin-bottom:0.5rem;">回答</div>
                        <div style="color:#374151;line-height:1.6;">{a[:500]}</div>
                        <div style="margin-top:0.8rem;">
                            <span class="tag tag-gray">📎 {src[:40] if src else '无来源'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 加载到问答页
                    if st.button("💬 继续这个话题", key=f"reload_{i}"):
                        st.session_state.pending_question = q
                        st.session_state.page = "问答"
                        st.rerun()

    # 导出功能
    if history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.download_button(
            "📥 导出对话历史 (JSON)",
            data=json.dumps(history, ensure_ascii=False, indent=2),
            file_name=f"chat_history_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
