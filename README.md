# Corporate Knowledge Agent

面向企业知识管理场景的 RAG + Agent 智能问答系统。上传企业内部文档后，系统自动完成分块、向量化与索引构建，通过 ReAct Agent 进行推理决策，结合语义检索与关键词检索的混合召回策略，最终由 LLM 以流式输出生成可追溯来源的回答。

---

## 检索管线

```
用户问题
  -> ReAct Agent 分析意图，决定是否检索及使用何种工具
  -> BGE Embedding 向量化
  -> FAISS 语义检索 + BM25 关键词检索（并行）
  -> RRF 融合排序
  -> Cross-Encoder 重排序
  -> LLM 生成回答（SSE 流式输出）
  -> 返回答案 + 来源引用 + Agent 推理链路
```

## Agent 推理

Agent 采用 **ReAct**（Reasoning + Acting）范式，不再依赖关键词规则，而是由 LLM 自主完成决策。系统定义了四类工具供 Agent 调用：

- `retrieve_documents` — 从知识库检索文档片段
- `compare_documents` — 对比不同文档中的规定或条款
- `summarize_topic` — 汇总某个主题下的所有相关内容
- `no_tools_needed` — 无需检索，直接回答（问候、闲聊等）

Agent 支持多步推理：对复杂问题会多次调用工具，每步输出思考过程，最终形成完整的可解释推理链路。当 API 不可用时，自动降级为关键词规则兜底。

## 混合检索

单一语义检索容易遗漏精确关键词匹配，单一关键词检索则无法理解语义相近的表达。系统并行执行两种检索后，通过 **RRF**（Reciprocal Rank Fusion）融合排序，再由 **Cross-Encoder Reranker** 对候选片段精排，输出时保证文档来源多样性，避免单一文档垄断结果。

## API 端点

后端基于 FastAPI，提供以下接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查，返回索引状态 |
| `GET` | `/documents` | 列出已上传文档及分块信息 |
| `POST` | `/documents/upload` | 上传 PDF/TXT 文件，自动分块索引 |
| `DELETE` | `/documents/{filename}` | 删除文档及其索引 |
| `GET` | `/chat/history` | 获取对话历史 |
| `DELETE` | `/chat/history` | 清空对话历史 |
| `POST` | `/query` | 同步问答，返回完整回答 + 推理计划 + 来源 |
| `POST` | `/query/stream` | SSE 流式问答，逐字输出 |

## 技术栈

| 组件 | 选型 |
|------|------|
| 后端框架 | FastAPI |
| 前端 | Streamlit |
| 向量索引 | FAISS |
| 语义检索 | BGE-large-zh-v1.5 |
| 关键词检索 | BM25（Okapi） |
| 融合策略 | RRF（Reciprocal Rank Fusion） |
| 重排序 | Cross-Encoder Reranker |
| Agent | ReAct + Function Calling |
| LLM | DeepSeek Chat API（兼容 OpenAI 格式） |
| 流式输出 | Server-Sent Events（SSE） |
| 日志 | Loguru |
| 部署 | Docker Compose |

## 项目结构

```
Corporate-Knowledge-Agent/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI 服务入口，SSE 流式端点
│   │   ├── rag.py           # RAG 核心：FAISS + BM25 + RRF + Reranker
│   │   ├── agent.py         # ReAct Agent：LLM Function Calling
│   │   ├── llm.py           # LLM 服务：同步 + 流式生成
│   │   ├── models.py        # Pydantic 数据模型
│   │   └── config.py        # 配置管理（pydantic-settings）
│   ├── data/uploads/        # 上传文档存储
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit UI：同步/流式切换
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   └── bge-large-zh-v1.5/   # 本地 Embedding 模型
├── docker-compose.yml
├── .env.example
└── README.md
```

## 快速开始

### Docker Compose

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

docker compose up -d
```

启动后访问前端 http://localhost:8501，API 文档 http://localhost:8000/docs。

### 手动启动

```bash
# 1. 安装后端依赖
cd backend
pip install -r requirements.txt

# 2. 配置环境变量
cp ../.env.example ../.env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. 启动后端
uvicorn app.main:app --reload --port 8000

# 4. 新终端，启动前端
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### 模型准备

系统需要本地 Embedding 模型，将 BGE-large-zh-v1.5 放入 `models/bge-large-zh-v1.5/` 目录。如需重排序功能，将 Cross-Encoder Reranker 模型放入 `models/bge-reranker-large/`。

Docker Compose 部署时，`models/` 目录以只读方式挂载到容器，无需重复下载。

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|------|------|
| `DEEPSEEK_API_KEY` | 是 | - | DeepSeek API 密钥 |
| `DEEPSEEK_API_URL` | 否 | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | 否 | `deepseek-chat` | 模型名称 |
| `EMBEDDING_MODEL_DIR` | 否 | `./models/bge-large-zh-v1.5` | Embedding 模型路径 |
| `RERANKER_MODEL_DIR` | 否 | `./models/bge-reranker-large` | Reranker 模型路径 |
| `RETRIEVAL_TOP_K` | 否 | `10` | 检索召回数量 |
| `HYBRID_ALPHA` | 否 | `0.5` | 语义检索权重（0~1） |

`.env` 不会被提交到版本控制，项目通过 `python-dotenv` 在启动时从项目根目录自动加载。
