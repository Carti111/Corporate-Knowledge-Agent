# Corporate Knowledge Agent

企业级 **RAG + Agent** 智能知识问答系统，支持文档上传、混合检索、LLM 推理与流式回答。

## 核心能力

- 📄 **文档管理** — 上传企业文档，自动分块与向量化索引
- 🔍 **混合检索** — BGE 语义检索 + BM25 关键词检索，RRF 融合排序
- 🧠 **ReAct Agent** — LLM 自主推理 + Function Calling，决定是否检索、使用何种工具
- 📊 **Cross-Encoder 重排** — 对召回结果精排，提升回答质量
- ⚡ **SSE 流式输出** — 逐字返回，即时响应
- 🐳 **一键部署** — Docker Compose 开箱即用

## 架构

```
用户问题
  → ReAct Agent（LLM 自主决策）
  → BGE Embedding 向量化
  → FAISS 语义检索 + BM25 关键词检索（并行）
  → RRF 融合
  → Cross-Encoder 重排序
  → LLM 生成回答（SSE 流式输出）
  → 返回答案 + 来源引用 + 推理链路
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 后端框架 | FastAPI（异步） |
| 前端 | Streamlit |
| 向量索引 | FAISS |
| 语义检索 | BGE-large-zh-v1.5 |
| 关键词检索 | BM25（Okapi） |
| 融合策略 | RRF（Reciprocal Rank Fusion） |
| 重排序 | Cross-Encoder Reranker |
| Agent 范式 | ReAct（Reasoning + Acting） |
| LLM | DeepSeek Chat API（兼容 OpenAI 格式） |
| 流式输出 | Server-Sent Events（SSE） |
| 部署 | Docker + Docker Compose |

## 快速开始

### Docker Compose（推荐）

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

docker compose up -d
```

访问：
- 前端：http://localhost:8501
- API 文档：http://localhost:8000/docs

### 手动启动

```bash
# 后端
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 前端（新终端）
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 目录结构

```
Corporate-Knowledge-Agent/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI 服务 & SSE 端点
│   │   ├── rag.py          # RAG 核心（FAISS + BM25 + RRF + Reranker）
│   │   ├── agent.py        # ReAct Agent（LLM Function Calling）
│   │   ├── llm.py          # LLM 服务（同步 + 流式）
│   │   ├── models.py       # Pydantic 数据模型
│   │   └── config.py       # 配置管理
│   ├── data/uploads/       # 上传文档
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py              # Streamlit UI
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   └── bge-large-zh-v1.5/  # 本地 Embedding 模型
├── docker-compose.yml
└── .env.example
```

## 环境变量

复制 [.env.example](.env.example) 为 `.env` 并填写：

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | ✅ | API 密钥 |
| `DEEPSEEK_API_URL` | ❌ | 自定义 API 地址（默认官方） |
| `DEEPSEEK_MODEL` | ❌ | 模型名称（默认 `deepseek-chat`） |

> `.env` 不会被提交到 Git，项目通过 `load_dotenv` 从根目录加载。
