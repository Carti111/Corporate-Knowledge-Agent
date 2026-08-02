# Corporate Knowledge Agent

面向企业知识管理场景的 **RAG + Agent 智能问答系统**。

该项目集成了：
- FastAPI 后端
- Streamlit 前端
- FAISS 向量检索
- BGE 中文 Embedding
- BM25 关键词召回
- Cross-Encoder 重排
- DeepSeek / OpenAI 兼容的 LLM 接口
- SSE 流式回答输出

## 项目目标

用于企业内部文档知识问答场景，支持：
- 文档上传与分块索引
- 文档检索 + Agent 推理
- 多轮问答历史记录
- Docker 一键部署

## 🚀 v2.0 核心升级

| 模块 | v1.0 (旧) | v2.0 (新) |
|------|-----------|-----------|
| **Agent** | 关键词 `if/else` 假 Agent | **ReAct** 范式 — LLM 自主推理 + Function Calling |
| **向量检索** | 内存 list + 暴力余弦相似度 | **FAISS** (Facebook AI Similarity Search) |
| **检索策略** | 单一语义检索 | **混合检索** (BGE 语义 + BM25 关键词) + RRF 融合 |
| **排序** | 无 | **Cross-Encoder Reranker** 重排序 |
| **LLM 输出** | 同步返回全文 | **SSE 流式**逐字输出 |
| **部署** | 手动启动两个进程 | **Docker Compose** 一键部署 |

## 🔧 技术栈

| 组件 | 技术选型 |
|------|----------|
| 后端框架 | FastAPI (异步) |
| 前端框架 | Streamlit |
| 向量索引 | **FAISS** |
| 语义检索 | BGE-large-zh-v1.5 (本地 Embedding) |
| 关键词检索 | **BM25** (Okapi) |
| 混合融合 | **RRF** (Reciprocal Rank Fusion) |
| 重排序 | **Cross-Encoder Reranker** |
| Agent 范式 | **ReAct** (Reasoning + Acting) |
| LLM | DeepSeek Chat API (兼容 OpenAI 格式) |
| 流式输出 | **Server-Sent Events (SSE)** |
| 部署 | **Docker + Docker Compose** |
| 日志 | Loguru 结构化日志 |

## 🔄 检索管线

```
用户问题
  → ReAct Agent 分析（LLM 自主决策：是否需要检索、用什么工具、是否多步推理）
  → BGE Embedding 向量化
  → FAISS 语义检索 + BM25 关键词检索（并行）
  → RRF 融合排序
  → Cross-Encoder 重排序
  → LLM 生成回答（SSE 流式逐字输出）
  → 附带来源引用 + Agent 推理链路
```

## 📁 目录结构

```
Ragagent/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI 服务（含 SSE 流式端点）
│   │   ├── rag.py            # RAG 核心（FAISS + BM25 + RRF + Reranker）
│   │   ├── agent.py          # ReAct Agent（LLM Function Calling）
│   │   ├── llm.py            # LLM 服务（同步 + 流式）
│   │   ├── models.py         # Pydantic 数据模型
│   │   └── config.py         # 配置管理（pydantic-settings）
│   ├── data/
│   │   ├── uploads/          # 上传文档目录
│   │   └── sample_docs/      # 示例文档
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                # Streamlit UI（支持流式/同步切换）
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   └── bge-large-zh-v1.5/    # 本地 Embedding 模型
├── docker-compose.yml        # 一键部署
└── README.md
```

## 🚀 快速运行

### 方式一：Docker Compose

```bash
# 1. 复制环境变量模板
cp .env.example .env
# 编辑 .env，至少补充 DEEPSEEK_API_KEY

# 2. 一键启动
docker compose up -d

# 3. 访问
# 前端: http://localhost:8501
# 后端 API: http://localhost:8000/docs
```

### 方式二：手动启动

```bash
# 1. 安装依赖
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

## 🔐 环境变量说明

请在项目根目录复制一份 [.env.example](.env.example) 为 `.env`，并填写：
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_API_URL`（可选）
- `DEEPSEEK_MODEL`（可选）

注意：
- `.env` 不会上传到 GitHub
- 代码通过 `load_dotenv` 从项目根目录读取本地配置

## ✅ 代码检查结论

本项目核心源码已完成语法级验证，前后端入口文件均为可执行状态。执行实际检验时，`python -m py_compile` 对关键 Python 文件没有报错，因此这次提交适合作为 GitHub 的公开代码仓库快照。