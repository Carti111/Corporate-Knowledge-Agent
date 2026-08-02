"""
RAG 服务：FAISS 向量索引 + BM25 混合检索 + Cross-Encoder 重排序。

架构：
  用户问题
    ├── Agent Router（ReAct 推理 → 决定是否检索）
    ├── 语义检索 (BGE-large-zh + FAISS)
    ├── 关键词检索 (BM25)
    ├── RRF 融合排序
    ├── Cross-Encoder 重排序
    └── LLM 生成回答（支持 SSE 流式）
"""

import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:  # pragma: no cover
    SentenceTransformer = None
    CrossEncoder = None

from loguru import logger

from .agent import ReActAgent
from .config import settings
from .llm import LLMService


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    id: str
    content: str
    source: str
    title: str
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FAISS 向量索引封装
# ---------------------------------------------------------------------------

class FAISSIndex:
    """FAISS 向量索引，支持增量添加和搜索。"""

    def __init__(self, dim: int = 1024, index_path: Optional[Path] = None):
        self.dim = dim
        self.index_path = index_path
        self.index = None
        self.id_map: Dict[int, int] = {}  # FAISS 内部 ID → chunks 列表索引
        self._init_faiss()

    def _init_faiss(self):
        """初始化 FAISS 索引（使用 Inner Product 模拟 Cosine Similarity）。"""
        try:
            import faiss

            # 使用 IndexFlatIP (Inner Product)，配合 L2 归一化向量 = Cosine Similarity
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            logger.info(f"FAISS 索引已初始化，维度: {self.dim}")
        except Exception as e:
            logger.warning(f"FAISS 初始化失败: {e}，将回退到暴力检索")
            self.index = None

    def add(self, vectors: List[List[float]], internal_ids: List[int]) -> None:
        """添加向量到索引。"""
        if self.index is None or not vectors:
            return
        try:
            import faiss
            import numpy as np

            vec_array = np.array(vectors, dtype=np.float32)
            id_array = np.array(internal_ids, dtype=np.int64)
            self.index.add_with_ids(vec_array, id_array)
        except Exception as e:
            logger.error(f"FAISS add 失败: {e}")

    def search(self, query_vector: List[float], top_k: int = 10) -> List[tuple]:
        """搜索返回 (internal_id, score) 列表。"""
        if self.index is None or not query_vector:
            return []
        try:
            import numpy as np

            q = np.array([query_vector], dtype=np.float32)
            scores, ids = self.index.search(q, top_k)
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx >= 0:
                    results.append((int(idx), float(score)))
            return results
        except Exception as e:
            logger.error(f"FAISS search 失败: {e}")
            return []

    def remove_ids(self, ids: List[int]) -> None:
        """从索引中移除指定 ID。"""
        if self.index is None or not ids:
            return
        try:
            import numpy as np

            id_array = np.array(ids, dtype=np.int64)
            self.index.remove_ids(id_array)
        except Exception as e:
            logger.error(f"FAISS remove_ids 失败: {e}")

    def save(self) -> None:
        """持久化索引到磁盘。"""
        if self.index is None or self.index_path is None:
            return
        try:
            import faiss

            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"FAISS 索引已保存到 {self.index_path}")
        except Exception as e:
            logger.warning(f"FAISS 索引保存失败: {e}")

    def load(self) -> bool:
        """从磁盘加载索引。"""
        if self.index_path is None or not self.index_path.exists():
            return False
        try:
            import faiss

            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS 索引已从 {self.index_path} 加载")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# BM25 关键词检索
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 关键词检索器，提供稀疏检索能力。"""

    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_refs: List[DocumentChunk] = []

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """用当前分块构建 BM25 索引。"""
        self.chunk_refs = list(chunks)
        self.tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception as e:
            logger.warning(f"BM25 索引构建失败: {e}")
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """搜索返回 (chunk_index, score) 列表。"""
        if self.bm25 is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        # 归一化 BM25 分数到 [0, 1]
        max_score = max(scores) if len(scores) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0
        return [(idx, float(score / max_score)) for idx, score in ranked]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文友好的分词（简单按字符 + 英文单词切分）。"""
        # 中文字符单独切分，英文单词保留
        tokens = []
        for char in text:
            if char.isalnum() or char.isspace():
                tokens.append(char.lower())
            else:
                tokens.append(" ")  # 标点替换为空格
        text_normalized = "".join(tokens)
        # 过滤过短的 token
        return [t for t in text_normalized.split() if len(t) >= 1]


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion) 融合
# ---------------------------------------------------------------------------

def _apply_fallback_scores(
    chunks: List[DocumentChunk],
    fallback_scores: Optional[Dict[int, float]] = None,
) -> List[tuple]:
    """当 Reranker 不可用时，用 RRF 融合分数替代 0.0 分数。

    如果连 fallback_scores 都没有，则按原始候选顺序分配递减分数。
    """
    if fallback_scores:
        # 获取每个 chunk 在原 chunks 列表中的索引对应的融合分数
        scored = []
        for i, chunk in enumerate(chunks):
            # fallback_scores 的 key 是 chunks 全局索引，这里用 enumerate 的 i 近似
            score = fallback_scores.get(i, 0.0)
            if score == 0.0:
                # 按位置给递减分数
                score = round(1.0 / (i + 1), 4)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    # 完全没有 fallback：按位置递减
    return [(chunk, round(1.0 / (i + 1), 4)) for i, chunk in enumerate(chunks)]


def _diverse_select(
    sorted_fused: List[tuple],
    chunks: List[DocumentChunk],
    max_slots: int = 30,
) -> List[int]:
    """多样性候选选择：每文档至少取 top-1，剩余按分数填充。

    这样即使某个文档语义匹配度极高，也不会垄断所有候选位。
    """
    # Round 1: 每个 unique source 取最高分的 chunk
    source_best: Dict[str, tuple] = {}  # source → (global_idx, score)
    for global_idx, score in sorted_fused:
        if global_idx >= len(chunks):
            continue
        source = chunks[global_idx].source
        if source not in source_best or score > source_best[source][1]:
            source_best[source] = (global_idx, score)

    selected: List[int] = []
    seen: set = set()

    # 先放入每文档的 top-1
    for global_idx, _ in sorted(source_best.values(), key=lambda x: x[1], reverse=True):
        selected.append(global_idx)
        seen.add(global_idx)

    # Round 2: 剩余按分数填充（跳过已选的）
    for global_idx, _ in sorted_fused:
        if len(selected) >= max_slots:
            break
        if global_idx >= len(chunks):
            continue
        if global_idx not in seen:
            selected.append(global_idx)
            seen.add(global_idx)

    return selected


def _diverse_final_output(
    reranked: List[tuple],
    top_k: int = 4,
) -> List[dict]:
    """最终多样性输出：每文档至少取 top-1，剩余按重排序分数填充。"""
    # Round 1: 每个 source 取最高分
    source_best: Dict[str, tuple] = {}  # source → (chunk, score)
    source_others: List[tuple] = []  # 其余 chunk

    for chunk, score in reranked:
        source = chunk.source
        if source not in source_best:
            source_best[source] = (chunk, round(float(score), 4))
        else:
            source_others.append((chunk, round(float(score), 4)))

    # 先输出每文档 top-1
    results: List[dict] = []
    seen_content: set = set()
    for chunk, score in sorted(source_best.values(), key=lambda x: x[1], reverse=True):
        content_key = chunk.content[:80]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        results.append({
            "score": score,
            "chunk": chunk.content,
            "source": chunk.source,
            "title": chunk.title,
            "page": chunk.page,
        })
        if len(results) >= top_k:
            return results

    # Round 2: 剩余按分数填充
    for chunk, score in sorted(source_others, key=lambda x: x[1], reverse=True):
        if len(results) >= top_k:
            break
        content_key = chunk.content[:80]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        results.append({
            "score": score,
            "chunk": chunk.content,
            "source": chunk.source,
            "title": chunk.title,
            "page": chunk.page,
        })

    return results


def reciprocal_rank_fusion(
    semantic_results: List[tuple],
    bm25_results: List[tuple],
    k: int = 60,
    alpha: float = 0.5,
) -> Dict[int, float]:
    """
    RRF 融合语义检索和关键词检索的结果。

    Args:
        semantic_results: [(chunk_idx, score), ...]
        bm25_results: [(chunk_idx, score), ...]
        k: RRF 平滑参数
        alpha: 语义检索权重 (0~1)，bm25 权重 = 1-alpha

    Returns:
        {chunk_idx: fused_score, ...}
    """
    scores: Dict[int, float] = {}

    # 语义检索 - 用 rank + score
    for rank, (idx, score) in enumerate(semantic_results):
        rrf_score = alpha * (1.0 / (k + rank + 1))
        scores[idx] = scores.get(idx, 0.0) + rrf_score + alpha * score * 0.1

    # BM25 关键词检索
    for rank, (idx, score) in enumerate(bm25_results):
        rrf_score = (1 - alpha) * (1.0 / (k + rank + 1))
        scores[idx] = scores.get(idx, 0.0) + rrf_score + (1 - alpha) * score * 0.1

    return scores


# ---------------------------------------------------------------------------
# Cross-Encoder 重排序
# ---------------------------------------------------------------------------

class Reranker:
    """Cross-Encoder 重排序器。"""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or settings.reranker_model_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        if CrossEncoder is None:
            return
        if not self.model_dir or not os.path.isdir(self.model_dir):
            logger.info("Reranker 模型未找到，将跳过重排序")
            return
        try:
            self.model = CrossEncoder(self.model_dir)
            logger.info(f"Reranker 模型加载成功: {self.model_dir}")
        except Exception as e:
            logger.warning(f"Reranker 加载失败: {e}")
            self.model = None

    def rerank(
        self,
        question: str,
        chunks: List[DocumentChunk],
        top_k: int = 4,
        fallback_scores: Optional[Dict[int, float]] = None,
    ) -> List[tuple]:
        """重排序返回 [(chunk, score), ...]，按分数降序。

        fallback_scores: {chunk_index: score} 当 reranker 不可用时使用 RRF 融合分数作为回退。
        """
        if self.model is None or not chunks:
            logger.info("Reranker 不可用，使用 RRF 融合分数")
            return _apply_fallback_scores(chunks, fallback_scores)[:top_k]

        try:
            pairs = [(question, chunk.content) for chunk in chunks]
            scores = self.model.predict(pairs)
            if isinstance(scores, float):
                scores = [scores]
            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"Reranker 执行失败: {e}")
            return _apply_fallback_scores(chunks, fallback_scores)[:top_k]


# ---------------------------------------------------------------------------
# Embedding 提供者
# ---------------------------------------------------------------------------

class LocalEmbeddingProvider:
    """本地 Embedding 模型（BGE-large-zh-v1.5）。"""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = model_dir or settings.embedding_model_dir
        self.model = None
        self._dim = 1024  # BGE-large 默认维度
        self._load_model()

    @property
    def dim(self) -> int:
        return self._dim

    def _load_model(self) -> None:
        if SentenceTransformer is None:
            return
        if not self.model_dir or not os.path.isdir(self.model_dir):
            logger.warning(f"Embedding 模型目录不存在: {self.model_dir}")
            return
        try:
            self.model = SentenceTransformer(self.model_dir)
            # 探测实际维度
            test_vec = self.model.encode(["test"], convert_to_numpy=True)
            self._dim = test_vec.shape[1]
            logger.info(f"Embedding 模型加载成功，维度: {self._dim}")
        except Exception as e:
            logger.warning(f"Embedding 模型加载失败: {e}")
            self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表。"""
        if self.model is not None:
            try:
                vectors = self.model.encode(
                    list(texts),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                return [list(map(float, vector)) for vector in vectors]
            except Exception as e:
                logger.error(f"Embedding 编码失败: {e}")
        return self._fallback_embed(texts)

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """TF-IDF 风格的降级向量化。"""
        vectors: List[List[float]] = []
        for text in texts:
            tokens = re.findall(r"\w+", text.lower())
            if not tokens:
                vectors.append([0.0])
                continue
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            vector = [float(frequencies[token]) for token in sorted(frequencies)]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


# ---------------------------------------------------------------------------
# RAG 服务主类
# ---------------------------------------------------------------------------

class RAGService:
    """企业知识库 RAG 服务。

    检索管线：
    1. Agent 分析问题 → 决定是否检索
    2. 语义检索 (FAISS) + 关键词检索 (BM25) → RRF 融合
    3. Cross-Encoder 重排序
    4. LLM 生成回答
    """

    def __init__(self) -> None:
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []
        self.documents: List[dict] = []
        self.history: List[dict] = []

        # 组件初始化
        self.embedding_provider = LocalEmbeddingProvider()
        self.faiss_index = FAISSIndex(
            dim=self.embedding_provider.dim,
            index_path=Path(settings.faiss_index_path),
        )
        self.bm25_retriever = BM25Retriever()
        self.reranker = Reranker()
        self.agent_router = ReActAgent(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_url,
            model=settings.deepseek_model,
        )
        self.llm_service = LLMService()

        self.history_path = settings.upload_dir.parent / "conversation_history.json"

        # 启动加载
        self._load_history()
        self._try_load_index()
        self._auto_ingest()

    # ------------------------------------------------------------------
    # 文档入库
    # ------------------------------------------------------------------

    def ingest_text(self, title: str, content: str, source: str) -> int:
        """入库一篇文档：分块 → embedding → FAISS → BM25。"""
        chunks = chunk_text(content, title=title, source=source, chunk_size=300, overlap=50)
        if not chunks:
            return 0

        # Embedding
        new_embeddings = self.embedding_provider.embed([chunk.content for chunk in chunks])

        # FAISS 索引
        start_id = len(self.chunks)
        faiss_ids = list(range(start_id, start_id + len(chunks)))
        self.faiss_index.add(new_embeddings, faiss_ids)

        # 更新内存
        self.chunks.extend(chunks)
        self.embeddings.extend(new_embeddings)

        # 重建 BM25（BM25 需要全量索引）
        self.bm25_retriever.build_index(self.chunks)

        # 保存
        self._save_index()

        preview = content[:300].replace("\n", " ")
        self.documents.append({
            "title": title,
            "source": source,
            "chunk_count": len(chunks),
            "preview": preview,
        })

        logger.info(f"文档入库完成: {title} → {len(chunks)} 个分块")
        return len(chunks)

    def delete_document(self, source: str) -> bool:
        """删除文档及其所有分块。"""
        if not source:
            return False

        # 找出需要删除的分块索引
        indices_to_remove = [
            i for i, chunk in enumerate(self.chunks) if chunk.source == source
        ]
        if not indices_to_remove:
            return False

        # FAISS 移除
        self.faiss_index.remove_ids(indices_to_remove)

        # 更新内存（注意：删除后索引会重新编号，需要重建 FAISS）
        new_chunks = []
        new_embeddings = []
        for i, (chunk, emb) in enumerate(zip(self.chunks, self.embeddings)):
            if chunk.source != source:
                new_chunks.append(chunk)
                new_embeddings.append(emb)

        self.chunks = new_chunks
        self.embeddings = new_embeddings
        self.documents = [d for d in self.documents if d.get("source") != source]

        # 重建 FAISS（因为 ID 重新编号）
        self._rebuild_faiss()
        self.bm25_retriever.build_index(self.chunks)
        self._save_index()

        logger.info(f"文档已删除: {source}")
        return True

    def list_documents(self) -> List[dict]:
        return list(self.documents)

    # ------------------------------------------------------------------
    # 检索管线
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 4) -> List[dict]:
        """完整的检索管线：语义 + BM25 → RRF → Reranker，带文档多样性保证。"""
        if not self.chunks:
            return []

        retrieval_top_k = max(settings.retrieval_top_k, top_k * 3)

        # 1. 语义检索 (FAISS)
        query_vector = self.embedding_provider.embed([question])[0]
        semantic_results = self.faiss_index.search(query_vector, top_k=retrieval_top_k)

        # 2. 关键词检索 (BM25)
        bm25_results = self.bm25_retriever.search(question, top_k=retrieval_top_k)

        # 3. RRF 融合
        fused = reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            alpha=settings.hybrid_alpha,
        )

        # 按融合分数排序
        sorted_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # 4. 多样性候选选择：每文档至少取 top-1，剩余按分数填充
        candidate_indices = _diverse_select(
            sorted_fused, self.chunks, max_slots=retrieval_top_k
        )
        candidate_chunks = [self.chunks[idx] for idx in candidate_indices if idx < len(self.chunks)]

        # 4b. 文档覆盖兜底：文档少时确保每份文档至少有一条候选
        if len(self.documents) <= 20:
            covered_sources = {self.chunks[idx].source for idx in candidate_indices}
            all_sources = {doc["source"] for doc in self.documents}
            missing = all_sources - covered_sources
            for src in missing:
                # 从该文档的 chunk 中取第一个（它们同属一个文档，内容有代表性）
                for i, chunk in enumerate(self.chunks):
                    if chunk.source == src:
                        candidate_chunks.append(chunk)
                        candidate_indices.append(i)
                        break

        # 构建 chunk 在候选列表中的位置 → RRF 融合分数 的映射
        fallback_scores: Dict[int, float] = {}
        for pos, idx in enumerate(candidate_indices):
            if idx in fused:
                fallback_scores[pos] = fused[idx]

        # 5. Cross-Encoder 重排序（传入 fallback 分数）
        rerank_k = max(settings.rerank_top_k, top_k * 2)
        reranked = self.reranker.rerank(
            question, candidate_chunks, top_k=rerank_k, fallback_scores=fallback_scores
        )

        # 6. 最终多样性输出：每文档至少取 top-1，剩余按重排序分数填充
        results = _diverse_final_output(reranked, top_k=top_k)

        # 6b. 输出端文档覆盖兜底：确保每份文档至少有一条
        #     如果 results 已满，替换掉最低分的重复来源项
        covered_sources = {r["source"] for r in results}
        all_sources = {doc["source"] for doc in self.documents}
        missing_sources = all_sources - covered_sources
        if missing_sources and len(self.documents) <= 20:
            for src in missing_sources:
                # 找到该文档的代表 chunk（优先 reranked，再 candidate，再全量）
                found_data = None
                for chunk, score in reranked:
                    if chunk.source == src:
                        found_data = {
                            "score": round(float(score), 4),
                            "chunk": chunk.content,
                            "source": chunk.source,
                            "title": chunk.title,
                            "page": chunk.page,
                        }
                        break
                if found_data is None:
                    for chunk in candidate_chunks:
                        if chunk.source == src:
                            found_data = {
                                "score": 0.01, "chunk": chunk.content,
                                "source": chunk.source, "title": chunk.title,
                                "page": chunk.page,
                            }
                            break
                if found_data is None:
                    for chunk in self.chunks:
                        if chunk.source == src:
                            found_data = {
                                "score": 0.01, "chunk": chunk.content,
                                "source": chunk.source, "title": chunk.title,
                                "page": chunk.page,
                            }
                            break
                if found_data is None:
                    continue

                if len(results) < top_k:
                    results.append(found_data)
                else:
                    # 替换掉最低分的、且来源有重复的结果
                    # 按 score 升序找第一个来源重复的项
                    for i in sorted(range(len(results)), key=lambda j: results[j]["score"]):
                        src_i = results[i]["source"]
                        if sum(1 for r in results if r["source"] == src_i) > 1:
                            results[i] = found_data
                            break

        logger.info(
            f"检索完成: 语义={len(semantic_results)}, BM25={len(bm25_results)}, "
            f"融合候选={len(candidate_chunks)}, 最终={len(results)}, "
            f"覆盖文档={len(set(r['source'] for r in results))}/{len(self.documents)}"
        )
        return results

    # ------------------------------------------------------------------
    # Agent 推理
    # ------------------------------------------------------------------

    def plan_query(self, question: str) -> dict:
        """Agent 分析问题并生成推理计划。"""
        plan = self.agent_router.plan(question)
        plan_dict = {
            "reasoning": plan.reasoning,
            "steps": [
                {
                    "thought": step.thought,
                    "action": {
                        "tool_name": step.action.tool_name if step.action else "none",
                        "arguments": step.action.arguments if step.action else {},
                        "result": step.action.result if step.action else "",
                    },
                    "observation": step.observation or "",
                }
                for step in plan.steps
            ],
            "needs_retrieval": plan.needs_retrieval,
            "needs_multistep": plan.needs_multistep,
            "tools_used": plan.tools_used,
        }
        if not plan_dict.get("reasoning"):
            plan_dict["reasoning"] = (
                "需要检索企业文档并生成可追溯回答"
                if plan_dict["needs_retrieval"]
                else "直接回答，无需检索"
            )
        return plan_dict

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------

    def answer_question(self, question: str, top_k: int = 4) -> dict:
        """同步问答接口。"""
        plan = self.plan_query(question)

        if not plan["needs_retrieval"]:
            return {
                "answer": "你好，我可以协助你查询企业知识库中的内容。请直接提出一个关于制度、流程或文档的问题。",
                "sources": [],
                "plan": plan,
            }

        contexts = self.retrieve(question, top_k=top_k)
        if not contexts:
            return {
                "answer": "当前知识库中还没有可用文档，请先上传企业文档后再提问。",
                "sources": [],
                "plan": plan,
            }

        answer = self.llm_service.generate_answer(question, contexts, plan)
        sources = [
            {
                "title": item["title"],
                "source": item["source"],
                "score": item["score"],
                "chunk": item["chunk"],
            }
            for item in contexts
        ]
        self.append_history(
            question,
            answer,
            ", ".join(item["source"] for item in sources),
        )
        return {"answer": answer, "sources": sources, "plan": plan}

    # ------------------------------------------------------------------
    # 流式问答 (SSE)
    # ------------------------------------------------------------------

    async def answer_question_stream(self, question: str, top_k: int = 4):
        """异步流式问答：先检索，再流式生成回答。"""
        plan = self.plan_query(question)

        # 发送 plan
        yield {"type": "plan", "data": plan}

        if not plan["needs_retrieval"]:
            yield {
                "type": "chunk",
                "data": "你好，我可以协助你查询企业知识库中的内容。请直接提出一个关于制度、流程或文档的问题。",
            }
            yield {"type": "done", "data": ""}
            return

        contexts = self.retrieve(question, top_k=top_k)
        if not contexts:
            yield {"type": "chunk", "data": "当前知识库中还没有可用文档，请先上传企业文档后再提问。"}
            yield {"type": "done", "data": ""}
            return

        # 发送 sources
        sources = [
            {
                "title": item["title"],
                "source": item["source"],
                "score": item["score"],
                "chunk": item["chunk"],
            }
            for item in contexts
        ]
        yield {"type": "sources", "data": sources}

        # 流式生成
        full_answer = ""
        async for token in self.llm_service.generate_answer_stream(question, contexts, plan):
            full_answer += token
            yield {"type": "chunk", "data": token}

        # 保存历史
        self.append_history(
            question,
            full_answer,
            ", ".join(item["source"] for item in sources),
        )

        yield {"type": "done", "data": ""}

    # ------------------------------------------------------------------
    # 对话历史
    # ------------------------------------------------------------------

    def append_history(self, question: str, answer: str, source: str) -> None:
        entry = {
            "question": question,
            "answer": answer,
            "source": source,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.history.append(entry)
        self._save_history()

    def get_history(self) -> List[dict]:
        return list(self.history)

    def clear_history(self) -> None:
        self.history = []
        self._save_history()

    def _load_history(self) -> None:
        if not self.history_path.exists():
            return
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                self.history = json.load(handle)
        except Exception:
            self.history = []

    def _save_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # FAISS 索引持久化
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        self.faiss_index.save()
        # 同时保存 chunks metadata（FAISS 只存向量，不存文本）
        meta_path = Path(settings.faiss_index_path).with_suffix(".meta.json")
        meta = {
            "documents": self.documents,
            "chunks": [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "title": chunk.title,
                    "page": chunk.page,
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks
            ],
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _try_load_index(self) -> None:
        """尝试从磁盘恢复索引。"""
        if self.faiss_index.load():
            meta_path = Path(settings.faiss_index_path).with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.documents = meta.get("documents", [])
                    self.chunks = [
                        DocumentChunk(
                            id=c["id"],
                            content=c["content"],
                            source=c["source"],
                            title=c["title"],
                            page=c.get("page"),
                            metadata=c.get("metadata", {}),
                        )
                        for c in meta.get("chunks", [])
                    ]
                    self.bm25_retriever.build_index(self.chunks)
                    logger.info(f"从磁盘恢复 {len(self.chunks)} 个分块")
                except Exception as e:
                    logger.warning(f"索引元数据恢复失败: {e}")

    def _rebuild_faiss(self) -> None:
        """删除文档后重建 FAISS 索引。"""
        self.faiss_index = FAISSIndex(
            dim=self.embedding_provider.dim,
            index_path=Path(settings.faiss_index_path),
        )
        if self.embeddings:
            ids = list(range(len(self.chunks)))
            self.faiss_index.add(self.embeddings, ids)
        self._save_index()

    def _auto_ingest(self) -> None:
        """启动时自动扫描 upload 目录并建索引。"""
        upload_dir = settings.upload_dir
        if not upload_dir.exists():
            return
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in (".txt", ".pdf"):
                filename = file_path.name
                if any(doc["source"] == filename for doc in self.documents):
                    continue
                try:
                    text = extract_text_from_file(file_path)
                    title = file_path.stem
                    self.ingest_text(title=title, content=text, source=filename)
                except Exception as e:
                    logger.error(f"自动入库失败 {filename}: {e}")


# ---------------------------------------------------------------------------
# 文本分块（改进版：sliding window + overlap）
# ---------------------------------------------------------------------------

def chunk_text(
    content: str,
    title: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[DocumentChunk]:
    """改进的分块策略：基于中文自然段落 + sliding window overlap。

    - 优先按段落（\\n\\n）切分
    - 过长的段落按字符滑动窗口切分
    - overlap 保证上下文连贯
    """
    normalized = re.sub(r"\s+", " ", content).strip()
    if not normalized:
        return []

    # 先按段落切分
    paragraphs = normalized.split("。")
    paragraphs = [p.strip() + "。" for p in paragraphs if p.strip()]

    chunks: List[DocumentChunk] = []
    buffer = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(buffer) + len(para) <= chunk_size:
            buffer += para
        else:
            # 保存当前 buffer
            if len(buffer.strip()) >= 20:
                chunks.append(
                    DocumentChunk(
                        id=f"{title}-{chunk_idx}",
                        content=buffer.strip(),
                        source=source,
                        title=title,
                        page=None,
                    )
                )
                chunk_idx += 1

                # overlap: 保留 buffer 末尾 overlap 个字符作为新 buffer 的开头
                if overlap > 0 and len(buffer) > overlap:
                    buffer = buffer[-overlap:] + para
                else:
                    buffer = para
            else:
                buffer += para

    # 最后一个 buffer
    if len(buffer.strip()) >= 20:
        chunks.append(
            DocumentChunk(
                id=f"{title}-{chunk_idx}",
                content=buffer.strip(),
                source=source,
                title=title,
                page=None,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# PDF 文本提取
# ---------------------------------------------------------------------------

def extract_text_from_file(path: Path) -> str:
    """从 PDF 或 TXT 文件中提取文本。"""
    if path.suffix.lower() == ".pdf":
        if PdfReader is None:
            raise RuntimeError("当前环境缺少 pypdf，请先安装 backend/requirements.txt 中的依赖")
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")
