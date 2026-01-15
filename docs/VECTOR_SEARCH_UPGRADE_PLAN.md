# Memory 系统向量化搜索升级方案

> **版本**: v1.0  
> **日期**: 2025-12-20  
> **状态**: 设计阶段

## 1. 概述

### 1.1 背景

当前 EpisodicMemory 使用关键词匹配进行搜索，存在以下局限：
- **语义理解不足**：无法理解同义词、近义词
- **数据冗余**：`reasoning` 和 `reasons` 字段重复存储
- **检索精度有限**：依赖硬编码关键词列表

### 1.2 目标

构建**生产级向量搜索系统**，实现：
- 语义级别的决策历史检索
- 消除 `reasoning` 字段冗余
- 支持复杂查询（如"类似的市场环境下的成功决策"）
- 为未来 RAG（检索增强生成）奠定基础

### 1.3 设计原则

1. **渐进式升级**：保持向后兼容，支持回退
2. **轻量级优先**：优先使用本地模型，避免 API 依赖
3. **生产就绪**：考虑性能、存储、可维护性
4. **可扩展架构**：支持未来接入更强大的模型

---

## 2. 技术选型

### 2.1 Embedding 模型对比

| 模型 | 维度 | 大小 | 速度 | 质量 | 推荐场景 |
|------|------|------|------|------|---------|
| **all-MiniLM-L6-v2** | 384 | 80MB | ⚡⚡⚡ | ⭐⭐⭐ | 开发/测试 |
| **bge-small-en-v1.5** | 384 | 130MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 生产推荐 |
| **bge-base-en-v1.5** | 768 | 440MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 高精度需求 |
| **text-embedding-3-small** | 1536 | API | ⚡ | ⭐⭐⭐⭐⭐ | 最高质量 |
| **nomic-embed-text-v1.5** | 768 | 550MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 长文本支持 |

**推荐选择**：`bge-small-en-v1.5`
- 中英文双语支持
- 本地运行，无 API 成本
- 在 MTEB 排行榜表现优异
- 384 维度平衡了质量和存储

### 2.2 向量数据库对比

| 方案 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| **Qdrant** | 独立服务 | 高性能、支持过滤、Rust 实现 | 生产环境 |
| **ChromaDB** | 嵌入式 | 简单易用、Python 原生 | 快速原型 |
| **LanceDB** | 嵌入式 | 零配置、支持 SQL | 中小规模 |
| **FAISS** | 库 | 极致性能、Facebook 出品 | 大规模检索 |
| **SQLite-VSS** | 扩展 | 与现有 SQLite 集成 | 轻量级 |

**推荐选择**：`LanceDB`
- 零配置，嵌入式部署
- 支持元数据过滤（按 symbol、date、action）
- 原生支持 pandas/pyarrow
- 自动持久化，无需额外服务

### 2.3 混合检索策略

采用 **Hybrid Search** 架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Input                            │
│              "类似 GS 突破行情的成功买入"                      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Vector Search  │ │  Keyword Search │ │ Metadata Filter │
│  (语义相似度)    │ │  (BM25/关键词)   │ │ (symbol/action) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Reciprocal Rank Fusion                    │
│                      (RRF 融合排序)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Re-ranking (可选)                         │
│              使用 Cross-Encoder 精排                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Top-K Results                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 架构设计

### 3.1 系统架构

```
stockbench/memory/
├── layers/
│   ├── episodic.py          # 现有：关键词搜索（保留兼容）
│   ├── episodic_vector.py   # 新增：向量搜索层
│   └── working.py           # 现有：工作记忆
├── embeddings/
│   ├── __init__.py
│   ├── base.py              # 抽象基类
│   ├── local_embedder.py    # 本地模型（bge/MiniLM）
│   ├── openai_embedder.py   # OpenAI API（可选）
│   └── cache.py             # Embedding 缓存
├── vectordb/
│   ├── __init__.py
│   ├── base.py              # 抽象基类
│   ├── lancedb_backend.py   # LanceDB 实现
│   └── qdrant_backend.py    # Qdrant 实现（可选）
├── search/
│   ├── __init__.py
│   ├── hybrid_search.py     # 混合检索
│   ├── reranker.py          # 重排序器
│   └── query_parser.py      # 查询解析
├── schemas.py               # 数据模型
└── store.py                 # 统一入口
```

### 3.2 核心类设计

#### 3.2.1 Embedder 抽象

```python
# stockbench/memory/embeddings/base.py

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class BaseEmbedder(ABC):
    """Embedding 模型抽象基类"""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        生成文本向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            向量数组，shape: (n, dimension)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        生成查询向量（可能有不同的处理）
        
        Args:
            query: 查询文本
            
        Returns:
            向量数组，shape: (dimension,)
        """
        pass
```

#### 3.2.2 本地 Embedder 实现

```python
# stockbench/memory/embeddings/local_embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np
from .base import BaseEmbedder

class LocalEmbedder(BaseEmbedder):
    """本地 Embedding 模型"""
    
    # 支持的模型配置
    MODELS = {
        "bge-small": {
            "name": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "query_prefix": "Represent this sentence for searching relevant passages: "
        },
        "bge-base": {
            "name": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "query_prefix": "Represent this sentence for searching relevant passages: "
        },
        "minilm": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "query_prefix": ""
        },
        "nomic": {
            "name": "nomic-ai/nomic-embed-text-v1.5",
            "dimension": 768,
            "query_prefix": "search_query: "
        }
    }
    
    def __init__(self, model_key: str = "bge-small", device: str = "auto"):
        """
        Args:
            model_key: 模型标识符
            device: 运行设备 ("auto", "cpu", "cuda")
        """
        config = self.MODELS.get(model_key)
        if not config:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODELS.keys())}")
        
        self._model_name = config["name"]
        self._dimension = config["dimension"]
        self._query_prefix = config["query_prefix"]
        
        # 延迟加载模型
        self._model = None
        self._device = device
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def _load_model(self):
        if self._model is None:
            device = self._device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._model = SentenceTransformer(self._model_name, device=device)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(texts, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        # BGE 模型需要添加查询前缀
        prefixed_query = self._query_prefix + query
        return self._model.encode(prefixed_query, normalize_embeddings=True)
```

#### 3.2.3 向量数据库抽象

```python
# stockbench/memory/vectordb/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None

class BaseVectorDB(ABC):
    """向量数据库抽象基类"""
    
    @abstractmethod
    def create_collection(self, name: str, dimension: int, **kwargs):
        """创建集合"""
        pass
    
    @abstractmethod
    def insert(self, collection: str, ids: List[str], vectors: np.ndarray, 
               metadata: List[Dict[str, Any]], texts: Optional[List[str]] = None):
        """插入向量"""
        pass
    
    @abstractmethod
    def search(self, collection: str, query_vector: np.ndarray, 
               top_k: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """向量搜索"""
        pass
    
    @abstractmethod
    def delete(self, collection: str, ids: List[str]):
        """删除向量"""
        pass
    
    @abstractmethod
    def update_metadata(self, collection: str, id: str, metadata: Dict[str, Any]):
        """更新元数据"""
        pass
```

#### 3.2.4 LanceDB 实现

```python
# stockbench/memory/vectordb/lancedb_backend.py

import lancedb
import pyarrow as pa
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseVectorDB, SearchResult

class LanceDBBackend(BaseVectorDB):
    """LanceDB 向量数据库实现"""
    
    def __init__(self, db_path: str = "storage/memory/vectordb"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._collections = {}
    
    def create_collection(self, name: str, dimension: int, **kwargs):
        """创建或获取集合"""
        if name not in self._collections:
            # 定义 schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dimension)),
                pa.field("text", pa.string()),
                pa.field("symbol", pa.string()),
                pa.field("action", pa.string()),
                pa.field("date", pa.string()),
                pa.field("confidence", pa.float32()),
                pa.field("actual_result", pa.float32()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("metadata", pa.string()),  # JSON 序列化的完整元数据
            ])
            
            try:
                table = self.db.open_table(name)
            except:
                table = self.db.create_table(name, schema=schema)
            
            self._collections[name] = table
        
        return self._collections[name]
    
    def insert(self, collection: str, ids: List[str], vectors: np.ndarray,
               metadata: List[Dict[str, Any]], texts: Optional[List[str]] = None):
        """插入向量"""
        import json
        
        table = self._collections.get(collection)
        if table is None:
            raise ValueError(f"Collection {collection} not found")
        
        data = []
        for i, (id_, vector, meta) in enumerate(zip(ids, vectors, metadata)):
            row = {
                "id": id_,
                "vector": vector.tolist(),
                "text": texts[i] if texts else "",
                "symbol": meta.get("symbol", ""),
                "action": meta.get("action", ""),
                "date": meta.get("date", ""),
                "confidence": float(meta.get("confidence", 0.5)),
                "actual_result": float(meta.get("actual_result", 0.0)) if meta.get("actual_result") else 0.0,
                "tags": meta.get("tags", []),
                "metadata": json.dumps(meta, ensure_ascii=False),
            }
            data.append(row)
        
        table.add(data)
    
    def search(self, collection: str, query_vector: np.ndarray,
               top_k: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """向量搜索（支持元数据过滤）"""
        import json
        
        table = self._collections.get(collection)
        if table is None:
            raise ValueError(f"Collection {collection} not found")
        
        # 构建查询
        query = table.search(query_vector.tolist()).limit(top_k)
        
        # 应用过滤条件
        if filters:
            filter_conditions = []
            if "symbol" in filters:
                filter_conditions.append(f"symbol = '{filters['symbol']}'")
            if "action" in filters:
                filter_conditions.append(f"action = '{filters['action']}'")
            if "date_gte" in filters:
                filter_conditions.append(f"date >= '{filters['date_gte']}'")
            if "min_confidence" in filters:
                filter_conditions.append(f"confidence >= {filters['min_confidence']}")
            
            if filter_conditions:
                query = query.where(" AND ".join(filter_conditions))
        
        results = query.to_pandas()
        
        return [
            SearchResult(
                id=row["id"],
                score=1 - row["_distance"],  # LanceDB 返回距离，转换为相似度
                metadata=json.loads(row["metadata"]),
                text=row["text"]
            )
            for _, row in results.iterrows()
        ]
    
    def delete(self, collection: str, ids: List[str]):
        """删除向量"""
        table = self._collections.get(collection)
        if table:
            table.delete(f"id IN {tuple(ids)}")
    
    def update_metadata(self, collection: str, id: str, metadata: Dict[str, Any]):
        """更新元数据（用于回填 actual_result）"""
        import json
        
        table = self._collections.get(collection)
        if table:
            # LanceDB 通过删除+插入实现更新
            existing = table.search().where(f"id = '{id}'").limit(1).to_pandas()
            if len(existing) > 0:
                row = existing.iloc[0].to_dict()
                row["metadata"] = json.dumps(metadata, ensure_ascii=False)
                row["actual_result"] = float(metadata.get("actual_result", 0.0)) if metadata.get("actual_result") else 0.0
                table.delete(f"id = '{id}'")
                table.add([row])
```

#### 3.2.5 混合搜索实现

```python
# stockbench/memory/search/hybrid_search.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from ..embeddings.base import BaseEmbedder
from ..vectordb.base import BaseVectorDB, SearchResult

@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    id: str
    vector_score: float
    keyword_score: float
    final_score: float
    metadata: Dict[str, Any]
    text: str

class HybridSearch:
    """
    混合搜索引擎
    
    结合向量搜索和关键词搜索，使用 RRF 融合排序
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vectordb: BaseVectorDB,
        collection: str,
        rrf_k: int = 60  # RRF 参数
    ):
        self.embedder = embedder
        self.vectordb = vectordb
        self.collection = collection
        self.rrf_k = rrf_k
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_reranker: bool = False
    ) -> List[HybridSearchResult]:
        """
        混合搜索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 元数据过滤条件
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重
            use_reranker: 是否使用重排序
            
        Returns:
            排序后的搜索结果
        """
        # 1. 向量搜索
        query_vector = self.embedder.embed_query(query)
        vector_results = self.vectordb.search(
            self.collection, 
            query_vector, 
            top_k=top_k * 3,  # 多取一些用于融合
            filters=filters
        )
        
        # 2. 关键词搜索（BM25 简化版）
        keyword_results = self._keyword_search(query, top_k * 3, filters)
        
        # 3. RRF 融合
        fused_results = self._rrf_fusion(
            vector_results, 
            keyword_results,
            vector_weight,
            keyword_weight
        )
        
        # 4. 可选：重排序
        if use_reranker:
            fused_results = self._rerank(query, fused_results)
        
        return fused_results[:top_k]
    
    def _keyword_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict]
    ) -> List[SearchResult]:
        """简化的关键词搜索（基于 TF-IDF 或 BM25）"""
        # 这里使用简化实现，实际可接入 Elasticsearch 或 rank_bm25
        query_words = set(query.lower().split())
        
        # 从向量数据库获取所有文档（实际应使用倒排索引）
        all_docs = self.vectordb.search(
            self.collection,
            np.zeros(self.embedder.dimension),  # dummy vector
            top_k=1000,
            filters=filters
        )
        
        scored = []
        for doc in all_docs:
            text = doc.text.lower() if doc.text else ""
            # 简单的词频匹配
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                scored.append(SearchResult(
                    id=doc.id,
                    score=score / len(query_words),
                    metadata=doc.metadata,
                    text=doc.text
                ))
        
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
    
    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float,
        keyword_weight: float
    ) -> List[HybridSearchResult]:
        """Reciprocal Rank Fusion 融合"""
        scores = {}
        
        # 向量搜索排名
        for rank, result in enumerate(vector_results):
            if result.id not in scores:
                scores[result.id] = {
                    "vector_score": 0,
                    "keyword_score": 0,
                    "metadata": result.metadata,
                    "text": result.text
                }
            scores[result.id]["vector_score"] = 1 / (self.rrf_k + rank + 1)
        
        # 关键词搜索排名
        for rank, result in enumerate(keyword_results):
            if result.id not in scores:
                scores[result.id] = {
                    "vector_score": 0,
                    "keyword_score": 0,
                    "metadata": result.metadata,
                    "text": result.text
                }
            scores[result.id]["keyword_score"] = 1 / (self.rrf_k + rank + 1)
        
        # 计算最终分数
        results = []
        for id_, data in scores.items():
            final_score = (
                vector_weight * data["vector_score"] + 
                keyword_weight * data["keyword_score"]
            )
            results.append(HybridSearchResult(
                id=id_,
                vector_score=data["vector_score"],
                keyword_score=data["keyword_score"],
                final_score=final_score,
                metadata=data["metadata"],
                text=data["text"]
            ))
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    def _rerank(
        self, 
        query: str, 
        results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
        """使用 Cross-Encoder 重排序"""
        try:
            from sentence_transformers import CrossEncoder
            
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            pairs = [(query, r.text) for r in results]
            scores = reranker.predict(pairs)
            
            for i, score in enumerate(scores):
                results[i].final_score = float(score)
            
            results.sort(key=lambda x: x.final_score, reverse=True)
        except ImportError:
            pass  # 如果没有安装，跳过重排序
        
        return results
```

---

## 4. 数据模型升级

### 4.1 DecisionEpisode 修改

```python
# 移除 reasoning 字段，保留 reasons
# 新增 embedding 字段（可选，用于缓存）

@dataclass
class DecisionEpisode:
    # ... 现有字段 ...
    
    # 移除: reasoning: str = ""  # 冗余，可从 reasons 生成
    
    # 新增: 向量缓存（可选）
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    def get_searchable_text(self) -> str:
        """生成用于向量化的文本"""
        parts = [
            f"Symbol: {self.symbol}",
            f"Action: {self.action}",
            f"Confidence: {self.confidence:.2f}",
            f"Reasons: {'; '.join(self.reasons)}",
        ]
        
        # 添加市场上下文摘要
        if self.market_context:
            news = self.market_context.get("news_events", {}).get("top_k_events", [])
            if news and news != ["No news available"]:
                parts.append(f"News: {'; '.join(news[:2])}")
            
            filter_reasoning = self.market_context.get("filter_reasoning", "")
            if filter_reasoning:
                parts.append(f"Analysis: {filter_reasoning}")
        
        # 添加标签
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        
        return " | ".join(parts)
```

### 4.2 向量化存储流程

```python
# 在 EpisodicMemory.add() 中集成向量化

def add(self, episode: DecisionEpisode) -> str:
    # 1. 生成可搜索文本
    searchable_text = episode.get_searchable_text()
    
    # 2. 生成向量
    embedding = self.embedder.embed(searchable_text)
    
    # 3. 存储到向量数据库
    self.vectordb.insert(
        collection="episodes",
        ids=[episode.id],
        vectors=embedding.reshape(1, -1),
        metadata=[{
            "symbol": episode.symbol,
            "action": episode.action,
            "date": episode.date,
            "confidence": episode.confidence,
            "actual_result": episode.actual_result,
            "tags": episode.tags,
            # 完整元数据
            "full": episode.to_dict()
        }],
        texts=[searchable_text]
    )
    
    # 4. 同时存储到 JSONL（保持兼容）
    self.backend.append_jsonl(file_path, episode.to_dict())
    
    return episode.id
```

---

## 5. 查询接口设计

### 5.1 语义搜索

```python
# 新增方法
def semantic_search(
    self,
    query: str,
    top_k: int = 10,
    symbol: Optional[str] = None,
    action: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    min_confidence: Optional[float] = None,
    only_successful: bool = False  # 只返回 actual_result > 0 的决策
) -> List[DecisionEpisode]:
    """
    语义搜索历史决策
    
    示例查询:
    - "类似的突破行情买入决策"
    - "高置信度的减仓决策"
    - "新闻驱动的交易机会"
    
    Args:
        query: 自然语言查询
        top_k: 返回数量
        symbol: 过滤特定股票
        action: 过滤特定动作
        date_range: 日期范围 (start, end)
        min_confidence: 最低置信度
        only_successful: 只返回成功决策
        
    Returns:
        匹配的决策列表
    """
    filters = {}
    if symbol:
        filters["symbol"] = symbol
    if action:
        filters["action"] = action
    if date_range:
        filters["date_gte"] = date_range[0]
    if min_confidence:
        filters["min_confidence"] = min_confidence
    
    results = self.hybrid_search.search(
        query=query,
        top_k=top_k,
        filters=filters
    )
    
    episodes = []
    for result in results:
        episode = DecisionEpisode.from_dict(result.metadata.get("full", {}))
        if only_successful and (episode.actual_result is None or episode.actual_result <= 0):
            continue
        episodes.append(episode)
    
    return episodes
```

### 5.2 相似决策查找

```python
def find_similar_decisions(
    self,
    episode: DecisionEpisode,
    top_k: int = 5,
    exclude_self: bool = True
) -> List[Tuple[DecisionEpisode, float]]:
    """
    查找与给定决策相似的历史决策
    
    用于：
    - 决策一致性检查
    - 历史表现参考
    - 策略回顾
    
    Args:
        episode: 参考决策
        top_k: 返回数量
        exclude_self: 是否排除自身
        
    Returns:
        (决策, 相似度) 列表
    """
    query_text = episode.get_searchable_text()
    query_vector = self.embedder.embed_query(query_text)
    
    results = self.vectordb.search(
        collection="episodes",
        query_vector=query_vector,
        top_k=top_k + (1 if exclude_self else 0)
    )
    
    similar = []
    for result in results:
        if exclude_self and result.id == episode.id:
            continue
        ep = DecisionEpisode.from_dict(result.metadata.get("full", {}))
        similar.append((ep, result.score))
    
    return similar[:top_k]
```

---

## 6. 配置设计

### 6.1 config.yaml 扩展

```yaml
memory:
  enabled: true
  storage_path: "storage/memory"
  
  # 向量搜索配置
  vector_search:
    enabled: true
    
    # Embedding 模型
    embedding:
      provider: "local"  # local | openai
      model: "bge-small"  # bge-small | bge-base | minilm | nomic
      device: "auto"  # auto | cpu | cuda
      cache_enabled: true
      cache_path: "storage/memory/embeddings_cache"
    
    # 向量数据库
    vectordb:
      provider: "lancedb"  # lancedb | qdrant | chroma
      path: "storage/memory/vectordb"
      
      # Qdrant 特定配置（如果使用）
      qdrant:
        host: "localhost"
        port: 6333
    
    # 混合搜索
    hybrid:
      enabled: true
      vector_weight: 0.7
      keyword_weight: 0.3
      rrf_k: 60
    
    # 重排序
    reranker:
      enabled: false
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  
  # 情景记忆配置
  episodic_memory:
    max_days: 30
    save_hold_decisions: false
```

---

## 7. 迁移计划

### 7.1 阶段一：基础设施（1-2 天）

- [ ] 安装依赖：`sentence-transformers`, `lancedb`
- [ ] 实现 `LocalEmbedder` 类
- [ ] 实现 `LanceDBBackend` 类
- [ ] 编写单元测试

### 7.2 阶段二：集成（2-3 天）

- [ ] 修改 `DecisionEpisode`，移除 `reasoning` 字段
- [ ] 在 `EpisodicMemory.add()` 中集成向量化
- [ ] 实现 `semantic_search()` 方法
- [ ] 实现 `find_similar_decisions()` 方法

### 7.3 阶段三：混合搜索（1-2 天）

- [ ] 实现 `HybridSearch` 类
- [ ] 实现 RRF 融合算法
- [ ] 可选：集成 Cross-Encoder 重排序

### 7.4 阶段四：测试与优化（2-3 天）

- [ ] 性能基准测试
- [ ] 搜索质量评估
- [ ] 内存/存储优化
- [ ] 文档更新

### 7.5 回滚方案

- 保留 JSONL 存储作为主存储
- 向量数据库作为索引层
- 配置开关 `vector_search.enabled` 控制是否启用
- 关闭向量搜索时自动回退到关键词搜索

---

## 8. 性能预估

### 8.1 资源消耗

| 指标 | 估算值 | 说明 |
|------|--------|------|
| **模型加载** | ~500MB RAM | bge-small 模型 |
| **单次 Embedding** | ~10ms | CPU，批量更快 |
| **向量存储** | ~1.5KB/条 | 384 维 float32 + 元数据 |
| **搜索延迟** | <50ms | top-10，10K 条记录 |

### 8.2 扩展性

| 数据规模 | 搜索延迟 | 存储空间 |
|---------|---------|---------|
| 1K 条 | <10ms | ~2MB |
| 10K 条 | <50ms | ~20MB |
| 100K 条 | <200ms | ~200MB |
| 1M 条 | 需要 ANN 索引 | ~2GB |

---

## 9. 未来扩展

### 9.1 RAG 集成

```python
# 未来可实现的 RAG 增强决策
def generate_decision_with_rag(
    self,
    current_features: Dict,
    query: str = "类似市场环境下的成功决策"
) -> Dict:
    """
    RAG 增强决策生成
    
    1. 检索相似历史决策
    2. 构建增强 prompt
    3. 调用 LLM 生成决策
    """
    # 检索相关历史
    similar_decisions = self.semantic_search(query, top_k=3, only_successful=True)
    
    # 构建 RAG prompt
    context = self._format_decisions_for_prompt(similar_decisions)
    
    # 增强 prompt
    enhanced_prompt = f"""
    参考以下历史成功决策：
    {context}
    
    当前市场数据：
    {json.dumps(current_features)}
    
    请基于历史经验和当前数据做出决策。
    """
    
    return self.llm.generate(enhanced_prompt)
```

### 9.2 多模态支持

- 图表向量化（技术分析图）
- 音频转文本（财报电话会议）

### 9.3 在线学习

- 基于 actual_result 的向量微调
- 决策质量反馈循环

---

## 10. 参考资料

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding 模型排行
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Sentence Transformers](https://www.sbert.net/)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BGE Embedding Models](https://huggingface.co/BAAI/bge-small-en-v1.5)
