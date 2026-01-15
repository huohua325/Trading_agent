"""
缓存层 - 兼容现有 storage/cache/ 结构

统一缓存管理系统 - 融合版（Phase 10增强）

融合了原CacheStore的简洁接口和CacheManager的高级管理功能：
- 基础功能：get/set/delete + TTL过期 + StorageBackend抽象
- 管理功能：统计、发现、清理、文件列表
- 兼容性：向后兼容现有storage/cache/结构
"""

from typing import Any, Optional, Callable, List, Dict
from pathlib import Path
from datetime import datetime

from ..backends.base import StorageBackend


class CacheStore:
    """
    缓存层 - 兼容现有 storage/cache/ 结构
    
    现有代码不需要改动，新代码可以通过 MemoryStore.cache 访问。
    
    支持的命名空间（与现有结构一致）：
    - llm: LLM 响应缓存
    - news: 新闻数据缓存
    - financials: 财务数据缓存
    - corporate_actions: 公司行动缓存
    """
    
    def __init__(self, backend: StorageBackend, cache_dir: Path):
        """
        Args:
            backend: 存储后端
            cache_dir: 缓存目录
        """
        self.backend = backend
        self.cache_dir = Path(cache_dir)
    
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            namespace: "llm" | "news" | "financials" 等
            key: 缓存键
            
        Returns:
            缓存值，不存在或过期返回 None
            
        兼容现有路径: storage/cache/{namespace}/{key}.json
        """
        path = self.cache_dir / namespace / f"{key}.json"
        data = self.backend.read_json(path)
        
        if data is None:
            return None
        
        # 检查是否过期
        if "expires_at" in data and data["expires_at"]:
            try:
                expires_at = datetime.fromisoformat(data["expires_at"])
                if datetime.now() > expires_at:
                    return None
            except (ValueError, TypeError):
                pass
        
        return data.get("value")
    
    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ):
        """
        设置缓存
        
        Args:
            namespace: 命名空间
            key: 缓存键
            value: 缓存值
            ttl_seconds: 过期时间（秒），None 表示永不过期
        """
        path = self.cache_dir / namespace / f"{key}.json"
        
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.fromtimestamp(
                datetime.now().timestamp() + ttl_seconds
            ).isoformat()
        
        data = {
            "value": value,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at,
        }
        
        self.backend.write_json(path, data)
    
    def get_or_compute(
        self,
        namespace: str,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: Optional[int] = None
    ) -> Any:
        """
        有则返回，无则计算并缓存
        
        Args:
            namespace: 命名空间
            key: 缓存键
            compute_fn: 计算函数
            ttl_seconds: 过期时间
            
        Returns:
            缓存值或计算结果
        """
        cached = self.get(namespace, key)
        if cached is not None:
            return cached
        
        value = compute_fn()
        self.set(namespace, key, value, ttl_seconds)
        return value
    
    def delete(self, namespace: str, key: str) -> bool:
        """删除缓存"""
        path = self.cache_dir / namespace / f"{key}.json"
        return self.backend.delete(path)
    
    def exists(self, namespace: str, key: str) -> bool:
        """检查缓存是否存在（且未过期）"""
        return self.get(namespace, key) is not None
    
    def list_namespaces(self) -> List[str]:
        """列出所有命名空间"""
        if not self.cache_dir.exists():
            return []
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
    
    def clear_namespace(self, namespace: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        清空指定命名空间的所有缓存
        
        Args:
            namespace: 命名空间名称
            dry_run: 是否仅模拟运行（不实际删除）
            
        Returns:
            删除统计信息
        """
        from loguru import logger
        
        path = self.cache_dir / namespace
        if not path.exists():
            return {"status": "not_found", "deleted_count": 0}
        
        files = list(path.rglob("*.*"))
        file_count = len(files)
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        if not dry_run:
            import shutil
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"[CacheStore] Cleared namespace: {namespace} ({file_count} files, {total_size/(1024*1024):.2f}MB)")
        else:
            logger.info(f"[CacheStore] DRY RUN - Would delete {file_count} files ({total_size/(1024*1024):.2f}MB) from {namespace}")
        
        return {
            "status": "success" if not dry_run else "dry_run",
            "deleted_count": file_count,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }
    
    # ==================== 高级管理功能（Phase 10融合）====================
    
    def discover_namespaces(self) -> List[str]:
        """
        自动发现所有缓存命名空间（子目录）
        
        Returns:
            命名空间列表
        """
        if not self.cache_dir.exists():
            return []
        
        namespaces = []
        for item in self.cache_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                namespaces.append(item.name)
        
        return sorted(namespaces)
    
    def get_namespace_path(self, namespace: str) -> Path:
        """
        获取指定命名空间的完整路径
        
        Args:
            namespace: 命名空间名称
            
        Returns:
            完整路径
        """
        return self.cache_dir / namespace
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有缓存的统计信息
        
        Returns:
            统计信息字典，格式:
            {
                "namespace_name": {
                    "file_count": int,
                    "total_size_mb": float,
                    "subdirs": List[str]
                }
            }
        """
        stats = {}
        
        for namespace in self.discover_namespaces():
            ns_path = self.get_namespace_path(namespace)
            
            # 统计文件
            files = list(ns_path.rglob("*.*"))
            file_count = len(files)
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            total_size_mb = round(total_size / (1024 * 1024), 2)
            
            # 统计子目录
            subdirs = [d.name for d in ns_path.iterdir() if d.is_dir()]
            
            stats[namespace] = {
                "file_count": file_count,
                "total_size_mb": total_size_mb,
                "subdirs": subdirs[:10]  # 限制显示前10个
            }
        
        return stats
    
    def list_cache_files(
        self, 
        namespace: str, 
        pattern: str = "*", 
        recursive: bool = False
    ) -> List[Path]:
        """
        列出指定命名空间下的缓存文件
        
        Args:
            namespace: 命名空间
            pattern: 文件匹配模式（glob格式）
            recursive: 是否递归搜索
            
        Returns:
            文件路径列表
        """
        ns_path = self.get_namespace_path(namespace)
        if not ns_path.exists():
            return []
        
        if recursive:
            return list(ns_path.rglob(pattern))
        else:
            return list(ns_path.glob(pattern))
    
    def ensure_namespace(self, namespace: str) -> Path:
        """
        确保命名空间目录存在
        
        Args:
            namespace: 命名空间名称
            
        Returns:
            命名空间路径
        """
        ns_path = self.get_namespace_path(namespace)
        ns_path.mkdir(parents=True, exist_ok=True)
        return ns_path


__all__ = ["CacheStore"]
