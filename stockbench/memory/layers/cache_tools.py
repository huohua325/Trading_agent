"""
缓存高级工具 - Phase 10融合到Memory系统

提供缓存健康检查、清理、完整性验证等高级功能
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import json
from loguru import logger


class CacheCleanupTool:
    """缓存清理和管理工具"""
    
    def __init__(self, cache_store):
        """
        Args:
            cache_store: CacheStore实例（来自memory.layers.cache）
        """
        self.cache = cache_store
    
    def analyze_cache_health(self) -> Dict[str, Any]:
        """
        分析缓存健康状况
        
        Returns:
            健康状况报告
        """
        stats = self.cache.get_cache_stats()
        
        total_files = sum(ns["file_count"] for ns in stats.values())
        total_size_mb = sum(ns["total_size_mb"] for ns in stats.values())
        
        # 检查异常情况
        warnings = []
        
        # 1. 检查缓存大小
        for ns, info in stats.items():
            if info["total_size_mb"] > 100:  # 单个命名空间超过100MB
                warnings.append({
                    "type": "large_cache",
                    "namespace": ns,
                    "size_mb": info["total_size_mb"],
                    "message": f"{ns} cache is large ({info['total_size_mb']}MB), consider cleanup"
                })
        
        # 2. 检查文件数量
        for ns, info in stats.items():
            if info["file_count"] > 5000:
                warnings.append({
                    "type": "many_files",
                    "namespace": ns,
                    "file_count": info["file_count"],
                    "message": f"{ns} has many files ({info['file_count']}), may slow down access"
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_namespaces": len(stats),
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "total_size_gb": round(total_size_mb / 1024, 2)
            },
            "namespaces": stats,
            "warnings": warnings,
            "health_score": self._calculate_health_score(stats, warnings)
        }
    
    def _calculate_health_score(self, stats: Dict, warnings: List) -> str:
        """计算健康评分"""
        total_size_gb = sum(ns["total_size_mb"] for ns in stats.values()) / 1024
        
        if len(warnings) == 0 and total_size_gb < 1:
            return "excellent"
        elif len(warnings) <= 2 and total_size_gb < 5:
            return "good"
        elif len(warnings) <= 5 and total_size_gb < 10:
            return "fair"
        else:
            return "needs_attention"
    
    def find_old_files(
        self, 
        namespace: str, 
        days_old: int = 30
    ) -> List[Dict[str, Any]]:
        """
        查找指定天数之前的旧文件
        
        Args:
            namespace: 命名空间
            days_old: 天数阈值
            
        Returns:
            旧文件列表
        """
        cutoff_time = datetime.now() - timedelta(days=days_old)
        old_files = []
        
        for file in self.cache.list_cache_files(namespace, "*", recursive=True):
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff_time:
                    old_files.append({
                        "path": str(file.relative_to(self.cache.cache_dir)),
                        "size_kb": round(file.stat().st_size / 1024, 2),
                        "modified": mtime.isoformat(),
                        "age_days": (datetime.now() - mtime).days
                    })
        
        return old_files
    
    def cleanup_old_files(
        self, 
        namespace: str, 
        days_old: int = 30,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        清理旧文件
        
        Args:
            namespace: 命名空间
            days_old: 天数阈值
            dry_run: 是否仅模拟运行
            
        Returns:
            清理结果
        """
        old_files = self.find_old_files(namespace, days_old)
        
        total_size_kb = sum(f["size_kb"] for f in old_files)
        
        if not dry_run:
            deleted_count = 0
            for file_info in old_files:
                file_path = self.cache.cache_dir / file_info["path"]
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
            
            logger.info(f"[CacheCleanup] Deleted {deleted_count} old files from {namespace}")
        else:
            logger.info(f"[CacheCleanup] DRY RUN - Would delete {len(old_files)} files ({total_size_kb/1024:.2f}MB) from {namespace}")
        
        return {
            "status": "success" if not dry_run else "dry_run",
            "file_count": len(old_files),
            "size_mb": round(total_size_kb / 1024, 2),
            "files": old_files[:10]  # 显示前10个文件
        }
    
    def find_duplicate_files(self, namespace: str) -> List[Dict[str, Any]]:
        """
        查找可能的重复文件（基于内容哈希）
        
        Args:
            namespace: 命名空间
            
        Returns:
            重复文件组列表
        """
        import hashlib
        
        file_hashes: Dict[str, List[Path]] = {}
        
        for file in self.cache.list_cache_files(namespace, "*.json", recursive=True):
            if file.is_file() and file.stat().st_size < 10 * 1024 * 1024:  # 跳过大文件
                try:
                    with open(file, 'rb') as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if content_hash not in file_hashes:
                        file_hashes[content_hash] = []
                    file_hashes[content_hash].append(file)
                except Exception:
                    continue
        
        # 找出重复的文件组
        duplicates = []
        for content_hash, files in file_hashes.items():
            if len(files) > 1:
                duplicates.append({
                    "hash": content_hash,
                    "count": len(files),
                    "size_kb": round(files[0].stat().st_size / 1024, 2),
                    "files": [str(f.relative_to(self.cache.cache_dir)) for f in files]
                })
        
        return duplicates
    
    def validate_cache_integrity(self, namespace: str) -> Dict[str, Any]:
        """
        验证缓存完整性（检查损坏的JSON文件）
        
        Args:
            namespace: 命名空间
            
        Returns:
            验证结果
        """
        corrupted_files = []
        valid_count = 0
        
        for file in self.cache.list_cache_files(namespace, "*.json", recursive=True):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    valid_count += 1
                except Exception as e:
                    corrupted_files.append({
                        "path": str(file.relative_to(self.cache.cache_dir)),
                        "error": str(e)
                    })
        
        return {
            "namespace": namespace,
            "valid_files": valid_count,
            "corrupted_files": len(corrupted_files),
            "integrity_score": round(valid_count / (valid_count + len(corrupted_files)) * 100, 2) if (valid_count + len(corrupted_files)) > 0 else 100,
            "corrupted_list": corrupted_files[:10]  # 显示前10个
        }
    
    def export_cache_manifest(self, output_path: Optional[str] = None) -> str:
        """
        导出缓存清单（用于备份/审计）
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            导出的文件路径
        """
        manifest = {
            "export_time": datetime.now().isoformat(),
            "base_dir": str(self.cache.cache_dir),
            "namespaces": {}
        }
        
        for namespace in self.cache.discover_namespaces():
            files = []
            for file in self.cache.list_cache_files(namespace, "*", recursive=True):
                if file.is_file():
                    files.append({
                        "path": str(file.relative_to(self.cache.cache_dir)),
                        "size": file.stat().st_size,
                        "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                    })
            
            manifest["namespaces"][namespace] = {
                "file_count": len(files),
                "files": files
            }
        
        if output_path is None:
            output_path = f"cache_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[CacheCleanup] Exported cache manifest to {output_path}")
        return output_path


__all__ = ["CacheCleanupTool"]
