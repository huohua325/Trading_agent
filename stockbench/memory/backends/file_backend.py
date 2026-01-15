"""
文件存储后端实现

使用 JSON/JSONL 文件进行持久化存储，
兼容现有的 storage/cache/ 目录结构。
"""

from typing import Any, Optional, List
from pathlib import Path
import json

from .base import StorageBackend


class FileBackend(StorageBackend):
    """
    文件存储后端
    
    基于本地文件系统的存储实现，支持 JSON 和 JSONL 格式。
    
    Args:
        base_path: 基础路径，相对路径会基于此路径解析
    """
    
    def __init__(self, base_path: Path = None):
        self.base_path = Path(base_path) if base_path else Path(".")
    
    def read_json(self, path: Path) -> Optional[Any]:
        """读取 JSON 文件"""
        full_path = self._resolve_path(path)
        if not full_path.exists():
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def write_json(self, path: Path, data: Any):
        """写入 JSON 文件"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def read_jsonl(self, path: Path) -> List[Any]:
        """读取 JSONL 文件"""
        full_path = self._resolve_path(path)
        if not full_path.exists():
            return []
        
        results = []
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        except (json.JSONDecodeError, IOError):
            pass
        
        return results
    
    def write_jsonl(self, path: Path, data: List[Any]):
        """写入 JSONL 文件（覆盖）"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def append_jsonl(self, path: Path, data: Any):
        """追加到 JSONL 文件"""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def delete(self, path: Path) -> bool:
        """删除文件"""
        full_path = self._resolve_path(path)
        if full_path.exists():
            full_path.unlink()
            return True
        return False
    
    def exists(self, path: Path) -> bool:
        """检查文件是否存在"""
        return self._resolve_path(path).exists()
    
    def _resolve_path(self, path: Path) -> Path:
        """
        解析路径（相对路径转绝对路径）
        
        Args:
            path: 输入路径
            
        Returns:
            解析后的完整路径
        """
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path


__all__ = ["FileBackend"]
