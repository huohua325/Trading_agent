"""
存储后端抽象基类

定义统一的存储接口，支持：
- JSON 文件读写
- JSONL 文件读写（追加模式）
- 文件删除和存在检查
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, List


class StorageBackend(ABC):
    """
    存储后端抽象基类
    
    所有存储后端实现必须继承此类并实现所有抽象方法。
    """
    
    @abstractmethod
    def read_json(self, path: Path) -> Optional[Any]:
        """
        读取 JSON 文件
        
        Args:
            path: 文件路径
            
        Returns:
            解析后的数据，文件不存在或解析失败返回 None
        """
        pass
    
    @abstractmethod
    def write_json(self, path: Path, data: Any):
        """
        写入 JSON 文件
        
        Args:
            path: 文件路径
            data: 要写入的数据
        """
        pass
    
    @abstractmethod
    def read_jsonl(self, path: Path) -> List[Any]:
        """
        读取 JSONL 文件（每行一个 JSON）
        
        Args:
            path: 文件路径
            
        Returns:
            数据列表，文件不存在返回空列表
        """
        pass
    
    @abstractmethod
    def write_jsonl(self, path: Path, data: List[Any]):
        """
        写入 JSONL 文件（覆盖）
        
        Args:
            path: 文件路径
            data: 数据列表
        """
        pass
    
    @abstractmethod
    def append_jsonl(self, path: Path, data: Any):
        """
        追加到 JSONL 文件
        
        Args:
            path: 文件路径
            data: 单条数据
        """
        pass
    
    @abstractmethod
    def delete(self, path: Path) -> bool:
        """
        删除文件
        
        Args:
            path: 文件路径
            
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def exists(self, path: Path) -> bool:
        """
        检查文件是否存在
        
        Args:
            path: 文件路径
            
        Returns:
            文件是否存在
        """
        pass


__all__ = ["StorageBackend"]
