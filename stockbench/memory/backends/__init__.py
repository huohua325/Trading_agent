"""存储后端模块"""

from .base import StorageBackend
from .file_backend import FileBackend

__all__ = ["StorageBackend", "FileBackend"]
