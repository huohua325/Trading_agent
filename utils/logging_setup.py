from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any
from loguru import logger


def setup_json_logging(base_dir: str | None = None) -> str:
    base_dir = base_dir or os.path.join(os.getcwd(), "trading_agent_v2", "storage", "logs")
    os.makedirs(base_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(base_dir, f"{date_str}.log")
    # 清除默认 handler，避免重复
    logger.remove()
    logger.add(log_path, serialize=True, enqueue=True, backtrace=False, diagnose=False, rotation="00:00")
    return log_path


class Metrics:
    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.timings_ms: Dict[str, float] = {}

    def incr(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(value)

    def gauge(self, key: str, value: float) -> None:
        self.gauges[key] = float(value)

    def timing(self, key: str, ms: float) -> None:
        self.timings_ms[key] = float(ms)

    def flush(self, extra: Dict[str, Any] | None = None) -> None:
        payload = {
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "metrics",
            "counters": self.counters,
            "gauges": self.gauges,
            "timings_ms": self.timings_ms,
        }
        if extra:
            payload.update(extra)
        logger.bind(component="metrics").info(payload)


def alert(message: str, severity: str = "warning", **kwargs: Any) -> None:
    payload = {"ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "type": "alert", "severity": severity, "message": message}
    if kwargs:
        payload.update(kwargs)
    logger.bind(component="alert").warning(payload) 