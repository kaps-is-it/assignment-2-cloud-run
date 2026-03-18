from __future__ import annotations

import os
import threading
import time
from collections import deque
from statistics import mean
from typing import Deque, Dict

import psutil


class RuntimeMetrics:
    def __init__(self, expected_daily_requests: int, history_size: int) -> None:
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
        self._started_at = time.time()
        self._durations_ms: Deque[float] = deque(maxlen=history_size)
        initial_memory = self.current_memory_mb()
        self._min_memory_mb = initial_memory
        self._max_memory_mb = initial_memory
        self._latest_memory_mb = initial_memory
        self._request_count = 0
        self._expected_daily_requests = expected_daily_requests

    def current_memory_mb(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)

    def observe(self, duration_ms: float, memory_mb: float) -> None:
        with self._lock:
            self._request_count += 1
            self._durations_ms.append(duration_ms)
            self._latest_memory_mb = memory_mb
            self._min_memory_mb = min(self._min_memory_mb, memory_mb)
            self._max_memory_mb = max(self._max_memory_mb, memory_mb)

    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            durations = list(self._durations_ms)
            if durations:
                sorted_durations = sorted(durations)
                p95_index = max(0, int(len(sorted_durations) * 0.95) - 1)
                avg_duration = float(mean(sorted_durations))
                p95_duration = float(sorted_durations[p95_index])
                min_duration = float(sorted_durations[0])
                max_duration = float(sorted_durations[-1])
            else:
                avg_duration = 0.0
                p95_duration = 0.0
                min_duration = 0.0
                max_duration = 0.0

            return {
                "uptime_seconds": self.uptime_seconds(),
                "request_count": float(self._request_count),
                "avg_response_time_ms": avg_duration,
                "p95_response_time_ms": p95_duration,
                "min_response_time_ms": min_duration,
                "max_response_time_ms": max_duration,
                "min_memory_mb": float(self._min_memory_mb),
                "max_memory_mb": float(self._max_memory_mb),
                "latest_memory_mb": float(self._latest_memory_mb),
                "expected_daily_requests": float(self._expected_daily_requests),
            }
