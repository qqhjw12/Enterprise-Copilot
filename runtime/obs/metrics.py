"""轻量指标收集（Prometheus 文本格式）。

不引入 prometheus_client 的原因：本项目指标只有十来个 counter/histogram，
自实现 100 行比拖一个依赖 + 处理多进程 registry 更省心。
需要与 Prometheus 深度集成时，把 render() 换成 official client 即可。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)
_MAX_SAMPLES = 2000


def _key(name: str, labels: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((labels or {}).items()))


def inc(name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
    with _lock:
        _counters[_key(name, labels)] += value


def observe(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    with _lock:
        bucket = _histograms[_key(name, labels)]
        bucket.append(value)
        if len(bucket) > _MAX_SAMPLES:
            del bucket[: len(bucket) - _MAX_SAMPLES]


def timer(name: str, labels: dict[str, str] | None = None):
    """with timer("rag_retrieve"): ..."""

    class _Timer:
        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *exc):
            observe(f"{name}_ms", (time.perf_counter() - self._start) * 1000, labels)
            return False

    return _Timer()


def _fmt_labels(labels: tuple[tuple[str, str], ...], extra: tuple[str, str] | None = None) -> str:
    items = list(labels)
    if extra:
        items.append(extra)
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in items) + "}"


def render() -> str:
    lines: list[str] = []
    with _lock:
        for (name, labels), value in _counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{_fmt_labels(labels)} {value}")
        for (name, labels), samples in _histograms.items():
            if not samples:
                continue
            ordered = sorted(samples)
            total = sum(ordered)
            count = len(ordered)
            lines.append(f"# TYPE {name} summary")
            lines.append(f"{name}_count{_fmt_labels(labels)} {count}")
            lines.append(f"{name}_sum{_fmt_labels(labels)} {total:.3f}")
            for qname, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99)):
                idx = min(count - 1, int(q * (count - 1)))
                lines.append(f"{name}{_fmt_labels(labels, ('quantile', qname))} {ordered[idx]:.3f}")
    return "\n".join(lines) + "\n"


def reset() -> None:
    with _lock:
        _counters.clear()
        _histograms.clear()
