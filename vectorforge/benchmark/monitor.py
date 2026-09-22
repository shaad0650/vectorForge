from __future__ import annotations
import time
from typing import Dict, Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class ResourceMonitor:
    """Simple resource monitor capturing RSS memory and CPU percent snapshots.

    psutil is optional; when not installed `snapshot` returns None fields.
    """

    def __init__(self):
        self._available = psutil is not None
        if self._available:
            self._proc = psutil.Process()

    def snapshot(self) -> Dict[str, Any]:
        if not self._available:
            return {"rss": None, "vms": None, "cpu_percent": None, "note": "psutil not installed"}
        m = self._proc.memory_info()
        return {"rss": m.rss, "vms": m.vms, "cpu_percent": self._proc.cpu_percent(interval=0.0)}
