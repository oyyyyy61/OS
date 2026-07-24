"""Durable append-only JSONL diagnostics."""

import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class DurableJSONLSink:
    """Append one JSON object and fsync it before returning."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("dagkv_diagnostic_trace_file must be an absolute path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(
                dict(row),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            fd = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                written = 0
                while written < len(payload):
                    count = os.write(fd, payload[written:])
                    if count <= 0:
                        raise OSError("short write while appending diagnostic JSONL")
                    written += count
                os.fsync(fd)
            finally:
                os.close(fd)
