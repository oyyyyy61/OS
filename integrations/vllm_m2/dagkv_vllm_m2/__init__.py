"""Fail-closed vLLM diagnostics for DAGKV milestone M2.

Put ``integrations/vllm_m2`` on ``PYTHONPATH``.  vLLM loads the connector and
offloading spec through their explicit module paths, so the integration does
not require a patch to the vLLM source tree.
"""

from typing import Any

__all__ = [
    "DAGKVDiagnosticCPUOffloadingSpec",
    "DAGKVDiagnosticCPUOffloadingWorker",
    "DAGKVDiagnosticOffloadingConnector",
    "ProbedGPULoadStoreSpec",
]


def __getattr__(name: str) -> Any:
    if name == "DAGKVDiagnosticOffloadingConnector":
        from .connector import DAGKVDiagnosticOffloadingConnector

        return DAGKVDiagnosticOffloadingConnector
    if name == "ProbedGPULoadStoreSpec":
        from .metadata import ProbedGPULoadStoreSpec

        return ProbedGPULoadStoreSpec
    if name in {
        "DAGKVDiagnosticCPUOffloadingSpec",
        "DAGKVDiagnosticCPUOffloadingWorker",
    }:
        from .spec import (
            DAGKVDiagnosticCPUOffloadingSpec,
            DAGKVDiagnosticCPUOffloadingWorker,
        )

        return {
            "DAGKVDiagnosticCPUOffloadingSpec": DAGKVDiagnosticCPUOffloadingSpec,
            "DAGKVDiagnosticCPUOffloadingWorker": (DAGKVDiagnosticCPUOffloadingWorker),
        }[name]
    raise AttributeError(name)
