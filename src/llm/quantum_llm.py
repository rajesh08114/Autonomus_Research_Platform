from __future__ import annotations

import time
from typing import Any

import httpx

from src.config.settings import settings
from src.core.logger import get_logger
from src.db.repository import ExperimentRepository

logger = get_logger(__name__)


def _local_quantum_template(framework: str) -> str:
    if framework == "qiskit":
        return (
            "from __future__ import annotations\n\n"
            "QUBIT_COUNT = 4\nCIRCUIT_LAYERS = 3\nBACKEND = 'aer_simulator'\n\n"
            "class QuantumLayer:\n"
            "    def forward(self, x):\n"
            "        return [sum(x) / max(len(x), 1)]\n\n"
            "def get_circuit_diagram():\n"
            "    return 'qiskit_circuit_stub'\n"
        )
    if framework == "cirq":
        return (
            "from __future__ import annotations\n\n"
            "QUBIT_COUNT = 4\nCIRCUIT_LAYERS = 3\nBACKEND = 'cirq-simulator'\n\n"
            "class QuantumLayer:\n"
            "    def forward(self, x):\n"
            "        return [sum(x) / max(len(x), 1)]\n\n"
            "def get_circuit_diagram():\n"
            "    return 'cirq_circuit_stub'\n"
        )
    return (
        "from __future__ import annotations\n\n"
        "QUBIT_COUNT = 4\nCIRCUIT_LAYERS = 3\nBACKEND = 'default.qubit'\n\n"
        "class QuantumLayer:\n"
        "    def forward(self, x):\n"
        "        return [sum(x) / max(len(x), 1)]\n\n"
        "def get_circuit_diagram():\n"
        "    return 'pennylane_circuit_stub'\n"
    )


async def generate_quantum_code(delegation_spec: dict[str, Any], experiment_id: str | None = None) -> str:
    framework = str(delegation_spec.get("framework", "pennylane"))
    started = time.time()
    if not settings.QUANTUM_LLM_ENDPOINT:
        code = _local_quantum_template(framework)
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase="quantum_gate",
            provider="quantum_local_template",
            model=f"{framework}_template",
            latency_ms=(time.time() - started) * 1000.0,
            success=True,
        )
        return code

    try:
        async with httpx.AsyncClient(timeout=settings.QUANTUM_LLM_TIMEOUT) as client:
            response = await client.post(
                settings.QUANTUM_LLM_ENDPOINT,
                headers={"Authorization": f"Bearer {settings.QUANTUM_LLM_API_KEY}"},
                json={"delegation_spec": delegation_spec},
            )
            response.raise_for_status()
            body = response.json()
        code = str(body.get("generated_code", _local_quantum_template(framework)))
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase="quantum_gate",
            provider="quantum_remote_endpoint",
            model=str(delegation_spec.get("framework", "quantum_remote")),
            latency_ms=(time.time() - started) * 1000.0,
            success=True,
        )
        return code
    except Exception as exc:
        logger.exception("llm.quantum.error")
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase="quantum_gate",
            provider="quantum_remote_endpoint",
            model=str(delegation_spec.get("framework", "quantum_remote")),
            latency_ms=(time.time() - started) * 1000.0,
            success=False,
            error_message=str(exc),
        )
        # Safe offline fallback for local/test environments.
        return _local_quantum_template(framework)
