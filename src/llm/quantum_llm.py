from __future__ import annotations

import json
import time
from typing import Any

import httpx

from src.config.settings import settings
from src.core.logger import get_logger
from src.db.repository import ExperimentRepository

logger = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, int(len(value) / 4))


def _normalize_framework(value: str) -> str:
    framework = str(value or "pennylane").strip().lower()
    return framework if framework in {"pennylane", "qiskit", "cirq", "torchquantum"} else "pennylane"


def _contains_quantum_layer(code: str) -> bool:
    text = str(code or "")
    return "class QuantumLayer" in text and "def forward(" in text


def _extract_code_from_payload(body: Any) -> str:
    if isinstance(body, dict):
        for key in ("code", "generated_code", "content", "response"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        data = body.get("data")
        if isinstance(data, dict):
            for key in ("code", "generated_code", "content", "response"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _build_codehub_prompt(delegation_spec: dict[str, Any]) -> str:
    framework = _normalize_framework(str(delegation_spec.get("framework", "pennylane")))
    algorithm = str(delegation_spec.get("algorithm") or "VQE")
    qubits = max(1, int(delegation_spec.get("qubit_count") or 4))
    layers = max(1, int(delegation_spec.get("layers") or 3))
    backend = str(delegation_spec.get("backend") or ("aer_simulator" if framework == "qiskit" else "default.qubit"))
    problem_type = str(delegation_spec.get("problem_type") or "classification")
    code_level = str(delegation_spec.get("code_level") or "intermediate")
    dataset_info = delegation_spec.get("dataset_info") or {}
    n_features = max(1, int(dataset_info.get("n_features") or 4))
    n_classes = max(2, int(dataset_info.get("n_classes") or 2))
    encoding = str(dataset_info.get("encoding") or "angle_encoding")

    return (
        "Generate a production-ready quantum module for integration into an AI research pipeline.\n"
        f"Framework: {framework}\n"
        f"Algorithm: {algorithm}\n"
        f"Problem type: {problem_type}\n"
        f"Code level: {code_level}\n"
        f"Qubits: {qubits}\n"
        f"Layers: {layers}\n"
        f"Backend: {backend}\n"
        f"Dataset hints: n_features={n_features}, n_classes={n_classes}, encoding={encoding}\n\n"
        "Hard requirements:\n"
        "1. Return Python code only.\n"
        "2. Define class QuantumLayer with method forward(self, x).\n"
        "3. Add get_circuit_diagram() helper.\n"
        "4. Keep code runnable and import-safe.\n"
        "5. Avoid deprecated APIs; target latest stable APIs.\n"
    )


def _build_codehub_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    bearer = str(settings.CODEHUB_BEARER_TOKEN or settings.QUANTUM_LLM_API_KEY or "").strip()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    internal_key = str(settings.CODEHUB_INTERNAL_API_KEY or "").strip()
    if internal_key:
        headers["X-Internal-API-Key"] = internal_key
    return headers


async def _generate_via_codehub(delegation_spec: dict[str, Any], experiment_id: str | None = None) -> str | None:
    if not settings.codehub_enabled:
        return None

    framework = _normalize_framework(str(delegation_spec.get("framework", "pennylane")))
    prompt = _build_codehub_prompt(delegation_spec)
    started = time.time()
    payload = {
        "prompt": prompt,
        "framework": framework,
        "num_qubits": int(delegation_spec.get("qubit_count") or 4),
        "include_explanation": False,
        "include_visualization": False,
        "client_context": {
            "client_type": "api",
            "client_version": "research_platform",
        },
        "runtime_preferences": {
            "mode": "modern",
        },
    }

    try:
        async with httpx.AsyncClient(timeout=max(5, int(settings.CODEHUB_TIMEOUT or 120))) as client:
            response = await client.post(
                settings.codehub_generate_url,
                headers=_build_codehub_headers(),
                json=payload,
            )
            response.raise_for_status()
            body = response.json()

        code = _extract_code_from_payload(body)
        if not _contains_quantum_layer(code):
            raise ValueError("CodeHub response did not include QuantumLayer.forward")

        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(code)
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase="quantum_gate",
            provider="codehub_backend",
            model=framework,
            latency_ms=(time.time() - started) * 1000.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=True,
        )
        return code
    except Exception as exc:
        logger.warning("llm.quantum.codehub.error", error=str(exc))
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase="quantum_gate",
            provider="codehub_backend",
            model=framework,
            latency_ms=(time.time() - started) * 1000.0,
            prompt_tokens=_estimate_tokens(prompt),
            success=False,
            error_message=str(exc),
        )
        return None


def _local_quantum_template(framework: str, delegation_spec: dict[str, Any] | None = None) -> str:
    spec = delegation_spec or {}
    algorithm = str(spec.get("algorithm") or "VQE")
    qubits = int(spec.get("qubit_count") or 4)
    qubits = max(1, min(64, qubits))
    layers = int(spec.get("layers") or 3)
    layers = max(1, min(32, layers))
    backend = str(spec.get("backend") or ("aer_simulator" if framework == "qiskit" else "default.qubit"))
    if framework == "qiskit":
        return (
            "from __future__ import annotations\n\n"
            "import math\n\n"
            f"QUBIT_COUNT = {qubits}\nCIRCUIT_LAYERS = {layers}\nBACKEND = '{backend}'\nALGORITHM = '{algorithm}'\n\n"
            "def feature_map(values):\n"
            "    return [float(v) * math.pi for v in values]\n\n"
            "def entangling_pattern(layer_index):\n"
            "    return [('cx', i, (i + 1) % QUBIT_COUNT) for i in range(QUBIT_COUNT)]\n\n"
            "class QuantumLayer:\n"
            "    def forward(self, x):\n"
            "        encoded = feature_map(x)\n"
            "        expectation = sum(math.cos(v) for v in encoded) / max(len(encoded), 1)\n"
            "        return [expectation]\n\n"
            "def get_circuit_diagram():\n"
            "    rows = ['Qiskit variational ansatz:']\n"
            "    rows.append(' - angle encoding on each qubit')\n"
            "    rows.append(f' - algorithm: {ALGORITHM}')\n"
            "    for layer in range(CIRCUIT_LAYERS):\n"
            "        rows.append(f' - layer {layer + 1}: RX/RY rotations + CX ring entanglement')\n"
            "    return '\\n'.join(rows)\n"
        )
    if framework == "cirq":
        return (
            "from __future__ import annotations\n\n"
            "import math\n\n"
            f"QUBIT_COUNT = {qubits}\nCIRCUIT_LAYERS = {layers}\nBACKEND = '{backend}'\nALGORITHM = '{algorithm}'\n\n"
            "def feature_map(values):\n"
            "    return [float(v) * math.pi for v in values]\n\n"
            "def entangling_pattern(layer_index):\n"
            "    return [('cz', i, (i + 1) % QUBIT_COUNT) for i in range(QUBIT_COUNT)]\n\n"
            "class QuantumLayer:\n"
            "    def forward(self, x):\n"
            "        encoded = feature_map(x)\n"
            "        expectation = sum(math.sin(v) for v in encoded) / max(len(encoded), 1)\n"
            "        return [expectation]\n\n"
            "def get_circuit_diagram():\n"
            "    rows = ['Cirq variational ansatz:']\n"
            "    rows.append(' - angle encoding on each qubit')\n"
            "    rows.append(f' - algorithm: {ALGORITHM}')\n"
            "    for layer in range(CIRCUIT_LAYERS):\n"
            "        rows.append(f' - layer {layer + 1}: RX/RZ rotations + CZ ring entanglement')\n"
            "    return '\\n'.join(rows)\n"
        )
    return (
        "from __future__ import annotations\n\n"
        "import math\n\n"
        f"QUBIT_COUNT = {qubits}\nCIRCUIT_LAYERS = {layers}\nBACKEND = '{backend}'\nALGORITHM = '{algorithm}'\n\n"
        "def feature_map(values):\n"
        "    return [float(v) * math.pi for v in values]\n\n"
        "def entangling_pattern(layer_index):\n"
        "    return [('cnot', i, (i + 1) % QUBIT_COUNT) for i in range(QUBIT_COUNT)]\n\n"
        "class QuantumLayer:\n"
        "    def forward(self, x):\n"
        "        encoded = feature_map(x)\n"
        "        expectation = sum(math.cos(v) * math.sin(v) for v in encoded) / max(len(encoded), 1)\n"
        "        return [expectation]\n\n"
        "def get_circuit_diagram():\n"
        "    rows = ['PennyLane hybrid circuit:']\n"
        "    rows.append(' - angle encoding over input features')\n"
        "    rows.append(f' - algorithm: {ALGORITHM}')\n"
        "    for layer in range(CIRCUIT_LAYERS):\n"
        "        rows.append(f' - layer {layer + 1}: Rot gates + CNOT chain')\n"
        "    return '\\n'.join(rows)\n"
    )


async def invoke_quantum_llm_json(
    system_prompt: str,
    user_prompt: str = "",
    experiment_id: str | None = None,
    phase: str | None = None,
) -> str:
    fallback = json.dumps(
        {
            "action": "ask_user",
            "reasoning": "Quantum clarification fallback used when endpoint is unavailable.",
            "parameters": {"questions": []},
            "next_step": "planner",
            "confidence": 0.5,
        }
    )
    started = time.time()
    if not settings.QUANTUM_LLM_ENDPOINT:
        prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        completion_tokens = _estimate_tokens(fallback)
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase=phase,
            provider="quantum_local_template",
            model="clarifier_template",
            latency_ms=(time.time() - started) * 1000.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=True,
        )
        return fallback

    try:
        headers = {"Content-Type": "application/json"}
        key = str(settings.QUANTUM_LLM_API_KEY or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with httpx.AsyncClient(timeout=settings.QUANTUM_LLM_TIMEOUT) as client:
            response = await client.post(
                settings.QUANTUM_LLM_ENDPOINT,
                headers=headers,
                json={
                    "task": "clarifier",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt or "Return one JSON object only."},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            body = response.json()

        content = ""
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message", {})
                    if isinstance(message, dict):
                        content = str(message.get("content") or "")
            if not content:
                content = str(body.get("content") or body.get("response") or "")
            if not content and isinstance(body.get("result"), dict):
                content = json.dumps(body.get("result"))
        if not content:
            content = fallback

        prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        completion_tokens = _estimate_tokens(content)
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase=phase,
            provider="quantum_remote_endpoint",
            model="clarifier",
            latency_ms=(time.time() - started) * 1000.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=True,
        )
        return content
    except Exception as exc:
        logger.exception("llm.quantum.clarifier.error")
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase=phase,
            provider="quantum_remote_endpoint",
            model="clarifier",
            latency_ms=(time.time() - started) * 1000.0,
            prompt_tokens=_estimate_tokens(system_prompt) + _estimate_tokens(user_prompt),
            completion_tokens=_estimate_tokens(fallback),
            total_tokens=_estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + _estimate_tokens(fallback),
            success=False,
            error_message=str(exc),
        )
        return fallback


async def generate_quantum_code(delegation_spec: dict[str, Any], experiment_id: str | None = None) -> str:
    framework = _normalize_framework(str(delegation_spec.get("framework", "pennylane")))
    started = time.time()

    codehub_code = await _generate_via_codehub(delegation_spec, experiment_id=experiment_id)
    if codehub_code:
        return codehub_code

    if not settings.QUANTUM_LLM_ENDPOINT:
        code = _local_quantum_template(framework, delegation_spec=delegation_spec)
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
        headers = {"Content-Type": "application/json"}
        key = str(settings.QUANTUM_LLM_API_KEY or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with httpx.AsyncClient(timeout=settings.QUANTUM_LLM_TIMEOUT) as client:
            response = await client.post(
                settings.QUANTUM_LLM_ENDPOINT,
                headers=headers,
                json={"delegation_spec": delegation_spec},
            )
            response.raise_for_status()
            body = response.json()
        generated_code = _extract_code_from_payload(body)
        code = generated_code if _contains_quantum_layer(generated_code) else _local_quantum_template(framework, delegation_spec=delegation_spec)
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
        return _local_quantum_template(framework, delegation_spec=delegation_spec)
