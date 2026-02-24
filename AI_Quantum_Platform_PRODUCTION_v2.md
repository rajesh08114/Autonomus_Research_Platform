# ⚛ AI + QUANTUM RESEARCH PLATFORM
## Complete Production-Level System Design
### HLD · LLD · LangGraph Implementation · Agent Prompts · API Contracts

**Version:** 2.0.0 · **Status:** Production · **Backend-Only · LLM-Orchestrated · LangGraph**

---

## Table of Contents

| # | Section |
|---|---------|
| **PART I** | **High-Level Design (HLD)** |
| 1 | [Executive Overview & Platform Vision](#1-executive-overview--platform-vision) |
| 2 | [System Goals & Non-Functional Requirements](#2-system-goals--non-functional-requirements) |
| 3 | [Full Platform Architecture](#3-full-platform-architecture) |
| 4 | [LangGraph Workflow — Complete Graph Definition](#4-langgraph-workflow--complete-graph-definition) |
| 5 | [Agent Ecosystem Map](#5-agent-ecosystem-map) |
| 6 | [Quantum Gate Architecture](#6-quantum-gate-architecture) |
| 7 | [Data Flow & State Lifecycle](#7-data-flow--state-lifecycle) |
| 8 | [Technology Stack & Justification](#8-technology-stack--justification) |
| **PART II** | **Low-Level Design (LLD) — Agent Implementation** |
| 9 | [ResearchState — Complete Schema](#9-researchstate--complete-schema) |
| 10 | [JSON Action Contract — Full Specification](#10-json-action-contract--full-specification) |
| 11 | [Agent 1 — Clarification Agent](#11-agent-1--clarification-agent) |
| 12 | [Agent 2 — Research Planner Agent](#12-agent-2--research-planner-agent) |
| 13 | [Agent 3 — Environment Manager Agent](#13-agent-3--environment-manager-agent) |
| 14 | [Agent 4 — Dataset Acquisition Agent](#14-agent-4--dataset-acquisition-agent) |
| 15 | [Agent 5 — Code Generation Agent](#15-agent-5--code-generation-agent) |
| 16 | [Agent 6 — Quantum Delegation Gate](#16-agent-6--quantum-delegation-gate) |
| 17 | [Agent 7 — Job Scheduler Agent](#17-agent-7--job-scheduler-agent) |
| 18 | [Agent 8 — Error Recovery Agent](#18-agent-8--error-recovery-agent) |
| 19 | [Agent 9 — Results Evaluator Agent](#19-agent-9--results-evaluator-agent) |
| 20 | [Agent 10 — Documentation Generator Agent](#20-agent-10--documentation-generator-agent) |
| **PART III** | **Complete API Specification** |
| 21 | [API Architecture & Auth](#21-api-architecture--auth) |
| 22 | [Experiment Lifecycle Endpoints](#22-experiment-lifecycle-endpoints) |
| 23 | [Interaction Endpoints](#23-interaction-endpoints) |
| 24 | [Monitoring & Observability Endpoints](#24-monitoring--observability-endpoints) |
| 25 | [Admin & Control Endpoints](#25-admin--control-endpoints) |
| **PART IV** | **Project Implementation** |
| 26 | [Complete Project File Structure](#26-complete-project-file-structure) |
| 27 | [Core LangGraph Implementation Code](#27-core-langgraph-implementation-code) |
| 28 | [Database Schema](#28-database-schema) |
| 29 | [Configuration & Environment](#29-configuration--environment) |
| 30 | [Security, Deployment & Operations](#30-security-deployment--operations) |

---

# PART I — HIGH-LEVEL DESIGN (HLD)

---

## 1. Executive Overview & Platform Vision

### 1.1 What Is This System

The **AI + Quantum Research Platform** is a production-grade, fully autonomous research orchestration backend. A researcher provides a single natural-language prompt. The system — without any further manual intervention — conducts the entire research lifecycle:

```
Natural Language Prompt
        │
        ▼
  Structured Clarification
        │
        ▼
  Reproducible Research Plan
        │
        ▼
  Automated Environment Setup
        │
        ▼
  Dataset Acquisition & Validation
        │
        ▼
  Code Generation (Classical or Quantum)
        │
        ▼
  Job Execution with Error Self-Repair
        │
        ▼
  Results Evaluation & Metric Extraction
        │
        ▼
  Final Research Documentation
```

### 1.2 What Makes This Production-Grade

| Aspect | Implementation |
|--------|---------------|
| **Single LLM, Multiple Agents** | One LLM model; role-switch via phase-specific system prompts |
| **Quantum Specialization** | Hard gate: only `delegate_quantum_code` action routes to Quantum LLM |
| **Structured Contract** | All agent I/O is validated JSON — zero prose contamination |
| **Self-Healing** | Error Recovery Agent auto-diagnoses and patches code; retry ≤ 5 |
| **User Sovereignty** | Every denial is respected; alternative is offered; plan adapts |
| **Full Observability** | Every state transition logged; ResearchState versioned per phase |
| **Reproducibility** | Seeds pinned; package versions locked; commands recorded |
| **Zero Trust Execution** | Subprocess isolation; no shell=True; path whitelist enforced |

### 1.3 System Boundaries

```
┌──────────────────────────────────────────────────────────────┐
│                    PLATFORM BOUNDARY                         │
│                                                              │
│  IN SCOPE:                                                   │
│  • FastAPI backend server                                    │
│  • LangGraph orchestration engine                            │
│  • All 10 LLM-driven agents                                  │
│  • Subprocess execution runtime                              │
│  • SQLite state persistence                                  │
│  • Local file system management                              │
│  • Quantum LLM delegation interface                          │
│                                                              │
│  OUT OF SCOPE:                                               │
│  • Frontend / UI                                             │
│  • Cloud quantum hardware (IBM Q) — pluggable               │
│  • External model training infrastructure                    │
│  • Billing / multi-tenancy                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. System Goals & Non-Functional Requirements

### 2.1 Functional Goals

| ID | Goal | Success Criterion |
|----|------|-------------------|
| G-01 | Full autonomy from prompt to documentation | Zero manual code writing after clarification phase |
| G-02 | Single LLM orchestrates all non-quantum decisions | One API key, one model, ten role contexts |
| G-03 | Quantum circuit generation is specialized | `delegate_quantum_code` is the only path to Quantum LLM |
| G-04 | Error self-repair | System heals 80%+ of common runtime errors without user input |
| G-05 | User denial handling | Any denial triggers graceful alternative within same turn |
| G-06 | Reproducible experiments | Re-running same state produces identical outputs |
| G-07 | Complete audit trail | Every decision, action, and state change is persisted |

### 2.2 Non-Functional Requirements

| Category | Requirement | Target |
|----------|-------------|--------|
| **Reliability** | Experiment completion rate | ≥ 85% (excl. unsupported hardware) |
| **Latency** | Clarification → first code file | < 60 seconds |
| **Timeout** | Max subprocess execution | 3600 seconds (configurable) |
| **Retry limit** | Error recovery attempts | ≤ 5 per script |
| **State size** | ResearchState JSON cap | ≤ 500 KB per phase |
| **Concurrency** | Simultaneous experiments | ≤ 10 (resource-gated) |
| **Log retention** | Experiment artifacts | 30 days (configurable) |
| **Security** | Path access | Project directory only (whitelist) |

---

## 3. Full Platform Architecture

### 3.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI + QUANTUM RESEARCH PLATFORM                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FASTAPI LAYER (:8000)                        │   │
│  │  POST /research/start  │  POST /research/{id}/answer                │   │
│  │  POST /research/{id}/confirm  │  GET /research/{id}/status          │   │
│  │  GET /research/{id}/logs  │  GET /research/{id}/results             │   │
│  │  GET /research/{id}/report  │  DELETE /research/{id}/abort          │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                    LANGGRAPH ORCHESTRATION ENGINE                    │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ Clarifier   │→ │   Planner    │→ │  Env Manager │               │   │
│  │  │  Agent [1]  │  │  Agent [2]   │  │   Agent [3]  │               │   │
│  │  └─────────────┘  └──────────────┘  └──────┬───────┘               │   │
│  │                                             │                       │   │
│  │  ┌──────────────────────────────────────────▼───────────────────┐  │   │
│  │  │                   Dataset Agent [4]                           │  │   │
│  │  └──────────────────────────────────────────┬───────────────────┘  │   │
│  │                                             │                       │   │
│  │  ┌──────────────────────────────────────────▼───────────────────┐  │   │
│  │  │                Code Generation Agent [5]                      │  │   │
│  │  │         ┌───────────────────┐    ┌─────────────────────┐     │  │   │
│  │  │         │ requires_quantum  │    │  General Code Gen   │     │  │   │
│  │  │         │     = TRUE        │    │  (LLM writes .py)   │     │  │   │
│  │  │         └────────┬──────────┘    └──────────┬──────────┘     │  │   │
│  │  └──────────────────┼────────────────────────────┼──────────────┘  │   │
│  │                     │                            │                  │   │
│  │  ┌──────────────────▼──────────┐                │                  │   │
│  │  │  QUANTUM GATE Agent [6]     │                │                  │   │
│  │  │  delegate_quantum_code ──▶  │                │                  │   │
│  │  │  Quantum LLM API            │                │                  │   │
│  │  │  ◀── returns circuit code   │                │                  │   │
│  │  └──────────────────┬──────────┘                │                  │   │
│  │                     └──────────────┬─────────────┘                 │   │
│  │                                    │                                │   │
│  │  ┌─────────────────────────────────▼──────────────────────────┐   │   │
│  │  │              Job Scheduler Agent [7]                        │   │   │
│  │  │              Subprocess Runner (isolated)                   │   │   │
│  │  └──────────────────────────┬─────────────────────────────────┘   │   │
│  │                             │                                       │   │
│  │              ┌──────────────┴──────────────┐                       │   │
│  │              ▼ [error]              ▼ [ok]                          │   │
│  │  ┌───────────────────┐   ┌──────────────────────┐                  │   │
│  │  │ Error Recovery    │   │  Results Evaluator   │                  │   │
│  │  │   Agent [8]       │   │     Agent [9]        │                  │   │
│  │  │ retry ≤ 5 ──────▶ │   └──────────┬───────────┘                  │   │
│  │  │ retry > 5 → ABORT │              │                              │   │
│  │  └───────────────────┘   ┌──────────▼───────────┐                  │   │
│  │                          │  Doc Generator       │                  │   │
│  │                          │    Agent [10]        │                  │   │
│  │                          └──────────┬───────────┘                  │   │
│  └─────────────────────────────────────┼───────────────────────────────┘  │
│                                        │                                   │
│  ┌─────────────────────────────────────▼───────────────────────────────┐  │
│  │              PERSISTENCE LAYER                                       │  │
│  │  SQLite (ResearchState)  │  Local FS (/projects/)  │  JSON Logs      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. LangGraph Workflow — Complete Graph Definition

### 4.1 Graph Nodes and Edges

```python
# ─────────────────────────────────────────────────────────────────────────────
# langgraph_definition.py  (conceptual — full impl in Part IV)
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END
from typing import Literal

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # ── Register all agent nodes ─────────────────────────────────────────────
    graph.add_node("clarifier",          clarifier_agent_node)
    graph.add_node("planner",            planner_agent_node)
    graph.add_node("env_manager",        env_manager_agent_node)
    graph.add_node("dataset_manager",    dataset_agent_node)
    graph.add_node("code_generator",     code_gen_agent_node)
    graph.add_node("quantum_gate",       quantum_gate_node)
    graph.add_node("job_scheduler",      job_scheduler_agent_node)
    graph.add_node("subprocess_runner",  subprocess_runner_node)
    graph.add_node("error_recovery",     error_recovery_agent_node)
    graph.add_node("results_evaluator",  results_evaluator_agent_node)
    graph.add_node("doc_generator",      doc_generator_agent_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("clarifier")

    # ── Linear edges (happy path) ────────────────────────────────────────────
    graph.add_edge("clarifier",       "planner")
    graph.add_edge("planner",         "env_manager")
    graph.add_edge("env_manager",     "dataset_manager")
    graph.add_edge("dataset_manager", "code_generator")

    # ── Conditional: quantum or direct to scheduler ───────────────────────────
    graph.add_conditional_edges(
        "code_generator",
        route_quantum_or_direct,
        {
            "quantum":    "quantum_gate",
            "no_quantum": "job_scheduler",
        }
    )

    # ── Quantum gate merges back to scheduler ────────────────────────────────
    graph.add_edge("quantum_gate",    "job_scheduler")

    # ── Scheduler triggers subprocess runner ─────────────────────────────────
    graph.add_edge("job_scheduler",   "subprocess_runner")

    # ── Conditional: success or error ────────────────────────────────────────
    graph.add_conditional_edges(
        "subprocess_runner",
        route_success_or_error,
        {
            "success": "results_evaluator",
            "error":   "error_recovery",
            "abort":   END,
        }
    )

    # ── Error recovery loops back to runner ──────────────────────────────────
    graph.add_conditional_edges(
        "error_recovery",
        route_retry_or_abort,
        {
            "retry": "subprocess_runner",
            "abort": END,
        }
    )

    # ── Final path ───────────────────────────────────────────────────────────
    graph.add_edge("results_evaluator", "doc_generator")
    graph.add_edge("doc_generator",     END)

    return graph.compile()
```

### 4.2 Routing Functions

```python
def route_quantum_or_direct(state: ResearchState) -> Literal["quantum", "no_quantum"]:
    """Route code generation: quantum circuit needed or not."""
    return "quantum" if state["requires_quantum"] else "no_quantum"


def route_success_or_error(state: ResearchState) -> Literal["success", "error", "abort"]:
    """Route after subprocess execution."""
    last_log = state["execution_logs"][-1] if state["execution_logs"] else {}
    if last_log.get("returncode", -1) == 0:
        return "success"
    if state["retry_count"] >= 5:
        return "abort"
    return "error"


def route_retry_or_abort(state: ResearchState) -> Literal["retry", "abort"]:
    """Route error recovery: retry or final abort."""
    if state["retry_count"] >= 5:
        return "abort"
    return "retry"
```

### 4.3 State Persistence Checkpointing

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Every node transition is checkpointed to SQLite
checkpointer = SqliteSaver.from_conn_string("/workspace/state.db")
compiled_graph = build_research_graph().with_config(
    {"checkpointer": checkpointer}
)
```

---

## 5. Agent Ecosystem Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENT ECOSYSTEM                                  │
│                                                                         │
│  Agent [1]  CLARIFIER          Goal: Extract structured research intent │
│  Agent [2]  PLANNER            Goal: Design reproducible research plan  │
│  Agent [3]  ENV MANAGER        Goal: Provision exact execution env      │
│  Agent [4]  DATASET AGENT      Goal: Acquire and validate data          │
│  Agent [5]  CODE GEN           Goal: Write complete, runnable code      │
│  Agent [6]  QUANTUM GATE       Goal: Delegate circuit specs to QLL M    │
│  Agent [7]  JOB SCHEDULER      Goal: Sequence and schedule execution    │
│  Agent [8]  ERROR RECOVERY     Goal: Diagnose and repair failures       │
│  Agent [9]  EVALUATOR          Goal: Extract metrics and generate plots │
│  Agent [10] DOC GENERATOR      Goal: Produce final research document    │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  SHARED:  ResearchState (TypedDict)  ←── flows through ALL agents       │
│  LLM:     Single master LLM  ←── drives agents [1–5], [7–10]           │
│  QUANTUM: Specialized LLM  ←── drives agent [6] ONLY                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Quantum Gate Architecture

### 6.1 Gate Decision Logic

```
                    CODE GENERATION AGENT
                             │
                 ┌───────────▼────────────┐
                 │  Inspect research_plan │
                 │  Does task require:    │
                 │  - Quantum circuits?   │
                 │  - Qubit manipulation? │
                 │  - VQE/QAOA/QNN?      │
                 └───────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
          YES │                             │ NO
              ▼                             ▼
   emit: delegate_quantum_code    emit: write_code
              │                             │
              ▼                             ▼
    ┌─────────────────┐           Write .py files
    │  QUANTUM GATE   │           using master LLM
    │  validates spec │
    │  routes to      │
    │  Quantum LLM    │
    └────────┬────────┘
             │
             ▼
    Quantum LLM generates:
    - quantum_circuit.py
    - Gate sequences
    - Measurement ops
    - Backend config
             │
             ▼
    Code Gen Agent merges
    circuit into project
             │
             ▼
    Job Scheduler continues
```

### 6.2 What the Quantum LLM Receives

```json
{
  "role": "system",
  "content": "You are a specialized Quantum Circuit Code Generator. Output ONLY valid Python code using the specified framework. No explanations. No prose.",
  "delegation_spec": {
    "framework":          "pennylane",
    "algorithm":          "VQE",
    "qubit_count":        4,
    "layers":             3,
    "dataset_info": {
      "n_features":       4,
      "encoding":         "angle_encoding",
      "n_classes":        3
    },
    "training_strategy":  "hybrid",
    "optimizer":          "adam",
    "backend":            "default.qubit",
    "return_expectation": "PauliZ",
    "integration_point":  "model.py::QuantumLayer.forward()"
  }
}
```

---

## 7. Data Flow & State Lifecycle

### 7.1 ResearchState Evolution Per Phase

```
PHASE         STATE KEYS WRITTEN
──────────────────────────────────────────────────────────────────
[1] Clarifier    user_prompt, clarifications, phase="clarifier"
[2] Planner      research_plan, requires_quantum, framework,
                 required_packages, project_id, project_path
[3] Env Manager  installed_packages, venv_path, denied_actions[]
[4] Dataset      dataset_path, data_report, dataset_source
[5] Code Gen     created_files[], quantum_circuit_code (if delegated)
[6] Quantum Gate quantum_circuit_code, created_files (updated)
[7] Scheduler    execution_order[], current_script
[8] Runner       execution_logs[], errors[]
[8] Error Recov  errors[], retry_count, repair_history[], denied_actions[]
[9] Evaluator    metrics{}, plots_generated[], evaluation_summary{}
[10] Doc Gen     documentation_path, status="success"
──────────────────────────────────────────────────────────────────
```

---

## 8. Technology Stack & Justification

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| API Framework | FastAPI | 0.111+ | Async, auto-docs, Pydantic validation |
| Workflow Engine | LangGraph | 0.2+ | Stateful agent graphs, checkpointing |
| LLM SDK | LangChain Core | 0.2+ | Provider abstraction for master LLM |
| Master LLM | Claude 3.5 Sonnet / GPT-4o | Latest | Instruction following, JSON reliability |
| Quantum LLM | Custom fine-tuned endpoint | - | Specialized circuit generation |
| Quantum (primary) | PennyLane | 0.36+ | Hybrid quantum-classical ML |
| Quantum (alt) | Qiskit / Cirq | 1.x / 1.x | Alternative backends |
| ML Framework | PyTorch | 2.2+ | Classical ML; integrates with PennyLane |
| ML (lightweight) | scikit-learn | 1.4+ | Fallback; denial-safe |
| Data | Pandas + NumPy | 2.x / 1.26+ | Data processing |
| Visualization | Matplotlib + Seaborn | 3.8+ / 0.13+ | Plots and charts |
| Database | SQLite + SQLAlchemy | - | State persistence, checkpointing |
| HTTP Client | HTTPX | 0.27+ | Async API calls |
| Process Exec | Python subprocess | stdlib | Isolated script execution |
| Config | Pydantic Settings | 2.x | Type-safe .env parsing |
| Logging | Python structlog | 24.x | Structured JSON logging |
| Testing | Pytest + Pytest-asyncio | - | Async test support |
| Packaging | Poetry | 1.8+ | Dependency locking |

---

# PART II — LOW-LEVEL DESIGN (LLD) — AGENT IMPLEMENTATION

---

## 9. ResearchState — Complete Schema

```python
# ─────────────────────────────────────────────────────────────────────────────
# src/state/research_state.py
# ─────────────────────────────────────────────────────────────────────────────
from typing import TypedDict, Optional, Any
from enum import Enum

class ExperimentStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    WAITING    = "waiting_user"
    SUCCESS    = "success"
    ABORTED    = "aborted"
    FAILED     = "failed"

class ErrorRecord(TypedDict):
    category:       str   # "ModuleNotFoundError" | "SyntaxError" | ...
    message:        str   # raw exception message
    file_path:      str   # which script failed
    line_number:    int
    traceback:      str
    timestamp:      float

class ExecutionLog(TypedDict):
    script_path:    str
    returncode:     int
    stdout:         str   # capped at 10k chars
    stderr:         str   # capped at 5k chars
    duration_sec:   float
    timestamp:      float

class RepairRecord(TypedDict):
    attempt:        int
    error_category: str
    fix_description: str
    file_changed:   str
    find_text:      str
    replace_text:   str
    timestamp:      float

class DenialRecord(TypedDict):
    action:         str
    denied_item:    str   # package name or action type
    reason:         str   # user-provided reason (optional)
    alternative_offered: str
    alternative_accepted: bool
    timestamp:      float

class ResearchState(TypedDict):
    # ── Identity ────────────────────────────────────────────────────────────
    experiment_id:          str           # UUID4
    project_path:           str           # /workspace/projects/{experiment_id}

    # ── Phase tracking ──────────────────────────────────────────────────────
    phase:                  str           # current active agent phase
    status:                 ExperimentStatus
    timestamp_start:        float
    timestamp_end:          Optional[float]

    # ── Input ───────────────────────────────────────────────────────────────
    user_prompt:            str           # original user input
    clarifications:         dict          # structured answers from Phase 1
    research_plan:          dict          # full plan from Planner Agent

    # ── Quantum configuration ───────────────────────────────────────────────
    requires_quantum:       bool
    quantum_framework:      Optional[str]   # "pennylane"|"qiskit"|"cirq"|None
    quantum_algorithm:      Optional[str]   # "VQE"|"QAOA"|"QNN"|"Grover"|None
    quantum_qubit_count:    Optional[int]
    quantum_circuit_code:   Optional[str]   # returned by Quantum LLM
    quantum_backend:        Optional[str]   # "default.qubit"|"ibmq_qasm"|...

    # ── Environment ─────────────────────────────────────────────────────────
    framework:              str           # "pytorch"|"sklearn"|"tensorflow"
    python_version:         str           # "3.11"
    required_packages:      list[str]     # ["pennylane==0.36.0", "torch==2.2.0"]
    installed_packages:     list[str]     # confirmed installed
    venv_path:              str
    output_format:          str           # ".py" | ".ipynb"

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset_source:         str           # "kaggle"|"sklearn"|"synthetic"|"upload"
    dataset_path:           str           # /workspace/projects/{id}/data/raw/
    kaggle_dataset_id:      Optional[str] # "username/dataset-name"
    data_report:            dict          # shape, columns, nulls, dtypes

    # ── Project files ───────────────────────────────────────────────────────
    created_files:          list[str]     # ordered list of generated files
    execution_order:        list[str]     # dependency-resolved run order

    # ── Execution ───────────────────────────────────────────────────────────
    execution_logs:         list[ExecutionLog]
    current_script:         Optional[str]
    total_duration_sec:     Optional[float]

    # ── Error & recovery ────────────────────────────────────────────────────
    errors:                 list[ErrorRecord]
    retry_count:            int             # 0-5; abort at 5
    last_error_category:    Optional[str]
    consecutive_same_error: int             # triggers strategy change at 2
    repair_history:         list[RepairRecord]

    # ── User interaction ────────────────────────────────────────────────────
    denied_actions:         list[DenialRecord]
    pending_user_question:  Optional[dict]  # set when waiting for user input
    pending_user_confirm:   Optional[dict]  # set when waiting for confirmation

    # ── Results ─────────────────────────────────────────────────────────────
    metrics:                dict
    plots_generated:        list[str]
    evaluation_summary:     dict

    # ── Documentation ───────────────────────────────────────────────────────
    documentation_path:     Optional[str]
    report_sections:        list[str]

    # ── Hyperparameters (from clarifications) ───────────────────────────────
    target_metric:          str           # "accuracy"|"f1"|"rmse"|"fidelity"
    hardware_target:        str           # "cpu"|"cuda"|"ibmq"
    random_seed:            int           # default 42
    max_epochs:             Optional[int]
    batch_size:             Optional[int]

    # ── Metadata ────────────────────────────────────────────────────────────
    llm_calls_count:        int           # total LLM invocations
    total_tokens_used:      int
    phase_timings:          dict          # {phase_name: duration_sec}
```

---

## 10. JSON Action Contract — Full Specification

### 10.1 Master Schema

```typescript
// Every LLM response MUST conform to this schema exactly
interface AgentAction {
  action:     ActionType;          // required — one of 12 allowed values
  reasoning:  string;              // required — why this action; min 10 chars
  parameters: ActionParameters;    // required — action-specific payload
  next_step:  string;              // required — next node name or "finish"/"abort"
  confidence: number;              // required — 0.0 to 1.0
  warnings?:  string[];            // optional — non-blocking concerns
}
```

### 10.2 All Action Parameter Schemas

```typescript
// ── ask_user ────────────────────────────────────────────────────────────────
interface AskUserParams {
  questions: Array<{
    id:          string;           // "Q1"–"Q12"
    text:        string;
    type:        "choice" | "boolean" | "text" | "number";
    options?:    string[];         // for type="choice"
    default?:    string | number | boolean;
    required:    boolean;
  }>;
}

// ── create_project ───────────────────────────────────────────────────────────
interface CreateProjectParams {
  project_id:     string;
  base_path:      string;          // always under PROJECT_ROOT
  directories:    string[];        // relative paths to create
  init_files:     string[];        // e.g., ["__init__.py", ".gitignore"]
}

// ── write_code ───────────────────────────────────────────────────────────────
interface WriteCodeParams {
  file_path:      string;          // absolute; must be under project_path
  language:       "python";
  content:        string;          // full file content
  depends_on:     string[];        // other files this depends on
  test_command:   string;          // syntax validation command
  encoding:       "utf-8";
}

// ── modify_file ──────────────────────────────────────────────────────────────
interface ModifyFileParams {
  file_path:      string;
  find:           string;          // exact string to find (unique in file)
  replace:        string;          // replacement string
  reason:         string;          // why this change fixes the error
  backup:         boolean;         // always true in production
}

// ── install_package ──────────────────────────────────────────────────────────
interface InstallPackageParams {
  package:             string;     // "pennylane"
  version:             string;     // "0.36.0" — always pin
  pip_flags:           string[];   // ["--quiet", "--no-cache-dir"]
  fallback_if_denied:  string;     // alternative if user denies
  reason:              string;     // why this package is needed
}

// ── run_python ───────────────────────────────────────────────────────────────
interface RunPythonParams {
  script_path:        string;      // must be under project_path
  args:               string[];
  timeout_seconds:    number;
  capture_output:     boolean;     // always true
  working_directory:  string;
  env_vars:           Record<string, string>;
  log_path:           string;
}

// ── delegate_quantum_code ────────────────────────────────────────────────────
interface DelegateQuantumCodeParams {
  target_file:         string;
  framework:           "pennylane" | "qiskit" | "cirq";
  algorithm:           "VQE" | "QAOA" | "QNN" | "Grover" | "custom";
  qubit_count:         number;
  layers:              number;
  dataset_info: {
    n_features:        number;
    n_classes:         number;
    encoding:          "angle_encoding" | "amplitude_encoding" | "basis_encoding";
  };
  training_strategy:   "hybrid" | "pure-quantum";
  optimizer:           string;
  backend:             string;
  return_expectation:  string;
  integration_point:   string;
}

// ── analyze_results ──────────────────────────────────────────────────────────
interface AnalyzeResultsParams {
  metrics_file:     string;
  log_file:         string;
  generate_plots:   boolean;
  plot_types:       ("loss_curve" | "confusion_matrix" | "roc_curve" | "circuit_diagram")[];
  output_dir:       string;
  baseline_compare: boolean;
}

// ── generate_documentation ───────────────────────────────────────────────────
interface GenerateDocumentationParams {
  output_path:       string;
  include_sections:  string[];
  format:            "markdown" | "html";
  include_code:      boolean;
  include_plots:     boolean;
}
```

### 10.3 Action Validation Rules

```python
# ─────────────────────────────────────────────────────────────────────────────
# src/core/action_validator.py
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_ACTIONS = {
    "ask_user", "create_project", "create_file", "write_code",
    "modify_file", "install_package", "run_python", "analyze_results",
    "generate_documentation", "delegate_quantum_code", "finish", "abort"
}

REQUIRED_FIELDS = {
    "run_python":             ["script_path", "timeout_seconds"],
    "delegate_quantum_code":  ["framework", "algorithm", "qubit_count"],
    "write_code":             ["file_path", "content"],
    "modify_file":            ["file_path", "find", "replace"],
    "install_package":        ["package", "version"],
}

PATH_RESTRICTED_ACTIONS = {"write_code", "modify_file", "run_python", "create_file"}

def validate_action(action: dict, state: ResearchState) -> tuple[bool, str]:
    if action.get("action") not in ALLOWED_ACTIONS:
        return False, f"Unknown action: {action.get('action')}"
    if not action.get("reasoning") or len(action["reasoning"]) < 10:
        return False, "reasoning field missing or too short"
    for field in REQUIRED_FIELDS.get(action["action"], []):
        if field not in action.get("parameters", {}):
            return False, f"Missing required parameter: {field}"
    # Path whitelist check
    if action["action"] in PATH_RESTRICTED_ACTIONS:
        path = action["parameters"].get("file_path") or action["parameters"].get("script_path")
        if path and not path.startswith(state["project_path"]):
            return False, f"Path traversal blocked: {path}"
    return True, "ok"
```
---

## 11. Agent 1 — Clarification Agent

### 11.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `clarifier` |
| **Goal** | Transform raw user prompt into a fully-structured, unambiguous research specification |
| **Input** | `user_prompt` (raw string) |
| **Output action** | `ask_user` |
| **Allowed actions** | `ask_user` only |
| **LLM role context** | Research Intent Extractor |
| **Blocks until** | All critical Q&A answers received from user |

### 11.2 Production System Prompt

```
SYSTEM:
You are the Clarification Agent for the AI + Quantum Research Platform.
Your ONLY goal is to extract every piece of information required to plan
a reproducible quantum or AI research experiment.

You operate as a strict JSON machine. You NEVER output prose.
You ONLY output a single valid JSON object matching the AgentAction schema.

CURRENT STATE:
{state_json}

YOUR TASK:
Analyze the user_prompt carefully. Identify ALL ambiguities and missing
parameters that the Planner Agent will need. Generate a minimal, targeted
set of clarification questions — ask ONLY what you cannot confidently infer.

QUESTION GENERATION RULES:
1. If the prompt clearly mentions "quantum", ask Q3=True, then ask Q4 (which framework).
2. If the prompt mentions a specific dataset by name, skip Q5; infer dataset_source="kaggle".
3. Never ask more than 8 questions. Prioritize: format, quantum, data, metric, hardware.
4. Group related questions. Use "choice" type with options whenever possible.
5. Infer Python version as 3.11 unless the user's prompt specifies otherwise.
6. Infer seed=42 unless user mentions reproducibility specifically.

INFERENCE RULES (do NOT ask if inferable):
- "classify" or "classification" → algorithm_class = "supervised"
- "cluster" or "unsupervised" → algorithm_class = "unsupervised"
- "VQE", "QAOA", "quantum circuit" → requires_quantum = true
- "train on" or "dataset" mentioned → ask for source
- "GPU" or "CUDA" mentioned → hardware_target = "cuda"
- "notebook" or "jupyter" → output_format = ".ipynb"

OUTPUT FORMAT (strict):
{
  "action": "ask_user",
  "reasoning": "<why these specific questions are needed>",
  "parameters": {
    "questions": [
      {
        "id": "Q1",
        "text": "<clear, concise question>",
        "type": "choice|boolean|text|number",
        "options": ["opt1", "opt2"],
        "default": "<sensible default>",
        "required": true|false
      }
    ]
  },
  "next_step": "planner",
  "confidence": 0.95
}

NEVER output anything outside the JSON object.
NEVER ask for information you can confidently infer.
NEVER use more than 8 questions.
```

### 11.3 Question Bank (All 12 Possible Questions)

```python
QUESTION_BANK = {
    "Q1":  {"topic": "output_format",     "text": "Do you want .py scripts or .ipynb notebooks?",                      "type": "choice",  "options": [".py", ".ipynb"],                                           "default": ".py"},
    "Q2":  {"topic": "algorithm_class",   "text": "What type of ML/AI task is this?",                                  "type": "choice",  "options": ["supervised","unsupervised","reinforcement","quantum_ml"],   "default": "supervised"},
    "Q3":  {"topic": "requires_quantum",  "text": "Does this experiment require quantum circuits?",                     "type": "boolean", "default": False},
    "Q4":  {"topic": "quantum_framework", "text": "Which quantum framework should be used?",                           "type": "choice",  "options": ["pennylane","qiskit","cirq","no_preference"],                 "default": "pennylane"},
    "Q5":  {"topic": "dataset_source",    "text": "Where should the dataset come from?",                               "type": "choice",  "options": ["kaggle","sklearn","synthetic","upload"],                      "default": "sklearn"},
    "Q6":  {"topic": "kaggle_dataset_id", "text": "Enter the Kaggle dataset path (e.g., username/dataset-name):",      "type": "text",    "default": None},
    "Q7":  {"topic": "target_metric",     "text": "What is the primary evaluation metric?",                            "type": "choice",  "options": ["accuracy","f1_macro","rmse","roc_auc","fidelity"],           "default": "accuracy"},
    "Q8":  {"topic": "hardware_target",   "text": "Target hardware for execution?",                                    "type": "choice",  "options": ["cpu","cuda","ibmq"],                                        "default": "cpu"},
    "Q9":  {"topic": "python_version",    "text": "Which Python version?",                                             "type": "choice",  "options": ["3.10","3.11","3.12"],                                       "default": "3.11"},
    "Q10": {"topic": "random_seed",       "text": "Random seed for reproducibility?",                                  "type": "number",  "default": 42},
    "Q11": {"topic": "max_epochs",        "text": "Maximum training epochs or circuit iterations?",                    "type": "number",  "default": 50},
    "Q12": {"topic": "data_sensitivity",  "text": "Is the dataset private or sensitive?",                              "type": "boolean", "default": False},
}
```

### 11.4 Node Implementation

```python
# src/agents/clarifier_agent.py
import json
from src.state.research_state import ResearchState
from src.llm.master_llm import invoke_master_llm
from src.core.action_validator import validate_action

CLARIFIER_SYSTEM_PROMPT = """..."""  # full prompt above

async def clarifier_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "clarifier"
    compressed_state = compress_state_for_prompt(state)

    prompt = CLARIFIER_SYSTEM_PROMPT.replace("{state_json}", json.dumps(compressed_state))
    raw_response = await invoke_master_llm(prompt, state["user_prompt"])

    action = parse_json_response(raw_response)
    valid, error = validate_action(action, state)

    if not valid:
        # Re-invoke with validation error context
        action = await invoke_master_llm(
            prompt,
            f"Previous output failed validation: {error}. Fix and retry."
        )

    # Set state to waiting for user
    state["pending_user_question"] = action["parameters"]
    state["status"] = "waiting_user"
    state["llm_calls_count"] += 1

    return state
```

---

## 12. Agent 2 — Research Planner Agent

### 12.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `planner` |
| **Goal** | Produce a complete, deterministic, reproducible research plan from clarified parameters |
| **Input** | `clarifications` dict (all Q&A answers) |
| **Output action** | `create_project` |
| **Allowed actions** | `create_project` |
| **LLM role context** | Research Architect |

### 12.2 Production System Prompt

```
SYSTEM:
You are the Research Planner Agent for the AI + Quantum Research Platform.
Your ONLY goal is to produce a complete, deterministic, executable research plan
from the structured clarifications provided.

You operate as a strict JSON machine. You NEVER output prose.
You ONLY output a single valid JSON object matching the AgentAction schema.

CURRENT STATE (clarifications available):
{state_json}

YOUR TASK:
Using state.clarifications, produce:
1. A complete research_plan object (methodology, algorithm, data strategy, metrics)
2. A create_project action with all directories and init files
3. A complete required_packages list with exact pinned versions

PLANNING RULES:
1. requires_quantum must be set based on clarifications.requires_quantum.
2. If requires_quantum=true, set quantum_framework from clarifications.quantum_framework.
3. Always include: numpy, pandas, matplotlib, scikit-learn in required_packages.
4. If pytorch used: torch==2.2.0, torchvision==0.17.0
5. If pennylane used: pennylane==0.36.0, pennylane-lightning==0.36.0
6. If qiskit used: qiskit==1.1.0, qiskit-aer==0.14.0
7. If kaggle source: add kaggle==1.6.12 to packages.
8. Always add structlog==24.1.0 for structured logging.
9. Project ID format: exp_{YYYYMMDD}_{6-char-random}
10. Output format determines if jupyter should be in packages.

RESEARCH PLAN MUST INCLUDE:
- objective: one-sentence goal
- methodology: step-by-step approach
- algorithm: specific algorithm name and variant
- framework: primary ML/quantum framework
- dataset: source, expected shape, preprocessing
- metrics: list of evaluation metrics + primary metric
- hardware: cpu|cuda|ibmq + fallback
- reproducibility: seed, version pins, command log strategy
- estimated_duration_minutes: realistic estimate

OUTPUT FORMAT (strict):
{
  "action": "create_project",
  "reasoning": "<why this structure and these packages>",
  "parameters": {
    "project_id": "exp_20240315_a1b2c3",
    "base_path": "/workspace/projects/exp_20240315_a1b2c3",
    "directories": [
      "data/raw", "data/processed", "src", "outputs/plots",
      "outputs/model_checkpoint", "logs", "docs"
    ],
    "init_files": ["src/__init__.py", ".gitignore", "requirements.txt"]
  },
  "next_step": "env_manager",
  "confidence": 0.92
}

STATE UPDATES YOU MUST EMBED IN reasoning (the orchestrator extracts these):
  research_plan: { complete plan object }
  requires_quantum: true|false
  quantum_framework: "pennylane"|"qiskit"|"cirq"|null
  framework: "pytorch"|"sklearn"|"tensorflow"
  required_packages: ["package==version", ...]
  output_format: ".py"|".ipynb"
  target_metric: "accuracy"|"f1_macro"|...
  hardware_target: "cpu"|"cuda"|"ibmq"
  random_seed: 42

NEVER omit required_packages. NEVER use unpinned versions.
NEVER create directories outside /workspace/projects/.
```

---

## 13. Agent 3 — Environment Manager Agent

### 13.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `env_manager` |
| **Goal** | Provision the exact Python environment the research plan requires — safely, minimally, and with graceful denial handling |
| **Input** | `required_packages`, `python_version`, `project_path` |
| **Output actions** | `install_package` (one per missing package) |
| **Allowed actions** | `install_package`, `ask_user` |

### 13.2 Production System Prompt

```
SYSTEM:
You are the Environment Manager Agent for the AI + Quantum Research Platform.
Your ONLY goal is to ensure every required package is installed in the correct
version before any code runs.

You operate as a strict JSON machine. ONLY output valid JSON.

CURRENT STATE:
{state_json}

ENVIRONMENT RULES:
1. Compare required_packages vs installed_packages — only install MISSING ones.
2. NEVER reinstall already installed packages.
3. Always pin exact versions (package==X.Y.Z).
4. Issue ONE install_package action per LLM call — the orchestrator loops you.
5. If a package install was denied, it appears in denied_actions[]. Do NOT retry it.
   Instead, immediately offer the fallback specified in FALLBACK_MAP.
6. If the fallback is also denied, remove that capability from research_plan.
7. After all packages processed, set next_step="dataset_manager".

FALLBACK MAP (use exact alternatives):
  torch / tensorflow   → scikit-learn==1.4.2
  pennylane           → qiskit-aer==0.14.0
  qiskit              → pennylane==0.36.0 (or cirq==1.3.0)
  kaggle              → requests==2.31.0 (manual download)
  jupyter             → skip; convert to .py format
  cuda/gpu packages   → cpu-only variants

DRY-RUN VALIDATION (always include in reasoning):
  Run: pip install {package}=={version} --dry-run --quiet
  If dry-run fails → package doesn't exist; abort with error
  If dry-run succeeds → emit real install

OUTPUT FORMAT (strict, one action at a time):
{
  "action": "install_package",
  "reasoning": "<what this package does; why needed; dry-run result>",
  "parameters": {
    "package": "pennylane",
    "version": "0.36.0",
    "pip_flags": ["--quiet", "--no-cache-dir"],
    "fallback_if_denied": "qiskit-aer==0.14.0",
    "reason": "Required for hybrid quantum-classical circuit execution"
  },
  "next_step": "env_manager",
  "confidence": 0.99
}

When ALL packages are installed or handled, emit:
{
  "action": "install_package",
  "reasoning": "All packages provisioned. Environment ready.",
  "parameters": { "package": "__complete__", "version": "0.0.0", "pip_flags": [] },
  "next_step": "dataset_manager",
  "confidence": 1.0
}
```

### 13.3 Orchestrator Loop Logic

```python
# src/agents/env_manager_agent.py
async def env_manager_agent_node(state: ResearchState) -> ResearchState:
    state["phase"] = "env_manager"
    pending = [
        p for p in state["required_packages"]
        if p not in state["installed_packages"]
        and p not in [d["denied_item"] for d in state["denied_actions"]]
    ]

    for package_spec in pending:
        action = await ask_llm_for_install_action(state, package_spec)

        if action["parameters"]["package"] == "__complete__":
            break

        # Check if denied
        if is_denied(action, state):
            fallback = action["parameters"].get("fallback_if_denied")
            if fallback:
                state = await offer_fallback(state, fallback)
            continue

        # Execute install
        result = await execute_pip_install(
            action["parameters"]["package"],
            action["parameters"]["version"],
            action["parameters"]["pip_flags"]
        )

        if result.returncode == 0:
            state["installed_packages"].append(package_spec)
        else:
            state["errors"].append({
                "category": "InstallError",
                "message": result.stderr,
                "file_path": "env_manager",
                "line_number": 0,
                "traceback": result.stderr,
                "timestamp": time.time()
            })

    return state
```

---

## 14. Agent 4 — Dataset Acquisition Agent

### 14.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `dataset_manager` |
| **Goal** | Acquire the dataset from the specified source, validate its structure, and produce a `data_report` |
| **Input** | `dataset_source`, `kaggle_dataset_id`, `project_path` |
| **Output actions** | `run_python` (download/generate), `write_code` (validation script) |
| **Allowed actions** | `write_code`, `run_python`, `ask_user` |

### 14.2 Production System Prompt

```
SYSTEM:
You are the Dataset Acquisition Agent for the AI + Quantum Research Platform.
Your ONLY goal is to acquire, validate, and report on the dataset for the experiment.

You operate as a strict JSON machine. ONLY output valid JSON.

CURRENT STATE:
{state_json}

ACQUISITION STRATEGY:

CASE dataset_source == "kaggle":
  1. Check that /home/user/.kaggle/kaggle.json exists.
     If missing → emit ask_user for kaggle credentials.
  2. Write a Python script: /project/data/download_kaggle.py
     Content: kaggle datasets download -d {kaggle_dataset_id} -p /project/data/raw/ --unzip
  3. Run the script via run_python.
  4. After success, run validation script.

CASE dataset_source == "sklearn":
  1. Write a Python script: /project/data/load_sklearn.py
     Map user's algorithm_class to correct sklearn.datasets loader:
       classification → load_iris / load_breast_cancer / make_classification
       regression → load_boston / make_regression
       clustering → make_blobs
  2. Script saves to /project/data/raw/dataset.csv
  3. Run validation.

CASE dataset_source == "synthetic":
  1. Write a Python script: /project/data/generate_synthetic.py
     Use sklearn.datasets.make_classification or make_regression.
     Parameters: n_samples=1000, n_features=from clarifications, random_state=seed
  2. Run validation.

CASE dataset_source == "upload":
  1. Check state.uploaded_file_path — if empty, emit ask_user.
  2. Copy file to /project/data/raw/.
  3. Run validation.

VALIDATION SCRIPT (always run after acquisition):
  The script must output a JSON line: DATA_REPORT: {...}
  Fields: shape, columns, dtypes, null_counts, class_distribution (if clf), sample_rows

OUTPUT FORMAT (strict):
{
  "action": "write_code",
  "reasoning": "<why this acquisition strategy was chosen>",
  "parameters": {
    "file_path": "/workspace/projects/{id}/data/acquire_data.py",
    "language": "python",
    "content": "<complete Python script — no truncation>",
    "depends_on": [],
    "test_command": "python -m py_compile /workspace/projects/{id}/data/acquire_data.py",
    "encoding": "utf-8"
  },
  "next_step": "dataset_manager",
  "confidence": 0.95
}

CRITICAL:
- Write COMPLETE scripts — never use placeholder comments like "# add code here".
- All file paths must be absolute and under project_path.
- The data_report JSON must be parseable by the Evaluator Agent.
```

### 14.3 Auto-Generated Validation Script Template

```python
# Auto-generated by Dataset Acquisition Agent
# /workspace/projects/{id}/data/validate_data.py

import pandas as pd
import json
import sys
import os

DATA_PATH = "/workspace/projects/{id}/data/raw/"

def validate_dataset():
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    if not csv_files:
        print("DATA_ERROR: No CSV files found in data/raw/")
        sys.exit(1)

    df = pd.read_csv(os.path.join(DATA_PATH, csv_files[0]))

    report = {
        "filename":    csv_files[0],
        "shape":       list(df.shape),
        "columns":     list(df.columns),
        "dtypes":      df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct":    (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "memory_mb":   round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    # Class distribution if target column exists
    if "target" in df.columns or "label" in df.columns:
        col = "target" if "target" in df.columns else "label"
        report["class_distribution"] = df[col].value_counts().to_dict()

    # Save report
    with open("/workspace/projects/{id}/data/data_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print parseable line for state extraction
    print(f"DATA_REPORT: {json.dumps(report)}")
    print("DATA_VALID: true")

if __name__ == "__main__":
    validate_dataset()
```

---

## 15. Agent 5 — Code Generation Agent

### 15.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `code_generator` |
| **Goal** | Write all project Python files — complete, runnable, production-quality — then route to Quantum Gate or Scheduler |
| **Input** | `research_plan`, `data_report`, `framework`, `requires_quantum` |
| **Output actions** | `write_code` (multiple), `delegate_quantum_code` (if quantum) |
| **Allowed actions** | `write_code`, `delegate_quantum_code` |

### 15.2 Production System Prompt

```
SYSTEM:
You are the Code Generation Agent for the AI + Quantum Research Platform.
Your ONLY goal is to write complete, runnable, production-quality Python code
for every file in the research project.

You operate as a strict JSON machine. ONLY output valid JSON.
You emit ONE write_code action per LLM call. The orchestrator loops you until
all files are written.

CURRENT STATE:
{state_json}

FILES YOU MUST GENERATE (in this exact order):
1. config.py          — all constants, paths, hyperparameters, seeds
2. src/utils.py       — logging setup, helper functions
3. src/preprocessing.py — data loading, cleaning, encoding, splitting
4. src/model.py       — model class definition (classical or hybrid)
5. src/train.py       — training loop with metric tracking
6. src/evaluate.py    — evaluation, metric computation, plot generation
7. main.py            — orchestrates all modules; entry point

IF requires_quantum == true:
  After config.py and utils.py, emit delegate_quantum_code.
  Wait for quantum_circuit_code to be set in state.
  Then write model.py importing from quantum_circuit.py.

CODE QUALITY RULES:
1. Every file must be self-contained and importable.
2. Use absolute imports only.
3. Every function must have a docstring.
4. Use type hints throughout.
5. All magic numbers go to config.py — never hardcode in logic files.
6. Logging: use structlog, not print().
7. All file paths: read from config.py, never hardcode.
8. Random seed: set in BOTH numpy.random.seed() AND torch.manual_seed() AND random.seed().
9. Model checkpoint: save to outputs/model_checkpoint/ after training.
10. Metrics: save to outputs/metrics.json in the evaluate.py script.
11. Plots: save to outputs/plots/ as PNG files.

config.py MUST CONTAIN:
  PROJECT_ROOT, DATA_RAW_PATH, DATA_PROCESSED_PATH, MODEL_CHECKPOINT_PATH
  METRICS_PATH, PLOTS_PATH, LOG_PATH
  RANDOM_SEED, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE
  DEVICE (cpu/cuda), FRAMEWORK, TARGET_METRIC

WHEN ALL FILES ARE WRITTEN:
{
  "action": "write_code",
  "reasoning": "All project files generated. Ready for scheduling.",
  "parameters": {
    "file_path": "__complete__",
    "language":  "python",
    "content":   "",
    "depends_on": [],
    "test_command": ""
  },
  "next_step": "job_scheduler",
  "confidence": 1.0
}

CRITICAL:
- NEVER write placeholder code. Every function must be fully implemented.
- NEVER skip a file. Write ALL 7 files (8 if quantum).
- NEVER import a package not in state.installed_packages.
- The quantum_circuit.py file is ONLY written by the Quantum LLM via delegation.
- You MAY write the model.py import stub for quantum_circuit.py.
```

### 15.3 Generated config.py Template

```python
# Auto-generated by Code Generation Agent
# config.py

import os
import torch

# ── Identity ──────────────────────────────────────────────────────────────────
EXPERIMENT_ID   = "{experiment_id}"
PROJECT_ROOT    = "/workspace/projects/{experiment_id}"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_PATH         = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_PATH   = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_checkpoint")
METRICS_PATH          = os.path.join(PROJECT_ROOT, "outputs", "metrics.json")
PLOTS_PATH            = os.path.join(PROJECT_ROOT, "outputs", "plots")
LOG_PATH              = os.path.join(PROJECT_ROOT, "logs", "run.log")
STATE_HISTORY_PATH    = os.path.join(PROJECT_ROOT, "logs", "state_history.json")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = {seed}

# ── Training ──────────────────────────────────────────────────────────────────
MAX_EPOCHS     = {max_epochs}
BATCH_SIZE     = {batch_size}
LEARNING_RATE  = {learning_rate}
TRAIN_SPLIT    = 0.8
VAL_SPLIT      = 0.1
TEST_SPLIT     = 0.1

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() and "{hardware_target}" == "cuda" else "cpu"

# ── Framework ─────────────────────────────────────────────────────────────────
FRAMEWORK      = "{framework}"
TARGET_METRIC  = "{target_metric}"

# ── Quantum (populated if requires_quantum=True) ──────────────────────────────
QUANTUM_FRAMEWORK  = "{quantum_framework}"
QUANTUM_BACKEND    = "{quantum_backend}"
QUBIT_COUNT        = {qubit_count}
CIRCUIT_LAYERS     = {circuit_layers}
```

---

## 16. Agent 6 — Quantum Delegation Gate

### 16.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `quantum_gate` |
| **Goal** | Validate quantum specs and route to Quantum LLM; integrate returned circuit code into the project |
| **Input** | `delegate_quantum_code` action parameters from Code Gen Agent |
| **Output** | `quantum_circuit.py` written to project |
| **LLM used** | Specialized Quantum LLM (NOT the master LLM) |

### 16.2 Quantum LLM System Prompt

```
SYSTEM:
You are a specialized Quantum Circuit Code Generator.
You generate ONLY Python quantum circuit code using the specified framework.
You NEVER output prose. You NEVER output explanations.
You output ONLY a complete, runnable Python file.

DELEGATION SPEC:
{delegation_spec_json}

GENERATION RULES:

IF framework == "pennylane":
  1. Import pennylane as qml
  2. Define device = qml.device(backend, wires=qubit_count)
  3. Implement data encoding using encoding strategy from spec
  4. Implement variational layers (parameterized rotations + entanglement)
  5. Define quantum node with @qml.qnode decorator
  6. Implement QuantumLayer class (torch.nn.Module if hybrid)
  7. forward() encodes input features and returns expectation values
  8. Include draw_circuit() helper for visualization

IF framework == "qiskit":
  1. Import qiskit, qiskit_aer
  2. Implement QuantumCircuit class
  3. Apply feature map (ZZFeatureMap or custom)
  4. Apply variational ansatz (RealAmplitudes or custom)
  5. Implement run_circuit() that returns measurement probabilities
  6. Wrap in QuantumLayer class

IF framework == "cirq":
  1. Import cirq
  2. Implement circuit with qubits, moments, gates
  3. Wrap in QuantumLayer class

REQUIRED IN OUTPUT FILE:
  - QuantumLayer class with forward(x) method
  - get_circuit_diagram() → returns ASCII string of circuit
  - QUBIT_COUNT, CIRCUIT_LAYERS, BACKEND constants
  - if __name__ == "__main__": test run

OUTPUT: Complete Python file content as a single string.
NO markdown. NO backticks. NO explanations.
```

### 16.3 Gate Node Implementation

```python
# src/agents/quantum_gate.py
import httpx
import json

QUANTUM_LLM_ENDPOINT = os.getenv("QUANTUM_LLM_ENDPOINT")
QUANTUM_LLM_API_KEY  = os.getenv("QUANTUM_LLM_API_KEY")

async def quantum_gate_node(state: ResearchState) -> ResearchState:
    state["phase"] = "quantum_gate"

    # Build delegation spec from state
    delegation_spec = {
        "framework":         state["quantum_framework"],
        "algorithm":         state["quantum_algorithm"],
        "qubit_count":       state["quantum_qubit_count"],
        "layers":            state["research_plan"].get("circuit_layers", 3),
        "dataset_info": {
            "n_features":    state["data_report"]["shape"][1] - 1,
            "n_classes":     len(state["data_report"].get("class_distribution", {3: 0})),
            "encoding":      state["research_plan"].get("encoding", "angle_encoding"),
        },
        "training_strategy": state["research_plan"].get("training_strategy", "hybrid"),
        "optimizer":         state["research_plan"].get("optimizer", "adam"),
        "backend":           state["quantum_backend"],
        "return_expectation":"PauliZ",
        "integration_point": "model.py::QuantumLayer.forward()",
    }

    # Call Quantum LLM
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            QUANTUM_LLM_ENDPOINT,
            headers={"Authorization": f"Bearer {QUANTUM_LLM_API_KEY}"},
            json={
                "system":           QUANTUM_LLM_SYSTEM_PROMPT,
                "delegation_spec":  delegation_spec,
                "max_tokens":       4096,
            }
        )

    circuit_code = response.json()["generated_code"]

    # Validate: must contain QuantumLayer class
    if "class QuantumLayer" not in circuit_code:
        raise ValueError("Quantum LLM did not return QuantumLayer class")

    # Write quantum_circuit.py
    circuit_path = f"{state['project_path']}/src/quantum_circuit.py"
    with open(circuit_path, "w", encoding="utf-8") as f:
        f.write(circuit_code)

    state["quantum_circuit_code"] = circuit_code
    state["created_files"].append(circuit_path)
    state["llm_calls_count"] += 1

    return state
```

---

## 17. Agent 7 — Job Scheduler Agent

### 17.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `job_scheduler` |
| **Goal** | Resolve file execution order from dependency graph; emit `run_python` actions in correct sequence |
| **Input** | `created_files`, `research_plan`, `project_path` |
| **Output action** | `run_python` |
| **Allowed actions** | `run_python` |

### 17.2 Production System Prompt

```
SYSTEM:
You are the Job Scheduler Agent for the AI + Quantum Research Platform.
Your ONLY goal is to determine the correct execution order for all project
Python files and emit run_python actions in that order.

You operate as a strict JSON machine. ONLY output valid JSON.
You emit ONE run_python action per call. The orchestrator loops you.

CURRENT STATE:
{state_json}

DEPENDENCY RESOLUTION RULES:
1. config.py has NO dependencies. Always first (syntax check only, don't run).
2. src/utils.py depends on config.py.
3. data/acquire_data.py depends on config.py.
4. data/validate_data.py depends on acquire_data.py.
5. src/preprocessing.py depends on validate_data.py + config.py.
6. src/quantum_circuit.py (if exists) depends on preprocessing.py.
7. src/model.py depends on quantum_circuit.py (if exists) OR config.py.
8. src/train.py depends on model.py + preprocessing.py.
9. src/evaluate.py depends on train.py.
10. main.py is the FINAL script — depends on all others.

EXECUTION RULES:
- Syntax check config.py: python -m py_compile config.py
- Run data scripts before any ML scripts.
- Run preprocessing before model or training.
- Set PYTHONPATH to project root in env_vars.
- Set RANDOM_SEED in env_vars.
- Timeout: data scripts 300s, training scripts 3600s, eval scripts 600s.
- LOG every script's output to logs/{script_name}.log.

OUTPUT FORMAT:
{
  "action": "run_python",
  "reasoning": "<which script, why now, what it produces>",
  "parameters": {
    "script_path":        "/workspace/projects/{id}/src/train.py",
    "args":               ["--config", "/workspace/projects/{id}/config.py"],
    "timeout_seconds":    3600,
    "capture_output":     true,
    "working_directory":  "/workspace/projects/{id}",
    "env_vars": {
      "PYTHONPATH":       "/workspace/projects/{id}",
      "RANDOM_SEED":      "42",
      "EXPERIMENT_ID":    "{id}"
    },
    "log_path":           "/workspace/projects/{id}/logs/train.log"
  },
  "next_step": "subprocess_runner",
  "confidence": 0.98
}
```

---

## 18. Agent 8 — Error Recovery Agent

### 18.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `error_recovery` |
| **Goal** | Diagnose runtime failures, produce the minimal targeted fix, and return the system to execution without repeating the same mistake |
| **Input** | `execution_logs[-1]`, `errors[-1]`, `retry_count`, `repair_history` |
| **Output actions** | `modify_file`, `install_package`, `ask_user`, `abort` |
| **Allowed actions** | `modify_file`, `install_package`, `ask_user`, `abort` |

### 18.2 Production System Prompt

```
SYSTEM:
You are the Error Recovery Agent for the AI + Quantum Research Platform.
Your ONLY goal is to diagnose the cause of a runtime failure and produce
a minimal, targeted fix that resolves it without breaking other functionality.

You operate as a strict JSON machine. ONLY output valid JSON.

CURRENT STATE:
{state_json}

LATEST FAILURE:
Script:     {failed_script}
Returncode: {returncode}
STDERR:     {stderr_last_5000_chars}
STDOUT:     {stdout_last_5000_chars}

RETRY STATUS:
retry_count:            {retry_count} / 5
last_error_category:    {last_error_category}
consecutive_same_error: {consecutive_same_error}

DIAGNOSIS DECISION TREE:

STEP 1 — Classify the error:
  - "ModuleNotFoundError: No module named 'X'" → category: "missing_module"
  - "ImportError: cannot import name 'X' from 'Y'" → category: "import_error"
  - "SyntaxError" → category: "syntax_error"
  - "AttributeError: 'X' object has no attribute 'Y'" → category: "api_change"
  - "RuntimeError: CUDA out of memory" → category: "gpu_memory"
  - "RuntimeError: Expected all tensors on same device" → category: "device_mismatch"
  - "ValueError: shapes X and Y not aligned" → category: "shape_mismatch"
  - "FileNotFoundError" → category: "file_not_found"
  - "TimeoutExpired" → category: "timeout"
  - "kaggle.rest.ApiException: 401" → category: "kaggle_auth"
  - "pennylane.DeviceError" → category: "quantum_backend_error"
  - other → category: "unknown"

STEP 2 — Select fix strategy:
  missing_module:       emit install_package for exact module name
  import_error:         emit modify_file to fix import statement
  syntax_error:         emit modify_file to fix the exact syntax error line
  api_change:           emit modify_file to use correct API (check error message)
  gpu_memory:           emit modify_file to add DEVICE="cpu" override in config.py
  device_mismatch:      emit modify_file to add .to(DEVICE) call
  shape_mismatch:       emit modify_file to add reshape/squeeze/unsqueeze
  file_not_found:       emit modify_file to fix path to use config.py constant
  timeout:              emit modify_file to reduce MAX_EPOCHS by 50%
  kaggle_auth:          emit ask_user for kaggle credentials
  quantum_backend_error: emit modify_file to switch backend to "default.qubit"
  unknown (first time): emit modify_file with best guess
  unknown (2nd time):   CHANGE STRATEGY (see below)

STRATEGY CHANGE (if consecutive_same_error >= 2):
  1. Switch ML framework: torch → sklearn
  2. Reduce model complexity: halve layers, halve qubits
  3. Switch quantum backend: ibmq → statevector_simulator
  4. Use synthetic data instead of real dataset
  5. If none work: emit abort

ABORT CONDITION:
  IF retry_count >= 5 OR unfixable error:
  {
    "action": "abort",
    "reasoning": "<exact reason; what was tried; why unrecoverable>",
    "parameters": {
      "error_summary":    "<1-2 sentences>",
      "attempted_fixes":  ["<fix1>", "<fix2>"],
      "partial_output":   "/workspace/projects/{id}/outputs/"
    },
    "next_step": "abort",
    "confidence": 1.0
  }

FIX OUTPUT FORMAT:
{
  "action": "modify_file",
  "reasoning": "<exact diagnosis; what line/pattern causes it; why this fix resolves it>",
  "parameters": {
    "file_path":   "/workspace/projects/{id}/config.py",
    "find":        "DEVICE = 'cuda'",
    "replace":     "DEVICE = 'cpu'  # auto-fixed: CUDA unavailable on this machine",
    "reason":      "CUDA not available; forcing CPU to resolve device error",
    "backup":      true
  },
  "next_step": "subprocess_runner",
  "confidence": 0.88
}

CRITICAL RULES:
- NEVER repeat a fix that appears in repair_history[].
- NEVER guess randomly — base every fix on the actual error message.
- The find string MUST appear exactly once in the target file.
- The replace string MUST fix the actual error.
- Always set backup=true to preserve original.
```

### 18.3 Error Category → Fix Matrix

```python
ERROR_FIX_MATRIX = {
    "missing_module": {
        "action":   "install_package",
        "strategy": "install exact module from error message",
        "example":  "ModuleNotFoundError: No module named 'pennylane' → install pennylane==0.36.0"
    },
    "gpu_memory": {
        "action":   "modify_file",
        "file":     "config.py",
        "find":     "DEVICE = \"cuda\"",
        "replace":  "DEVICE = \"cpu\"  # auto-fixed: GPU OOM"
    },
    "shape_mismatch": {
        "action":   "modify_file",
        "strategy": "add .reshape() or .unsqueeze() at the exact operation that failed"
    },
    "timeout": {
        "action":   "modify_file",
        "file":     "config.py",
        "strategy": "halve MAX_EPOCHS value"
    },
    "quantum_backend_error": {
        "action":   "modify_file",
        "file":     "src/quantum_circuit.py",
        "find":     "ibmq_qasm",
        "replace":  "default.qubit  # auto-fixed: IBMQ unavailable"
    },
}
```

---

## 19. Agent 9 — Results Evaluator Agent

### 19.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `results_evaluator` |
| **Goal** | Parse all execution outputs, compute and structure evaluation metrics, generate visualizations, and produce `evaluation_summary.json` |
| **Input** | `execution_logs`, `project_path`, `target_metric` |
| **Output action** | `analyze_results` |
| **Allowed actions** | `analyze_results`, `run_python` |

### 19.2 Production System Prompt

```
SYSTEM:
You are the Results Evaluator Agent for the AI + Quantum Research Platform.
Your ONLY goal is to extract all metrics from experiment outputs, validate them,
generate plots, and produce a structured evaluation summary.

You operate as a strict JSON machine. ONLY output valid JSON.

CURRENT STATE:
{state_json}

EVALUATION STRATEGY:
1. Parse /project/outputs/metrics.json (written by evaluate.py).
2. If metrics.json missing or empty → parse stdout from execution_logs for metric lines.
3. Extract at minimum: {target_metric}, training_loss, duration_sec.
4. Generate plots using a run_python action first, then analyze_results.
5. Compare to baseline if research_plan includes baseline_metric.

METRIC EXTRACTION (from stdout fallback):
  Look for lines matching: METRIC: {key}={value}
  Look for lines matching: {"accuracy": X, "f1": Y, ...}

PLOT GENERATION (write + run a plot script):
  For classification: confusion_matrix, roc_curve, loss_curve
  For regression: residuals_plot, prediction_vs_actual
  For quantum: circuit_diagram (from quantum_circuit.get_circuit_diagram()), loss_curve
  Save all plots as PNG to /project/outputs/plots/

EVALUATION SUMMARY SCHEMA:
{
  "experiment_id":        "{id}",
  "status":               "success",
  "algorithm":            "{algorithm}",
  "framework":            "{framework}",
  "dataset":              "{dataset_source}",
  "training_duration_sec": 0.0,
  "metrics": {
    "primary": { "name": "{target_metric}", "value": 0.0 },
    "all": {
      "accuracy":   0.0,
      "f1_macro":   0.0,
      "roc_auc":    0.0,
      "train_loss": 0.0,
      "val_loss":   0.0
    }
  },
  "quantum_metrics": {
    "circuit_depth":     0,
    "gate_count":        0,
    "fidelity":          0.0
  },
  "hardware":      "{hardware_target}",
  "seed":          42,
  "reproducible":  true,
  "plots":         ["confusion_matrix.png", "roc_curve.png"],
  "baseline_comparison": null
}

OUTPUT FORMAT:
{
  "action": "analyze_results",
  "reasoning": "<what metrics were found; what plots were generated>",
  "parameters": {
    "metrics_file":     "/workspace/projects/{id}/outputs/metrics.json",
    "log_file":         "/workspace/projects/{id}/logs/train.log",
    "generate_plots":   true,
    "plot_types":       ["loss_curve", "confusion_matrix", "roc_curve"],
    "output_dir":       "/workspace/projects/{id}/outputs/plots/",
    "baseline_compare": false
  },
  "next_step": "doc_generator",
  "confidence": 0.95
}
```

---

## 20. Agent 10 — Documentation Generator Agent

### 20.1 Agent Identity

| Property | Value |
|----------|-------|
| **Node name** | `doc_generator` |
| **Goal** | Produce a complete, publication-ready research documentation file covering every phase of the experiment |
| **Input** | Full `ResearchState` snapshot |
| **Output action** | `generate_documentation` then `finish` |
| **Allowed actions** | `generate_documentation`, `finish` |

### 20.2 Production System Prompt

```
SYSTEM:
You are the Documentation Generator Agent for the AI + Quantum Research Platform.
Your ONLY goal is to produce a comprehensive, publication-ready research document
that covers every phase of the experiment from intent to results.

You operate as a strict JSON machine. ONLY output valid JSON.

CURRENT STATE (full snapshot):
{full_state_json}

DOCUMENTATION GENERATION RULES:
1. Write in academic/technical style — clear, precise, reproducible.
2. Include ALL sections listed below — none may be omitted.
3. Abstract: 2-3 sentences covering goal, method, primary result.
4. Code snippets: include key snippets from generated files (config, model, circuit).
5. Results table: format ALL metrics in a Markdown table.
6. Reproducibility: include exact pip install commands and run commands.
7. Quantum section: only if requires_quantum=true.
8. Error section: include if any errors were encountered and how they were resolved.
9. Conclusion: interpret the primary metric result; is it significant?
10. Next steps: 3 concrete follow-up experiments.

REQUIRED SECTIONS:
  1.  Abstract
  2.  Research Objective
  3.  Methodology
  4.  Environment Setup (Python version, packages, hardware)
  5.  Dataset Description (source, shape, preprocessing)
  6.  Model Architecture (layers, parameters, circuit if quantum)
  7.  Training Configuration (epochs, LR, batch, seed)
  8.  Quantum Circuit Description (only if quantum)
  9.  Experimental Results (metrics table, plots reference)
  10. Error Incidents & Resolutions (if any)
  11. Reproducibility Guide (step-by-step commands)
  12. Conclusion & Interpretation
  13. Future Work & Next Experiments
  14. Appendix: Full State Snapshot (JSON)

OUTPUT FORMAT:
{
  "action": "generate_documentation",
  "reasoning": "<what is being documented; key findings to highlight>",
  "parameters": {
    "output_path":      "/workspace/projects/{id}/docs/final_report.md",
    "include_sections": ["abstract", "objective", "methodology", ...all 14...],
    "format":           "markdown",
    "include_code":     true,
    "include_plots":    true
  },
  "next_step": "finish",
  "confidence": 0.99
}

After generate_documentation succeeds, emit finish:
{
  "action": "finish",
  "reasoning": "Experiment complete. Documentation generated. All artifacts saved.",
  "parameters": {
    "experiment_id":    "{id}",
    "status":           "success",
    "primary_metric":   "{target_metric}={value}",
    "documentation":    "/workspace/projects/{id}/docs/final_report.md",
    "artifacts_path":   "/workspace/projects/{id}/outputs/"
  },
  "next_step": "END",
  "confidence": 1.0
}
```
---

# PART III — COMPLETE API SPECIFICATION

---

## 21. API Architecture & Auth

### 21.1 Base Configuration

```
Base URL:       http://localhost:8000/api/v1
Content-Type:   application/json
Auth:           Bearer token (X-API-Key header)
Rate Limiting:  100 req/min per API key
Versioning:     URI versioning (/api/v1/)
Async:          All endpoints are async (FastAPI + asyncio)
```

### 21.2 Common Response Envelope

```json
{
  "success":    true,
  "data":       { },
  "error":      null,
  "request_id": "req_01j2k3l4m5n6",
  "timestamp":  "2024-03-15T10:30:00Z",
  "version":    "1.0.0"
}
```

### 21.3 Common Error Response

```json
{
  "success":    false,
  "data":       null,
  "error": {
    "code":     "EXPERIMENT_NOT_FOUND",
    "message":  "Experiment exp_20240315_a1b2c3 does not exist",
    "details":  { "experiment_id": "exp_20240315_a1b2c3" }
  },
  "request_id": "req_01j2k3l4m5n6",
  "timestamp":  "2024-03-15T10:30:00Z"
}
```

### 21.4 HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | OK — synchronous success |
| 201 | Created — new experiment started |
| 202 | Accepted — async task queued |
| 400 | Bad Request — validation error |
| 401 | Unauthorized — invalid API key |
| 403 | Forbidden — action not permitted |
| 404 | Not Found — experiment/resource missing |
| 409 | Conflict — experiment already in terminal state |
| 422 | Unprocessable — Pydantic validation failed |
| 429 | Too Many Requests — rate limit exceeded |
| 500 | Internal Server Error — unhandled exception |
| 503 | Service Unavailable — LLM API unreachable |

---

## 22. Experiment Lifecycle Endpoints

### `POST /api/v1/research/start`
**Start a new research experiment**

**Request Body:**
```json
{
  "prompt":       "Build a hybrid quantum-classical classifier using VQE to classify the Iris dataset with PennyLane",
  "priority":     "normal",
  "tags":         ["quantum", "classification", "vqe"],
  "webhook_url":  "https://my-server.com/callbacks/experiment",
  "config_overrides": {
    "max_epochs":     100,
    "random_seed":    123,
    "hardware_target": "cpu"
  }
}
```

**Field Constraints:**
| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| prompt | string | ✅ | 10–2000 chars |
| priority | enum | ❌ | "low"\|"normal"\|"high", default "normal" |
| tags | string[] | ❌ | max 10 tags, each ≤ 50 chars |
| webhook_url | string | ❌ | valid HTTPS URL |
| config_overrides | object | ❌ | overrides clarification defaults |

**Response `201 Created`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":    "exp_20240315_a1b2c3",
    "status":           "pending",
    "phase":            "clarifier",
    "created_at":       "2024-03-15T10:30:00Z",
    "estimated_duration_minutes": 15,
    "pending_questions": {
      "questions": [
        {
          "id":       "Q1",
          "text":     "Do you want .py scripts or .ipynb notebooks?",
          "type":     "choice",
          "options":  [".py", ".ipynb"],
          "default":  ".py",
          "required": true
        },
        {
          "id":       "Q3",
          "text":     "Does this require quantum circuits?",
          "type":     "boolean",
          "default":  true,
          "required": true
        },
        {
          "id":       "Q7",
          "text":     "What is the primary evaluation metric?",
          "type":     "choice",
          "options":  ["accuracy","f1_macro","rmse","roc_auc","fidelity"],
          "default":  "accuracy",
          "required": true
        }
      ]
    },
    "links": {
      "self":    "/api/v1/research/exp_20240315_a1b2c3",
      "answer":  "/api/v1/research/exp_20240315_a1b2c3/answer",
      "status":  "/api/v1/research/exp_20240315_a1b2c3/status",
      "logs":    "/api/v1/research/exp_20240315_a1b2c3/logs"
    }
  },
  "request_id": "req_abc123",
  "timestamp":  "2024-03-15T10:30:00Z"
}
```

---

### `POST /api/v1/research/{experiment_id}/answer`
**Submit answers to clarification questions**

**Path Params:** `experiment_id` — UUID format

**Request Body:**
```json
{
  "answers": {
    "Q1":  ".py",
    "Q3":  true,
    "Q4":  "pennylane",
    "Q5":  "sklearn",
    "Q7":  "accuracy",
    "Q8":  "cpu",
    "Q10": 42,
    "Q11": 50
  }
}
```

**Validation Rules:**
- `experiment_id` must be in status `waiting_user`
- All `required: true` questions must be answered
- Answer types must match question type (boolean/number/choice/text)
- Choice answers must be one of the provided `options`

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":   "exp_20240315_a1b2c3",
    "answers_received": 8,
    "status":          "running",
    "phase":           "planner",
    "message":         "Answers accepted. Research planning in progress.",
    "next_action":     "wait",
    "estimated_next_update_seconds": 30
  },
  "request_id": "req_def456",
  "timestamp":  "2024-03-15T10:31:00Z"
}
```

**Error `409 Conflict`:**
```json
{
  "success": false,
  "data":    null,
  "error": {
    "code":    "WRONG_STATUS",
    "message": "Experiment is not waiting for answers. Current status: running",
    "details": { "current_status": "running", "required_status": "waiting_user" }
  }
}
```

---

### `POST /api/v1/research/{experiment_id}/confirm`
**Confirm or deny a pending agent action (e.g., package install)**

**Request Body:**
```json
{
  "action_id":  "act_xyz789",
  "decision":   "deny",
  "reason":     "I don't want to install PyTorch — too large",
  "alternative_preference": "scikit-learn"
}
```

**Field Constraints:**
| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| action_id | string | ✅ | ID from pending_user_confirm.action_id |
| decision | enum | ✅ | "confirm" \| "deny" |
| reason | string | ❌ | ≤ 500 chars; helps LLM choose alternative |
| alternative_preference | string | ❌ | user's preferred alternative |

**Response `200 OK` (confirmed):**
```json
{
  "success": true,
  "data": {
    "experiment_id":  "exp_20240315_a1b2c3",
    "action_id":      "act_xyz789",
    "decision":       "confirm",
    "action":         "install_package",
    "package":        "torch==2.2.0",
    "status":         "running",
    "phase":          "env_manager",
    "message":        "Package installation confirmed. Installing torch==2.2.0..."
  }
}
```

**Response `200 OK` (denied):**
```json
{
  "success": true,
  "data": {
    "experiment_id":   "exp_20240315_a1b2c3",
    "action_id":       "act_xyz789",
    "decision":        "deny",
    "denied_package":  "torch==2.2.0",
    "alternative_offered": {
      "action_id":   "act_xyz790",
      "action":      "install_package",
      "package":     "scikit-learn==1.4.2",
      "reason":      "Lightweight alternative; no GPU required; no heavy dependencies",
      "note":        "Quantum ML capabilities will be adapted to use sklearn backend"
    },
    "status":          "waiting_user",
    "phase":           "env_manager",
    "message":         "Denial recorded. Alternative proposed — please confirm or deny."
  }
}
```

---

### `GET /api/v1/research/{experiment_id}`
**Get complete experiment details**

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":    "exp_20240315_a1b2c3",
    "status":           "running",
    "phase":            "code_generator",
    "created_at":       "2024-03-15T10:30:00Z",
    "updated_at":       "2024-03-15T10:35:00Z",
    "prompt":           "Build a hybrid quantum-classical VQE classifier...",
    "requires_quantum": true,
    "quantum_framework":"pennylane",
    "framework":        "pytorch",
    "dataset_source":   "sklearn",
    "hardware_target":  "cpu",
    "retry_count":      0,
    "phase_timings": {
      "clarifier":      12.3,
      "planner":        8.7,
      "env_manager":    45.2,
      "dataset_manager":6.1
    },
    "created_files": [
      "/workspace/projects/exp_20240315_a1b2c3/config.py",
      "/workspace/projects/exp_20240315_a1b2c3/src/utils.py"
    ],
    "installed_packages": ["pennylane==0.36.0", "torch==2.2.0"],
    "denied_actions":   [],
    "errors":           [],
    "metrics":          {},
    "links": {
      "logs":    "/api/v1/research/exp_20240315_a1b2c3/logs",
      "files":   "/api/v1/research/exp_20240315_a1b2c3/files",
      "results": "/api/v1/research/exp_20240315_a1b2c3/results",
      "report":  "/api/v1/research/exp_20240315_a1b2c3/report",
      "abort":   "/api/v1/research/exp_20240315_a1b2c3/abort"
    }
  }
}
```

---

## 23. Interaction Endpoints

### `GET /api/v1/research/{experiment_id}/status`
**Lightweight polling endpoint for status updates**

**Query Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| include_phase_timings | boolean | false | Include time per phase |
| include_last_action | boolean | false | Include last LLM action |

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":   "exp_20240315_a1b2c3",
    "status":          "running",
    "phase":           "subprocess_runner",
    "retry_count":     1,
    "current_script":  "src/train.py",
    "progress_pct":    65,
    "waiting_for_user":false,
    "pending_action":  null,
    "last_updated":    "2024-03-15T10:42:00Z",
    "elapsed_seconds": 720
  }
}
```

---

### `GET /api/v1/research/{experiment_id}/logs`
**Get execution logs for the experiment**

**Query Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| phase | string | all | Filter by phase name |
| level | string | all | "info"\|"error"\|"warning" |
| limit | integer | 100 | Max log entries |
| offset | integer | 0 | Pagination offset |
| since | datetime | - | ISO8601 filter |

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":  "exp_20240315_a1b2c3",
    "total_entries":  347,
    "returned":       100,
    "logs": [
      {
        "id":          "log_001",
        "phase":       "env_manager",
        "level":       "info",
        "timestamp":   "2024-03-15T10:33:15Z",
        "message":     "Installing pennylane==0.36.0",
        "details":     { "package": "pennylane", "version": "0.36.0" }
      },
      {
        "id":          "log_002",
        "phase":       "subprocess_runner",
        "level":       "error",
        "timestamp":   "2024-03-15T10:40:22Z",
        "message":     "Script failed: src/train.py — CUDA not available",
        "details": {
          "returncode": 1,
          "stderr":     "RuntimeError: CUDA error: no kernel image...",
          "duration_sec": 3.2
        }
      },
      {
        "id":          "log_003",
        "phase":       "error_recovery",
        "level":       "info",
        "timestamp":   "2024-03-15T10:40:25Z",
        "message":     "Auto-fix applied: DEVICE=cuda → DEVICE=cpu in config.py",
        "details": {
          "error_category": "gpu_unavailable",
          "retry_count":    1,
          "fix_applied":    "modify_file: config.py DEVICE='cpu'"
        }
      }
    ],
    "pagination": {
      "limit":  100,
      "offset": 0,
      "next":   "/api/v1/research/exp_20240315_a1b2c3/logs?offset=100"
    }
  }
}
```

---

### `GET /api/v1/research/{experiment_id}/files`
**List all generated project files**

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id": "exp_20240315_a1b2c3",
    "project_path":  "/workspace/projects/exp_20240315_a1b2c3",
    "files": [
      {
        "path":         "config.py",
        "absolute_path":"/workspace/projects/exp_20240315_a1b2c3/config.py",
        "size_bytes":   1204,
        "created_at":   "2024-03-15T10:36:00Z",
        "phase":        "code_generator",
        "is_quantum":   false
      },
      {
        "path":         "src/quantum_circuit.py",
        "absolute_path":"/workspace/projects/exp_20240315_a1b2c3/src/quantum_circuit.py",
        "size_bytes":   3847,
        "created_at":   "2024-03-15T10:37:30Z",
        "phase":        "quantum_gate",
        "is_quantum":   true
      }
    ],
    "total_files":    8,
    "total_size_bytes": 24680
  }
}
```

---

### `GET /api/v1/research/{experiment_id}/files/{file_path}`
**Download/view a specific generated file**

**Path Params:** `file_path` — URL-encoded relative path (e.g., `src%2Fmodel.py`)

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":  "exp_20240315_a1b2c3",
    "file_path":      "src/model.py",
    "content":        "# Auto-generated by Code Generation Agent\nimport torch...",
    "size_bytes":     2847,
    "language":       "python",
    "created_at":     "2024-03-15T10:37:00Z",
    "phase":          "code_generator"
  }
}
```

---

## 24. Monitoring & Observability Endpoints

### `GET /api/v1/research/{experiment_id}/results`
**Get experiment metrics and evaluation results**

**Response `200 OK` (experiment complete):**
```json
{
  "success": true,
  "data": {
    "experiment_id": "exp_20240315_a1b2c3",
    "status":        "success",
    "algorithm":     "VQE_hybrid_classifier",
    "framework":     "pennylane",
    "dataset":       "sklearn_iris",
    "training": {
      "epochs":        50,
      "final_loss":    0.142,
      "duration_sec":  312.4,
      "convergence_epoch": 38
    },
    "evaluation": {
      "primary_metric": {
        "name":  "accuracy",
        "value": 0.947,
        "interpretation": "94.7% accuracy — excellent classification performance"
      },
      "all_metrics": {
        "accuracy":         0.947,
        "f1_macro":         0.943,
        "f1_weighted":      0.946,
        "roc_auc":          0.991,
        "precision_macro":  0.945,
        "recall_macro":     0.941
      }
    },
    "quantum_metrics": {
      "qubit_count":    4,
      "circuit_depth":  9,
      "gate_count":     24,
      "circuit_layers": 3,
      "encoding":       "angle_encoding",
      "backend":        "default.qubit"
    },
    "hardware":      "cpu",
    "seed":          42,
    "reproducible":  true,
    "plots": [
      {
        "name":  "loss_curve.png",
        "path":  "/api/v1/research/exp_20240315_a1b2c3/plots/loss_curve.png",
        "type":  "loss_curve"
      },
      {
        "name":  "confusion_matrix.png",
        "path":  "/api/v1/research/exp_20240315_a1b2c3/plots/confusion_matrix.png",
        "type":  "confusion_matrix"
      }
    ]
  }
}
```

**Response `202 Accepted` (experiment still running):**
```json
{
  "success": true,
  "data": {
    "experiment_id": "exp_20240315_a1b2c3",
    "status":        "running",
    "phase":         "subprocess_runner",
    "message":       "Experiment still in progress. Results not yet available.",
    "progress_pct":  65
  }
}
```

---

### `GET /api/v1/research/{experiment_id}/report`
**Get the final Markdown research report**

**Query Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| format | string | markdown | "markdown" \| "json" |
| download | boolean | false | Trigger file download |

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id": "exp_20240315_a1b2c3",
    "report_path":   "/workspace/projects/exp_20240315_a1b2c3/docs/final_report.md",
    "generated_at":  "2024-03-15T10:55:00Z",
    "word_count":    3847,
    "sections": [
      "abstract", "objective", "methodology", "environment_setup",
      "dataset_description", "model_architecture", "training_configuration",
      "quantum_circuit_description", "experimental_results",
      "error_incidents", "reproducibility_guide", "conclusion",
      "future_work", "appendix"
    ],
    "content": "# Experiment Report: VQE Hybrid Classifier\n\n## Abstract\n..."
  }
}
```

---

### `GET /api/v1/research/{experiment_id}/plots/{plot_name}`
**Get a generated plot image**

**Response:** Binary PNG file with `Content-Type: image/png`

---

### `GET /api/v1/research`
**List all experiments (paginated)**

**Query Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| status | string | all | Filter by status |
| phase | string | all | Filter by phase |
| requires_quantum | boolean | - | Filter quantum experiments |
| limit | integer | 20 | Page size (max 100) |
| offset | integer | 0 | Pagination offset |
| sort | string | created_at:desc | Sort field:direction |
| tag | string | - | Filter by tag |

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiments": [
      {
        "experiment_id":    "exp_20240315_a1b2c3",
        "status":           "success",
        "phase":            "finished",
        "prompt_preview":   "Build a hybrid quantum-classical VQE classifier...",
        "requires_quantum": true,
        "framework":        "pennylane",
        "primary_metric":   { "name": "accuracy", "value": 0.947 },
        "created_at":       "2024-03-15T10:30:00Z",
        "duration_sec":     1542
      }
    ],
    "total":   47,
    "limit":   20,
    "offset":  0,
    "next":    "/api/v1/research?offset=20"
  }
}
```

---

## 25. Admin & Control Endpoints

### `DELETE /api/v1/research/{experiment_id}/abort`
**Force-abort a running experiment**

**Request Body:**
```json
{
  "reason":         "User requested cancellation",
  "save_partial":   true
}
```

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":  "exp_20240315_a1b2c3",
    "status":         "aborted",
    "aborted_at":     "2024-03-15T10:45:00Z",
    "aborted_phase":  "subprocess_runner",
    "partial_saved":  true,
    "partial_path":   "/workspace/projects/exp_20240315_a1b2c3/outputs/partial/",
    "error_report":   "/workspace/projects/exp_20240315_a1b2c3/docs/error_report.md"
  }
}
```

---

### `POST /api/v1/research/{experiment_id}/retry`
**Retry a failed or aborted experiment from last checkpoint**

**Request Body:**
```json
{
  "from_phase":     "error_recovery",
  "reset_retries":  true,
  "override_config": {
    "hardware_target": "cpu",
    "max_epochs":      25
  }
}
```

**Response `202 Accepted`:**
```json
{
  "success": true,
  "data": {
    "experiment_id":  "exp_20240315_a1b2c3",
    "status":         "running",
    "restarted_from": "error_recovery",
    "retry_count":    0,
    "message":        "Experiment resumed from error_recovery checkpoint"
  }
}
```

---

### `GET /api/v1/system/health`
**Platform health check**

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "status":     "healthy",
    "version":    "2.0.0",
    "timestamp":  "2024-03-15T10:30:00Z",
    "components": {
      "api":              { "status": "up", "latency_ms": 2 },
      "langgraph":        { "status": "up" },
      "master_llm":       { "status": "up", "provider": "anthropic", "model": "claude-3-5-sonnet" },
      "quantum_llm":      { "status": "up", "endpoint": "https://quantum-llm.internal" },
      "database":         { "status": "up", "type": "sqlite", "size_mb": 24.5 },
      "filesystem":       { "status": "up", "free_gb": 45.2 }
    },
    "active_experiments": 3,
    "total_experiments":  47
  }
}
```

---

### `GET /api/v1/system/metrics`
**Platform usage and performance metrics**

**Response `200 OK`:**
```json
{
  "success": true,
  "data": {
    "experiments": {
      "total":      47,
      "success":    38,
      "failed":     4,
      "aborted":    5,
      "running":    3,
      "success_rate": 0.809
    },
    "performance": {
      "avg_duration_sec":    1234,
      "median_duration_sec": 987,
      "avg_retry_count":     0.8,
      "error_self_heal_rate":0.82
    },
    "llm_usage": {
      "total_calls":   1847,
      "total_tokens":  2340000,
      "avg_calls_per_experiment": 39.3
    },
    "quantum_experiments": {
      "total":     12,
      "frameworks": { "pennylane": 8, "qiskit": 3, "cirq": 1 }
    }
  }
}
```

---

# PART IV — PROJECT IMPLEMENTATION

---

## 26. Complete Project File Structure

```
ai-quantum-research-platform/
│
├── pyproject.toml                    # Poetry dependency management
├── poetry.lock                       # Locked dependency versions
├── .env.example                      # Environment variable template
├── .env                              # Local secrets (gitignored)
├── .gitignore
├── README.md
├── Makefile                          # Dev commands: make run, make test, make lint
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/                          # FastAPI application
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI app factory
│   │   ├── router.py                 # Main router aggregator
│   │   ├── dependencies.py           # Auth, DB session injection
│   │   ├── middleware.py             # CORS, logging, rate limiting
│   │   │
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── research.py           # /research/* endpoints
│   │       ├── system.py             # /system/* endpoints
│   │       └── files.py              # /files/* endpoints
│   │
│   ├── state/
│   │   ├── __init__.py
│   │   └── research_state.py         # ResearchState TypedDict + enums
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py                # build_research_graph() function
│   │   ├── routers.py                # Conditional edge routing functions
│   │   └── runner.py                 # Graph execution manager
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py             # Abstract base: invoke_llm, parse_response
│   │   ├── clarifier_agent.py        # Agent 1
│   │   ├── planner_agent.py          # Agent 2
│   │   ├── env_manager_agent.py      # Agent 3
│   │   ├── dataset_agent.py          # Agent 4
│   │   ├── code_gen_agent.py         # Agent 5
│   │   ├── quantum_gate.py           # Agent 6 (Quantum LLM)
│   │   ├── job_scheduler_agent.py    # Agent 7
│   │   ├── error_recovery_agent.py   # Agent 8
│   │   ├── evaluator_agent.py        # Agent 9
│   │   └── doc_generator_agent.py    # Agent 10
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── clarifier.py              # Clarifier system prompt
│   │   ├── planner.py                # Planner system prompt
│   │   ├── env_manager.py
│   │   ├── dataset_agent.py
│   │   ├── code_gen.py
│   │   ├── quantum_llm.py            # Quantum LLM prompt
│   │   ├── job_scheduler.py
│   │   ├── error_recovery.py
│   │   ├── evaluator.py
│   │   └── doc_generator.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── master_llm.py             # Master LLM client (Anthropic/OpenAI)
│   │   ├── quantum_llm.py            # Quantum LLM HTTP client
│   │   └── response_parser.py        # JSON extraction + validation
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── action_validator.py       # JSON action schema validation
│   │   ├── subprocess_runner.py      # Isolated Python subprocess executor
│   │   ├── file_manager.py           # Safe file read/write (path whitelist)
│   │   ├── package_installer.py      # pip install with dry-run validation
│   │   └── state_compressor.py       # Compress state for LLM context
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py                 # SQLAlchemy ORM models
│   │   ├── database.py               # SQLite engine + session factory
│   │   ├── repository.py             # CRUD operations for experiments
│   │   └── migrations/               # Alembic migration scripts
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request_schemas.py        # Pydantic request models
│   │   └── response_schemas.py       # Pydantic response models
│   │
│   └── config/
│       ├── __init__.py
│       └── settings.py               # Pydantic Settings from .env
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_api/
│   │   ├── test_research_endpoints.py
│   │   └── test_system_endpoints.py
│   ├── test_agents/
│   │   ├── test_clarifier.py
│   │   ├── test_planner.py
│   │   ├── test_error_recovery.py
│   │   └── test_quantum_gate.py
│   ├── test_core/
│   │   ├── test_action_validator.py
│   │   ├── test_subprocess_runner.py
│   │   └── test_file_manager.py
│   └── test_graph/
│       ├── test_routing.py
│       └── test_full_workflow.py     # Integration tests
│
├── scripts/
│   ├── start_server.sh
│   ├── run_tests.sh
│   └── clean_experiments.py
│
└── workspace/
    ├── state.db                      # SQLite state + checkpoints
    └── projects/                     # Experiment outputs
        └── exp_{id}/                 # One folder per experiment
```

---

## 27. Core LangGraph Implementation Code

### 27.1 FastAPI App Factory

```python
# src/api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.router import api_router
from src.db.database import init_db
from src.config.settings import settings
import structlog

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Platform starting", version="2.0.0")
    await init_db()
    yield
    logger.info("Platform shutting down")

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI + Quantum Research Platform",
        version="2.0.0",
        description="Autonomous LLM-orchestrated quantum and AI research backend",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    return app

app = create_app()
```

### 27.2 Graph Runner

```python
# src/graph/runner.py
import uuid, time, asyncio
from src.graph.builder import build_research_graph
from src.db.repository import ExperimentRepository
from src.state.research_state import ResearchState, ExperimentStatus

compiled_graph = build_research_graph()

async def start_experiment(prompt: str, config_overrides: dict) -> str:
    experiment_id = f"exp_{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    project_path  = f"/workspace/projects/{experiment_id}"

    initial_state: ResearchState = {
        "experiment_id":          experiment_id,
        "project_path":           project_path,
        "phase":                  "clarifier",
        "status":                 ExperimentStatus.PENDING,
        "timestamp_start":        time.time(),
        "timestamp_end":          None,
        "user_prompt":            prompt,
        "clarifications":         {},
        "research_plan":          {},
        "requires_quantum":       False,
        "quantum_framework":      None,
        "quantum_algorithm":      None,
        "quantum_qubit_count":    None,
        "quantum_circuit_code":   None,
        "quantum_backend":        None,
        "framework":              "sklearn",
        "python_version":         "3.11",
        "required_packages":      [],
        "installed_packages":     [],
        "venv_path":              f"{project_path}/.venv",
        "output_format":          ".py",
        "dataset_source":         "sklearn",
        "dataset_path":           f"{project_path}/data/raw",
        "kaggle_dataset_id":      None,
        "data_report":            {},
        "created_files":          [],
        "execution_order":        [],
        "execution_logs":         [],
        "current_script":         None,
        "total_duration_sec":     None,
        "errors":                 [],
        "retry_count":            0,
        "last_error_category":    None,
        "consecutive_same_error": 0,
        "repair_history":         [],
        "denied_actions":         [],
        "pending_user_question":  None,
        "pending_user_confirm":   None,
        "metrics":                {},
        "plots_generated":        [],
        "evaluation_summary":     {},
        "documentation_path":     None,
        "report_sections":        [],
        "target_metric":          config_overrides.get("target_metric", "accuracy"),
        "hardware_target":        config_overrides.get("hardware_target", "cpu"),
        "random_seed":            config_overrides.get("random_seed", 42),
        "max_epochs":             config_overrides.get("max_epochs", 50),
        "batch_size":             config_overrides.get("batch_size", 32),
        "llm_calls_count":        0,
        "total_tokens_used":      0,
        "phase_timings":          {},
    }

    # Persist initial state
    await ExperimentRepository.create(initial_state)

    # Run graph in background task
    asyncio.create_task(
        run_graph_async(experiment_id, initial_state)
    )

    return experiment_id


async def run_graph_async(experiment_id: str, state: ResearchState):
    config = {"configurable": {"thread_id": experiment_id}}
    try:
        async for event in compiled_graph.astream(state, config):
            # Persist state after each node
            updated_state = list(event.values())[0]
            await ExperimentRepository.update(experiment_id, updated_state)
    except Exception as e:
        await ExperimentRepository.mark_failed(experiment_id, str(e))
```

### 27.3 Base Agent

```python
# src/agents/base_agent.py
import json, time
from abc import ABC, abstractmethod
from src.state.research_state import ResearchState
from src.llm.master_llm import invoke_master_llm
from src.core.action_validator import validate_action
from src.core.state_compressor import compress_state
import structlog

logger = structlog.get_logger()

class BaseAgent(ABC):
    MAX_RETRIES = 3       # LLM self-correction retries (separate from execution retries)

    @property
    @abstractmethod
    def phase_name(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt_template(self) -> str: ...

    async def invoke(self, state: ResearchState) -> ResearchState:
        t_start = time.time()
        state["phase"] = self.phase_name
        logger.info("Agent invoked", phase=self.phase_name, experiment=state["experiment_id"])

        compressed = compress_state(state)
        system_prompt = self.system_prompt_template.format(state_json=json.dumps(compressed, indent=2))

        action = await self._invoke_with_retry(system_prompt, state)
        state["llm_calls_count"] += 1

        state = await self.execute_action(action, state)

        elapsed = time.time() - t_start
        state["phase_timings"][self.phase_name] = elapsed
        return state

    async def _invoke_with_retry(self, system_prompt: str, state: ResearchState) -> dict:
        for attempt in range(self.MAX_RETRIES):
            try:
                raw = await invoke_master_llm(system_prompt)
                action = parse_json_response(raw)
                valid, error = validate_action(action, state)
                if valid:
                    return action
                logger.warning("Action validation failed", error=error, attempt=attempt)
                system_prompt += f"\n\nPREVIOUS OUTPUT FAILED VALIDATION: {error}\nFix and retry."
            except Exception as e:
                logger.error("LLM invocation failed", error=str(e), attempt=attempt)
                if attempt == self.MAX_RETRIES - 1:
                    raise
        raise RuntimeError(f"Agent {self.phase_name} failed after {self.MAX_RETRIES} LLM retries")

    @abstractmethod
    async def execute_action(self, action: dict, state: ResearchState) -> ResearchState: ...
```

### 27.4 Subprocess Runner Node

```python
# src/core/subprocess_runner.py
import subprocess, json, time, os
from src.state.research_state import ResearchState, ExecutionLog
import structlog

logger = structlog.get_logger()

BLOCKED_PATHS = ["/etc", "/usr", "/bin", "/sbin", "/root", "/home"]

async def subprocess_runner_node(state: ResearchState) -> ResearchState:
    """Execute the current script as an isolated subprocess."""
    if not state["execution_order"]:
        return state  # nothing to run

    script = state["execution_order"][state["execution_logs"].__len__()]
    state["current_script"] = script

    # Safety: path must be under project_path
    if not script.startswith(state["project_path"]):
        raise PermissionError(f"Blocked: script {script} not in project_path")
    for blocked in BLOCKED_PATHS:
        if script.startswith(blocked):
            raise PermissionError(f"Blocked: system path access {blocked}")

    env = {
        **os.environ,
        "PYTHONPATH":    state["project_path"],
        "RANDOM_SEED":   str(state["random_seed"]),
        "EXPERIMENT_ID": state["experiment_id"],
    }

    log_path = f"{state['project_path']}/logs/{os.path.basename(script)}.log"
    t_start  = time.time()

    try:
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=state["project_path"],
            env=env
        )
        elapsed = time.time() - t_start

        log: ExecutionLog = {
            "script_path":  script,
            "returncode":   result.returncode,
            "stdout":       result.stdout[-10000:],
            "stderr":       result.stderr[-5000:],
            "duration_sec": elapsed,
            "timestamp":    time.time()
        }

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        state["execution_logs"].append(log)

        if result.returncode != 0:
            state["errors"].append({
                "category":    classify_error(result.stderr),
                "message":     result.stderr[:1000],
                "file_path":   script,
                "line_number": extract_line_number(result.stderr),
                "traceback":   result.stderr,
                "timestamp":   time.time()
            })

    except subprocess.TimeoutExpired:
        state["execution_logs"].append({
            "script_path":  script,
            "returncode":   -1,
            "stdout":       "",
            "stderr":       "TimeoutExpired: script ran longer than 3600 seconds",
            "duration_sec": 3600.0,
            "timestamp":    time.time()
        })

    return state


def classify_error(stderr: str) -> str:
    for err_type in [
        "ModuleNotFoundError", "ImportError", "SyntaxError",
        "AttributeError", "ValueError", "RuntimeError",
        "FileNotFoundError", "TimeoutExpired", "MemoryError"
    ]:
        if err_type in stderr:
            return err_type
    if "CUDA" in stderr:  return "gpu_unavailable"
    if "shape" in stderr: return "shape_mismatch"
    if "kaggle" in stderr.lower(): return "kaggle_auth"
    return "unknown"
```

---

## 28. Database Schema

```sql
-- SQLAlchemy models map to these tables
-- workspace/state.db (SQLite)

CREATE TABLE experiments (
    id                  TEXT PRIMARY KEY,    -- exp_20240315_a1b2c3
    status              TEXT NOT NULL,       -- ExperimentStatus enum
    phase               TEXT NOT NULL,
    prompt              TEXT NOT NULL,
    requires_quantum    BOOLEAN DEFAULT FALSE,
    quantum_framework   TEXT,
    framework           TEXT,
    dataset_source      TEXT,
    hardware_target     TEXT DEFAULT 'cpu',
    target_metric       TEXT DEFAULT 'accuracy',
    random_seed         INTEGER DEFAULT 42,
    retry_count         INTEGER DEFAULT 0,
    llm_calls_count     INTEGER DEFAULT 0,
    total_tokens_used   INTEGER DEFAULT 0,
    project_path        TEXT,
    documentation_path  TEXT,
    state_json          TEXT,               -- Full ResearchState as JSON
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at        DATETIME
);

CREATE TABLE experiment_logs (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT REFERENCES experiments(id),
    phase           TEXT,
    level           TEXT,       -- info | warning | error
    message         TEXT,
    details_json    TEXT,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE experiment_metrics (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT REFERENCES experiments(id),
    metric_name     TEXT,
    metric_value    REAL,
    metric_type     TEXT,       -- training | evaluation | quantum
    recorded_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE denial_records (
    id                  TEXT PRIMARY KEY,
    experiment_id       TEXT REFERENCES experiments(id),
    action              TEXT,
    denied_item         TEXT,
    reason              TEXT,
    alternative_offered TEXT,
    alternative_accepted BOOLEAN DEFAULT FALSE,
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE repair_history (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT REFERENCES experiments(id),
    attempt         INTEGER,
    error_category  TEXT,
    fix_description TEXT,
    file_changed    TEXT,
    success         BOOLEAN,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_experiments_status    ON experiments(status);
CREATE INDEX idx_experiments_phase     ON experiments(phase);
CREATE INDEX idx_logs_experiment       ON experiment_logs(experiment_id);
CREATE INDEX idx_logs_phase            ON experiment_logs(phase);
CREATE INDEX idx_metrics_experiment    ON experiment_metrics(experiment_id);
```

---

## 29. Configuration & Environment

### 29.1 Settings (Pydantic)

```python
# src/config/settings.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # ── API ───────────────────────────────────────────────────────────────────
    APP_ENV:             str   = "production"
    API_KEY:             str                    # required
    ALLOWED_ORIGINS:     List[str] = ["http://localhost:3000"]
    LOG_LEVEL:           str   = "INFO"

    # ── Master LLM ───────────────────────────────────────────────────────────
    MASTER_LLM_PROVIDER: str   = "anthropic"    # anthropic | openai | google
    MASTER_LLM_MODEL:    str   = "claude-3-5-sonnet-20241022"
    MASTER_LLM_API_KEY:  str                    # required
    MASTER_LLM_MAX_TOKENS: int = 4096
    MASTER_LLM_TEMPERATURE: float = 0.1         # low temp for JSON reliability

    # ── Quantum LLM ──────────────────────────────────────────────────────────
    QUANTUM_LLM_ENDPOINT: str                   # required if quantum supported
    QUANTUM_LLM_API_KEY:  str = ""
    QUANTUM_LLM_TIMEOUT:  int = 120

    # ── Platform Limits ───────────────────────────────────────────────────────
    MAX_RETRY_COUNT:        int   = 5
    MAX_LLM_RETRIES:        int   = 3
    SUBPROCESS_TIMEOUT:     int   = 3600
    MAX_CONCURRENT_EXPS:    int   = 10
    MAX_STATE_SIZE_KB:      int   = 500
    STDOUT_CAP_CHARS:       int   = 10000
    STDERR_CAP_CHARS:       int   = 5000

    # ── Paths ─────────────────────────────────────────────────────────────────
    PROJECT_ROOT:       str   = "/workspace/projects"
    STATE_DB_PATH:      str   = "/workspace/state.db"
    KAGGLE_CONFIG_DIR:  str   = "/home/user/.kaggle"

    # ── Features ──────────────────────────────────────────────────────────────
    QUANTUM_ENABLED:    bool  = True
    KAGGLE_ENABLED:     bool  = True
    GPU_ALLOWED:        bool  = False           # set True if GPU available
    WEBHOOK_ENABLED:    bool  = True

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 29.2 .env.example

```env
# ── Required ─────────────────────────────────────────────────────────────────
API_KEY=your-platform-api-key-here
MASTER_LLM_API_KEY=sk-ant-...
QUANTUM_LLM_ENDPOINT=https://quantum-llm.your-domain.com/v1/generate
QUANTUM_LLM_API_KEY=qk-...

# ── Master LLM ────────────────────────────────────────────────────────────────
MASTER_LLM_PROVIDER=anthropic
MASTER_LLM_MODEL=claude-3-5-sonnet-20241022
MASTER_LLM_MAX_TOKENS=4096
MASTER_LLM_TEMPERATURE=0.1

# ── Platform ──────────────────────────────────────────────────────────────────
APP_ENV=production
LOG_LEVEL=INFO
MAX_RETRY_COUNT=5
SUBPROCESS_TIMEOUT=3600
MAX_CONCURRENT_EXPS=10

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/workspace/projects
STATE_DB_PATH=/workspace/state.db
KAGGLE_CONFIG_DIR=/home/user/.kaggle

# ── Features ──────────────────────────────────────────────────────────────────
QUANTUM_ENABLED=true
KAGGLE_ENABLED=true
GPU_ALLOWED=false
WEBHOOK_ENABLED=true
```

---

## 30. Security, Deployment & Operations

### 30.1 Security Guardrails (Implementation)

```python
# src/core/security.py

PATH_WHITELIST_PREFIX = settings.PROJECT_ROOT

def validate_path(path: str, project_path: str) -> bool:
    """Ensure path is within project directory only."""
    resolved = os.path.realpath(path)
    return resolved.startswith(os.path.realpath(project_path))

def sanitize_subprocess_args(args: list) -> list:
    """Ensure no shell injection in subprocess arguments."""
    # Only allow alphanumeric, dash, underscore, dot, slash in args
    import re
    safe = []
    for arg in args:
        if not re.match(r'^[a-zA-Z0-9/_\-\.=@]+$', arg):
            raise ValueError(f"Unsafe subprocess argument: {arg}")
        safe.append(arg)
    return safe
```

### 30.2 Deployment (Docker)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install poetry==1.8.0
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN mkdir -p /workspace/projects

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  platform:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./workspace:/workspace
      - ~/.kaggle:/home/user/.kaggle:ro
    env_file: .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/system/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 30.3 Makefile

```makefile
.PHONY: run test lint clean install

run:
	poetry run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest tests/ -v --asyncio-mode=auto --cov=src --cov-report=html

lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/

install:
	poetry install

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

docker-build:
	docker build -t ai-quantum-platform:latest .

docker-run:
	docker-compose up -d
```

### 30.4 Operational Runbook

```
STARTING THE PLATFORM:
  1. cp .env.example .env && fill in all required values
  2. make install
  3. make run
  4. Verify: curl http://localhost:8000/api/v1/system/health

RUNNING AN EXPERIMENT (curl example):
  # Step 1: Start
  curl -X POST http://localhost:8000/api/v1/research/start \
    -H "X-API-Key: your-key" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Build VQE classifier for Iris with PennyLane"}'

  # Step 2: Answer questions (use experiment_id from step 1)
  curl -X POST http://localhost:8000/api/v1/research/{id}/answer \
    -H "X-API-Key: your-key" \
    -d '{"answers": {"Q1": ".py", "Q3": true, "Q4": "pennylane", "Q7": "accuracy"}}'

  # Step 3: Poll status
  curl http://localhost:8000/api/v1/research/{id}/status -H "X-API-Key: your-key"

  # Step 4: Get results (when status=success)
  curl http://localhost:8000/api/v1/research/{id}/results -H "X-API-Key: your-key"

  # Step 5: Download report
  curl http://localhost:8000/api/v1/research/{id}/report -H "X-API-Key: your-key"

MONITORING:
  - Logs:    tail -f workspace/projects/{id}/logs/run.log
  - DB:      sqlite3 workspace/state.db "SELECT id,status,phase FROM experiments"
  - Health:  curl http://localhost:8000/api/v1/system/health
  - Metrics: curl http://localhost:8000/api/v1/system/metrics

COMMON ISSUES:
  LLM API unreachable     → Check MASTER_LLM_API_KEY in .env
  Quantum LLM fails       → Check QUANTUM_LLM_ENDPOINT; set QUANTUM_ENABLED=false
  Kaggle download fails   → Check ~/.kaggle/kaggle.json exists and has valid token
  CUDA errors             → Set GPU_ALLOWED=false; error recovery will auto-fix
  Infinite retry loop     → Should not occur (retry_count ≤ 5 enforced); check logs
```

---

*⚛ AI + Quantum Research Platform — Production Architecture v2.0.0*
*Backend-Only · LLM-Orchestrated · LangGraph · 10-Agent Pipeline*
