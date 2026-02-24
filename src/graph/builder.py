from __future__ import annotations

from typing import Any

from src.agents.clarifier_agent import clarifier_agent_node
from src.agents.code_gen_agent import code_gen_agent_node
from src.agents.dataset_agent import dataset_agent_node
from src.agents.doc_generator_agent import doc_generator_agent_node
from src.agents.env_manager_agent import env_manager_agent_node
from src.agents.error_recovery_agent import error_recovery_agent_node
from src.agents.evaluator_agent import evaluator_agent_node
from src.agents.job_scheduler_agent import job_scheduler_agent_node
from src.agents.planner_agent import planner_agent_node
from src.agents.quantum_gate import quantum_gate_node
from src.core.subprocess_runner import subprocess_runner_node
from src.graph.routers import route_quantum_or_direct, route_retry_or_abort, route_success_or_error

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    END = "__end__"
    StateGraph = None  # type: ignore


def build_research_graph() -> Any:
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(dict)
    graph.add_node("clarifier", clarifier_agent_node)
    graph.add_node("planner", planner_agent_node)
    graph.add_node("env_manager", env_manager_agent_node)
    graph.add_node("dataset_manager", dataset_agent_node)
    graph.add_node("code_generator", code_gen_agent_node)
    graph.add_node("quantum_gate", quantum_gate_node)
    graph.add_node("job_scheduler", job_scheduler_agent_node)
    graph.add_node("subprocess_runner", subprocess_runner_node)
    graph.add_node("error_recovery", error_recovery_agent_node)
    graph.add_node("results_evaluator", evaluator_agent_node)
    graph.add_node("doc_generator", doc_generator_agent_node)

    graph.set_entry_point("clarifier")
    graph.add_edge("clarifier", "planner")
    graph.add_edge("planner", "env_manager")
    graph.add_edge("env_manager", "dataset_manager")
    graph.add_edge("dataset_manager", "code_generator")
    graph.add_conditional_edges(
        "code_generator",
        route_quantum_or_direct,
        {"quantum": "quantum_gate", "no_quantum": "job_scheduler"},
    )
    graph.add_edge("quantum_gate", "job_scheduler")
    graph.add_edge("job_scheduler", "subprocess_runner")
    graph.add_conditional_edges(
        "subprocess_runner",
        route_success_or_error,
        {"success": "results_evaluator", "error": "error_recovery", "abort": END},
    )
    graph.add_conditional_edges(
        "error_recovery",
        route_retry_or_abort,
        {"retry": "subprocess_runner", "abort": END},
    )
    graph.add_edge("results_evaluator", "doc_generator")
    graph.add_edge("doc_generator", END)
    return graph.compile()

