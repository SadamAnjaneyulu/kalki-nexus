"""
Kalki Nexus - LangGraph Orchestration

Assembles the multi-agent graph:

    START -> supervisor -> (parallel fan-out to N specialist agents)
          -> aggregator (fan-in / merge node)
          -> error_node -> {supervisor (bounded retry) | fallback_agent}
          -> human_approval_node (interrupt(), only if an AgentResult flagged it)
          -> END

Every specialist agent node is auto-discovered from agents/ via
core.registry.discover_agents() - there is no hardcoded per-agent import
list to maintain here.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict


from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.aggregator import aggregator_node
from agents.fallback_agent import FallbackAgent
from agents.supervisor import SupervisorAgent
from core.base_agent import AgentResult
from core.observability import get_logger
from core.registry import discover_agents

logger = get_logger("kalki.graph")

MAX_GRAPH_RETRIES = 1


def merge_agent_results(
    left: Optional[Dict[str, AgentResult]], right: Optional[Dict[str, AgentResult]]
) -> Dict[str, AgentResult]:
    """Reducer: union two agent_results dicts, letting the newer write win per key.

    This is what allows a parallel fan-out (e.g. Python + Docker running at
    once) to both land in shared state without clobbering each other.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def merge_metadata(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer: shallow-merge state metadata, one level deep (so nested dicts like
    `timings` accumulate per-agent entries instead of one branch overwriting another)."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


class KalkiState(TypedDict):
    """Shared state passed between every node in the Kalki Nexus graph."""

    messages: Annotated[List[Any], add_messages]
    user_input: str
    discord_channel: Optional[str]
    attached_files: List[str]
    requested_tools: List[str]
    route: List[str]
    route_reasoning: Optional[str]
    route_confidence: Optional[float]
    agent_results: Annotated[Dict[str, AgentResult], merge_agent_results]
    final_answer: Optional[str]
    sources: List[str]
    error: Optional[Dict[str, Any]]
    retry_count: int
    metadata: Annotated[Dict[str, Any], merge_metadata]


def route_from_supervisor(state: KalkiState) -> List[str]:
    """Fan out to every agent selected by the Supervisor's routing decision.

    Returning a list of node names is what triggers LangGraph's parallel
    fan-out: compound requests (e.g. "Python + Docker") run more than one
    specialist agent for a single request, in the same superstep.
    """
    return state["route"] or [END]


def route_after_aggregator(state: KalkiState) -> str:
    """After merging every agent's result: escalate errors, gate on human
    approval, or finish."""
    if state.get("error"):
        return "error_node"
    if any(result.metadata.get("requires_approval") for result in (state.get("agent_results") or {}).values()):
        return "human_approval_node"
    return END


def route_after_error(state: KalkiState) -> str:
    """Bounded retry: re-run the Supervisor once for a retryable error, otherwise hand off."""
    error = state.get("error") or {}
    retry_count = state.get("retry_count", 0)
    if error.get("retryable") and retry_count < MAX_GRAPH_RETRIES:
        return "supervisor"
    return "fallback_agent"


async def error_node(state: KalkiState) -> Dict[str, Any]:
    """Log the current error and bump the graph-level retry counter.

    Note this is distinct from (and layered on top of) the per-call retries
    every BaseAgent already performs internally via core.resilience -
    reaching this node means an agent's own retries were already exhausted.
    """
    error = state.get("error") or {}
    logger.warning("graph error_node: %s", error.get("message", "unknown error"))
    return {"retry_count": state.get("retry_count", 0) + 1}


async def human_approval_node(state: KalkiState) -> Dict[str, Any]:
    """Pause the run for human sign-off on any AgentResult flagged
    `metadata["requires_approval"]` (e.g. a destructive GitHub or terminal
    action). Requires the graph to be compiled with a checkpointer."""
    try:
        from langgraph.types import interrupt
    except ImportError:  # pragma: no cover - older langgraph without interrupt()
        logger.warning("langgraph.types.interrupt unavailable; auto-approving.")
        return {"metadata": {"human_approval": "auto-approved (interrupt() unavailable)"}}

    pending = {
        name: result.metadata.get("approval_reason", "no reason given")
        for name, result in (state.get("agent_results") or {}).items()
        if result.metadata.get("requires_approval")
    }
    decision = interrupt({"message": "Human approval required before finalizing.", "pending": pending})
    approved = decision.get("approved", False) if isinstance(decision, dict) else bool(decision)

    if not approved:
        return {
            "final_answer": "The pending action was not approved by a human reviewer. No changes were made.",
            "metadata": {"human_approval": "denied"},
        }
    return {"metadata": {"human_approval": "approved"}}


def build_graph() -> StateGraph:
    """Assemble the Kalki Nexus LangGraph state graph from auto-discovered agents."""
    graph = StateGraph(KalkiState)

    specialist_classes = {
        name: cls for name, cls in discover_agents().items() if name != "fallback_agent"
    }
    specialists = {name: cls() for name, cls in specialist_classes.items()}

    graph.add_node("supervisor", SupervisorAgent())
    for name, instance in specialists.items():
        graph.add_node(name, instance)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("error_node", error_node)
    graph.add_node("fallback_agent", FallbackAgent())
    graph.add_node("human_approval_node", human_approval_node)

    graph.add_edge(START, "supervisor")

    path_map: Dict[str, str] = {name: name for name in specialists}
    path_map[END] = END
    graph.add_conditional_edges("supervisor", route_from_supervisor, path_map)

    for name in specialists:
        graph.add_edge(name, "aggregator")

    graph.add_conditional_edges(
        "aggregator",
        route_after_aggregator,
        {"error_node": "error_node", "human_approval_node": "human_approval_node", END: END},
    )
    graph.add_conditional_edges(
        "error_node",
        route_after_error,
        {"supervisor": "supervisor", "fallback_agent": "fallback_agent"},
    )
    graph.add_edge("fallback_agent", END)
    graph.add_edge("human_approval_node", END)

    return graph


# Process-wide singleton — compiled once, reused for every request.
# No LangGraph checkpointer: we use our own SQLite memory layer instead.
# MemorySaver caused 'threads can only be started once' crashes because
# it stores AgentResult (a Pydantic model) via msgpack and the deserialization
# triggered internal threading conflicts on every second Discord message.
_APP: Optional[Any] = None


def compiled_graph():
    """Return the compiled Kalki Nexus graph, built once per process."""
    global _APP
    if _APP is None:
        graph = build_graph()
        _APP = graph.compile(checkpointer=None)
    return _APP


async def invoke(
    user_input: str,
    discord_channel: Optional[str] = None,
    attached_files: Optional[List[str]] = None,
    requested_tools: Optional[List[str]] = None,
    thread_id: str = "default",
) -> KalkiState:
    """Run a single request through the graph, reusing the compiled singleton."""
    app = compiled_graph()
    initial_state: KalkiState = {
        "messages": [],
        "user_input": user_input,
        "discord_channel": discord_channel,
        "attached_files": attached_files or [],
        "requested_tools": requested_tools or [],
        "route": [],
        "route_reasoning": None,
        "route_confidence": None,
        "agent_results": {},
        "final_answer": None,
        "sources": [],
        "error": None,
        "retry_count": 0,
        "metadata": {},
    }
    return await app.ainvoke(initial_state)
