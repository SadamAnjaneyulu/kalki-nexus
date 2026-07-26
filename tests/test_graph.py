"""
Unit tests for Kalki Nexus Graph Assembly and Routing.
"""
import pytest
from graph import build_graph, route_from_supervisor, route_after_aggregator, route_after_error, merge_agent_results
from core.base_agent import AgentResult


def test_build_graph():
    graph = build_graph()
    assert graph is not None


def test_route_from_supervisor():
    state = {"route": ["python_agent", "quant_agent"]}
    routes = route_from_supervisor(state)
    assert routes == ["python_agent", "quant_agent"]


def test_route_after_aggregator():
    # Test error routing
    state_err = {"error": {"message": "fatal error"}}
    assert route_after_aggregator(state_err) == "error_node"

    # Test clean end
    state_clean = {"error": None, "agent_results": {}}
    assert route_after_aggregator(state_clean) == "__end__"


def test_route_after_error():
    # Retryable under max retries
    state_retry = {"error": {"retryable": True}, "retry_count": 0}
    assert route_after_error(state_retry) == "supervisor"

    # Exhausted retries -> fallback
    state_fallback = {"error": {"retryable": True}, "retry_count": 2}
    assert route_after_error(state_fallback) == "fallback_agent"


def test_merge_agent_results():
    res1 = AgentResult(agent="agent1", answer="ans1")
    res2 = AgentResult(agent="agent2", answer="ans2")
    merged = merge_agent_results({"agent1": res1}, {"agent2": res2})
    assert "agent1" in merged
    assert "agent2" in merged


def test_normalize_channel():
    from agents.supervisor import normalize_channel, CHANNEL_HINTS
    assert normalize_channel("📜-rajya-grantham") == "rajya-grantham"
    assert CHANNEL_HINTS[normalize_channel("📜-rajya-grantham")] == "research_agent"
    assert CHANNEL_HINTS[normalize_channel("🐍-vajra-python")] == "python_agent"
    assert CHANNEL_HINTS[normalize_channel("🐳-docker")] == "docker_agent"
    assert CHANNEL_HINTS[normalize_channel("🛰️-mcp-mantra")] == "mcp_agent"
