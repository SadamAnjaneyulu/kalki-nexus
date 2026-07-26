"""
Kalki Nexus - Result Aggregator

Fan-in node: every specialist agent that ran writes an AgentResult into
state["agent_results"][agent_name]. The aggregator merges them into a single
final_answer - a straight pass-through when exactly one agent contributed,
or a short LLM synthesis pass when more than one agent's output needs to be
combined into a coherent response. It is also where a per-agent failure
becomes a graph-level `state["error"]`, driving the error/retry/fallback
path in graph.py.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from core.observability import get_logger

logger = get_logger("kalki.agents.aggregator")

_SYNTHESIS_PROMPT = (
    "You are combining the outputs of multiple specialist agents into one coherent "
    "answer for the user. Preserve every concrete detail (code, commands, numbers, "
    "file names) from each agent's answer - do not summarize away specifics. "
    "Organize the combined answer with a short heading per agent's contribution."
)


async def _synthesize(user_input: str, successful: Dict[str, Any]) -> str:
    if len(successful) == 1:
        return next(iter(successful.values())).answer

    settings = get_settings()
    try:
        llm = settings.build_chat_model(temperature=0.2)
    except Exception as exc:  # noqa: BLE001 - no configured provider: fall back to a plain concatenation
        logger.warning("synthesis LLM unavailable (%s); concatenating agent answers instead.", exc)
        return "\n\n".join(f"## {name}\n{result.answer}" for name, result in successful.items())

    parts = "\n\n".join(f"### {name}\n{result.answer}" for name, result in successful.items())
    messages = [
        SystemMessage(content=_SYNTHESIS_PROMPT),
        HumanMessage(content=f"Original request:\n{user_input}\n\nAgent outputs:\n{parts}"),
    ]
    response = await llm.ainvoke(messages)
    return str(response.content)


async def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: merge every agent_results entry into final_answer / error."""
    results = state.get("agent_results") or {}
    if not results:
        return {"final_answer": "No agent produced a result.", "error": None}

    failed = {name: result for name, result in results.items() if result.metadata.get("error")}
    successful = {name: result for name, result in results.items() if name not in failed}

    if failed and not successful:
        first_error = next(iter(failed.values())).metadata["error"]
        logger.warning("aggregator: every contributing agent failed (%s)", list(failed))
        return {"final_answer": None, "error": first_error}

    final_answer = await _synthesize(state.get("user_input", ""), successful)
    sources = sorted({source for result in successful.values() for source in result.sources})
    overall_confidence = sum(result.confidence for result in successful.values()) / len(successful)

    return {
        "final_answer": final_answer,
        "sources": sources,
        "error": None,  # a partial failure alongside a successful agent is not graph-fatal
        "metadata": {
            "contributing_agents": sorted(successful),
            "failed_agents": sorted(failed),
            "overall_confidence": round(overall_confidence, 3),
        },
    }
