"""
Kalki Nexus - Quant Agent

Handles quantitative finance requests: strategy design, backtesting logic,
risk metrics, and portfolio analysis. Every request is recorded into this
agent's namespaced Agent Memory for later review.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent


class QuantAgent(BaseAgent):
    name = "quant_agent"
    description = "Designs and evaluates trading strategies, backtests, and risk metrics."
    channel_hints: ClassVar[List[str]] = ["quant"]
    default_tool_categories: ClassVar[List[str]] = ["quant"]
    temperature = 0.1

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)

    async def post_process(self, state: Dict[str, Any], result: AgentResult) -> AgentResult:
        memory = self.load_memory()
        strategy_name = state.get("user_input", "")[:40] or "unnamed_strategy"
        await memory.set(f"backtest:{strategy_name}", {"raw_response": result.answer[:500]})
        return result
