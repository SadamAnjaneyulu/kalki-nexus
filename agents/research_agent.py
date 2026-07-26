"""
Kalki Nexus - Research Agent (Phase 6: RAG-enhanced)

Retrieves context from indexed documents before calling the LLM,
making every response grounded in your knowledge base.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent
from core.rag import Retriever


class ResearchAgent(BaseAgent):
    name = "research_agent"
    description = "Gathers and summarizes information from the web and internal knowledge base, with citations."
    channel_hints: ClassVar[List[str]] = ["research"]
    default_tool_categories: ClassVar[List[str]] = ["web", "browser"]
    temperature = 0.2
    rag_collection: ClassVar[str] = "kalki_knowledge"

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        user_input = state.get("user_input", "")

        # Phase 6: Retrieve relevant RAG context before LLM call
        retriever = Retriever()
        rag_context = await retriever.build_context(self.rag_collection, user_input, top_k=4)

        if rag_context:
            # Inject retrieved context into the user message
            enriched_state = {
                **state,
                "user_input": f"{rag_context}\n\n---\n\nUser question: {user_input}",
            }
        else:
            enriched_state = state

        return await self._default_llm_run(enriched_state)

    async def post_process(self, state: Dict[str, Any], result: AgentResult) -> AgentResult:
        memory = self.load_memory()
        topic = state.get("user_input", "")[:80]
        await memory.set(f"source:{topic}", {"summary": result.answer[:500]})
        return result
