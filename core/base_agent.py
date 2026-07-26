"""
Kalki Nexus - BaseAgent

Every specialist agent subclasses BaseAgent and, in the common case, only
sets a few class attributes (`name`, `description`, `channel_hints`,
`default_tool_categories`) - `load_prompt()`, `load_tools()`,
`load_memory()`, `run()`, and `post_process()` already do the right thing.
Agents override only the step that actually differs (e.g. Research Agent
overrides `post_process()` to cache sources; Quant Agent overrides it to
record a backtest).

`BaseAgent.__call__` is what actually gets registered as a LangGraph node:
it times the run, retries on retryable errors, wraps the outcome as an
`AgentResult`, and writes it into `state["agent_results"][self.name]` rather
than clobbering a single shared `final_answer` key - this is what lets the
Result Aggregator merge more than one agent's output for a single request.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import Settings, get_settings
from core.exceptions import AgentError, KalkiError
from core.observability import extract_token_usage, get_logger, timed
from core.permissions import DEFAULT_AGENT_PERMISSIONS, SecurityContext
from core.resilience import RetryPolicy, with_retry
from tools.registry import ToolLoader

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class AgentResult(BaseModel):
    """The uniform shape every agent hands back to the Result Aggregator."""

    agent: str
    answer: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.75
    sources: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class BaseAgent(ABC):
    """Shared behavior for every Kalki Nexus specialist agent."""

    name: ClassVar[str]
    description: ClassVar[str] = ""
    prompt_file: ClassVar[Optional[str]] = None  # defaults to f"{name.replace('_agent', '')}.md"
    channel_hints: ClassVar[List[str]] = []  # Discord channel names that bias routing toward this agent
    default_tool_categories: ClassVar[List[str]] = []
    temperature: ClassVar[float] = 0.2
    retry_policy: ClassVar[RetryPolicy] = RetryPolicy(max_attempts=2, backoff_seconds=1.0)

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(f"kalki.agents.{self.name}")
        self.security_context = SecurityContext(DEFAULT_AGENT_PERMISSIONS.get(self.name, []))

    # -- overridable steps ---------------------------------------------------

    def load_prompt(self) -> str:
        """Load this agent's system prompt from prompts/<prompt_file>."""
        filename = self.prompt_file or f"{self.name.replace('_agent', '')}.md"
        return (PROMPTS_DIR / filename).read_text(encoding="utf-8")

    def load_tools(self) -> List[Any]:
        """Return the LangChain tools this agent is allowed to use, permission-checked."""
        return ToolLoader.load_categories(self.default_tool_categories, self.security_context)

    def load_memory(self):
        """Return this agent's namespaced Agent Memory. Imported lazily to avoid a cycle."""
        from memory.factory import MemoryFactory

        return MemoryFactory.agent_memory(self.name, self.settings)

    @abstractmethod
    async def run(self, state: Dict[str, Any]) -> AgentResult:
        """Do the actual work and return a populated AgentResult."""

    async def post_process(self, state: Dict[str, Any], result: AgentResult) -> AgentResult:
        """Hook for side effects after `run()` (e.g. persisting to Agent Memory). No-op by default."""
        return result

    # -- default LLM-backed run() building blocks ----------------------------

    async def _default_llm_run(self, state: Dict[str, Any]) -> AgentResult:
        """A ready-to-use `run()` body for simple "prompt + tools + one LLM call" agents."""
        tools = self.load_tools()
        llm = self.settings.build_chat_model(temperature=self.temperature, tools=tools or None)
        messages = [
            SystemMessage(content=self.load_prompt()),
            HumanMessage(content=state.get("user_input", "")),
        ]
        response: AIMessage = await llm.ainvoke(messages)
        tool_calls = [
            {"name": call.get("name"), "args": call.get("args")} for call in (response.tool_calls or [])
        ]
        tokens = extract_token_usage(response)
        return AgentResult(
            agent=self.name,
            answer=str(response.content),
            tool_calls=tool_calls,
            metadata={
                "model": self.settings.model,
                "provider": self.settings.provider.value,
                "token_usage": tokens,
            },
        )

    # -- the LangGraph node -----------------------------------------------

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """The actual LangGraph node function: timed, retried, and error-isolated."""

        async def _attempt() -> AgentResult:
            result = await self.run(state)
            return await self.post_process(state, result)

        try:
            with timed(self.logger, f"agent:{self.name}") as timing:
                result = await with_retry(self.retry_policy, f"agent:{self.name}")(_attempt)()
            result.metadata.setdefault("duration_seconds", timing.elapsed)
        except KalkiError as exc:
            self.logger.error("agent %s failed: %s", self.name, exc.message)
            return {
                "agent_results": {self.name: AgentResult(agent=self.name, answer="", confidence=0.0, metadata={"error": exc.to_dict()})},
                "metadata": {"failures": {self.name: exc.message}},
            }
        except Exception as exc:  # noqa: BLE001 - normalize unexpected errors
            wrapped = AgentError(self.name, str(exc))
            self.logger.exception("agent %s raised an unexpected error", self.name)
            return {
                "agent_results": {self.name: AgentResult(agent=self.name, answer="", confidence=0.0, metadata={"error": wrapped.to_dict()})},
                "metadata": {"failures": {self.name: str(exc)}},
            }

        return {
            "agent_results": {self.name: result},
            "messages": [AIMessage(content=result.answer, name=self.name)],
            "metadata": {
                "timings": {self.name: result.metadata.get("duration_seconds", 0.0)},
                "tokens": {self.name: result.metadata.get("token_usage", {})},
            },
        }
