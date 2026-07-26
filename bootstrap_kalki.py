#!/usr/bin/env python3
"""
bootstrap_kalki.py

Bootstraps "Kalki Nexus": a modular, async-first, LangGraph-orchestrated
multi-agent AI Operating System scaffold with a Discord front end.

This is a *code generator*: running it materializes a full project (agents,
tools, memory, MCP registry, Discord bot, tests, docs) onto disk. It does not
itself run the agents.

Usage:
    python bootstrap_kalki.py                  # generate ./kalki-nexus (skips existing files)
    python bootstrap_kalki.py --overwrite       # regenerate, overwriting existing files
    python bootstrap_kalki.py --root some/dir   # generate into a custom target directory
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
except ImportError:  # pragma: no cover
    sys.stderr.write("This script requires 'rich'. Install it with: pip install rich\n")
    raise SystemExit(1)

console = Console()
PROJECT_NAME = "kalki-nexus"


@dataclass(frozen=True)
class GeneratedFile:
    """A single file to be materialized on disk, relative to the project root."""

    relative_path: str
    content: str


# ============================================================================
# Section 1: top-level project files
# ============================================================================

def content_readme() -> str:
    return '''# Kalki Nexus

Kalki Nexus is a modular, async-first **AI Operating System**: a
LangGraph-orchestrated team of specialist agents (Python, Docker, GitHub,
Research, Quant, Automation, MCP) coordinated by an LLM-powered Supervisor,
with a Discord front end and a plugin architecture for adding new agents,
tools, memory backends, MCP servers, and chat surfaces.

## Architecture

```
                              +-------------------+
                              |   Discord Bot      |
                              | (channel adapter)  |
                              +---------+-----------+
                                        |
                                        v
                              +-------------------+
                              |    Supervisor       |   <- LLM + Pydantic
                              | (structured routing)|      structured output
                              +---------+-----------+
                                        |
                    fan-out (parallel or sequential, Supervisor-decided)
                                        |
             +----------+--------+-----+-----+----------+----------+----------+
             |          |               |            |           |          |
             v          v               v            v           v          v
        +--------+ +--------+     +--------+   +---------+  +--------+ +--------+
        | Python | | Docker |     | GitHub |   |Research |  | Quant  | |  MCP   |
        | Agent  | | Agent  | ... | Agent  |   | Agent   |  | Agent  | | Agent  |
        +--------+ +--------+     +--------+   +---------+  +--------+ +--------+
             |          |               |            |           |          |
             +----------+--------+------+------------+-----------+----------+
                                        |
                                        v
                              +-------------------+
                              | Result Aggregator  |   <- merges AgentResult
                              |   (merge node)      |      objects, synthesizes
                              +---------+-----------+      a final answer
                                        |
                             +----------+-----------+
                             |                       |
                    needs human approval?      normal completion
                             |                       |
                             v                       v
                    +-----------------+       +-------------+
                    | Human Approval  |------>|     END      |
                    |  (interrupt())  |       +-------------+
                    +-----------------+

                    Any node -> on exception -> Retry (bounded) -> Error Node -> Fallback Agent -> END
```

The **Supervisor** (`agents/supervisor.py`) is an LLM call with a Pydantic
`RouteDecision` schema (`agents`, `reasoning`, `confidence`) produced via
`llm.with_structured_output(RouteDecision)`. It inspects the user message,
the Discord channel, attached files, requested tools, and prior conversation
state. The Discord channel (e.g. `#docker`, `#quant`, `#vajra-python`)
contributes a **hint**, not a hard rule: it is folded into the routing
prompt so the Supervisor is *biased* toward the matching agent without being
forced to pick it. If the LLM call is unavailable (no API key, offline
tests), routing falls back to a deterministic heuristic
(`agents/supervisor.py::heuristic_routes`) so the graph always degrades
gracefully instead of failing closed.

Requests can fan out to more than one agent at once (e.g. Python + Docker,
Research + Quant). LangGraph runs the selected agent nodes as parallel
branches; each branch writes into the shared `agent_results` dict (merged via
a custom reducer), and the **Result Aggregator** node fans them back in,
merging every agent's `AgentResult` (`answer`, `metadata`, `confidence`,
`sources`, `tool_calls`) into one final response - synthesizing across
agents with a short LLM pass when more than one agent contributed.

## Core Abstractions

| Abstraction  | Location                | Purpose                                             |
|--------------|--------------------------|------------------------------------------------------|
| `BaseAgent`  | `core/base_agent.py`     | `load_prompt()`, `load_tools()`, `load_memory()`, `run()`, `post_process()` - specialists override only what differs. |
| `BaseTool`   | `core/base_tool.py`      | Every tool declares `name`, `description`, `permissions`, `category`, and exposes `to_langchain_tool()`. |
| `BaseMemory` | `core/base_memory.py`    | Every memory layer implements `get/set/delete/list_keys` against a pluggable `StorageBackend`. |
| `SecurityContext` | `core/permissions.py` | Checks a tool's declared `Permission`s against what an agent has been granted before it runs. |
| `ToolRegistry` | `tools/registry.py`    | Central place tools register into; agents *request* tools by category instead of importing/binding them manually. |
| `MCPRegistry`  | `mcp/registry.py`      | Server registration, dynamic tool discovery/loading, capability caching with TTL, hot reload. |

## Memory Architecture

Kalki Nexus separates memory by *scope*, not by agent:

- **Working Memory** - scratch space for the current graph run only (in-process, not persisted).
- **Conversation Memory** - the running message history for a thread/session.
- **Session Memory** - per-session key/value state (e.g. active workflow step).
- **Agent Memory** - namespaced, per-agent persisted state (what used to be `research_memory.py` / `quant_memory.py`).
- **Shared Memory** - cross-agent persisted state.
- **Long-Term Memory** - the durable interface (`memory/long_term_memory.py`) all of the above are built on.

Every layer is a thin wrapper around a `StorageBackend` (`memory/backends/`).
The default backend is SQLite (`memory/backends/sqlite_backend.py`, fully
functional). `memory/backends/stub_backends.py` defines
`PostgresBackend`, `RedisBackend`, `QdrantBackend`, and `ChromaBackend` with
the *same* `StorageBackend` interface, so swapping the backend is a one-line
change in `config.py` (`MEMORY_BACKEND=postgres`) with **zero agent code
changes** - agents only ever talk to `BaseMemory`.

## MCP (Model Context Protocol)

`mcp/registry.py` implements `MCPRegistry`: server registration
(`MCPServerConfig`), dynamic tool discovery per server with TTL-based
capability caching, hot reload (invalidate + re-discover), and tool
resolution/selection by name. `mcp/client.py` documents exactly where to
plug in a real transport (e.g. `langchain_mcp_adapters.MultiServerMCPClient`
for stdio/SSE/streamable-HTTP servers) - the actual network calls are the
one piece intentionally left as a `TODO`, per the brief; the surrounding
architecture (registry, caching, hot reload, multi-server support, tool
resolution) is real and exercised by `agents/mcp_agent.py`.

## Configuration

`config.py` exposes a single `Settings` class (env-driven, Pydantic) with a
`provider` field (`openai` | `anthropic` | `openrouter` | `ollama` |
`azure_openai`) and one factory method, `Settings.build_chat_model(...)`,
that returns the right LangChain chat model for whichever provider is
configured. Agents and the Supervisor call `settings.build_chat_model(...)`
and never import a provider-specific class directly - swapping providers is
an environment variable change, not a code change.

## Security & Permissions

Every tool declares `required_permissions: List[Permission]` (filesystem
read/write/delete, terminal exec, Docker, GitHub, browser automation,
network, Discord send). `core/permissions.py::SecurityContext` is built per
agent from `DEFAULT_AGENT_PERMISSIONS` and checked before every tool call in
`BaseTool.__call__`, so an agent can only reach the tools it has been
explicitly granted.

## Error Handling & Resilience

`core/resilience.py` provides `RetryPolicy` + `with_retry(...)` (bounded
retries with backoff around any agent/tool call), and `agents/fallback_agent.py`
is a last-resort agent the graph routes to once retries are exhausted. The
graph itself has a dedicated `error_node` that captures structured
exceptions (`core/exceptions.py`) into `state["error"]` and routes to the
Fallback Agent rather than crashing the run.

## Observability

`core/observability.py` wires up:
- Rich logging (`get_logger`)
- `LangSmith` tracing when `LANGSMITH_API_KEY` is set (`setup_langsmith`)
- Per-node and per-tool timing (`@timed`, recorded into `state["metadata"]["timings"]`)
- Graph visualization (`render_graph_mermaid`, writes `graph.mmd` / `graph.png` when Graphviz/Mermaid deps are present)

## Human-in-the-Loop

Any `AgentResult` can set `metadata["requires_approval"] = True` (e.g. a
destructive GitHub or terminal action). The graph's conditional edge routes
those results to `human_approval_node`, which calls LangGraph's
`interrupt()` primitive and pauses the run at a checkpoint
(`MemorySaver`) until a human resumes it with an approve/deny `Command`.

## Installation

1. Create and activate a virtual environment (Python 3.12+):
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv && .venv\\Scripts\\activate`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy environment variables and fill them in:
   `cp .env.example .env`

## Environment Variables

| Variable                 | Description                                                      |
|---------------------------|-------------------------------------------------------------------|
| MODEL_PROVIDER            | `openai` \\| `anthropic` \\| `openrouter` \\| `ollama` \\| `azure_openai` |
| MODEL                     | Chat model name for the configured provider                       |
| OPENAI_API_KEY            | Required when `MODEL_PROVIDER=openai` (or `openrouter`)           |
| ANTHROPIC_API_KEY         | Required when `MODEL_PROVIDER=anthropic`                          |
| OPENROUTER_API_KEY        | Required when `MODEL_PROVIDER=openrouter`                         |
| OLLAMA_BASE_URL           | Used when `MODEL_PROVIDER=ollama` (default `http://localhost:11434`) |
| AZURE_OPENAI_ENDPOINT     | Required when `MODEL_PROVIDER=azure_openai`                       |
| AZURE_OPENAI_DEPLOYMENT   | Required when `MODEL_PROVIDER=azure_openai`                       |
| AZURE_OPENAI_API_VERSION  | Required when `MODEL_PROVIDER=azure_openai`                       |
| DISCORD_TOKEN             | Bot token from the Discord Developer Portal                       |
| GITHUB_TOKEN              | Personal access token used by the GitHub tools                    |
| LANGSMITH_API_KEY         | Optional: enables LangSmith tracing                               |
| LANGSMITH_PROJECT         | Optional: LangSmith project name                                  |
| MEMORY_BACKEND            | `sqlite` (default) \\| `postgres` \\| `redis` \\| `qdrant` \\| `chroma` |
| LOG_LEVEL                 | Python logging level, e.g. `INFO` or `DEBUG`                      |

## Running

Run a single example request through the graph:

```
python app.py
```

Run the Discord bot:

```
python app.py --discord
```

Render the graph structure without running it:

```
python app.py --render-graph
```

Run the test suite:

```
pytest
```

## Adding a New Agent

Agents are auto-discovered - no manual registration needed:

1. Add `agents/<name>_agent.py`.
2. Define a class subclassing `core.base_agent.BaseAgent`, setting `name`,
   `description`, `channel_hints` (Discord channels that should bias routing
   toward this agent), and `default_tool_categories`. Override `load_tools()`
   / `load_memory()` / `post_process()` only if you need non-default
   behavior - `run()` already handles prompt loading, tool binding, LLM
   invocation, and wrapping the response into an `AgentResult`.
3. Add `prompts/<name>.md` with that agent's system prompt.
4. That's it. `core/registry.py::discover_agents()` walks the `agents/`
   package at startup, finds every `BaseAgent` subclass, and both
   `graph.py` (node + edge wiring) and `agents/supervisor.py` (the list of
   agents the Supervisor is allowed to route to) pick it up automatically.

## Adding a New Tool

1. Add a class to the relevant file under `tools/` subclassing
   `core.base_tool.BaseTool`, declaring `name`, `description`,
   `required_permissions`, and `category`, and implementing `async def run(self, **kwargs)`.
2. Decorate it with `@ToolRegistry.register()` (from `tools/registry.py`).
3. Any agent whose `default_tool_categories` includes that tool's category
   picks it up automatically via `ToolLoader.load_categories(...)` - no
   manual `llm.bind_tools([...])` list to maintain per agent.

## Adding a New MCP Server

1. Register it in `config.py` / `mcp/registry.py::default_servers()` as an
   `MCPServerConfig` (name, transport, command/url, env).
2. `MCPRegistry.discover_tools()` will pick up its tools on the next
   discovery pass (or call `MCPRegistry.hot_reload()` to force it
   immediately); `agents/mcp_agent.py` resolves and binds them dynamically.
3. Wire the actual transport call in `mcp/client.py` (see the `TODO` there)
   when you're ready to connect a real server - the registry, caching, and
   agent-facing API do not change.

## Adding a New Chat Surface (Telegram, Slack, WhatsApp, Web UI, Voice)

The graph has no Discord-specific code in it. `core/channel_adapter.py`
defines a small `ChannelAdapter` protocol (`send`, `stream`, `typing`) that
`discord/bot.py` implements; a new surface implements the same protocol and
calls `graph.py::invoke(...)` / `compiled_graph().astream(...)` exactly the
way the Discord bot does.

## Project Layout

```
kalki-nexus/
    app.py                entrypoint: logging, config, example run, --discord / --render-graph flags
    graph.py               LangGraph state, auto-discovered node wiring, invoke()
    config.py               single Settings class, multi-provider model factory
    core/                   BaseAgent, BaseTool, BaseMemory, permissions, resilience,
                             observability, plugin registry, channel adapter protocol
    agents/                 one module per specialist agent (BaseAgent subclasses),
                             the LLM-powered Supervisor, the Result Aggregator, the Fallback Agent
    prompts/                one system prompt per agent
    tools/                  BaseTool subclasses (Docker, filesystem, GitHub, browser,
                             terminal, Discord, web) plus the ToolRegistry/Loader/Resolver
    mcp/                    MCPRegistry, MCPServerConfig, capability cache, client transport stub
    workflows/               fixed multi-agent pipelines outside Supervisor routing
    memory/                  Working/Conversation/Session/Agent/Shared/Long-Term memory,
                             pluggable storage backends (SQLite real, Postgres/Redis/Qdrant/Chroma stubbed)
    discord/                Discord bot: slash commands, threads, streaming, typing indicator
    data/                   SQLite DB + agent memory persistence
    logs/                   application logs
    tests/                  pytest smoke tests for the graph, supervisor, and tool registry
```
'''


def content_gitignore() -> str:
    return '''__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.venv/
venv/
env/
.env
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
.ruff_cache/
logs/*.log
data/*.json
data/*.db
data/*.sqlite3
!data/.gitkeep
!logs/.gitkeep
graph.mmd
graph.png
.DS_Store
.idea/
.vscode/
'''


def content_env_example() -> str:
    return '''# --- Model provider -----------------------------------------------------
# One of: openai | anthropic | openrouter | ollama | azure_openai
MODEL_PROVIDER=openai
MODEL=gpt-5

OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_API_KEY=

# --- Integrations ---------------------------------------------------------
DISCORD_TOKEN=
GITHUB_TOKEN=

# --- Observability ---------------------------------------------------------
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=kalki-nexus

# --- Memory ------------------------------------------------------------
# One of: sqlite | postgres | redis | qdrant | chroma
MEMORY_BACKEND=sqlite
POSTGRES_DSN=
REDIS_URL=
QDRANT_URL=
CHROMA_PATH=

LOG_LEVEL=INFO
'''


def content_requirements() -> str:
    return '''langgraph>=0.2.0
langgraph-checkpoint>=2.0.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0
langchain-community>=0.3.0
openai>=1.40.0
anthropic>=0.34.0
discord.py>=2.4.0
python-dotenv>=1.0.1
rich>=13.7.0
pydantic>=2.8.0
httpx>=0.27.0
aiofiles>=24.1.0
aiosqlite>=0.20.0
uvicorn>=0.30.0
fastapi>=0.112.0
typing_extensions>=4.12.0
PyGithub>=2.4.0
langsmith>=0.1.100
pytest>=8.3.0
pytest-asyncio>=0.24.0

# --- Optional, feature-gated at runtime (import lazily / behind TODOs) -----
# langchain-mcp-adapters>=0.1.0   # real MCP transport (see mcp/client.py)
# playwright>=1.47.0               # real browser automation (see tools/browser_tools.py)
# psycopg[binary]>=3.2.0           # Postgres memory backend
# redis>=5.0.0                     # Redis memory backend
# qdrant-client>=1.11.0            # Qdrant memory backend
# chromadb>=0.5.0                  # Chroma memory backend
'''


def content_pyproject() -> str:
    return '''[project]
name = "kalki-nexus"
version = "0.2.0"
description = "Kalki Nexus: a modular, LangGraph-orchestrated AI Operating System."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.2.0",
    "langgraph-checkpoint>=2.0.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",
    "langchain-community>=0.3.0",
    "openai>=1.40.0",
    "anthropic>=0.34.0",
    "discord.py>=2.4.0",
    "python-dotenv>=1.0.1",
    "rich>=13.7.0",
    "pydantic>=2.8.0",
    "httpx>=0.27.0",
    "aiofiles>=24.1.0",
    "aiosqlite>=0.20.0",
    "uvicorn>=0.30.0",
    "fastapi>=0.112.0",
    "typing_extensions>=4.12.0",
    "PyGithub>=2.4.0",
    "langsmith>=0.1.100",
]

[project.optional-dependencies]
dev = ["pytest>=8.3.0", "pytest-asyncio>=0.24.0"]
mcp = ["langchain-mcp-adapters>=0.1.0"]
browser = ["playwright>=1.47.0"]
memory-postgres = ["psycopg[binary]>=3.2.0"]
memory-redis = ["redis>=5.0.0"]
memory-qdrant = ["qdrant-client>=1.11.0"]
memory-chroma = ["chromadb>=0.5.0"]

[build-system]
requires = ["setuptools>=69.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
# NOTE: "discord" and "mcp" are intentionally excluded from this list.
# Packaging a local "discord" module alongside the installed "discord.py"
# dependency (which also installs a top-level "discord" package), or a local
# "mcp" module alongside the official "mcp" SDK (a transitive dependency of
# langchain-mcp-adapters), would collide in site-packages if this project
# were ever installed rather than run in place from source. Both packages
# work fine run-in-place because the project root is on sys.path first; see
# the README's "Known gotcha" section if you ever do package/install this.
packages = ["core", "agents", "tools", "memory", "workflows"]
'''


def content_config_py() -> str:
    return '''"""
Kalki Nexus - Configuration

A single, typed Settings class loaded from environment variables (and a
local `.env` file via python-dotenv). `Settings.build_chat_model(...)` is
the one factory every agent uses to get an LLM - it dispatches to whichever
provider is configured (OpenAI, Anthropic, OpenRouter, Ollama, Azure OpenAI)
so agent code never imports a provider-specific class directly.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

PROJECT_ROOT = Path(__file__).resolve().parent


class ModelProvider(str, Enum):
    """Supported chat model providers, selected via MODEL_PROVIDER."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


class MemoryBackendKind(str, Enum):
    """Supported long-term memory storage backends, selected via MEMORY_BACKEND."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"
    REDIS = "redis"
    QDRANT = "qdrant"
    CHROMA = "chroma"


class Settings(BaseModel):
    """Runtime configuration for Kalki Nexus, sourced from environment variables."""

    provider: ModelProvider = Field(
        default_factory=lambda: ModelProvider(os.getenv("MODEL_PROVIDER", "openai"))
    )
    model: str = Field(default_factory=lambda: os.getenv("MODEL", "gpt-5"))

    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    azure_openai_endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_openai_deployment: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))
    azure_openai_api_version: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    )
    azure_openai_api_key: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))

    discord_token: str = Field(default_factory=lambda: os.getenv("DISCORD_TOKEN", ""))
    github_token: str = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))

    langsmith_api_key: str = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = Field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "kalki-nexus"))

    memory_backend: MemoryBackendKind = Field(
        default_factory=lambda: MemoryBackendKind(os.getenv("MEMORY_BACKEND", "sqlite"))
    )
    postgres_dsn: str = Field(default_factory=lambda: os.getenv("POSTGRES_DSN", ""))
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    chroma_path: str = Field(default_factory=lambda: os.getenv("CHROMA_PATH", ""))

    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def require_key_for_provider(self) -> None:
        """Raise a clear error if the configured provider is missing its credential."""
        missing = {
            ModelProvider.OPENAI: ("OPENAI_API_KEY", self.openai_api_key),
            ModelProvider.ANTHROPIC: ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            ModelProvider.OPENROUTER: ("OPENROUTER_API_KEY", self.openrouter_api_key),
            ModelProvider.AZURE_OPENAI: ("AZURE_OPENAI_API_KEY", self.azure_openai_api_key),
            ModelProvider.OLLAMA: ("", "ok"),  # Ollama is typically unauthenticated / local.
        }[self.provider]
        env_name, value = missing
        if env_name and not value:
            raise RuntimeError(
                f"{env_name} is not set for MODEL_PROVIDER={self.provider.value}. "
                "Copy .env.example to .env and fill it in."
            )

    def build_chat_model(
        self,
        temperature: float = 0.2,
        tools: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ):
        """Return a LangChain chat model for whichever provider is configured.

        This is the single seam agents use to get an LLM. Adding a provider
        means adding one branch here, not touching every agent module.
        """
        self.require_key_for_provider()
        model_name = model_override or self.model
        llm: Any

        if self.provider is ModelProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=model_name, api_key=self.openai_api_key, temperature=temperature)

        elif self.provider is ModelProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model=model_name, api_key=self.anthropic_api_key, temperature=temperature)

        elif self.provider is ModelProvider.OPENROUTER:
            # OpenRouter speaks the OpenAI wire protocol; only the base_url and key differ.
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model_name,
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
            )

        elif self.provider is ModelProvider.OLLAMA:
            from langchain_community.chat_models import ChatOllama

            llm = ChatOllama(model=model_name, base_url=self.ollama_base_url, temperature=temperature)

        elif self.provider is ModelProvider.AZURE_OPENAI:
            from langchain_openai import AzureChatOpenAI

            llm = AzureChatOpenAI(
                azure_endpoint=self.azure_openai_endpoint,
                azure_deployment=self.azure_openai_deployment,
                api_version=self.azure_openai_api_version,
                api_key=self.azure_openai_api_key,
                temperature=temperature,
            )
        else:  # pragma: no cover - guarded by the ModelProvider enum
            raise ValueError(f"Unsupported provider: {self.provider}")

        return llm.bind_tools(tools) if tools else llm


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
'''


# ============================================================================
# Section 2: core/ - shared abstractions (BaseAgent, BaseTool, BaseMemory,
# permissions, resilience, observability, plugin registry, channel adapter)
# ============================================================================

def content_core_init() -> str:
    return '''"""Kalki Nexus core package: cross-cutting abstractions shared by every agent, tool, and memory layer."""
'''


def content_core_exceptions() -> str:
    return '''"""
Kalki Nexus - Structured Exceptions

Every failure mode in the graph should raise one of these instead of a bare
Exception, so `core/resilience.py` and the graph's error node can make
informed retry/fallback decisions instead of treating all failures alike.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class KalkiError(Exception):
    """Base class for every structured Kalki Nexus exception."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class AgentError(KalkiError):
    """Raised when a specialist agent fails to produce a result."""

    def __init__(self, agent_name: str, message: str, *, details: Optional[Dict[str, Any]] = None, retryable: bool = True) -> None:
        super().__init__(message, details={"agent": agent_name, **(details or {})}, retryable=retryable)
        self.agent_name = agent_name


class ToolError(KalkiError):
    """Raised when a tool call fails."""

    def __init__(self, tool_name: str, message: str, *, details: Optional[Dict[str, Any]] = None, retryable: bool = True) -> None:
        super().__init__(message, details={"tool": tool_name, **(details or {})}, retryable=retryable)
        self.tool_name = tool_name


class PermissionDeniedError(KalkiError):
    """Raised when a tool is invoked without the permissions it declares as required."""

    def __init__(self, tool_name: str, missing: Any) -> None:
        super().__init__(
            f"Tool '{tool_name}' requires permissions the caller was not granted: {missing}",
            details={"tool": tool_name, "missing_permissions": [str(p) for p in missing]},
            retryable=False,
        )
        self.tool_name = tool_name
        self.missing = missing


class RoutingError(KalkiError):
    """Raised when the Supervisor cannot produce a valid routing decision."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details=details, retryable=True)


class MemoryError_(KalkiError):
    """Raised on a memory backend failure. Named with a trailing underscore to avoid
    shadowing the built-in MemoryError."""

    def __init__(self, backend: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details={"backend": backend, **(details or {})}, retryable=True)
        self.backend = backend


class MCPError(KalkiError):
    """Raised on an MCP server/tool discovery or invocation failure."""

    def __init__(self, server_name: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details={"server": server_name, **(details or {})}, retryable=True)
        self.server_name = server_name


class RetryExhaustedError(KalkiError):
    """Raised by core.resilience.with_retry once every attempt has failed."""

    def __init__(self, operation: str, attempts: int, last_error: Optional[BaseException]) -> None:
        super().__init__(
            f"'{operation}' failed after {attempts} attempt(s): {last_error}",
            details={"operation": operation, "attempts": attempts, "last_error": str(last_error)},
            retryable=False,
        )
        self.last_error = last_error
'''


def content_core_permissions() -> str:
    return '''"""
Kalki Nexus - Permissions & Security Context

Every tool declares the permissions it needs. Every agent is granted a set
of permissions (see config.py-adjacent DEFAULT_AGENT_PERMISSIONS below).
`SecurityContext.check(...)` is called by `BaseTool.__call__` before a tool
actually runs, so an agent can only reach what it has been explicitly
granted - not everything that happens to be importable.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Set

from core.exceptions import PermissionDeniedError


class Permission(str, Enum):
    """A single grantable capability a tool may require."""

    FS_READ = "fs:read"
    FS_WRITE = "fs:write"
    FS_DELETE = "fs:delete"
    TERMINAL = "terminal:exec"
    DOCKER = "docker:manage"
    GITHUB = "github:manage"
    BROWSER = "browser:automate"
    NETWORK = "network:http"
    DISCORD = "discord:send"


class SecurityContext:
    """The set of permissions a particular agent (or tool call) has been granted."""

    def __init__(self, granted: Iterable[Permission]) -> None:
        self.granted: Set[Permission] = set(granted)

    def check(self, required: List[Permission], tool_name: str) -> None:
        """Raise PermissionDeniedError if any required permission is not granted."""
        missing = [permission for permission in required if permission not in self.granted]
        if missing:
            raise PermissionDeniedError(tool_name, missing)

    def has(self, permission: Permission) -> bool:
        return permission in self.granted

    @classmethod
    def full_access(cls) -> "SecurityContext":
        """A SecurityContext granting every permission - useful for tests/admin tooling."""
        return cls(list(Permission))


# Default permission grants per agent name. Agents call
# `SecurityContext(DEFAULT_AGENT_PERMISSIONS.get(self.name, []))` inside
# `load_tools()` (see core/base_agent.py) rather than hardcoding this per module.
DEFAULT_AGENT_PERMISSIONS: Dict[str, List[Permission]] = {
    "python_agent": [Permission.FS_READ, Permission.FS_WRITE],
    "docker_agent": [Permission.DOCKER, Permission.FS_READ],
    "github_agent": [Permission.GITHUB, Permission.NETWORK],
    "research_agent": [Permission.NETWORK, Permission.BROWSER],
    "quant_agent": [Permission.FS_READ, Permission.FS_WRITE],
    "automation_agent": [Permission.FS_READ, Permission.FS_WRITE, Permission.TERMINAL],
    "mcp_agent": [Permission.NETWORK],
    "fallback_agent": [],
}
'''


def content_core_base_tool() -> str:
    return '''"""
Kalki Nexus - BaseTool

Every tool in `tools/` subclasses BaseTool instead of being a bare
`@tool`-decorated function. This gives every tool a uniform shape - a
declared name, description, category, and required permissions - and a
single place (`__call__`) where permission checks and structured errors are
enforced before the tool logic runs.
"""
from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional

from langchain_core.tools import StructuredTool

from core.exceptions import ToolError
from core.permissions import Permission, SecurityContext


class BaseTool(ABC):
    """Shared interface every Kalki Nexus tool implements."""

    name: ClassVar[str]
    description: ClassVar[str]
    category: ClassVar[str] = "general"
    required_permissions: ClassVar[List[Permission]] = []

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the tool. Subclasses implement the actual side effect here."""

    async def __call__(self, *, security_context: Optional[SecurityContext] = None, **kwargs: Any) -> Any:
        """Permission-checked entrypoint used by ToolLoader-bound LangChain tools."""
        if security_context is not None:
            security_context.check(self.required_permissions, self.name)
        try:
            return await self.run(**kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize into a structured ToolError
            raise ToolError(self.name, str(exc)) from exc

    def to_langchain_tool(self, security_context: Optional[SecurityContext] = None) -> StructuredTool:
        """Bind this tool (with its permission check already applied) as a LangChain StructuredTool.

        Uses functools.wraps on the bound run() method so LangChain's
        signature introspection (used to build the tool's argument schema
        for the LLM) sees run()'s real parameters instead of a bare **kwargs.
        """
        bound_run = self.run

        @functools.wraps(bound_run)
        async def _invoke(**kwargs: Any) -> Any:
            if security_context is not None:
                security_context.check(self.required_permissions, self.name)
            try:
                return await bound_run(**kwargs)
            except ToolError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize into a structured ToolError
                raise ToolError(self.name, str(exc)) from exc

        return StructuredTool.from_function(
            coroutine=_invoke,
            name=self.name,
            description=self.description,
        )
'''


def content_core_base_memory() -> str:
    return '''"""
Kalki Nexus - BaseMemory & StorageBackend

BaseMemory is the interface every memory layer (Working, Conversation,
Session, Agent, Shared, Long-Term) implements. StorageBackend is the
interface every physical store (SQLite, Postgres, Redis, Qdrant, Chroma)
implements. Memory layers hold a StorageBackend by composition, so swapping
SQLite for Postgres/Redis/Qdrant/Chroma is a config change, not an agent
code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class StorageBackend(ABC):
    """Physical storage interface: get/set/delete/list_keys against a namespaced store."""

    @abstractmethod
    async def get(self, namespace: str, key: str) -> Optional[Any]: ...

    @abstractmethod
    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None: ...

    @abstractmethod
    async def delete(self, namespace: str, key: str) -> None: ...

    @abstractmethod
    async def list_keys(self, namespace: str, prefix: str = "") -> List[str]: ...


class BaseMemory(ABC):
    """Interface every memory layer implements, regardless of scope or backend."""

    namespace: str

    def __init__(self, backend: StorageBackend, namespace: str) -> None:
        self.backend = backend
        self.namespace = namespace

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]: ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]: ...
'''


def content_core_base_agent() -> str:
    return '''"""
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
from core.observability import get_logger, timed
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
        return AgentResult(
            agent=self.name,
            answer=str(response.content),
            tool_calls=tool_calls,
            metadata={"model": self.settings.model, "provider": self.settings.provider.value},
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
                "error": exc.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001 - normalize unexpected errors
            wrapped = AgentError(self.name, str(exc))
            self.logger.exception("agent %s raised an unexpected error", self.name)
            return {
                "agent_results": {self.name: AgentResult(agent=self.name, answer="", confidence=0.0, metadata={"error": wrapped.to_dict()})},
                "error": wrapped.to_dict(),
            }

        return {
            "agent_results": {self.name: result},
            "messages": [AIMessage(content=result.answer, name=self.name)],
            "metadata": {"timings": {self.name: result.metadata.get("duration_seconds", 0.0)}},
        }
'''


def content_core_observability() -> str:
    return '''"""
Kalki Nexus - Observability

Rich logging, LangSmith tracing setup, and lightweight timing utilities used
by every agent and tool call. `state["metadata"]["timings"]` accumulates a
{node_name: seconds} map across a single graph run so slow nodes are visible
without attaching a profiler.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from rich.logging import RichHandler

from config import Settings

_CONFIGURED = False


def configure_logging(settings: Settings) -> None:
    """Configure Rich-powered logging for the whole application. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. get_logger('kalki.agents.python_agent')."""
    return logging.getLogger(name)


def setup_langsmith(settings: Settings) -> bool:
    """Enable LangSmith tracing via environment variables if an API key is configured.

    Returns True if tracing was enabled. LangChain/LangGraph read these
    LANGCHAIN_* variables automatically - no code-level SDK call is required.
    """
    if not settings.langsmith_api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    return True


@dataclass
class _Timing:
    label: str
    start: float = field(default_factory=time.perf_counter)
    elapsed: float = 0.0


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[_Timing]:
    """Context manager that logs and records the wall-clock duration of a block.

    Usage:
        with timed(logger, "agent:python_agent") as timing:
            ...
        # timing.elapsed is now populated
    """
    timing = _Timing(label=label)
    try:
        yield timing
    finally:
        timing.elapsed = time.perf_counter() - timing.start
        logger.debug("%s took %.3fs", label, timing.elapsed)


def render_graph_mermaid(app: object, output_path: str = "graph.mmd") -> Optional[str]:
    """Write a Mermaid diagram of a compiled graph's structure. Best-effort.

    Requires nothing beyond langgraph itself for the `.mmd` text; a `.png`
    render additionally requires the optional `pygraphviz`/`grandalf`
    dependencies, so PNG export is attempted but never required.
    """
    try:
        mermaid = app.get_graph().draw_mermaid()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - visualization is best-effort
        get_logger("kalki.observability").warning("could not render graph: %s", exc)
        return None

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(mermaid)

    try:
        png_bytes = app.get_graph().draw_mermaid_png()  # type: ignore[attr-defined]
        png_path = output_path.rsplit(".", 1)[0] + ".png"
        with open(png_path, "wb") as handle:
            handle.write(png_bytes)
    except Exception:  # noqa: BLE001 - PNG export needs extra deps; text export already succeeded
        pass

    return output_path
'''


def content_core_resilience() -> str:
    return '''"""
Kalki Nexus - Resilience

RetryPolicy + with_retry wrap any async callable with bounded retries and
linear backoff, retrying only on exceptions marked `retryable=True` (see
core/exceptions.py). Once every attempt is exhausted, RetryExhaustedError is
raised so the graph's error node / Fallback Agent can take over deliberately
instead of the process crashing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from core.exceptions import KalkiError, RetryExhaustedError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry configuration: how many attempts, and how long to back off between them."""

    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0


def with_retry(policy: RetryPolicy, operation_name: str):
    """Decorator factory: retries an async function per `policy`, re-raising
    non-retryable KalkiErrors immediately and wrapping exhaustion in
    RetryExhaustedError."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = policy.backoff_seconds
            last_error: BaseException | None = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except KalkiError as exc:
                    last_error = exc
                    if not exc.retryable or attempt == policy.max_attempts:
                        raise
                except Exception as exc:  # noqa: BLE001 - unexpected errors are retried too, up to the limit
                    last_error = exc
                    if attempt == policy.max_attempts:
                        raise RetryExhaustedError(operation_name, attempt, last_error) from exc
                await asyncio.sleep(delay)
                delay *= policy.backoff_multiplier
            raise RetryExhaustedError(operation_name, policy.max_attempts, last_error)

        return wrapper

    return decorator
'''


def content_core_registry() -> str:
    return '''"""
Kalki Nexus - Plugin Registry (Agent Auto-Discovery)

Drop a new `agents/<name>_agent.py` defining a BaseAgent subclass and it is
picked up automatically - graph.py and the Supervisor never need to import
it by name. This is what "add an agent by dropping a file into agents/"
means in practice: `discover_agents()` walks the package, imports every
module, and collects every concrete BaseAgent subclass it finds.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import lru_cache
from typing import Dict, Type

import agents as agents_package
from core.base_agent import BaseAgent


@lru_cache(maxsize=1)
def discover_agents() -> Dict[str, Type[BaseAgent]]:
    """Import every module under agents/ and return {agent.name: AgentClass}
    for every concrete BaseAgent subclass found, excluding the Supervisor
    and Aggregator (which are orchestration nodes, not routable specialists)."""
    registry: Dict[str, Type[BaseAgent]] = {}
    skip_modules = {"supervisor", "aggregator"}

    for module_info in pkgutil.iter_modules(agents_package.__path__):
        if module_info.name in skip_modules or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"agents.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseAgent)
                and obj is not BaseAgent
                and obj.__module__ == module.__name__
                and getattr(obj, "name", None)
            ):
                registry[obj.name] = obj

    return registry


def clear_agent_cache() -> None:
    """Invalidate the discovery cache (used by tests and hot-reload tooling)."""
    discover_agents.cache_clear()
'''


def content_core_channel_adapter() -> str:
    return '''"""
Kalki Nexus - Channel Adapter Protocol

Any chat surface (Discord today; Telegram, Slack, WhatsApp, a web UI, or
voice tomorrow) implements this Protocol and calls graph.py the same way
discord/bot.py does. The LangGraph graph itself has no knowledge of Discord
or any other transport - it only knows KalkiState.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class ChannelAdapter(Protocol):
    """The minimum a chat surface needs to implement to front the Kalki Nexus graph."""

    async def send(self, text: str) -> None:
        """Send a complete message to the user/channel."""
        ...

    async def stream(self, chunks: AsyncIterator[str]) -> None:
        """Stream a response incrementally (e.g. progressive message edits)."""
        ...

    async def typing(self) -> None:
        """Signal that a response is being generated (typing indicator equivalent)."""
        ...
'''


def content_app_py() -> str:
    return '''"""
Kalki Nexus - Application Entrypoint

Boots Rich-powered logging + LangSmith tracing, loads configuration, and
either runs a sample graph invocation, renders the graph structure, or
launches the Discord bot.
"""
from __future__ import annotations

import asyncio
import sys

from config import get_settings
from core.observability import configure_logging, get_logger, render_graph_mermaid, setup_langsmith
from graph import compiled_graph, invoke

logger = get_logger("kalki.app")


async def run_example() -> None:
    """Run a single example request through the Kalki Nexus graph."""
    result = await invoke("Write a Python script that backtests a VWAP mean reversion strategy.")
    logger.info("Final answer: %s", result.get("final_answer"))


def render_graph() -> None:
    """Compile the graph and write graph.mmd (and graph.png, if renderable) without invoking it."""
    app = compiled_graph()
    path = render_graph_mermaid(app)
    if path:
        logger.info("Graph structure written to %s", path)
    else:
        logger.warning("Could not render the graph (see warning above).")


def main() -> None:
    """CLI entrypoint.

    --discord       launch the Discord bot instead of the example run
    --render-graph  write graph.mmd/graph.png and exit
    """
    settings = get_settings()
    configure_logging(settings)
    if setup_langsmith(settings):
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)

    if "--discord" in sys.argv:
        from discord.bot import run as run_discord_bot  # local import: only needed for this path

        run_discord_bot()
    elif "--render-graph" in sys.argv:
        render_graph()
    else:
        asyncio.run(run_example())


if __name__ == "__main__":
    main()
'''


def content_graph_py() -> str:
    return '''"""
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

from langgraph.checkpoint.memory import MemorySaver
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


def compiled_graph(checkpointer: Optional[Any] = None):
    """Return a compiled, ready-to-invoke Kalki Nexus graph.

    A checkpointer is required for the human-approval interrupt() to work
    across resumed runs; defaults to an in-memory saver so the scaffold runs
    out of the box. Swap in a persistent checkpointer (Postgres/SQLite/Redis)
    for a real deployment.
    """
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer or MemorySaver())


async def invoke(
    user_input: str,
    discord_channel: Optional[str] = None,
    attached_files: Optional[List[str]] = None,
    requested_tools: Optional[List[str]] = None,
    thread_id: str = "default",
) -> KalkiState:
    """Example entrypoint: run a single request through the graph end to end."""
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
    run_config = {"configurable": {"thread_id": thread_id}}
    return await app.ainvoke(initial_state, config=run_config)
'''


# ============================================================================
# Section 3: agents/
# ============================================================================

def content_agents_init() -> str:
    return '''"""Kalki Nexus agents package.

Every module here (except supervisor.py and aggregator.py, which are
orchestration nodes rather than routable specialists) is auto-discovered by
core.registry.discover_agents(). Add a new specialist by dropping a
BaseAgent subclass in a new module here - nothing else needs to change.
"""
'''


def content_agent_supervisor() -> str:
    return '''"""
Kalki Nexus - Supervisor

An LLM-powered router with a Pydantic structured-output schema
(`RouteDecision`). Inspects the user message, the Discord channel, attached
files, requested tools, and prior state, then decides which specialist
agent(s) should run.

The Discord channel contributes a *hint*, not a rule: CHANNEL_HINTS maps a
channel name to the agent it should bias the Supervisor toward, and that
hint is folded into the routing prompt rather than short-circuiting the LLM
call - the Supervisor can and does override it (e.g. someone asking a
research question in #docker still routes to the Research Agent).

If the structured LLM call is unavailable (no API key, offline dev/tests),
`heuristic_routes()` is used as a deterministic fallback so the graph always
degrades gracefully instead of failing closed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.base_agent import BaseAgent
from core.observability import get_logger
from core.registry import discover_agents

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "supervisor.md"

logger = get_logger("kalki.agents.supervisor")

# Discord channel name -> agent it should bias routing toward. "Bias", not
# "force": see module docstring.
CHANNEL_HINTS: Dict[str, str] = {
    "vajra-python": "python_agent",
    "python": "python_agent",
    "docker": "docker_agent",
    "github": "github_agent",
    "research": "research_agent",
    "quant": "quant_agent",
    "automation": "automation_agent",
    "mcp": "mcp_agent",
}

# Keyword hints used for the deterministic fallback router. This is
# intentionally kept in sync with, but independent of, the LLM path so the
# graph still works with zero configured API keys.
_KEYWORDS: Dict[str, List[str]] = {
    "python_agent": ["python", "script", "bug", "traceback", "refactor"],
    "docker_agent": ["docker", "container", "compose", "image", "dockerfile"],
    "github_agent": ["github", "pull request", "pr ", "repo", "commit", "issue"],
    "research_agent": ["research", "paper", "summarize", "compare", "sources"],
    "quant_agent": ["quant", "backtest", "strategy", "pnl", "sharpe", "risk"],
    "automation_agent": ["automate", "schedule", "workflow", "cron"],
    "mcp_agent": ["mcp", "connector", "tool call"],
}


class RouteDecision(BaseModel):
    """Structured output schema the Supervisor's LLM call is constrained to."""

    agents: List[str] = Field(..., description="Ordered list of specialist agent names that should run.")
    reasoning: str = Field(..., description="A short explanation of why these agent(s) were chosen.")
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="Confidence in this routing decision, 0-1.")


def normalize_channel(discord_channel: Optional[str]) -> Optional[str]:
    """Strip a leading '#' and lowercase a Discord channel name for lookup in CHANNEL_HINTS."""
    if not discord_channel:
        return None
    return discord_channel.lstrip("#").strip().lower()


def heuristic_routes(
    user_input: str,
    attached_files: List[str],
    requested_tools: List[str],
    channel_hint: Optional[str] = None,
) -> List[str]:
    """Deterministic keyword-based routing, used when the LLM path is unavailable.

    The channel hint is appended only if nothing else matched, matching the
    "hint, not a rule" contract described in the module docstring.
    """
    text = user_input.lower()
    selected = [
        agent for agent, keywords in _KEYWORDS.items() if any(keyword in text for keyword in keywords)
    ]

    for tool in requested_tools:
        tool_lower = tool.lower()
        for agent, keywords in _KEYWORDS.items():
            if agent in selected:
                continue
            if any(keyword in tool_lower for keyword in keywords):
                selected.append(agent)

    if attached_files and "python_agent" not in selected:
        selected.append("python_agent")

    if not selected and channel_hint:
        selected.append(channel_hint)

    return selected or ["python_agent"]


class SupervisorAgent(BaseAgent):
    """LangGraph node: populates state["route"] via an LLM structured-output call."""

    name = "supervisor"
    description = "Routes each request to one or more specialist agents."
    prompt_file = "supervisor.md"

    def load_prompt(self) -> str:
        return PROMPT_PATH.read_text(encoding="utf-8")

    def _build_routing_prompt(self, state: Dict[str, Any], channel_hint: Optional[str], valid_agents: List[str]) -> str:
        return (
            f"{self.load_prompt()}\\n\\n"
            f"Available agents: {', '.join(valid_agents)}\\n"
            f"Discord channel: {state.get('discord_channel') or 'n/a'}"
            f"{f' (hint: prefer {channel_hint}, but only if the message actually fits)' if channel_hint else ''}\\n"
            f"Attached files: {state.get('attached_files') or 'none'}\\n"
            f"Requested tools: {state.get('requested_tools') or 'none'}\\n"
            f"Prior route (if any): {state.get('route') or 'none'}\\n\\n"
            f"User message:\\n{state.get('user_input', '')}"
        )

    async def decide(self, state: Dict[str, Any]) -> RouteDecision:
        valid_agents = sorted(name for name in discover_agents() if name != "fallback_agent")
        channel_hint = CHANNEL_HINTS.get(normalize_channel(state.get("discord_channel")))

        try:
            llm = self.settings.build_chat_model(temperature=0.0).with_structured_output(RouteDecision)
            prompt = self._build_routing_prompt(state, channel_hint, valid_agents)
            decision: RouteDecision = await llm.ainvoke(prompt)
            decision.agents = [agent for agent in decision.agents if agent in valid_agents]
            if not decision.agents:
                decision.agents = heuristic_routes(
                    state.get("user_input", ""), state.get("attached_files", []),
                    state.get("requested_tools", []), channel_hint,
                )
            return decision
        except Exception as exc:  # noqa: BLE001 - any LLM/config failure falls back to the heuristic router
            logger.warning("LLM routing unavailable (%s); falling back to heuristic_routes().", exc)
            routes = heuristic_routes(
                state.get("user_input", ""), state.get("attached_files", []),
                state.get("requested_tools", []), channel_hint,
            )
            return RouteDecision(agents=routes, reasoning="heuristic fallback (LLM routing unavailable)", confidence=0.4)

    async def run(self, state: Dict[str, Any]):
        # SupervisorAgent does not produce an AgentResult; it overrides __call__ instead.
        raise NotImplementedError

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = await self.decide(state)
        self.logger.info("route=%s confidence=%.2f reasoning=%s", decision.agents, decision.confidence, decision.reasoning)
        return {
            "route": decision.agents,
            "route_reasoning": decision.reasoning,
            "route_confidence": decision.confidence,
        }
'''


def content_agent_aggregator() -> str:
    return '''"""
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
        return "\\n\\n".join(f"## {name}\\n{result.answer}" for name, result in successful.items())

    parts = "\\n\\n".join(f"### {name}\\n{result.answer}" for name, result in successful.items())
    messages = [
        SystemMessage(content=_SYNTHESIS_PROMPT),
        HumanMessage(content=f"Original request:\\n{user_input}\\n\\nAgent outputs:\\n{parts}"),
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
'''


def content_agent_fallback() -> str:
    return '''"""
Kalki Nexus - Fallback Agent

The last resort once the graph's error_node has exhausted its bounded
retries (see graph.py::route_after_error). Never raises: it always returns
a graceful, honest degraded response rather than letting the run crash, and
surfaces whatever partial agent_results exist so the user isn't left with
nothing.
"""
from __future__ import annotations

from typing import Any, Dict

from core.base_agent import AgentResult, BaseAgent


class FallbackAgent(BaseAgent):
    """Graph-level last resort: never raises, always returns a usable AgentResult."""

    name = "fallback_agent"
    description = "Produces a graceful degraded response when every retry has been exhausted."
    retry_policy = BaseAgent.retry_policy  # no further retries: this IS the end of the retry chain

    def load_tools(self):
        return []

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        error = state.get("error") or {}
        partial = {
            name: result.answer
            for name, result in (state.get("agent_results") or {}).items()
            if result.answer
        }

        if partial:
            body = "\\n\\n".join(f"[{name}] {answer}" for name, answer in partial.items())
            answer = (
                "One or more agents hit an error and retries were exhausted, but here is "
                f"the partial progress that was made before that happened:\\n\\n{body}"
            )
        else:
            answer = (
                "Sorry - this request could not be completed. "
                f"Last error: {error.get('message', 'unknown error')}. "
                "Please try rephrasing the request or check the service configuration."
            )

        return AgentResult(
            agent=self.name,
            answer=answer,
            confidence=0.2,
            metadata={"fallback": True, "original_error": error},
        )

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.run(state)
        return {
            "agent_results": {self.name: result},
            "final_answer": result.answer,
            "error": None,
        }
'''


def content_agent_python() -> str:
    return '''"""
Kalki Nexus - Python Agent

Handles Python code generation, debugging, and refactoring requests.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent


class PythonAgent(BaseAgent):
    name = "python_agent"
    description = "Writes, debugs, and refactors Python code."
    channel_hints: ClassVar[List[str]] = ["python", "vajra-python"]
    default_tool_categories: ClassVar[List[str]] = ["filesystem"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)
'''


def content_agent_docker() -> str:
    return '''"""
Kalki Nexus - Docker Agent

Handles containerization requests: writing Dockerfiles, docker-compose
configs, and diagnosing container build/runtime issues.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent


class DockerAgent(BaseAgent):
    name = "docker_agent"
    description = "Writes and debugs Docker/Compose configuration."
    channel_hints: ClassVar[List[str]] = ["docker"]
    default_tool_categories: ClassVar[List[str]] = ["docker"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)
'''


def content_agent_github() -> str:
    return '''"""
Kalki Nexus - GitHub Agent

Drafts and manages issues, pull requests, and repository content via the
GitHub tools. Flags destructive actions for human approval instead of
performing them unattended.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent

_DESTRUCTIVE_HINTS = ("force-push", "force push", "delete branch", "delete repo", "merge pull request", "merge pr")


class GithubAgent(BaseAgent):
    name = "github_agent"
    description = "Drafts and manages GitHub issues, pull requests, and repository content."
    channel_hints: ClassVar[List[str]] = ["github"]
    default_tool_categories: ClassVar[List[str]] = ["github"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        result = await self._default_llm_run(state)
        text = state.get("user_input", "").lower()
        if any(hint in text for hint in _DESTRUCTIVE_HINTS):
            result.metadata["requires_approval"] = True
            result.metadata["approval_reason"] = "Request appears to involve a destructive GitHub operation."
        return result
'''


def content_agent_mcp() -> str:
    return '''"""
Kalki Nexus - MCP Agent

Bridges requests to Model Context Protocol (MCP) servers and tools. Unlike
the other specialists, its tools are discovered dynamically at runtime from
the MCPRegistry rather than bound statically via ToolLoader - so it
overrides run() directly instead of relying on _default_llm_run's
synchronous load_tools() path.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import AgentResult, BaseAgent
from mcp.registry import get_mcp_registry


class McpAgent(BaseAgent):
    name = "mcp_agent"
    description = "Discovers and invokes tools exposed by connected MCP servers."
    channel_hints = ["mcp"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        registry = get_mcp_registry()
        capabilities = await registry.discover_tools()
        mcp_tools = await registry.load_tools()

        llm = self.settings.build_chat_model(temperature=self.temperature, tools=mcp_tools or None)
        capability_summary = "\\n".join(
            f"- {cap.server_name}.{cap.tool_name}: {cap.description}" for cap in capabilities
        ) or "(no MCP servers currently registered)"

        messages = [
            SystemMessage(content=f"{self.load_prompt()}\\n\\nKnown MCP tool capabilities:\\n{capability_summary}"),
            HumanMessage(content=state.get("user_input", "")),
        ]
        response = await llm.ainvoke(messages)
        tool_calls = [{"name": c.get("name"), "args": c.get("args")} for c in (response.tool_calls or [])]

        return AgentResult(
            agent=self.name,
            answer=str(response.content),
            tool_calls=tool_calls,
            metadata={"mcp_servers": sorted({cap.server_name for cap in capabilities})},
        )
'''


def content_agent_automation() -> str:
    return '''"""
Kalki Nexus - Automation Agent

Handles scheduling, workflow orchestration, and repetitive task automation
across the local filesystem and terminal. Flags destructive shell commands
for human approval rather than running them unattended.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent

_DESTRUCTIVE_HINTS = ("rm -rf", "format ", "drop table", "del /f", ":(){ :|:& };:")


class AutomationAgent(BaseAgent):
    name = "automation_agent"
    description = "Designs and executes repeatable filesystem/terminal workflows."
    channel_hints: ClassVar[List[str]] = ["automation"]
    default_tool_categories: ClassVar[List[str]] = ["filesystem", "terminal"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        result = await self._default_llm_run(state)
        text = state.get("user_input", "").lower()
        if any(hint in text for hint in _DESTRUCTIVE_HINTS):
            result.metadata["requires_approval"] = True
            result.metadata["approval_reason"] = "Request appears to involve a destructive shell command."
        return result
'''


def content_agent_research() -> str:
    return '''"""
Kalki Nexus - Research Agent

Gathers, summarizes, and cites information from the web to support other
agents (e.g. feeding the Quant Agent with instrument or market research).
Every finding is cached into this agent's namespaced Agent Memory.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from core.base_agent import AgentResult, BaseAgent


class ResearchAgent(BaseAgent):
    name = "research_agent"
    description = "Gathers and summarizes information from the web, with citations."
    channel_hints: ClassVar[List[str]] = ["research"]
    default_tool_categories: ClassVar[List[str]] = ["web", "browser"]
    temperature = 0.2

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)

    async def post_process(self, state: Dict[str, Any], result: AgentResult) -> AgentResult:
        memory = self.load_memory()
        topic = state.get("user_input", "")[:80]
        await memory.set(f"source:{topic}", {"summary": result.answer[:500]})
        return result
'''


def content_agent_quant() -> str:
    return '''"""
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
    default_tool_categories: ClassVar[List[str]] = []
    temperature = 0.1

    async def run(self, state: Dict[str, Any]) -> AgentResult:
        return await self._default_llm_run(state)

    async def post_process(self, state: Dict[str, Any], result: AgentResult) -> AgentResult:
        memory = self.load_memory()
        strategy_name = state.get("user_input", "")[:40] or "unnamed_strategy"
        await memory.set(f"backtest:{strategy_name}", {"raw_response": result.answer[:500]})
        return result
'''


# ============================================================================
# Section 4: prompts/
# ============================================================================

def content_prompt_supervisor() -> str:
    return '''# Supervisor Agent

You are the Supervisor for Kalki Nexus, a multi-agent AI Operating System.

## Responsibilities
- Read the incoming user request, the Discord channel it arrived on (if any,
  along with a channel hint), any attached files, any explicitly requested
  tools, and the prior routing state.
- Decide which specialist agent, or combination of agents, should handle the
  request, choosing only from the "Available agents" list you are given.
- Prefer routing to multiple agents when a request spans domains, for
  example "write a Python script and containerize it" routes to
  `python_agent` + `docker_agent`.
- Treat the channel hint as a bias, not a rule: only follow it if the
  message content actually fits that agent's domain.

## Constraints
- Do not attempt to answer the user's request yourself; only decide routing.
- If the request is ambiguous, default to the agent whose domain is
  mentioned first in the message.
- Keep routing decisions deterministic and explainable - your `reasoning`
  field should name the specific words/context that drove the decision.

## Output
Respond with the structured `RouteDecision` schema you have been bound to:
`agents` (ordered list of agent names to run), `reasoning` (why), and
`confidence` (0-1).
'''


def content_prompt_python() -> str:
    return '''# Python Agent

You are the Python specialist for Kalki Nexus.

## Responsibilities
- Write, debug, and refactor Python code.
- Diagnose tracebacks and propose minimal, correct fixes.
- Follow PEP 8, use type hints, and prefer standard library solutions before
  reaching for third-party dependencies.

## Constraints
- Never fabricate library APIs; if unsure a function exists, say so.
- Include a short explanation alongside any code you produce.
- Flag any security-sensitive code (subprocess, eval, pickle, etc.) explicitly.

## Output
Return runnable code blocks plus a concise explanation of the approach.
'''


def content_prompt_docker() -> str:
    return '''# Docker Agent

You are the Docker and containerization specialist for Kalki Nexus.

## Responsibilities
- Write Dockerfiles, docker-compose files, and multi-stage build configs.
- Diagnose container build and runtime failures from logs.
- Recommend image size, caching, and security best practices.

## Constraints
- Prefer official, minimal base images (e.g. `python:3.12-slim`).
- Never hardcode secrets into images; use build args or runtime env vars.
- Call out any privileged or `--network=host` requirements explicitly.

## Output
Return complete, ready-to-build Docker configuration plus a short rationale.
'''


def content_prompt_github() -> str:
    return '''# GitHub Agent

You are the GitHub specialist for Kalki Nexus.

## Responsibilities
- Draft and manage issues, pull requests, and repository content.
- Summarize diffs and review changes for correctness and style.
- Use the GitHub tools available to you rather than guessing repository state.

## Constraints
- Never fabricate PR numbers, commit SHAs, or file contents; look them up.
- Keep issue and PR descriptions concise and action-oriented.
- Flag destructive operations (force-push, branch deletion, merging) - these
  are routed to a human approval step; describe the action clearly so the
  reviewer can decide.

## Output
Return the action taken (or proposed) plus links/identifiers for the result.
'''


def content_prompt_mcp() -> str:
    return '''# MCP Agent

You are the Model Context Protocol (MCP) specialist for Kalki Nexus.

## Responsibilities
- Discover and invoke tools exposed by connected MCP servers.
- Translate natural-language requests into the correct MCP tool calls.
- Summarize MCP tool results back into plain language for the user.

## Constraints
- Only call tools that appear in the "Known MCP tool capabilities" list you
  are given for this run.
- If no MCP server exposes a needed capability, say so rather than guessing.
- Confirm before taking any destructive or irreversible action via MCP.

## Output
Return the tool call made (if any) and a plain-language summary of the result.
'''


def content_prompt_automation() -> str:
    return '''# Automation Agent

You are the automation specialist for Kalki Nexus.

## Responsibilities
- Design and execute repeatable workflows: scheduled jobs, file pipelines,
  and multi-step terminal/filesystem tasks.
- Break multi-step automation requests into an ordered, auditable plan
  before executing.

## Constraints
- Never run destructive shell commands (`rm -rf`, disk formatting, etc.)
  without explicit human approval - these are routed to a human approval
  step automatically; describe the risk clearly.
- Prefer idempotent operations that are safe to re-run.
- Log every step taken so runs can be audited after the fact.

## Output
Return the plan, the steps executed, and their results.
'''


def content_prompt_research() -> str:
    return '''# Research Agent

You are the research specialist for Kalki Nexus.

## Responsibilities
- Gather information from the web and other sources to answer a question
  or support another agent's work (e.g. market or instrument research for
  the Quant Agent).
- Summarize findings accurately and cite sources.

## Constraints
- Never invent sources, statistics, or quotes.
- Prefer primary sources over aggregators.
- Distinguish clearly between established fact and inference.

## Output
Return a concise summary plus a list of sources used.
'''


def content_prompt_quant() -> str:
    return '''# Quant Agent

You are the quantitative finance specialist for Kalki Nexus.

## Responsibilities
- Design, evaluate, and explain trading strategies, backtests, and risk
  metrics (Sharpe, drawdown, VaR, DV01, etc.).
- Keep a strict separation between training/in-sample and holdout/out-of-
  sample data when discussing or generating backtest code.

## Constraints
- Never present backtested or simulated performance as a guarantee of
  future results.
- Show assumptions (risk-free rate, transaction costs, slippage) explicitly
  rather than leaving them implicit.
- Flag look-ahead bias, survivorship bias, and overfitting risks proactively.

## Output
Return the strategy logic or metrics requested, plus the assumptions used.
'''


# ============================================================================
# Section 5: tools/
# ============================================================================

def content_tools_init() -> str:
    return '''"""Kalki Nexus tools package: BaseTool implementations plus the ToolRegistry.

Importing this package eagerly imports every tool module so their
`@ToolRegistry.register()` decorators run and the registry is fully
populated before any agent calls ToolLoader.load_categories(...).
"""
from __future__ import annotations

from tools import (  # noqa: F401 - imported for registration side effects
    browser_tools,
    discord_tools,
    docker_tools,
    filesystem_tools,
    github_tools,
    terminal_tools,
    web_tools,
)
'''


def content_tool_registry() -> str:
    return '''"""
Kalki Nexus - Tool Registry, Loader, and Resolver

Tools register themselves once (via the @ToolRegistry.register() class
decorator) instead of every agent maintaining its own manual
`llm.bind_tools([...])` list. Agents request tools by *category*
(`ToolLoader.load_categories`) or by explicit name
(`ToolResolver.resolve_requested`, used for the `requested_tools` the
Supervisor was handed).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from core.base_tool import BaseTool
from core.permissions import SecurityContext


class ToolRegistry:
    """Global registry of every BaseTool subclass, keyed by tool name."""

    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls):
        """Class decorator: `@ToolRegistry.register()` above a BaseTool subclass."""

        def decorator(tool_cls: Type[BaseTool]) -> Type[BaseTool]:
            cls._tools[tool_cls.name] = tool_cls
            return tool_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        tool_cls = cls._tools.get(name)
        return tool_cls() if tool_cls else None

    @classmethod
    def by_category(cls, category: str) -> List[BaseTool]:
        return [tool_cls() for tool_cls in cls._tools.values() if tool_cls.category == category]

    @classmethod
    def all(cls) -> List[BaseTool]:
        return [tool_cls() for tool_cls in cls._tools.values()]

    @classmethod
    def categories(cls) -> List[str]:
        return sorted({tool_cls.category for tool_cls in cls._tools.values()})


class ToolLoader:
    """Loads every registered tool in a set of categories as LangChain tools."""

    @staticmethod
    def load_categories(categories: List[str], security_context: Optional[SecurityContext] = None) -> List[Any]:
        tools: List[Any] = []
        for category in categories:
            for tool in ToolRegistry.by_category(category):
                tools.append(tool.to_langchain_tool(security_context))
        return tools


class ToolResolver:
    """Resolves explicit tool names (e.g. state["requested_tools"]) to LangChain tools."""

    @staticmethod
    def resolve_requested(names: List[str], security_context: Optional[SecurityContext] = None) -> List[Any]:
        resolved: List[Any] = []
        for name in names:
            tool = ToolRegistry.get(name)
            if tool is not None:
                resolved.append(tool.to_langchain_tool(security_context))
        return resolved
'''


def content_tool_docker() -> str:
    return '''"""
Kalki Nexus - Docker Tools

BaseTool wrappers around common Docker operations, registered under the
"docker" category. Wire the TODOs up to the `docker` CLI (via subprocess)
or the Docker SDK for Python (`docker-py`) before using these in production.
"""
from __future__ import annotations

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry


@ToolRegistry.register()
class DockerPsTool(BaseTool):
    name = "docker_ps"
    description = "List running Docker containers."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self) -> str:
        # TODO: shell out to `docker ps --format json` or use docker-py's client.containers.list()
        raise NotImplementedError("docker_ps: wire up the Docker CLI or SDK call.")


@ToolRegistry.register()
class DockerRunTool(BaseTool):
    name = "docker_run"
    description = "Run a Docker container from an image, optionally executing a command."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self, image: str, command: str = "") -> str:
        # TODO: shell out to `docker run {image} {command}` or use docker-py's client.containers.run()
        raise NotImplementedError("docker_run: wire up the Docker CLI or SDK call.")


@ToolRegistry.register()
class DockerBuildTool(BaseTool):
    name = "docker_build"
    description = "Build a Docker image from a Dockerfile and tag it."
    category = "docker"
    required_permissions = [Permission.DOCKER, Permission.FS_READ]

    async def run(self, dockerfile_path: str, tag: str) -> str:
        # TODO: shell out to `docker build -f {dockerfile_path} -t {tag} .` or use docker-py
        raise NotImplementedError("docker_build: wire up the Docker CLI or SDK call.")


@ToolRegistry.register()
class DockerLogsTool(BaseTool):
    name = "docker_logs"
    description = "Fetch the last N lines of logs for a container."
    category = "docker"
    required_permissions = [Permission.DOCKER]

    async def run(self, container_id: str, tail: int = 100) -> str:
        # TODO: shell out to `docker logs --tail {tail} {container_id}` or use docker-py
        raise NotImplementedError("docker_logs: wire up the Docker CLI or SDK call.")
'''


def content_tool_filesystem() -> str:
    return '''"""
Kalki Nexus - Filesystem Tools

Async filesystem access for agents, rooted at the current working directory,
registered under the "filesystem" category. Read/write/list each declare
their own permission so an agent granted only FS_READ cannot write files.
"""
from __future__ import annotations

from pathlib import Path

import aiofiles

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

# TODO: sandbox this to an allow-listed root directory before exposing it to
# agents in a production deployment.
WORKSPACE_ROOT = Path.cwd()


@ToolRegistry.register()
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read and return the text contents of a file relative to the workspace root."
    category = "filesystem"
    required_permissions = [Permission.FS_READ]

    async def run(self, path: str) -> str:
        target = WORKSPACE_ROOT / path
        async with aiofiles.open(target, mode="r", encoding="utf-8") as handle:
            return await handle.read()


@ToolRegistry.register()
class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file relative to the workspace root, creating parent dirs."
    category = "filesystem"
    required_permissions = [Permission.FS_WRITE]

    async def run(self, path: str, content: str) -> str:
        target = WORKSPACE_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, mode="w", encoding="utf-8") as handle:
            await handle.write(content)
        return f"wrote {len(content)} bytes to {target}"


@ToolRegistry.register()
class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List file and directory names under a path, relative to the workspace root."
    category = "filesystem"
    required_permissions = [Permission.FS_READ]

    async def run(self, path: str = ".") -> list:
        target = WORKSPACE_ROOT / path
        return sorted(item.name for item in target.iterdir())


@ToolRegistry.register()
class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Delete a file relative to the workspace root."
    category = "filesystem"
    required_permissions = [Permission.FS_DELETE]

    async def run(self, path: str) -> str:
        target = WORKSPACE_ROOT / path
        target.unlink(missing_ok=True)
        return f"deleted {target}"
'''


def content_tool_github() -> str:
    return '''"""
Kalki Nexus - GitHub Tools

BaseTool wrappers around the GitHub REST API via PyGithub, registered under
the "github" category. Requires GITHUB_TOKEN to be set in the environment.
"""
from __future__ import annotations

from github import Github

from config import get_settings
from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry


def _client() -> Github:
    settings = get_settings()
    if not settings.github_token:
        raise RuntimeError("GITHUB_TOKEN is not set. Copy .env.example to .env and fill it in.")
    return Github(settings.github_token)


@ToolRegistry.register()
class ListOpenPullRequestsTool(BaseTool):
    name = "list_open_pull_requests"
    description = "List the titles of open pull requests for owner/repo."
    category = "github"
    required_permissions = [Permission.GITHUB, Permission.NETWORK]

    async def run(self, repo_full_name: str) -> list:
        repo = _client().get_repo(repo_full_name)
        return [pr.title for pr in repo.get_pulls(state="open")]


@ToolRegistry.register()
class CreateIssueTool(BaseTool):
    name = "create_issue"
    description = "Create an issue in owner/repo and return its URL."
    category = "github"
    required_permissions = [Permission.GITHUB, Permission.NETWORK]

    async def run(self, repo_full_name: str, title: str, body: str = "") -> str:
        repo = _client().get_repo(repo_full_name)
        issue = repo.create_issue(title=title, body=body)
        return issue.html_url


@ToolRegistry.register()
class GetFileContentsTool(BaseTool):
    name = "get_file_contents"
    description = "Fetch the decoded text contents of a file at a path in owner/repo."
    category = "github"
    required_permissions = [Permission.GITHUB, Permission.NETWORK]

    async def run(self, repo_full_name: str, path: str, ref: str = "main") -> str:
        repo = _client().get_repo(repo_full_name)
        content_file = repo.get_contents(path, ref=ref)
        return content_file.decoded_content.decode("utf-8")


# TODO: add tools for merging PRs, requesting reviews, and posting review
# comments - route them through GithubAgent's destructive-action detection
# (agents/github_agent.py) so they require human approval.
'''


def content_tool_browser() -> str:
    return '''"""
Kalki Nexus - Browser Tools

Placeholder BaseTool wrappers for headless-browser automation (e.g.
Playwright), registered under the "browser" category. Install `playwright`
and run `playwright install` before wiring these up.
"""
from __future__ import annotations

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry


@ToolRegistry.register()
class OpenPageTool(BaseTool):
    name = "open_page"
    description = "Open a URL in a headless browser and return the rendered page title."
    category = "browser"
    required_permissions = [Permission.BROWSER, Permission.NETWORK]

    async def run(self, url: str) -> str:
        # TODO: launch Playwright (async_playwright), navigate to `url`, return page.title()
        raise NotImplementedError("open_page: wire up a headless browser (e.g. Playwright).")


@ToolRegistry.register()
class ClickSelectorTool(BaseTool):
    name = "click_selector"
    description = "Navigate to a URL and click the element matching a CSS selector."
    category = "browser"
    required_permissions = [Permission.BROWSER, Permission.NETWORK]

    async def run(self, url: str, selector: str) -> str:
        # TODO: launch Playwright, page.goto(url), then page.click(selector)
        raise NotImplementedError("click_selector: wire up a headless browser (e.g. Playwright).")


@ToolRegistry.register()
class ExtractTextTool(BaseTool):
    name = "extract_text"
    description = "Navigate to a URL and return the text content of a CSS selector."
    category = "browser"
    required_permissions = [Permission.BROWSER, Permission.NETWORK]

    async def run(self, url: str, selector: str = "body") -> str:
        # TODO: launch Playwright, page.goto(url), return page.inner_text(selector)
        raise NotImplementedError("extract_text: wire up a headless browser (e.g. Playwright).")
'''


def content_tool_terminal() -> str:
    return '''"""
Kalki Nexus - Terminal Tools

Real (but unsandboxed) shell command execution, registered under the
"terminal" category. Harden this before exposing it to untrusted input:
allow-list commands, drop privileges, or run inside a disposable container.
"""
from __future__ import annotations

import asyncio

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

# TODO: allow-list permitted commands and/or run inside a disposable container
# before exposing this tool to untrusted agents or users.
DEFAULT_TIMEOUT_SECONDS = 30


@ToolRegistry.register()
class RunShellCommandTool(BaseTool):
    name = "run_shell_command"
    description = "Run a shell command and return its combined stdout/stderr output."
    category = "terminal"
    required_permissions = [Permission.TERMINAL]

    async def run(self, command: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return f"command timed out after {timeout}s: {command}"
        return stdout.decode("utf-8", errors="replace")
'''


def content_tool_discord() -> str:
    return '''"""
Kalki Nexus - Discord Tools

BaseTool wrapper for sending messages back to Discord from an agent,
registered under the "discord" category. Expects a running Discord client
instance to be attached at startup via bind_client().
"""
from __future__ import annotations

from typing import Optional

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

_client: Optional[object] = None


def bind_client(client: object) -> None:
    """Attach the running Discord client so these tools can send messages through it.
    Call this from discord/bot.py's on_ready handler."""
    global _client
    _client = client


@ToolRegistry.register()
class SendChannelMessageTool(BaseTool):
    name = "send_channel_message"
    description = "Send a message to a Discord channel by its numeric ID."
    category = "discord"
    required_permissions = [Permission.DISCORD]

    async def run(self, channel_id: int, content: str) -> str:
        if _client is None:
            raise RuntimeError("Discord client is not bound. Call bind_client() from discord/bot.py at startup.")
        channel = _client.get_channel(channel_id)  # type: ignore[attr-defined]
        if channel is None:
            raise ValueError(f"Discord channel {channel_id} not found or not cached.")
        await channel.send(content)
        return f"sent {len(content)} chars to channel {channel_id}"
'''


def content_tool_web() -> str:
    return '''"""
Kalki Nexus - Web Tools

BaseTool wrappers for HTTP fetch and web search, registered under the "web"
category. fetch_url is fully implemented; web_search is a stub pending a
search provider (e.g. Tavily, Serper, Bing).
"""
from __future__ import annotations

import httpx

from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry

REQUEST_TIMEOUT_SECONDS = 15.0


@ToolRegistry.register()
class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch a URL over HTTP(S) and return the response body as text."
    category = "web"
    required_permissions = [Permission.NETWORK]

    async def run(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text


@ToolRegistry.register()
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for a query and return up to max_results result snippets."
    category = "web"
    required_permissions = [Permission.NETWORK]

    async def run(self, query: str, max_results: int = 5) -> list:
        # TODO: call a search provider such as Tavily, Serper, or Bing Web Search here.
        raise NotImplementedError("web_search: wire up a search provider (e.g. Tavily or Serper).")
'''


# ============================================================================
# Section 6: mcp/ - Model Context Protocol registry
# ============================================================================

def content_mcp_init() -> str:
    return '''"""Kalki Nexus mcp package: MCP server registration, tool discovery/caching, and hot reload."""
'''


def content_mcp_registry() -> str:
    return '''"""
Kalki Nexus - MCP Registry

Server registration, dynamic tool discovery, capability caching (TTL-based),
hot reload, and tool resolution/selection for Model Context Protocol
servers. The actual network transport is deliberately isolated in
mcp/client.py behind `discover_server_tools()` / `load_server_tools()` - see
that module for the one real TODO in this subsystem. Everything else here
(the registry, the cache, multi-server support, hot reload, and the
agent-facing API `discover_tools()` / `load_tools()` / `resolve_tool()`) is
real and exercised by agents/mcp_agent.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from core.exceptions import MCPError
from core.observability import get_logger
from mcp.client import discover_server_tools, load_server_tools

logger = get_logger("kalki.mcp.registry")


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: str  # "stdio" | "sse" | "streamable_http"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPToolCapability:
    """A single tool a server exposes, as last discovered."""

    server_name: str
    tool_name: str
    description: str
    input_schema: Dict[str, Any]
    cached_at: float


class MCPRegistry:
    """Registers MCP servers and manages discovery/caching/loading of their tools."""

    def __init__(self, cache_ttl_seconds: float = 300.0) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._servers: Dict[str, MCPServerConfig] = {}
        self._capability_cache: Dict[str, List[MCPToolCapability]] = {}

    def register_server(self, config: MCPServerConfig) -> None:
        self._servers[config.name] = config
        self._capability_cache.pop(config.name, None)  # a re-registered server needs fresh discovery

    def unregister_server(self, name: str) -> None:
        self._servers.pop(name, None)
        self._capability_cache.pop(name, None)

    def servers(self) -> List[MCPServerConfig]:
        return [config for config in self._servers.values() if config.enabled]

    def _cache_is_fresh(self, server_name: str) -> bool:
        cached = self._capability_cache.get(server_name)
        if not cached:
            return False
        return (time.time() - cached[0].cached_at) < self.cache_ttl_seconds

    async def discover_tools(
        self, server_name: Optional[str] = None, force_refresh: bool = False
    ) -> List[MCPToolCapability]:
        """Discover (or return cached) tool capabilities for one server, or all servers."""
        targets = [self._servers[server_name]] if server_name else self.servers()
        all_capabilities: List[MCPToolCapability] = []

        for config in targets:
            if not force_refresh and self._cache_is_fresh(config.name):
                all_capabilities.extend(self._capability_cache[config.name])
                continue
            try:
                raw_tools = await discover_server_tools(config)
            except Exception as exc:  # noqa: BLE001 - one server's outage shouldn't break the others
                logger.warning("MCP discovery failed for server '%s': %s", config.name, exc)
                raise MCPError(config.name, f"tool discovery failed: {exc}") from exc

            now = time.time()
            capabilities = [
                MCPToolCapability(
                    server_name=config.name,
                    tool_name=raw["name"],
                    description=raw.get("description", ""),
                    input_schema=raw.get("input_schema", {}),
                    cached_at=now,
                )
                for raw in raw_tools
            ]
            self._capability_cache[config.name] = capabilities
            all_capabilities.extend(capabilities)

        return all_capabilities

    async def load_tools(self, server_names: Optional[List[str]] = None) -> List[Any]:
        """Load live, callable LangChain-compatible tools for the given servers (or all)."""
        targets = [self._servers[name] for name in server_names] if server_names else self.servers()
        tools: List[Any] = []
        for config in targets:
            try:
                tools.extend(await load_server_tools(config))
            except Exception as exc:  # noqa: BLE001 - degrade gracefully per server
                logger.warning("MCP tool loading failed for server '%s': %s", config.name, exc)
        return tools

    async def hot_reload(self) -> List[MCPToolCapability]:
        """Invalidate every cached capability and re-discover from scratch."""
        self._capability_cache.clear()
        return await self.discover_tools(force_refresh=True)

    def resolve_tool(self, tool_name: str) -> Optional[MCPToolCapability]:
        """Find a cached tool capability by exact name, or by `server.tool` qualified name."""
        for capabilities in self._capability_cache.values():
            for capability in capabilities:
                if capability.tool_name == tool_name or f"{capability.server_name}.{capability.tool_name}" == tool_name:
                    return capability
        return None


def default_servers() -> List[MCPServerConfig]:
    """Starter MCP server list. Add real servers here, or call
    MCPRegistry.register_server(...) at application startup."""
    return [
        # Example (disabled by default until a real command/url is supplied):
        # MCPServerConfig(name="filesystem", transport="stdio", command="npx",
        #                  args=["-y", "@modelcontextprotocol/server-filesystem", "."], enabled=False),
    ]


@lru_cache(maxsize=1)
def get_mcp_registry() -> MCPRegistry:
    """Process-wide MCPRegistry singleton, pre-populated with default_servers()."""
    registry = MCPRegistry()
    for config in default_servers():
        registry.register_server(config)
    return registry
'''


def content_mcp_client() -> str:
    return '''"""
Kalki Nexus - MCP Client Transport

This is the one intentionally-placeholder piece of the MCP subsystem: the
actual network call to an MCP server. Everything upstream of this module
(MCPRegistry's server registration, capability caching, hot reload, and
resolution) is real; only the wire call is a TODO.

To make this real, install `langchain-mcp-adapters` and replace the bodies
below with something like:

    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def discover_server_tools(config: MCPServerConfig) -> list[dict]:
        client = MultiServerMCPClient({config.name: _connection_dict(config)})
        async with client.session(config.name) as session:
            response = await session.list_tools()
            return [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in response.tools
            ]

    async def load_server_tools(config: MCPServerConfig) -> list:
        client = MultiServerMCPClient({config.name: _connection_dict(config)})
        async with client.session(config.name) as session:
            return await load_mcp_tools(session)
"""
from __future__ import annotations

from typing import Any, Dict, List

from mcp.registry import MCPServerConfig  # noqa: F401 - re-exported for type hints in real implementations


async def discover_server_tools(config: "MCPServerConfig") -> List[Dict[str, Any]]:
    """Return this server's tool capabilities as plain dicts: {name, description, input_schema}.

    TODO: open a real session against `config` (stdio/sse/streamable_http)
    and call the MCP `tools/list` method. Left empty rather than raising so
    a Kalki Nexus deployment with zero MCP servers configured still runs
    cleanly end to end.
    """
    return []


async def load_server_tools(config: "MCPServerConfig") -> List[Any]:
    """Return live, callable LangChain-compatible tool objects for this server.

    TODO: open a real session against `config` and call
    `langchain_mcp_adapters.tools.load_mcp_tools(session)` (or equivalent).
    """
    return []
'''

FILES = [
    GeneratedFile('README.md', content_readme()),
    GeneratedFile('.gitignore', content_gitignore()),
    GeneratedFile('.env.example', content_env_example()),
    GeneratedFile('requirements.txt', content_requirements()),
    GeneratedFile('pyproject.toml', content_pyproject()),
    GeneratedFile('config.py', content_config_py()),
    GeneratedFile('core/__init__.py', content_core_init()),
    GeneratedFile('core/exceptions.py', content_core_exceptions()),
    GeneratedFile('core/permissions.py', content_core_permissions()),
    GeneratedFile('core/base_tool.py', content_core_base_tool()),
    GeneratedFile('core/base_memory.py', content_core_base_memory()),
    GeneratedFile('core/base_agent.py', content_core_base_agent()),
    GeneratedFile('core/observability.py', content_core_observability()),
    GeneratedFile('core/resilience.py', content_core_resilience()),
    GeneratedFile('core/registry.py', content_core_registry()),
    GeneratedFile('core/channel_adapter.py', content_core_channel_adapter()),
    GeneratedFile('app.py', content_app_py()),
    GeneratedFile('graph.py', content_graph_py()),
    GeneratedFile('agents/__init__.py', content_agents_init()),
    GeneratedFile('agents/supervisor.py', content_agent_supervisor()),
    GeneratedFile('agents/aggregator.py', content_agent_aggregator()),
    GeneratedFile('agents/fallback_agent.py', content_agent_fallback()),
    GeneratedFile('agents/python_agent.py', content_agent_python()),
    GeneratedFile('agents/docker_agent.py', content_agent_docker()),
    GeneratedFile('agents/github_agent.py', content_agent_github()),
    GeneratedFile('agents/mcp_agent.py', content_agent_mcp()),
    GeneratedFile('agents/automation_agent.py', content_agent_automation()),
    GeneratedFile('agents/research_agent.py', content_agent_research()),
    GeneratedFile('agents/quant_agent.py', content_agent_quant()),
    GeneratedFile('prompts/supervisor.md', content_prompt_supervisor()),
    GeneratedFile('prompts/python.md', content_prompt_python()),
    GeneratedFile('prompts/docker.md', content_prompt_docker()),
    GeneratedFile('prompts/github.md', content_prompt_github()),
    GeneratedFile('prompts/mcp.md', content_prompt_mcp()),
    GeneratedFile('prompts/automation.md', content_prompt_automation()),
    GeneratedFile('prompts/research.md', content_prompt_research()),
    GeneratedFile('prompts/quant.md', content_prompt_quant()),
    GeneratedFile('tools/__init__.py', content_tools_init()),
    GeneratedFile('tools/registry.py', content_tool_registry()),
    GeneratedFile('tools/docker_tools.py', content_tool_docker()),
    GeneratedFile('tools/filesystem_tools.py', content_tool_filesystem()),
    GeneratedFile('tools/github_tools.py', content_tool_github()),
    GeneratedFile('tools/browser_tools.py', content_tool_browser()),
    GeneratedFile('tools/terminal_tools.py', content_tool_terminal()),
    GeneratedFile('tools/discord_tools.py', content_tool_discord()),
    GeneratedFile('tools/web_tools.py', content_tool_web()),
    GeneratedFile('mcp/__init__.py', content_mcp_init()),
    GeneratedFile('mcp/registry.py', content_mcp_registry()),
    GeneratedFile('mcp/client.py', content_mcp_client()),
]

def generate_project(root_dir: str, overwrite: bool) -> None:
    root = Path(root_dir).resolve()
    console.print(f"Bootstrapping [bold cyan]{PROJECT_NAME}[/bold cyan] at: {root}")

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Generating files...", total=len(FILES))
        for gf in FILES:
            target_path = root / gf.relative_path
            if target_path.exists() and not overwrite:
                progress.console.print(f"[yellow]Skipping[/yellow] {gf.relative_path} (already exists)")
                progress.advance(task)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(gf.content, encoding="utf-8")
            progress.console.print(f"[green]Wrote[/green] {gf.relative_path}")
            progress.advance(task)
    console.print(f"\n[bold green]Success![/bold green] Generated {len(FILES)} files.")

def main() -> None:
    print("Inside main()")
    parser = argparse.ArgumentParser(description=f"Bootstrap {PROJECT_NAME}")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--root", default=PROJECT_NAME)
    args = parser.parse_args()
    generate_project(args.root, args.overwrite)

print("Script started")
print("FILES:", len(FILES))
if __name__ == '__main__':
    main()
