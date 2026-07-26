# Kalki Nexus

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
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy environment variables and fill them in:
   `cp .env.example .env`

## Environment Variables

| Variable                 | Description                                                      |
|---------------------------|-------------------------------------------------------------------|
| MODEL_PROVIDER            | `openai` \| `anthropic` \| `openrouter` \| `ollama` \| `azure_openai` |
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
| MEMORY_BACKEND            | `sqlite` (default) \| `postgres` \| `redis` \| `qdrant` \| `chroma` |
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
