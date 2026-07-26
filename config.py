"""
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
    NVIDIA = "nvidia"


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
    nvidia_api_key: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY", os.getenv("NVIDIA_NIM_API_KEY", ""))
    )
    nvidia_base_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )

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
            ModelProvider.NVIDIA: ("NVIDIA_API_KEY", self.nvidia_api_key),
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

        elif self.provider is ModelProvider.NVIDIA:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model_name,
                api_key=self.nvidia_api_key,
                base_url=self.nvidia_base_url,
                temperature=temperature,
            )
        else:  # pragma: no cover - guarded by the ModelProvider enum
            raise ValueError(f"Unsupported provider: {self.provider}")

        return llm.bind_tools(tools) if tools else llm


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
