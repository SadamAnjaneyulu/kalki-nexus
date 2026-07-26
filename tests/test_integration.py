"""
Integration tests for Kalki Nexus end-to-end workflows:
  - RAG Indexer & Retriever pipeline
  - Agent-to-Agent Delegation
  - Discord Message Chunking
"""
import pytest
from core.rag import Indexer, Retriever
from core.delegation import delegate_to
from discord_bot.bot import _chunk_message


@pytest.mark.asyncio
async def test_rag_end_to_end_pipeline(tmp_path):
    db_file = tmp_path / "test_rag.db"
    indexer = Indexer(db_path=db_file)
    retriever = Retriever(db_path=db_file)

    sample_doc = (
        "WorldQuant BRAIN is a quantitative trading platform. "
        "Alpha design requires calculating Volume Weighted Average Price (VWAP) "
        "and evaluating Sharpe and Sortino ratios for risk management."
    )
    chunks_count = await indexer.index("worldquant_docs", sample_doc, source="wq_manual.md")
    assert chunks_count > 0

    results = await retriever.retrieve("worldquant_docs", "Volume Weighted Average Price VWAP")
    assert len(results) > 0
    assert "VWAP" in results[0]["text"]
    assert results[0]["source"] == "wq_manual.md"

    context = await retriever.build_context("worldquant_docs", "Sharpe ratio")
    assert "Sharpe" in context


@pytest.mark.asyncio
async def test_agent_delegation_flow():
    parent_state = {"user_input": "Run backtest", "_caller": "python_agent"}
    # Delegate to research_agent with a sub-question
    res = await delegate_to("fallback_agent", parent_state, "Explain fallback policy")
    assert res.agent == "fallback_agent"
    assert res.answer != ""


def test_discord_message_chunking():
    short_msg = "Hello World"
    assert _chunk_message(short_msg) == ["Hello World"]

    long_msg = "Line 1\n" + ("X" * 1500) + "\nLine 2\n" + ("Y" * 1000)
    chunks = _chunk_message(long_msg, limit=1600)
    assert len(chunks) == 2
    assert chunks[0].startswith("Line 1")
