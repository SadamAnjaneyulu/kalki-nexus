"""
Unit tests for Kalki Nexus Tool Registry and Quantitative Tools.
"""
import pytest
from tools.quant_tools import VWAPCalculatorTool, SharpeRatioTool
from tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_vwap_calculator():
    tool = VWAPCalculatorTool()
    res = await tool.run(prices=[100.0, 102.0, 101.0], volumes=[10, 20, 30])
    assert "vwap" in res
    assert res["total_volume"] == 60
    assert res["data_points"] == 3


@pytest.mark.asyncio
async def test_sharpe_ratio_calculator():
    tool = SharpeRatioTool()
    returns = [0.01, 0.02, -0.005, 0.015, 0.008]
    res = await tool.run(returns=returns)
    assert "sharpe_ratio" in res
    assert "sortino_ratio" in res


def test_tool_registry_quant_category():
    tools = ToolRegistry.by_category("quant")
    tool_names = [t.name for t in tools]
    assert "vwap_calculator" in tool_names
    assert "sharpe_ratio_calculator" in tool_names
