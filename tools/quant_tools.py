"""
Kalki Nexus - Quantitative & WorldQuant Strategy Tools
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from core.base_tool import BaseTool
from tools.registry import ToolRegistry


@ToolRegistry.register()
class VWAPCalculatorTool(BaseTool):
    name = "vwap_calculator"
    description = "Calculates Volume Weighted Average Price (VWAP) given price and volume series."
    category = "quant"

    async def run(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        if not prices or not volumes or len(prices) != len(volumes):
            return {"error": "prices and volumes must be non-empty and equal length lists."}
        total_volume = sum(volumes)
        if total_volume == 0:
            return {"error": "total volume cannot be zero."}
        cumulative_pv = sum(p * v for p, v in zip(prices, volumes))
        vwap = cumulative_pv / total_volume
        return {
            "vwap": round(vwap, 4),
            "total_volume": total_volume,
            "data_points": len(prices),
        }


@ToolRegistry.register()
class SharpeRatioTool(BaseTool):
    name = "sharpe_ratio_calculator"
    description = "Calculates annualized Sharpe Ratio and Sortino Ratio given a list of periodic percentage returns."
    category = "quant"

    async def run(self, returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, Any]:
        if not returns or len(returns) < 2:
            return {"error": "At least 2 return data points required."}
        
        n = len(returns)
        mean_return = sum(returns) / n
        diffs = [r - mean_return for r in returns]
        variance = sum(d ** 2 for d in diffs) / (n - 1)
        std_dev = math.sqrt(variance)

        excess_return = mean_return - (risk_free_rate / periods_per_year)
        sharpe = (excess_return / std_dev) * math.sqrt(periods_per_year) if std_dev > 0 else 0.0

        downside_diffs = [min(0, r - (risk_free_rate / periods_per_year)) for r in returns]
        downside_variance = sum(d ** 2 for d in downside_diffs) / n
        downside_std = math.sqrt(downside_variance)
        sortino = (excess_return / downside_std) * math.sqrt(periods_per_year) if downside_std > 0 else 0.0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "mean_return_pct": round(mean_return * 100, 4),
            "annualized_std_dev_pct": round(std_dev * math.sqrt(periods_per_year) * 100, 4),
        }
