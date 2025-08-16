"""
Smart Portfolio Optimizer - Modern Portfolio Theory Implementation

A comprehensive quantitative finance library for portfolio optimization
using Markowitz mean-variance optimization and advanced risk metrics.
"""

from .optimizer import PortfolioOptimizer
from .data_manager import DataManager
from .risk_metrics import RiskMetrics

__version__ = "1.0.0"
__author__ = "Quantitative Finance Developer"

__all__ = [
    'PortfolioOptimizer',
    'DataManager', 
    'RiskMetrics'
]
