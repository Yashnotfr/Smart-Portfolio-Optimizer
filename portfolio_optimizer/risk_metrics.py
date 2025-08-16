"""
Risk Metrics for Portfolio Optimizer

Calculates various financial risk measures including VaR, CVaR, Sharpe ratio,
maximum drawdown, and other portfolio risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

class RiskMetrics:
    """
    Comprehensive risk metrics calculator for portfolio analysis.
    
    This class provides methods to calculate:
    - Value at Risk (VaR) - Historical and Parametric
    - Conditional Value at Risk (CVaR)
    - Sharpe Ratio and Information Ratio
    - Maximum Drawdown
    - Volatility and Beta
    - Skewness and Kurtosis
    - Correlation and Diversification metrics
    """
    
    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize RiskMetrics with returns data.
        
        Args:
            returns_data: DataFrame of asset returns (daily)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate basic statistics
        self.mean_returns = returns_data.mean() * 252  # Annualized
        self.volatility = returns_data.std() * np.sqrt(252)  # Annualized
        
    def calculate_var(self, weights: np.ndarray, confidence: float = 0.95, 
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR) for a portfolio.
        
        Args:
            weights: Portfolio weights
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR value (negative number representing potential loss)
        """
        if method == 'historical':
            return self._historical_var(weights, confidence)
        elif method == 'parametric':
            return self._parametric_var(weights, confidence)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")
    
    def _historical_var(self, weights: np.ndarray, confidence: float) -> float:
        """Calculate historical VaR."""
        portfolio_returns = self.returns_data @ weights
        return np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    def _parametric_var(self, weights: np.ndarray, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        portfolio_mean = self.mean_returns @ weights
        portfolio_vol = np.sqrt(weights.T @ self.returns_data.cov() @ weights * 252)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        return portfolio_mean + z_score * portfolio_vol
    
    def calculate_cvar(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            weights: Portfolio weights
            confidence: Confidence level
            
        Returns:
            CVaR value (expected loss beyond VaR)
        """
        portfolio_returns = self.returns_data @ weights
        var_threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        # Calculate expected value of returns below VaR threshold
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        return tail_returns.mean()
    
    def calculate_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate Sharpe ratio for portfolio.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Sharpe ratio (risk-adjusted return measure)
        """
        portfolio_return = self.mean_returns @ weights
        portfolio_vol = np.sqrt(weights.T @ self.returns_data.cov() @ weights * 252)
        
        excess_return = portfolio_return - self.risk_free_rate
        return excess_return / portfolio_vol if portfolio_vol > 0 else 0
    
    def calculate_information_ratio(self, weights: np.ndarray, 
                                  benchmark_weights: np.ndarray) -> float:
        """
        Calculate Information ratio relative to benchmark.
        
        Args:
            weights: Portfolio weights
            benchmark_weights: Benchmark portfolio weights
            
        Returns:
            Information ratio
        """
        portfolio_return = self.mean_returns @ weights
        benchmark_return = self.mean_returns @ benchmark_weights
        
        active_return = portfolio_return - benchmark_return
        
        # Calculate tracking error
        active_weights = weights - benchmark_weights
        tracking_error = np.sqrt(active_weights.T @ self.returns_data.cov() @ active_weights * 252)
        
        return active_return / tracking_error if tracking_error > 0 else 0
    
    def calculate_max_drawdown(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with max drawdown, duration, and recovery metrics
        """
        portfolio_returns = self.returns_data @ weights
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find peak before maximum drawdown
        peak_idx = running_max.loc[:max_dd_idx].idxmax()
        
        # Find recovery point (return to peak)
        recovery_data = cumulative_returns.loc[max_dd_idx:]
        recovery_idx = recovery_data[recovery_data >= running_max.loc[peak_idx]].index
        
        if len(recovery_idx) > 0:
            recovery_idx = recovery_idx[0]
            recovery_duration = (recovery_idx - max_dd_idx).days
        else:
            recovery_duration = None
        
        drawdown_duration = (max_dd_idx - peak_idx).days
        
        return {
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'recovery_duration': recovery_duration,
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'recovery_date': recovery_idx if len(recovery_idx) > 0 else None
        }
    
    def calculate_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Annualized portfolio volatility
        """
        return np.sqrt(weights.T @ self.returns_data.cov() @ weights * 252)
    
    def calculate_beta(self, weights: np.ndarray, market_weights: np.ndarray) -> float:
        """
        Calculate portfolio beta relative to market.
        
        Args:
            weights: Portfolio weights
            market_weights: Market portfolio weights
            
        Returns:
            Portfolio beta
        """
        portfolio_returns = self.returns_data @ weights
        market_returns = self.returns_data @ market_weights
        
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 0
    
    def calculate_treynor_ratio(self, weights: np.ndarray, 
                              market_weights: np.ndarray) -> float:
        """
        Calculate Treynor ratio (excess return per unit of systematic risk).
        
        Args:
            weights: Portfolio weights
            market_weights: Market portfolio weights
            
        Returns:
            Treynor ratio
        """
        portfolio_return = self.mean_returns @ weights
        beta = self.calculate_beta(weights, market_weights)
        
        excess_return = portfolio_return - self.risk_free_rate
        return excess_return / beta if beta != 0 else 0
    
    def calculate_jensen_alpha(self, weights: np.ndarray, 
                             market_weights: np.ndarray) -> float:
        """
        Calculate Jensen's alpha (excess return not explained by CAPM).
        
        Args:
            weights: Portfolio weights
            market_weights: Market portfolio weights
            
        Returns:
            Jensen's alpha
        """
        portfolio_return = self.mean_returns @ weights
        market_return = self.mean_returns @ market_weights
        beta = self.calculate_beta(weights, market_weights)
        
        expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        return portfolio_return - expected_return
    
    def calculate_sortino_ratio(self, weights: np.ndarray, 
                              target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside deviation-based risk-adjusted return).
        
        Args:
            weights: Portfolio weights
            target_return: Target return rate (default: 0%)
            
        Returns:
            Sortino ratio
        """
        portfolio_returns = self.returns_data @ weights
        portfolio_mean = self.mean_returns @ weights
        
        # Calculate downside deviation
        downside_returns = portfolio_returns[portfolio_returns < target_return]
        if len(downside_returns) == 0:
            return float('inf') if portfolio_mean > target_return else 0
        
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2) * 252)
        
        excess_return = portfolio_mean - target_return
        return excess_return / downside_deviation if downside_deviation > 0 else 0
    
    def calculate_calmar_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown).
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Calmar ratio
        """
        portfolio_return = self.mean_returns @ weights
        max_dd = self.calculate_max_drawdown(weights)['max_drawdown']
        
        return portfolio_return / abs(max_dd) if max_dd != 0 else 0
    
    def calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate diversification ratio (weighted average vol / portfolio vol).
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Diversification ratio
        """
        weighted_avg_vol = np.sum(weights * self.volatility)
        portfolio_vol = self.calculate_volatility(weights)
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
    
    def calculate_concentration_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio concentration metrics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with concentration metrics
        """
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights ** 2)
        
        # Effective number of assets
        effective_assets = 1 / hhi if hhi > 0 else 0
        
        # Largest position
        max_weight = np.max(weights)
        
        # Top 5 concentration
        top_5_concentration = np.sum(np.sort(weights)[-5:])
        
        return {
            'hhi': hhi,
            'effective_assets': effective_assets,
            'max_weight': max_weight,
            'top_5_concentration': top_5_concentration
        }
    
    def calculate_skewness_kurtosis(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio skewness and kurtosis.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with skewness and kurtosis
        """
        portfolio_returns = self.returns_data @ weights
        
        return {
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns)
        }
    
    def calculate_all_metrics(self, weights: np.ndarray, 
                            market_weights: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all risk metrics for a portfolio.
        
        Args:
            weights: Portfolio weights
            market_weights: Market portfolio weights (optional)
            
        Returns:
            Dictionary with all risk metrics
        """
        metrics = {
            'expected_return': self.mean_returns @ weights,
            'volatility': self.calculate_volatility(weights),
            'sharpe_ratio': self.calculate_sharpe_ratio(weights),
            'var_95': self.calculate_var(weights, 0.95),
            'cvar_95': self.calculate_cvar(weights, 0.95),
            'sortino_ratio': self.calculate_sortino_ratio(weights),
            'calmar_ratio': self.calculate_calmar_ratio(weights),
            'diversification_ratio': self.calculate_diversification_ratio(weights)
        }
        
        # Add drawdown metrics
        drawdown_metrics = self.calculate_max_drawdown(weights)
        metrics.update(drawdown_metrics)
        
        # Add concentration metrics
        concentration_metrics = self.calculate_concentration_metrics(weights)
        metrics.update(concentration_metrics)
        
        # Add distribution metrics
        dist_metrics = self.calculate_skewness_kurtosis(weights)
        metrics.update(dist_metrics)
        
        # Add market-relative metrics if market weights provided
        if market_weights is not None:
            metrics.update({
                'beta': self.calculate_beta(weights, market_weights),
                'treynor_ratio': self.calculate_treynor_ratio(weights, market_weights),
                'jensen_alpha': self.calculate_jensen_alpha(weights, market_weights),
                'information_ratio': self.calculate_information_ratio(weights, market_weights)
            })
        
        return metrics
