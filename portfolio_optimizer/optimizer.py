"""
Portfolio Optimizer - Modern Portfolio Theory Implementation

Main optimization engine using Markowitz mean-variance optimization
and various portfolio construction strategies.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize
import warnings

from .data_manager import DataManager
from .risk_metrics import RiskMetrics

class PortfolioOptimizer:
    """
    Main portfolio optimization engine implementing Modern Portfolio Theory.
    
    This class provides:
    - Markowitz Mean-Variance Optimization
    - Risk Parity Optimization
    - Maximum Sharpe Ratio Optimization
    - Minimum Variance Optimization
    - Efficient Frontier Generation
    - Portfolio Rebalancing
    """
    
    def __init__(self, symbols: List[str], start_date: str = None, 
                 end_date: str = None, risk_free_rate: float = 0.02):
        """
        Initialize PortfolioOptimizer.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            risk_free_rate: Annual risk-free rate
        """
        self.data_manager = DataManager(symbols, start_date, end_date)
        self.risk_metrics = RiskMetrics(
            self.data_manager.get_returns_data(), 
            risk_free_rate
        )
        self.risk_free_rate = risk_free_rate
        self.symbols = self.data_manager.symbols
        
        # Get data matrices
        self.mean_returns = self.data_manager.get_mean_returns()
        self.covariance_matrix = self.data_manager.get_covariance_matrix()
        self.returns_data = self.data_manager.get_returns_data()
        
        # Number of assets
        self.n_assets = len(self.symbols)
        
        # Initialize optimization results
        self.optimization_results = {}
        
    def optimize_portfolio(self, method: str = 'markowitz', **kwargs) -> Dict:
        """
        Optimize portfolio using specified method.
        
        Args:
            method: Optimization method ('markowitz', 'risk_parity', 'max_sharpe', 'min_variance')
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with optimization results
        """
        if method == 'markowitz':
            return self._markowitz_optimization(**kwargs)
        elif method == 'risk_parity':
            return self._risk_parity_optimization(**kwargs)
        elif method == 'max_sharpe':
            return self._max_sharpe_optimization(**kwargs)
        elif method == 'min_variance':
            return self._min_variance_optimization(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _markowitz_optimization(self, target_return: float = None, 
                              target_risk: float = None,
                              risk_aversion: float = 1.0) -> Dict:
        """
        Markowitz mean-variance optimization.
        
        Args:
            target_return: Target portfolio return (if None, maximize return)
            target_risk: Target portfolio risk (if None, minimize risk)
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            
        Returns:
            Optimization results dictionary
        """
        # Define variables
        weights = cp.Variable(self.n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0  # Long-only constraint
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append(self.mean_returns @ weights >= target_return)
        
        # Add target risk constraint if specified
        if target_risk is not None:
            constraints.append(cp.quad_form(weights, self.covariance_matrix) <= target_risk**2)
        
        # Define objective function
        if target_return is not None:
            # Minimize risk for given return
            objective = cp.Minimize(cp.quad_form(weights, self.covariance_matrix))
        elif target_risk is not None:
            # Maximize return for given risk
            objective = cp.Maximize(self.mean_returns @ weights)
        else:
            # Mean-variance optimization with risk aversion
            expected_return = self.mean_returns @ weights
            portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
            objective = cp.Maximize(expected_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != 'optimal':
            raise ValueError(f"Optimization failed: {problem.status}")
        
        # Calculate results
        optimal_weights = weights.value
        portfolio_return = self.mean_returns @ optimal_weights
        portfolio_risk = np.sqrt(optimal_weights.T @ self.covariance_matrix @ optimal_weights)
        
        results = {
            'method': 'markowitz',
            'weights': dict(zip(self.symbols, optimal_weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': self.risk_metrics.calculate_sharpe_ratio(optimal_weights),
            'status': problem.status,
            'optimal_value': problem.value
        }
        
        # Add additional risk metrics
        risk_metrics = self.risk_metrics.calculate_all_metrics(optimal_weights)
        results.update(risk_metrics)
        
        return results
    
    def _risk_parity_optimization(self, target_risk: float = None) -> Dict:
        """
        Risk parity optimization (equal risk contribution).
        
        Args:
            target_risk: Target portfolio risk (optional)
            
        Returns:
            Optimization results dictionary
        """
        # Define variables
        weights = cp.Variable(self.n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        # Risk parity objective: minimize variance of risk contributions
        risk_contributions = cp.multiply(weights, cp.sqrt(cp.diag(self.covariance_matrix)))
        
        # Objective: minimize variance of risk contributions
        objective = cp.Minimize(cp.sum_squares(risk_contributions - cp.mean(risk_contributions)))
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != 'optimal':
            raise ValueError(f"Risk parity optimization failed: {problem.status}")
        
        # Calculate results
        optimal_weights = weights.value
        portfolio_return = self.mean_returns @ optimal_weights
        portfolio_risk = np.sqrt(optimal_weights.T @ self.covariance_matrix @ optimal_weights)
        
        results = {
            'method': 'risk_parity',
            'weights': dict(zip(self.symbols, optimal_weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': self.risk_metrics.calculate_sharpe_ratio(optimal_weights),
            'status': problem.status,
            'optimal_value': problem.value
        }
        
        # Add risk metrics
        risk_metrics = self.risk_metrics.calculate_all_metrics(optimal_weights)
        results.update(risk_metrics)
        
        return results
    
    def _max_sharpe_optimization(self) -> Dict:
        """
        Maximum Sharpe ratio optimization.
        
        Returns:
            Optimization results dictionary
        """
        # Define variables
        weights = cp.Variable(self.n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        # Objective: maximize Sharpe ratio
        # Since cvxpy doesn't directly support ratio optimization,
        # we use the fact that maximizing Sharpe ratio is equivalent to
        # minimizing the negative Sharpe ratio
        expected_return = self.mean_returns @ weights
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
        
        # Use a reformulation: maximize excess return / volatility
        # This is equivalent to maximizing Sharpe ratio
        objective = cp.Maximize(expected_return - self.risk_free_rate)
        constraints.append(cp.quad_form(weights, self.covariance_matrix) <= 1)  # Normalize volatility
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != 'optimal':
            raise ValueError(f"Max Sharpe optimization failed: {problem.status}")
        
        # Calculate results
        optimal_weights = weights.value
        portfolio_return = self.mean_returns @ optimal_weights
        portfolio_risk = np.sqrt(optimal_weights.T @ self.covariance_matrix @ optimal_weights)
        
        results = {
            'method': 'max_sharpe',
            'weights': dict(zip(self.symbols, optimal_weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': self.risk_metrics.calculate_sharpe_ratio(optimal_weights),
            'status': problem.status,
            'optimal_value': problem.value
        }
        
        # Add risk metrics
        risk_metrics = self.risk_metrics.calculate_all_metrics(optimal_weights)
        results.update(risk_metrics)
        
        return results
    
    def _min_variance_optimization(self) -> Dict:
        """
        Minimum variance optimization.
        
        Returns:
            Optimization results dictionary
        """
        # Define variables
        weights = cp.Variable(self.n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(weights, self.covariance_matrix))
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != 'optimal':
            raise ValueError(f"Min variance optimization failed: {problem.status}")
        
        # Calculate results
        optimal_weights = weights.value
        portfolio_return = self.mean_returns @ optimal_weights
        portfolio_risk = np.sqrt(optimal_weights.T @ self.covariance_matrix @ optimal_weights)
        
        results = {
            'method': 'min_variance',
            'weights': dict(zip(self.symbols, optimal_weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': self.risk_metrics.calculate_sharpe_ratio(optimal_weights),
            'status': problem.status,
            'optimal_value': problem.value
        }
        
        # Add risk metrics
        risk_metrics = self.risk_metrics.calculate_all_metrics(optimal_weights)
        results.update(risk_metrics)
        
        return results
    
    def generate_efficient_frontier(self, num_portfolios: int = 100) -> Dict:
        """
        Generate efficient frontier by solving multiple optimization problems.
        
        Args:
            num_portfolios: Number of portfolios to generate
            
        Returns:
            Dictionary with efficient frontier data
        """
        # Find minimum and maximum returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                result = self._markowitz_optimization(target_return=target_return)
                efficient_portfolios.append({
                    'target_return': target_return,
                    'actual_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                })
            except ValueError:
                # Skip infeasible portfolios
                continue
        
        return {
            'portfolios': efficient_portfolios,
            'returns': [p['actual_return'] for p in efficient_portfolios],
            'volatilities': [p['volatility'] for p in efficient_portfolios],
            'sharpe_ratios': [p['sharpe_ratio'] for p in efficient_portfolios]
        }
    
    def calculate_portfolio_performance(self, weights: np.ndarray, 
                                      start_date: str = None, 
                                      end_date: str = None) -> Dict:
        """
        Calculate historical performance of a portfolio.
        
        Args:
            weights: Portfolio weights
            start_date: Start date for performance calculation
            end_date: End date for performance calculation
            
        Returns:
            Performance metrics dictionary
        """
        # Get returns data for the specified period
        if start_date or end_date:
            mask = pd.Series(True, index=self.returns_data.index)
            if start_date:
                mask &= self.returns_data.index >= start_date
            if end_date:
                mask &= self.returns_data.index <= end_date
            returns_subset = self.returns_data[mask]
        else:
            returns_subset = self.returns_data
        
        # Calculate portfolio returns
        portfolio_returns = returns_subset @ weights
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Calculate drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns,
            'portfolio_returns': portfolio_returns
        }
    
    def rebalance_portfolio(self, current_weights: Dict[str, float], 
                          target_weights: Dict[str, float],
                          transaction_costs: float = 0.001) -> Dict:
        """
        Calculate optimal rebalancing trades considering transaction costs.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_costs: Transaction cost as percentage
            
        Returns:
            Rebalancing analysis dictionary
        """
        # Convert to numpy arrays
        current = np.array([current_weights.get(sym, 0) for sym in self.symbols])
        target = np.array([target_weights.get(sym, 0) for sym in self.symbols])
        
        # Calculate trades
        trades = target - current
        
        # Calculate transaction costs
        total_cost = np.sum(np.abs(trades)) * transaction_costs
        
        # Calculate expected improvement
        current_metrics = self.risk_metrics.calculate_all_metrics(current)
        target_metrics = self.risk_metrics.calculate_all_metrics(target)
        
        return {
            'trades': dict(zip(self.symbols, trades)),
            'transaction_cost': total_cost,
            'current_sharpe': current_metrics['sharpe_ratio'],
            'target_sharpe': target_metrics['sharpe_ratio'],
            'sharpe_improvement': target_metrics['sharpe_ratio'] - current_metrics['sharpe_ratio'],
            'current_volatility': current_metrics['volatility'],
            'target_volatility': target_metrics['volatility'],
            'volatility_change': target_metrics['volatility'] - current_metrics['volatility']
        }
    
    def get_optimization_summary(self) -> Dict:
        """
        Get summary of all optimization methods.
        
        Returns:
            Summary dictionary with all optimization results
        """
        methods = ['markowitz', 'risk_parity', 'max_sharpe', 'min_variance']
        summary = {}
        
        for method in methods:
            try:
                result = self.optimize_portfolio(method=method)
                summary[method] = {
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown']
                }
            except Exception as e:
                summary[method] = {'error': str(e)}
        
        return summary
