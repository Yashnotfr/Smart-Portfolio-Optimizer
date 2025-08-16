"""
Unit tests for Smart Portfolio Optimizer

Tests the core functionality of the portfolio optimization engine.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer import PortfolioOptimizer, DataManager, RiskMetrics

class TestDataManager(unittest.TestCase):
    """Test cases for DataManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.data_manager = DataManager(self.symbols, start_date='2023-01-01', end_date='2023-12-31')
    
    def test_initialization(self):
        """Test DataManager initialization."""
        self.assertIsNotNone(self.data_manager.symbols)
        self.assertEqual(len(self.data_manager.symbols), 3)
        self.assertIsNotNone(self.data_manager.price_data)
        self.assertIsNotNone(self.data_manager.returns_data)
    
    def test_data_quality(self):
        """Test data quality validation."""
        quality_metrics = self.data_manager.validate_data_quality()
        self.assertIn('status', quality_metrics)
        self.assertIn('total_observations', quality_metrics)
        self.assertIn('missing_percentage', quality_metrics)
    
    def test_statistics_calculation(self):
        """Test statistical calculations."""
        mean_returns = self.data_manager.get_mean_returns()
        volatility = self.data_manager.get_volatility()
        correlation = self.data_manager.get_correlation_matrix()
        covariance = self.data_manager.get_covariance_matrix()
        
        self.assertIsNotNone(mean_returns)
        self.assertIsNotNone(volatility)
        self.assertIsNotNone(correlation)
        self.assertIsNotNone(covariance)
        
        # Check dimensions
        self.assertEqual(len(mean_returns), len(self.symbols))
        self.assertEqual(len(volatility), len(self.symbols))
        self.assertEqual(correlation.shape, (len(self.symbols), len(self.symbols)))
        self.assertEqual(covariance.shape, (len(self.symbols), len(self.symbols)))

class TestRiskMetrics(unittest.TestCase):
    """Test cases for RiskMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.001, 0.025, 252),
            'MSFT': np.random.normal(0.001, 0.018, 252)
        }, index=dates)
        
        self.risk_metrics = RiskMetrics(returns_data, risk_free_rate=0.02)
        self.weights = np.array([0.4, 0.3, 0.3])
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        var_historical = self.risk_metrics.calculate_var(self.weights, confidence=0.95, method='historical')
        var_parametric = self.risk_metrics.calculate_var(self.weights, confidence=0.95, method='parametric')
        
        self.assertIsInstance(var_historical, float)
        self.assertIsInstance(var_parametric, float)
        self.assertLess(var_historical, 0)  # VaR should be negative (loss)
        self.assertLess(var_parametric, 0)
    
    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation."""
        cvar = self.risk_metrics.calculate_cvar(self.weights, confidence=0.95)
        
        self.assertIsInstance(cvar, float)
        self.assertLess(cvar, 0)  # CVaR should be negative (loss)
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.risk_metrics.calculate_sharpe_ratio(self.weights)
        
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, -10)  # Reasonable range
        self.assertLess(sharpe, 10)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        drawdown_metrics = self.risk_metrics.calculate_max_drawdown(self.weights)
        
        self.assertIn('max_drawdown', drawdown_metrics)
        self.assertIn('drawdown_duration', drawdown_metrics)
        self.assertIsInstance(drawdown_metrics['max_drawdown'], float)
        self.assertLessEqual(drawdown_metrics['max_drawdown'], 0)
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        volatility = self.risk_metrics.calculate_volatility(self.weights)
        
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)
    
    def test_all_metrics(self):
        """Test calculation of all risk metrics."""
        all_metrics = self.risk_metrics.calculate_all_metrics(self.weights)
        
        required_metrics = ['expected_return', 'volatility', 'sharpe_ratio', 
                          'var_95', 'cvar_95', 'max_drawdown']
        
        for metric in required_metrics:
            self.assertIn(metric, all_metrics)
            self.assertIsInstance(all_metrics[metric], (int, float))

class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.optimizer = PortfolioOptimizer(
            symbols=self.symbols,
            start_date='2023-01-01',
            end_date='2023-12-31',
            risk_free_rate=0.02
        )
    
    def test_initialization(self):
        """Test PortfolioOptimizer initialization."""
        self.assertIsNotNone(self.optimizer.data_manager)
        self.assertIsNotNone(self.optimizer.risk_metrics)
        self.assertEqual(len(self.optimizer.symbols), 3)
        self.assertEqual(self.optimizer.n_assets, 3)
    
    def test_markowitz_optimization(self):
        """Test Markowitz mean-variance optimization."""
        result = self.optimizer.optimize_portfolio(method='markowitz')
        
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Check weights sum to 1
        weights_sum = sum(result['weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
        
        # Check all weights are non-negative
        for weight in result['weights'].values():
            self.assertGreaterEqual(weight, 0)
    
    def test_max_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        result = self.optimizer.optimize_portfolio(method='max_sharpe')
        
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Check weights sum to 1
        weights_sum = sum(result['weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
    
    def test_min_variance_optimization(self):
        """Test minimum variance optimization."""
        result = self.optimizer.optimize_portfolio(method='min_variance')
        
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Check weights sum to 1
        weights_sum = sum(result['weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        result = self.optimizer.optimize_portfolio(method='risk_parity')
        
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Check weights sum to 1
        weights_sum = sum(result['weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
    
    def test_efficient_frontier_generation(self):
        """Test efficient frontier generation."""
        frontier = self.optimizer.generate_efficient_frontier(num_portfolios=20)
        
        self.assertIn('portfolios', frontier)
        self.assertIn('returns', frontier)
        self.assertIn('volatilities', frontier)
        self.assertIn('sharpe_ratios', frontier)
        
        self.assertGreater(len(frontier['portfolios']), 0)
        self.assertEqual(len(frontier['returns']), len(frontier['portfolios']))
        self.assertEqual(len(frontier['volatilities']), len(frontier['portfolios']))
    
    def test_portfolio_performance(self):
        """Test portfolio performance calculation."""
        # Get weights from optimization
        result = self.optimizer.optimize_portfolio(method='markowitz')
        weights_array = np.array(list(result['weights'].values()))
        
        performance = self.optimizer.calculate_portfolio_performance(weights_array)
        
        self.assertIn('total_return', performance)
        self.assertIn('annualized_return', performance)
        self.assertIn('annualized_volatility', performance)
        self.assertIn('sharpe_ratio', performance)
        self.assertIn('max_drawdown', performance)
    
    def test_rebalancing(self):
        """Test portfolio rebalancing."""
        current_weights = {'AAPL': 0.33, 'GOOGL': 0.33, 'MSFT': 0.34}
        target_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        
        rebalance_result = self.optimizer.rebalance_portfolio(
            current_weights, target_weights, transaction_costs=0.001
        )
        
        self.assertIn('trades', rebalance_result)
        self.assertIn('transaction_cost', rebalance_result)
        self.assertIn('sharpe_improvement', rebalance_result)
        
        # Check that trades sum to approximately zero
        trades_sum = sum(rebalance_result['trades'].values())
        self.assertAlmostEqual(trades_sum, 0.0, places=6)
    
    def test_optimization_summary(self):
        """Test optimization summary generation."""
        summary = self.optimizer.get_optimization_summary()
        
        methods = ['markowitz', 'risk_parity', 'max_sharpe', 'min_variance']
        
        for method in methods:
            self.assertIn(method, summary)
            if 'error' not in summary[method]:
                self.assertIn('expected_return', summary[method])
                self.assertIn('volatility', summary[method])
                self.assertIn('sharpe_ratio', summary[method])
    
    def test_invalid_method(self):
        """Test handling of invalid optimization method."""
        with self.assertRaises(ValueError):
            self.optimizer.optimize_portfolio(method='invalid_method')
    
    def test_target_return_constraint(self):
        """Test Markowitz optimization with target return constraint."""
        result = self.optimizer.optimize_portfolio(
            method='markowitz',
            target_return=0.15
        )
        
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        
        # Check that expected return meets or exceeds target
        self.assertGreaterEqual(result['expected_return'], 0.15)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization workflow."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(symbols, risk_free_rate=0.02)
        
        # Perform optimization
        result = optimizer.optimize_portfolio(method='max_sharpe')
        
        # Validate results
        self.assertIsNotNone(result)
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Check that weights are valid
        weights_sum = sum(result['weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
        
        for weight in result['weights'].values():
            self.assertGreaterEqual(weight, 0)
            self.assertLessEqual(weight, 1)
    
    def test_data_consistency(self):
        """Test data consistency across different components."""
        symbols = ['AAPL', 'GOOGL']
        
        optimizer = PortfolioOptimizer(symbols, risk_free_rate=0.02)
        
        # Check that data manager and risk metrics use consistent data
        dm_returns = optimizer.data_manager.get_returns_data()
        rm_returns = optimizer.risk_metrics.returns_data
        
        pd.testing.assert_frame_equal(dm_returns, rm_returns)
        
        # Check that mean returns are consistent
        dm_mean = optimizer.data_manager.get_mean_returns()
        rm_mean = optimizer.risk_metrics.mean_returns
        
        pd.testing.assert_series_equal(dm_mean, rm_mean)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
