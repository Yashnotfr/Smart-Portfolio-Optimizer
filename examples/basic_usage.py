"""
Basic Usage Example for Smart Portfolio Optimizer

This example demonstrates how to use the portfolio optimizer
to create optimal portfolios using different methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer import PortfolioOptimizer
import numpy as np
import pandas as pd

def main():
    """Main example function."""
    print("🚀 Smart Portfolio Optimizer - Basic Usage Example")
    print("=" * 60)
    
    # Define assets
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META']
    print(f"📊 Analyzing portfolio with assets: {', '.join(symbols)}")
    
    # Initialize optimizer
    print("\n📈 Initializing portfolio optimizer...")
    optimizer = PortfolioOptimizer(
        symbols=symbols,
        start_date='2023-01-01',
        end_date='2024-01-01',
        risk_free_rate=0.02
    )
    
    # Get data summary
    print("\n📋 Data Summary:")
    data_summary = optimizer.data_manager.get_data_summary()
    print(f"   • Number of assets: {len(data_summary['symbols'])}")
    print(f"   • Date range: {data_summary['start_date']} to {data_summary['end_date']}")
    print(f"   • Observations: {data_summary['observations']}")
    
    # Check data quality
    print("\n🔍 Data Quality Check:")
    quality_metrics = optimizer.data_manager.validate_data_quality()
    print(f"   • Status: {quality_metrics['status']}")
    print(f"   • Missing data: {quality_metrics['missing_percentage']:.2f}%")
    
    # 1. Markowitz Mean-Variance Optimization
    print("\n🎯 1. Markowitz Mean-Variance Optimization")
    print("-" * 40)
    
    markowitz_result = optimizer.optimize_portfolio(method='markowitz')
    print(f"   • Expected Return: {markowitz_result['expected_return']:.4f} ({markowitz_result['expected_return']*100:.2f}%)")
    print(f"   • Volatility: {markowitz_result['volatility']:.4f} ({markowitz_result['volatility']*100:.2f}%)")
    print(f"   • Sharpe Ratio: {markowitz_result['sharpe_ratio']:.4f}")
    print(f"   • Maximum Drawdown: {markowitz_result['max_drawdown']:.4f} ({markowitz_result['max_drawdown']*100:.2f}%)")
    
    print("\n   Portfolio Weights:")
    for asset, weight in markowitz_result['weights'].items():
        print(f"   • {asset}: {weight:.4f} ({weight*100:.2f}%)")
    
    # 2. Maximum Sharpe Ratio Optimization
    print("\n📈 2. Maximum Sharpe Ratio Optimization")
    print("-" * 40)
    
    max_sharpe_result = optimizer.optimize_portfolio(method='max_sharpe')
    print(f"   • Expected Return: {max_sharpe_result['expected_return']:.4f} ({max_sharpe_result['expected_return']*100:.2f}%)")
    print(f"   • Volatility: {max_sharpe_result['volatility']:.4f} ({max_sharpe_result['volatility']*100:.2f}%)")
    print(f"   • Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.4f}")
    print(f"   • Maximum Drawdown: {max_sharpe_result['max_drawdown']:.4f} ({max_sharpe_result['max_drawdown']*100:.2f}%)")
    
    # 3. Minimum Variance Optimization
    print("\n🛡️ 3. Minimum Variance Optimization")
    print("-" * 40)
    
    min_var_result = optimizer.optimize_portfolio(method='min_variance')
    print(f"   • Expected Return: {min_var_result['expected_return']:.4f} ({min_var_result['expected_return']*100:.2f}%)")
    print(f"   • Volatility: {min_var_result['volatility']:.4f} ({min_var_result['volatility']*100:.2f}%)")
    print(f"   • Sharpe Ratio: {min_var_result['sharpe_ratio']:.4f}")
    print(f"   • Maximum Drawdown: {min_var_result['max_drawdown']:.4f} ({min_var_result['max_drawdown']*100:.2f}%)")
    
    # 4. Risk Parity Optimization
    print("\n⚖️ 4. Risk Parity Optimization")
    print("-" * 40)
    
    risk_parity_result = optimizer.optimize_portfolio(method='risk_parity')
    print(f"   • Expected Return: {risk_parity_result['expected_return']:.4f} ({risk_parity_result['expected_return']*100:.2f}%)")
    print(f"   • Volatility: {risk_parity_result['volatility']:.4f} ({risk_parity_result['volatility']*100:.2f}%)")
    print(f"   • Sharpe Ratio: {risk_parity_result['sharpe_ratio']:.4f}")
    print(f"   • Maximum Drawdown: {risk_parity_result['max_drawdown']:.4f} ({risk_parity_result['max_drawdown']*100:.2f}%)")
    
    # 5. Generate Efficient Frontier
    print("\n📊 5. Generating Efficient Frontier")
    print("-" * 40)
    
    efficient_frontier = optimizer.generate_efficient_frontier(num_portfolios=50)
    print(f"   • Generated {len(efficient_frontier['portfolios'])} efficient portfolios")
    
    # Find the portfolio with highest Sharpe ratio
    best_sharpe_idx = np.argmax(efficient_frontier['sharpe_ratios'])
    best_portfolio = efficient_frontier['portfolios'][best_sharpe_idx]
    
    print(f"   • Best Sharpe Ratio Portfolio:")
    print(f"     - Return: {best_portfolio['actual_return']:.4f} ({best_portfolio['actual_return']*100:.2f}%)")
    print(f"     - Volatility: {best_portfolio['volatility']:.4f} ({best_portfolio['volatility']*100:.2f}%)")
    print(f"     - Sharpe Ratio: {best_portfolio['sharpe_ratio']:.4f}")
    
    # 6. Portfolio Performance Analysis
    print("\n📈 6. Portfolio Performance Analysis")
    print("-" * 40)
    
    # Use Markowitz weights for performance analysis
    weights_array = np.array(list(markowitz_result['weights'].values()))
    performance = optimizer.calculate_portfolio_performance(weights_array)
    
    print(f"   • Total Return: {performance['total_return']:.4f} ({performance['total_return']*100:.2f}%)")
    print(f"   • Annualized Return: {performance['annualized_return']:.4f} ({performance['annualized_return']*100:.2f}%)")
    print(f"   • Annualized Volatility: {performance['annualized_volatility']:.4f} ({performance['annualized_volatility']*100:.2f}%)")
    print(f"   • Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
    print(f"   • Maximum Drawdown: {performance['max_drawdown']:.4f} ({performance['max_drawdown']*100:.2f}%)")
    
    # 7. Risk Metrics Comparison
    print("\n🔍 7. Risk Metrics Comparison")
    print("-" * 40)
    
    methods = ['markowitz', 'max_sharpe', 'min_variance', 'risk_parity']
    comparison_data = []
    
    for method in methods:
        result = optimizer.optimize_portfolio(method=method)
        comparison_data.append({
            'Method': method.replace('_', ' ').title(),
            'Expected Return (%)': result['expected_return'] * 100,
            'Volatility (%)': result['volatility'] * 100,
            'Sharpe Ratio': result['sharpe_ratio'],
            'Max Drawdown (%)': abs(result['max_drawdown']) * 100,
            'VaR (95%) (%)': abs(result['var_95']) * 100,
            'CVaR (95%) (%)': abs(result['cvar_95']) * 100
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    
    # 8. Rebalancing Example
    print("\n🔄 8. Portfolio Rebalancing Example")
    print("-" * 40)
    
    # Simulate current portfolio weights (equal weight)
    current_weights = {symbol: 1/len(symbols) for symbol in symbols}
    
    # Target weights from Markowitz optimization
    target_weights = markowitz_result['weights']
    
    # Calculate rebalancing trades
    rebalance_result = optimizer.rebalance_portfolio(
        current_weights=current_weights,
        target_weights=target_weights,
        transaction_costs=0.001
    )
    
    print(f"   • Transaction Cost: {rebalance_result['transaction_cost']:.4f}")
    print(f"   • Sharpe Ratio Improvement: {rebalance_result['sharpe_improvement']:.4f}")
    print(f"   • Volatility Change: {rebalance_result['volatility_change']:.4f}")
    
    print("\n   Rebalancing Trades:")
    for asset, trade in rebalance_result['trades'].items():
        if abs(trade) > 0.001:  # Only show significant trades
            print(f"   • {asset}: {trade:+.4f} ({trade*100:+.2f}%)")
    
    print("\n✅ Example completed successfully!")
    print("\n💡 Key Takeaways:")
    print("   • Different optimization methods produce different risk-return profiles")
    print("   • Maximum Sharpe Ratio typically provides the best risk-adjusted returns")
    print("   • Minimum Variance focuses on risk reduction")
    print("   • Risk Parity aims for equal risk contribution across assets")
    print("   • Rebalancing can improve portfolio performance but incurs transaction costs")

if __name__ == "__main__":
    main()
