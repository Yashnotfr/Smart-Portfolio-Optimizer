"""
Flask Dashboard for Smart Portfolio Optimizer

Main web application providing interactive interface for portfolio optimization.
"""

from flask import Flask, render_template, request, jsonify, session
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.utils
import sys
import os

# Add parent directory to path to import portfolio_optimizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer import PortfolioOptimizer
from utils.visualization import (
    create_efficient_frontier_plot,
    create_portfolio_weights_chart,
    create_correlation_heatmap,
    create_performance_chart,
    create_drawdown_chart,
    create_risk_metrics_dashboard,
    create_asset_allocation_pie,
    create_optimization_comparison_chart
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change in production

# Global variables to store optimization results
optimizer = None
current_results = {}

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio based on user inputs."""
    try:
        data = request.get_json()
        
        # Extract parameters
        symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        method = data.get('method', 'markowitz')
        target_return = data.get('target_return')
        risk_free_rate = data.get('risk_free_rate', 0.02)
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Initialize optimizer
        global optimizer
        optimizer = PortfolioOptimizer(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=risk_free_rate
        )
        
        # Perform optimization
        if method == 'markowitz' and target_return:
            result = optimizer.optimize_portfolio(
                method=method,
                target_return=target_return
            )
        else:
            result = optimizer.optimize_portfolio(method=method)
        
        # Store results globally
        global current_results
        current_results = result
        
        # Generate efficient frontier
        efficient_frontier = optimizer.generate_efficient_frontier()
        
        # Create individual assets data for plotting
        individual_assets = pd.DataFrame({
            'return': optimizer.mean_returns,
            'volatility': optimizer.data_manager.get_volatility()
        })
        
        # Create visualizations
        frontier_plot = create_efficient_frontier_plot(
            efficient_frontier, 
            individual_assets
        )
        weights_plot = create_portfolio_weights_chart(
            result['weights'], 
            f'{method.title()} Portfolio Weights'
        )
        correlation_plot = create_correlation_heatmap(
            optimizer.data_manager.get_correlation_matrix()
        )
        
        # Calculate portfolio performance
        weights_array = np.array(list(result['weights'].values()))
        performance = optimizer.calculate_portfolio_performance(weights_array)
        
        # Create performance charts
        performance_plot = create_performance_chart(
            performance['portfolio_returns'],
            title=f'{method.title()} Portfolio Performance'
        )
        drawdown_plot = create_drawdown_chart(performance['portfolio_returns'])
        
        # Create risk metrics dashboard
        risk_dashboard = create_risk_metrics_dashboard(result)
        
        # Convert plots to JSON
        plots = {
            'efficient_frontier': json.dumps(frontier_plot, cls=plotly.utils.PlotlyJSONEncoder),
            'weights': json.dumps(weights_plot, cls=plotly.utils.PlotlyJSONEncoder),
            'correlation': json.dumps(correlation_plot, cls=plotly.utils.PlotlyJSONEncoder),
            'performance': json.dumps(performance_plot, cls=plotly.utils.PlotlyJSONEncoder),
            'drawdown': json.dumps(drawdown_plot, cls=plotly.utils.PlotlyJSONEncoder),
            'risk_dashboard': json.dumps(risk_dashboard, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        # Prepare response data
        response_data = {
            'success': True,
            'result': result,
            'plots': plots,
            'efficient_frontier': efficient_frontier,
            'performance': performance,
            'data_summary': optimizer.data_manager.get_data_summary()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/compare_methods', methods=['POST'])
def compare_methods():
    """Compare different optimization methods."""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        risk_free_rate = data.get('risk_free_rate', 0.02)
        
        # Initialize optimizer
        global optimizer
        optimizer = PortfolioOptimizer(
            symbols=symbols,
            risk_free_rate=risk_free_rate
        )
        
        # Get optimization summary
        summary = optimizer.get_optimization_summary()
        
        # Create comparison chart
        comparison_plot = create_optimization_comparison_chart(summary)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'comparison_plot': json.dumps(comparison_plot, cls=plotly.utils.PlotlyJSONEncoder)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/rebalance', methods=['POST'])
def rebalance_portfolio():
    """Calculate rebalancing trades."""
    try:
        data = request.get_json()
        current_weights = data.get('current_weights', {})
        target_weights = data.get('target_weights', {})
        transaction_costs = data.get('transaction_costs', 0.001)
        
        if optimizer is None:
            return jsonify({
                'success': False,
                'error': 'No optimizer available. Please run optimization first.'
            }), 400
        
        # Calculate rebalancing
        rebalance_result = optimizer.rebalance_portfolio(
            current_weights,
            target_weights,
            transaction_costs
        )
        
        return jsonify({
            'success': True,
            'rebalance_result': rebalance_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/data_quality', methods=['POST'])
def check_data_quality():
    """Check data quality for given symbols."""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        
        # Create temporary data manager
        temp_optimizer = PortfolioOptimizer(symbols=symbols)
        quality_metrics = temp_optimizer.data_manager.validate_data_quality()
        
        return jsonify({
            'success': True,
            'quality_metrics': quality_metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/get_available_symbols')
def get_available_symbols():
    """Get list of popular stock symbols."""
    popular_symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX',
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
        'ADBE', 'CRM', 'INTC', 'VZ', 'CMCSA', 'PFE', 'TMO', 'ABT', 'KO',
        'PEP', 'AVGO', 'TXN', 'COST', 'DHR', 'ACN', 'NEE', 'LLY', 'UNP'
    ]
    
    return jsonify({
        'success': True,
        'symbols': popular_symbols
    })

@app.route('/get_optimization_history')
def get_optimization_history():
    """Get optimization history from session."""
    history = session.get('optimization_history', [])
    return jsonify({
        'success': True,
        'history': history
    })

@app.route('/save_optimization', methods=['POST'])
def save_optimization():
    """Save current optimization to history."""
    try:
        data = request.get_json()
        optimization_name = data.get('name', f'Optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        if optimizer is None or not current_results:
            return jsonify({
                'success': False,
                'error': 'No optimization results to save.'
            }), 400
        
        # Save to session
        history = session.get('optimization_history', [])
        history.append({
            'name': optimization_name,
            'timestamp': datetime.now().isoformat(),
            'symbols': optimizer.symbols,
            'method': current_results.get('method', 'unknown'),
            'result': current_results
        })
        session['optimization_history'] = history[-10:]  # Keep last 10
        
        return jsonify({
            'success': True,
            'message': f'Optimization "{optimization_name}" saved successfully.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/export_results', methods=['POST'])
def export_results():
    """Export optimization results to CSV/JSON."""
    try:
        data = request.get_json()
        export_format = data.get('format', 'json')
        
        if optimizer is None or not current_results:
            return jsonify({
                'success': False,
                'error': 'No optimization results to export.'
            }), 400
        
        if export_format == 'csv':
            # Create CSV data
            weights_df = pd.DataFrame(list(current_results['weights'].items()), 
                                    columns=['Asset', 'Weight'])
            
            # Convert to CSV string
            csv_data = weights_df.to_csv(index=False)
            
            return jsonify({
                'success': True,
                'data': csv_data,
                'filename': f'portfolio_weights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            })
        else:
            # Return JSON data
            return jsonify({
                'success': True,
                'data': current_results,
                'filename': f'portfolio_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
