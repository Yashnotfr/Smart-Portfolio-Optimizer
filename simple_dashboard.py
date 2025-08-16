#!/usr/bin/env python3
"""
Simplified Smart Portfolio Optimizer Dashboard
Uses mock data to demonstrate functionality
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.utils
import os

app = Flask(__name__)

def create_mock_data(symbols):
    """Create mock financial data for given symbols."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)  # For reproducible results
    
    prices_data = {}
    for symbol in symbols:
        start_price = np.random.uniform(50, 200)
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [start_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        prices_data[symbol] = prices
    
    return pd.DataFrame(prices_data, index=dates)

def calculate_portfolio_metrics(returns, weights):
    """Calculate portfolio metrics."""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe_ratio
    }

def optimize_portfolio(returns, method='equal_weight'):
    """Simple portfolio optimization."""
    n_assets = len(returns.columns)
    
    if method == 'equal_weight':
        weights = np.ones(n_assets) / n_assets
    elif method == 'max_return':
        mean_returns = returns.mean() * 252
        max_idx = np.argmax(mean_returns)
        weights = np.zeros(n_assets)
        weights[max_idx] = 1.0
    elif method == 'min_vol':
        cov_matrix = returns.cov() * 252
        # Simple minimum variance (equal weights for now)
        weights = np.ones(n_assets) / n_assets
    
    return weights

@app.route('/')
def index():
    return render_template('simple_index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
        method = data.get('method', 'equal_weight')
        
        # Create mock data
        price_data = create_mock_data(symbols)
        returns = price_data.pct_change().dropna()
        
        # Optimize portfolio
        weights = optimize_portfolio(returns, method)
        metrics = calculate_portfolio_metrics(returns, weights)
        
        # Create charts
        efficient_frontier = create_efficient_frontier_plot(returns)
        weights_chart = create_weights_chart(symbols, weights)
        correlation_chart = create_correlation_chart(returns)
        
        return jsonify({
            'success': True,
            'weights': weights.tolist(),
            'metrics': metrics,
            'charts': {
                'efficient_frontier': efficient_frontier,
                'weights': weights_chart,
                'correlation': correlation_chart
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def create_efficient_frontier_plot(returns):
    """Create efficient frontier plot."""
    # Generate random portfolios
    n_portfolios = 100
    n_assets = len(returns.columns)
    
    portfolio_returns = []
    portfolio_vols = []
    
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        metrics = calculate_portfolio_metrics(returns, weights)
        portfolio_returns.append(metrics['expected_return'])
        portfolio_vols.append(metrics['volatility'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_vols,
        y=portfolio_returns,
        mode='markers',
        marker=dict(
            size=10, 
            color=portfolio_returns, 
            colorscale='Viridis',
            opacity=0.7,
            line=dict(width=1, color='rgba(255,255,255,0.3)')
        ),
        name='Random Portfolios',
        hovertemplate='<b>Volatility:</b> %{x:.3f}<br><b>Return:</b> %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Efficient Frontier',
            font=dict(size=20, color='#ffffff'),
            x=0.5
        ),
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#2a2a2a',
            zerolinecolor='#2a2a2a',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#2a2a2a',
            zerolinecolor='#2a2a2a',
            showgrid=True
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=False
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_weights_chart(symbols, weights):
    """Create portfolio weights chart."""
    colors = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16']
    
    fig = go.Figure(data=[
        go.Bar(
            x=symbols, 
            y=weights, 
            marker_color=colors[:len(symbols)],
            marker_line=dict(width=2, color='rgba(255,255,255,0.3)'),
            hovertemplate='<b>%{x}</b><br>Weight: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Portfolio Allocation',
            font=dict(size=20, color='#ffffff'),
            x=0.5
        ),
        xaxis_title='Assets',
        yaxis_title='Weight',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#2a2a2a',
            zerolinecolor='#2a2a2a',
            showgrid=False
        ),
        yaxis=dict(
            gridcolor='#2a2a2a',
            zerolinecolor='#2a2a2a',
            showgrid=True
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=False
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_correlation_chart(returns):
    """Create correlation heatmap."""
    corr_matrix = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"}
    ))
    
    fig.update_layout(
        title=dict(
            text='Asset Correlation Matrix',
            font=dict(size=20, color='#ffffff'),
            x=0.5
        ),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ffffff'),
        margin=dict(l=60, r=60, t=80, b=60),
        xaxis=dict(
            side='bottom',
            tickfont=dict(size=12, color='#ffffff')
        ),
        yaxis=dict(
            tickfont=dict(size=12, color='#ffffff')
        )
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    print("🚀 Starting Simplified Smart Portfolio Optimizer Dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
