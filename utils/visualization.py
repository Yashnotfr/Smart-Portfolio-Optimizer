"""
Visualization utilities for portfolio analysis.

Provides functions to create interactive charts and plots for
portfolio optimization results and analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def create_efficient_frontier_plot(efficient_frontier_data: Dict, 
                                 individual_assets: pd.DataFrame = None) -> go.Figure:
    """
    Create interactive efficient frontier plot.
    
    Args:
        efficient_frontier_data: Dictionary with efficient frontier data
        individual_assets: DataFrame with individual asset returns and volatilities
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add efficient frontier
    fig.add_trace(go.Scatter(
        x=efficient_frontier_data['volatilities'],
        y=efficient_frontier_data['returns'],
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color='blue'),
        hovertemplate='<b>Volatility:</b> %{x:.3f}<br>' +
                     '<b>Return:</b> %{y:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add individual assets if provided
    if individual_assets is not None:
        fig.add_trace(go.Scatter(
            x=individual_assets['volatility'],
            y=individual_assets['return'],
            mode='markers+text',
            name='Individual Assets',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=individual_assets.index,
            textposition='top center',
            hovertemplate='<b>Asset:</b> %{text}<br>' +
                         '<b>Volatility:</b> %{x:.3f}<br>' +
                         '<b>Return:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Portfolio Volatility (Annualized)',
        yaxis_title='Expected Return (Annualized)',
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600
    )
    
    return fig

def create_portfolio_weights_chart(weights: Dict[str, float], 
                                 title: str = 'Portfolio Weights') -> go.Figure:
    """
    Create portfolio weights bar chart.
    
    Args:
        weights: Dictionary of asset weights
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
    df = df.sort_values('Weight', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Weight'],
        y=df['Asset'],
        orientation='h',
        marker=dict(
            color=df['Weight'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Weight')
        ),
        hovertemplate='<b>Asset:</b> %{y}<br>' +
                     '<b>Weight:</b> %{x:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Weight',
        yaxis_title='Asset',
        template='plotly_white',
        width=800,
        height=400,
        xaxis=dict(range=[0, max(weights.values()) * 1.1])
    )
    
    return fig

def create_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """
    Create correlation matrix heatmap.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='Correlation'),
        hovertemplate='<b>Assets:</b> %{y} vs %{x}<br>' +
                     '<b>Correlation:</b> %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Asset Correlation Matrix',
        template='plotly_white',
        width=700,
        height=600
    )
    
    return fig

def create_performance_chart(portfolio_returns: pd.Series, 
                           benchmark_returns: pd.Series = None,
                           title: str = 'Portfolio Performance') -> go.Figure:
    """
    Create portfolio performance chart with cumulative returns.
    
    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series (optional)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    fig = go.Figure()
    
    # Add portfolio performance
    fig.add_trace(go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date:</b> %{x}<br>' +
                     '<b>Cumulative Return:</b> %{y:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add benchmark if provided
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Cumulative Return:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template='plotly_white',
        width=900,
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_drawdown_chart(portfolio_returns: pd.Series) -> go.Figure:
    """
    Create drawdown chart.
    
    Args:
        portfolio_returns: Portfolio return series
        
    Returns:
        Plotly figure object
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,  # Convert to percentage
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=2),
        name='Drawdown',
        hovertemplate='<b>Date:</b> %{x}<br>' +
                     '<b>Drawdown:</b> %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        width=900,
        height=400,
        yaxis=dict(range=[drawdown.min() * 100 * 1.1, 5])
    )
    
    return fig

def create_risk_metrics_dashboard(risk_metrics: Dict) -> go.Figure:
    """
    Create dashboard with key risk metrics.
    
    Args:
        risk_metrics: Dictionary with risk metrics
        
    Returns:
        Plotly figure object with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sharpe Ratio', 'Maximum Drawdown', 'Volatility', 'VaR (95%)'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Sharpe Ratio
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=risk_metrics.get('sharpe_ratio', 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sharpe Ratio"},
        gauge={'axis': {'range': [None, 2]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 0.5], 'color': "lightgray"},
                   {'range': [0.5, 1], 'color': "yellow"},
                   {'range': [1, 2], 'color': "green"}
               ],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 1.5
               }}
    ), row=1, col=1)
    
    # Maximum Drawdown
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=abs(risk_metrics.get('max_drawdown', 0)) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Max Drawdown (%)"},
        gauge={'axis': {'range': [None, 50]},
               'bar': {'color': "darkred"},
               'steps': [
                   {'range': [0, 10], 'color': "green"},
                   {'range': [10, 20], 'color': "yellow"},
                   {'range': [20, 50], 'color': "red"}
               ],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 30
               }}
    ), row=1, col=2)
    
    # Volatility
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=risk_metrics.get('volatility', 0) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Volatility (%)"},
        gauge={'axis': {'range': [None, 30]},
               'bar': {'color': "darkgreen"},
               'steps': [
                   {'range': [0, 10], 'color': "green"},
                   {'range': [10, 20], 'color': "yellow"},
                   {'range': [20, 30], 'color': "red"}
               ],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 25
               }}
    ), row=2, col=1)
    
    # VaR
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=abs(risk_metrics.get('var_95', 0)) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "VaR (95%) (%)"},
        gauge={'axis': {'range': [None, 10]},
               'bar': {'color': "darkorange"},
               'steps': [
                   {'range': [0, 3], 'color': "green"},
                   {'range': [3, 6], 'color': "yellow"},
                   {'range': [6, 10], 'color': "red"}
               ],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 8
               }}
    ), row=2, col=2)
    
    fig.update_layout(
        title='Portfolio Risk Metrics Dashboard',
        template='plotly_white',
        width=1000,
        height=600
    )
    
    return fig

def create_rolling_metrics_chart(rolling_data: pd.DataFrame) -> go.Figure:
    """
    Create rolling metrics chart.
    
    Args:
        rolling_data: DataFrame with rolling metrics
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
        vertical_spacing=0.1
    )
    
    # Rolling Sharpe Ratio
    fig.add_trace(go.Scatter(
        x=rolling_data.index,
        y=rolling_data['rolling_sharpe'],
        mode='lines',
        name='Rolling Sharpe',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    # Rolling Volatility
    fig.add_trace(go.Scatter(
        x=rolling_data.index,
        y=rolling_data['rolling_vol'] * 100,
        mode='lines',
        name='Rolling Volatility',
        line=dict(color='red', width=2)
    ), row=2, col=1)
    
    fig.update_layout(
        title='Rolling Portfolio Metrics',
        template='plotly_white',
        width=900,
        height=600,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    
    return fig

def create_asset_allocation_pie(weights: Dict[str, float]) -> go.Figure:
    """
    Create asset allocation pie chart.
    
    Args:
        weights: Dictionary of asset weights
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.3,
        textinfo='label+percent',
        textposition='inside',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title='Asset Allocation',
        template='plotly_white',
        width=600,
        height=500
    )
    
    return fig

def create_optimization_comparison_chart(optimization_results: Dict) -> go.Figure:
    """
    Create comparison chart of different optimization methods.
    
    Args:
        optimization_results: Dictionary with optimization results
        
    Returns:
        Plotly figure object
    """
    methods = list(optimization_results.keys())
    returns = [optimization_results[m]['expected_return'] for m in methods]
    volatilities = [optimization_results[m]['volatility'] for m in methods]
    sharpe_ratios = [optimization_results[m]['sharpe_ratio'] for m in methods]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Expected Return', 'Volatility', 'Sharpe Ratio'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Expected Returns
    fig.add_trace(go.Bar(
        x=methods,
        y=returns,
        name='Expected Return',
        marker_color='green'
    ), row=1, col=1)
    
    # Volatilities
    fig.add_trace(go.Bar(
        x=methods,
        y=volatilities,
        name='Volatility',
        marker_color='red'
    ), row=1, col=2)
    
    # Sharpe Ratios
    fig.add_trace(go.Bar(
        x=methods,
        y=sharpe_ratios,
        name='Sharpe Ratio',
        marker_color='blue'
    ), row=1, col=3)
    
    fig.update_layout(
        title='Optimization Methods Comparison',
        template='plotly_white',
        width=1200,
        height=400,
        showlegend=False
    )
    
    return fig
