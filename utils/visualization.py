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
    Create interactive efficient frontier plot with modern styling.
    
    Args:
        efficient_frontier_data: Dictionary with efficient frontier data
        individual_assets: DataFrame with individual asset returns and volatilities
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add efficient frontier with gradient colors
    fig.add_trace(go.Scatter(
        x=efficient_frontier_data['volatilities'],
        y=efficient_frontier_data['returns'],
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='#6366f1', width=3),
        marker=dict(
            size=8, 
            color=efficient_frontier_data['returns'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Return',
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                outlinewidth=0,
                bgcolor='rgba(0,0,0,0)'
            )
        ),
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
            marker=dict(
                size=12, 
                color='#ef4444', 
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            text=individual_assets.index,
            textposition='top center',
            textfont=dict(color='white', size=12),
            hovertemplate='<b>Asset:</b> %{text}<br>' +
                         '<b>Volatility:</b> %{x:.3f}<br>' +
                         '<b>Return:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text='Efficient Frontier',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Portfolio Volatility (Annualized)',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Expected Return (Annualized)',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=80, r=80, t=80, b=80),
        width=800,
        height=600,
        hovermode='closest',
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def create_portfolio_weights_chart(weights: Dict[str, float], 
                                 title: str = 'Portfolio Weights') -> go.Figure:
    """
    Create portfolio weights bar chart with modern styling.
    
    Args:
        weights: Dictionary of asset weights
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
    df = df.sort_values('Weight', ascending=True)
    
    # Create a modern color palette
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#ef4444', '#f97316', 
              '#eab308', '#84cc16', '#22c55e', '#14b8a6', '#06b6d4']
    
    fig = go.Figure()
    
    # Create horizontal bar chart with gradient colors
    fig.add_trace(go.Bar(
        x=df['Weight'],
        y=df['Asset'],
        orientation='h',
        marker=dict(
            color=df['Weight'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Weight',
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                outlinewidth=0,
                bgcolor='rgba(0,0,0,0)'
            ),
            line=dict(width=0)
        ),
        hovertemplate='<b>%{y}</b><br>' +
                     '<b>Weight:</b> %{x:.1%}<br>' +
                     '<extra></extra>',
        text=[f'{w:.1%}' for w in df['Weight']],
        textposition='auto',
        textfont=dict(color='white', size=12)
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Portfolio Weight',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[0, max(weights.values()) * 1.1]
        ),
        yaxis=dict(
            title='Assets',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=80, r=80, t=80, b=80),
        width=800,
        height=400,
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """
    Create correlation matrix heatmap with modern styling.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        
    Returns:
        Plotly figure object
    """
    # Create custom colorscale for better visualization
    colorscale = [
        [0, '#1f2937'],      # Dark gray for negative correlations
        [0.25, '#374151'],   # Medium gray
        [0.5, '#6b7280'],    # Neutral gray
        [0.75, '#d1d5db'],   # Light gray
        [1, '#ffffff']       # White for positive correlations
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(
            title='Correlation',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            outlinewidth=0,
            bgcolor='rgba(0,0,0,0)',
            len=0.8
        ),
        hovertemplate='<b>%{y} vs %{x}</b><br>' +
                     '<b>Correlation:</b> %{z:.3f}<br>' +
                     '<extra></extra>',
        text=correlation_matrix.round(3).values,
        texttemplate='%{text}',
        textfont=dict(color='white', size=12),
        hoverongaps=False
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text='Asset Correlation Matrix',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Assets',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            side='bottom'
        ),
        yaxis=dict(
            title='Assets',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            side='left'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=80, r=80, t=80, b=80),
        width=700,
        height=600
    )
    
    return fig

def create_performance_chart(portfolio_returns: pd.Series, 
                           benchmark_returns: pd.Series = None,
                           title: str = 'Portfolio Performance') -> go.Figure:
    """
    Create portfolio performance chart with cumulative returns and modern styling.
    
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
    
    # Add portfolio performance with gradient fill
    fig.add_trace(go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#6366f1', width=3),
        fill='tonexty',
        fillcolor='rgba(99, 102, 241, 0.1)',
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
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Cumulative Return:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Date',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Cumulative Return',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=80, r=80, t=80, b=80),
        width=900,
        height=500,
        hovermode='x unified',
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def create_drawdown_chart(portfolio_returns: pd.Series) -> go.Figure:
    """
    Create drawdown chart with modern styling.
    
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
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='#ef4444', width=3),
        name='Drawdown',
        hovertemplate='<b>Date:</b> %{x}<br>' +
                     '<b>Drawdown:</b> %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text='Portfolio Drawdown',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Date',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Drawdown (%)',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[drawdown.min() * 100 * 1.1, 5]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=80, r=80, t=80, b=80),
        width=900,
        height=400,
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.2)'
        )
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
        title=dict(
            text='Portfolio Risk Metrics Dashboard',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
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
        line=dict(color='#6366f1', width=3)
    ), row=1, col=1)
    
    # Rolling Volatility
    fig.add_trace(go.Scatter(
        x=rolling_data.index,
        y=rolling_data['rolling_vol'] * 100,
        mode='lines',
        name='Rolling Volatility',
        line=dict(color='#ef4444', width=3)
    ), row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text='Rolling Portfolio Metrics',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        width=900,
        height=600,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1, titlefont=dict(color='white'), tickfont=dict(color='white'))
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1, titlefont=dict(color='white'), tickfont=dict(color='white'))
    fig.update_xaxes(titlefont=dict(color='white'), tickfont=dict(color='white'))
    
    return fig

def create_asset_allocation_pie(weights: Dict[str, float]) -> go.Figure:
    """
    Create asset allocation pie chart with modern styling.
    
    Args:
        weights: Dictionary of asset weights
        
    Returns:
        Plotly figure object
    """
    # Modern color palette
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#ef4444', '#f97316', 
              '#eab308', '#84cc16', '#22c55e', '#14b8a6', '#06b6d4']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.4,
        textinfo='label+percent',
        textposition='inside',
        textfont=dict(color='white', size=12),
        marker=dict(colors=colors[:len(weights)]),
        hovertemplate='<b>%{label}</b><br>' +
                     '<b>Weight:</b> %{percent}<br>' +
                     '<b>Value:</b> %{value:.3f}<br>' +
                     '<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text='Asset Allocation',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
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
        marker_color='#10b981'
    ), row=1, col=1)
    
    # Volatilities
    fig.add_trace(go.Bar(
        x=methods,
        y=volatilities,
        name='Volatility',
        marker_color='#ef4444'
    ), row=1, col=2)
    
    # Sharpe Ratios
    fig.add_trace(go.Bar(
        x=methods,
        y=sharpe_ratios,
        name='Sharpe Ratio',
        marker_color='#6366f1'
    ), row=1, col=3)
    
    fig.update_layout(
        title=dict(
            text='Optimization Methods Comparison',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        width=1200,
        height=400,
        showlegend=False
    )
    
    # Update axes for dark theme
    fig.update_xaxes(titlefont=dict(color='white'), tickfont=dict(color='white'))
    fig.update_yaxes(titlefont=dict(color='white'), tickfont=dict(color='white'))
    
    return fig
