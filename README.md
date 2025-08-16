# Smart Portfolio Optimizer

A comprehensive quantitative finance project implementing Modern Portfolio Theory (Markowitz optimization) to create optimal investment portfolios.

## 🎯 What It Does

- **Multi-Asset Portfolio Optimization**: Takes multiple assets and applies Modern Portfolio Theory
- **Risk-Return Optimization**: Outputs optimal portfolio weights that maximize return for given risk tolerance
- **Efficient Frontier Visualization**: Interactive charts showing the optimal risk-return trade-off
- **Real-time Data Integration**: Fetches live market data using Yahoo Finance API
- **Advanced Risk Metrics**: Calculates Sharpe ratio, VaR, and other key performance indicators

## 🚀 Why This Project Impresses

This project demonstrates core quantitative finance concepts that are highly relevant to:
- **Investment Banks**: Portfolio construction and risk management
- **Hedge Funds**: Quantitative trading strategies
- **Asset Management**: Client portfolio optimization
- **FinTech Companies**: Algorithmic investment platforms

**HR Takeaway**: "This candidate understands portfolio theory & can translate it into real tools."

## 🛠️ Tech Stack

- **Python**: Core optimization and data processing
- **NumPy/SciPy**: Mathematical computations and statistical analysis
- **cvxpy**: Convex optimization for portfolio weights calculation
- **Flask**: Web dashboard backend
- **Plotly/Dash**: Interactive visualizations
- **yfinance**: Real-time market data
- **scikit-learn**: Additional statistical tools

## 📁 Project Structure

```
Smart-Portfolio-Optimizer/
├── portfolio_optimizer/          # Core optimization engine
│   ├── __init__.py
│   ├── optimizer.py             # Main optimization logic
│   ├── data_manager.py          # Data fetching and processing
│   └── risk_metrics.py          # Risk calculations
├── dashboard/                   # Web interface
│   ├── __init__.py
│   ├── app.py                  # Flask application
│   ├── templates/              # HTML templates
│   └── static/                 # CSS/JS files
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── visualization.py        # Chart generation
├── tests/                      # Unit tests
├── examples/                   # Example usage
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   python dashboard/app.py
   ```

3. **Access the Application**:
   Open your browser and go to `http://localhost:5000`

## 📊 Features

### Core Optimization
- **Markowitz Mean-Variance Optimization**: Traditional risk-return optimization
- **Risk Parity**: Equal risk contribution across assets
- **Maximum Sharpe Ratio**: Optimal risk-adjusted returns
- **Minimum Variance**: Lowest possible portfolio volatility

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Value at Risk (VaR)**: Potential loss estimation
- **Conditional VaR**: Expected loss beyond VaR
- **Maximum Drawdown**: Worst historical decline
- **Volatility**: Portfolio standard deviation

### Visualization
- **Efficient Frontier**: Interactive risk-return curve
- **Portfolio Weights**: Asset allocation breakdown
- **Performance Charts**: Historical returns and drawdowns
- **Correlation Matrix**: Asset relationship heatmap
- **Risk-Return Scatter**: Individual asset positioning

### Data Management
- **Real-time Data**: Live market prices via Yahoo Finance
- **Historical Analysis**: Backtesting capabilities
- **Multiple Timeframes**: Daily, weekly, monthly analysis
- **Data Validation**: Quality checks and error handling

## 🎓 Educational Value

This project demonstrates understanding of:
- **Modern Portfolio Theory**: Harry Markowitz's Nobel Prize-winning work
- **Convex Optimization**: Mathematical programming techniques
- **Financial Mathematics**: Risk metrics and statistical analysis
- **Software Engineering**: Clean code architecture and testing
- **Data Science**: Time series analysis and visualization

## 🔬 Advanced Features

- **Monte Carlo Simulation**: Portfolio scenario analysis
- **Black-Litterman Model**: Bayesian portfolio optimization
- **Factor Models**: Multi-factor risk decomposition
- **Transaction Costs**: Realistic trading constraints
- **Rebalancing Strategies**: Dynamic portfolio management

## 📈 Example Usage

```python
from portfolio_optimizer import PortfolioOptimizer

# Initialize optimizer
optimizer = PortfolioOptimizer(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])

# Get optimal portfolio
optimal_weights = optimizer.optimize_portfolio(
    target_return=0.12,
    risk_free_rate=0.02
)

# Calculate risk metrics
sharpe_ratio = optimizer.calculate_sharpe_ratio(optimal_weights)
var_95 = optimizer.calculate_var(optimal_weights, confidence=0.95)

print(f"Optimal Weights: {optimal_weights}")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"95% VaR: {var_95:.3f}")
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📝 License

MIT License - feel free to use this project for educational and commercial purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This project is for educational purposes. Always consult with financial professionals before making investment decisions.
