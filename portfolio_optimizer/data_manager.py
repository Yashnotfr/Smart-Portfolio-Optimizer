"""
Data Manager for Portfolio Optimizer

Handles fetching, processing, and validating financial data from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages financial data fetching, processing, and validation.
    
    This class handles:
    - Real-time data fetching from Yahoo Finance
    - Historical price data retrieval
    - Data cleaning and validation
    - Return calculations
    - Correlation and covariance matrices
    """
    
    def __init__(self, symbols: List[str], start_date: str = None, end_date: str = None):
        """
        Initialize DataManager with asset symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
        """
        self.symbols = [s.upper() for s in symbols]
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        self.price_data = None
        self.returns_data = None
        self.correlation_matrix = None
        self.covariance_matrix = None
        self.mean_returns = None
        
        # Fetch data on initialization
        self._fetch_data()
    
    def _fetch_data(self) -> None:
        """Fetch historical price data for all symbols."""
        try:
            logger.info(f"Fetching data for symbols: {self.symbols}")
            
            # Download data using yfinance
            data = yf.download(
                self.symbols,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            # Handle single symbol case
            if len(self.symbols) == 1:
                data.columns = pd.MultiIndex.from_product([['Close'], self.symbols])
            
            # Extract closing prices
            if 'Close' in data.columns:
                self.price_data = data['Close']
            else:
                self.price_data = data
            
            # Clean data
            self._clean_data()
            
            # Calculate returns and statistics
            self._calculate_returns()
            self._calculate_statistics()
            
            logger.info(f"Successfully fetched data for {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _clean_data(self) -> None:
        """Clean and validate price data."""
        if self.price_data is None:
            return
        
        # Remove rows with all NaN values
        self.price_data = self.price_data.dropna(how='all')
        
        # Forward fill missing values (up to 5 consecutive days)
        self.price_data = self.price_data.fillna(method='ffill', limit=5)
        
        # Remove symbols with too much missing data (>20%)
        missing_pct = self.price_data.isnull().sum() / len(self.price_data)
        valid_symbols = missing_pct[missing_pct < 0.2].index
        self.price_data = self.price_data[valid_symbols]
        
        # Update symbols list
        self.symbols = list(valid_symbols)
        
        # Remove remaining rows with any NaN values
        self.price_data = self.price_data.dropna()
        
        logger.info(f"Cleaned data: {len(self.symbols)} symbols, {len(self.price_data)} observations")
    
    def _calculate_returns(self) -> None:
        """Calculate daily returns from price data."""
        if self.price_data is None:
            return
        
        # Calculate daily returns
        self.returns_data = self.price_data.pct_change().dropna()
        
        # Annualize returns (252 trading days)
        self.mean_returns = self.returns_data.mean() * 252
        
        logger.info("Calculated returns and mean returns")
    
    def _calculate_statistics(self) -> None:
        """Calculate correlation and covariance matrices."""
        if self.returns_data is None:
            return
        
        # Calculate correlation matrix
        self.correlation_matrix = self.returns_data.corr()
        
        # Calculate covariance matrix (annualized)
        self.covariance_matrix = self.returns_data.cov() * 252
        
        logger.info("Calculated correlation and covariance matrices")
    
    def get_price_data(self) -> pd.DataFrame:
        """Get historical price data."""
        return self.price_data
    
    def get_returns_data(self) -> pd.DataFrame:
        """Get daily returns data."""
        return self.returns_data
    
    def get_mean_returns(self) -> pd.Series:
        """Get annualized mean returns."""
        return self.mean_returns
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix."""
        return self.correlation_matrix
    
    def get_covariance_matrix(self) -> pd.DataFrame:
        """Get annualized covariance matrix."""
        return self.covariance_matrix
    
    def get_volatility(self) -> pd.Series:
        """Get annualized volatility for each asset."""
        if self.returns_data is None:
            return pd.Series()
        return self.returns_data.std() * np.sqrt(252)
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> pd.Series:
        """Calculate Sharpe ratio for each asset."""
        if self.mean_returns is None:
            return pd.Series()
        
        volatility = self.get_volatility()
        excess_returns = self.mean_returns - risk_free_rate
        return excess_returns / volatility
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive data summary."""
        if self.returns_data is None:
            return {}
        
        summary = {
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'observations': len(self.returns_data),
            'mean_returns': self.mean_returns.to_dict(),
            'volatility': self.get_volatility().to_dict(),
            'sharpe_ratio': self.get_sharpe_ratio().to_dict(),
            'min_return': self.returns_data.min().to_dict(),
            'max_return': self.returns_data.max().to_dict(),
            'skewness': self.returns_data.skew().to_dict(),
            'kurtosis': self.returns_data.kurtosis().to_dict()
        }
        
        return summary
    
    def update_data(self, symbols: List[str] = None, 
                   start_date: str = None, 
                   end_date: str = None) -> None:
        """Update data with new parameters."""
        if symbols:
            self.symbols = [s.upper() for s in symbols]
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        
        self._fetch_data()
    
    def get_rolling_statistics(self, window: int = 252) -> Dict:
        """Calculate rolling statistics for portfolio analysis."""
        if self.returns_data is None:
            return {}
        
        rolling_stats = {
            'rolling_mean': self.returns_data.rolling(window=window).mean() * 252,
            'rolling_vol': self.returns_data.rolling(window=window).std() * np.sqrt(252),
            'rolling_sharpe': (self.returns_data.rolling(window=window).mean() * 252 - 0.02) / 
                             (self.returns_data.rolling(window=window).std() * np.sqrt(252))
        }
        
        return rolling_stats
    
    def validate_data_quality(self) -> Dict:
        """Validate data quality and return quality metrics."""
        if self.returns_data is None:
            return {'status': 'No data available'}
        
        quality_metrics = {
            'total_observations': len(self.returns_data),
            'missing_values': self.returns_data.isnull().sum().sum(),
            'missing_percentage': (self.returns_data.isnull().sum().sum() / 
                                 (len(self.returns_data) * len(self.symbols))) * 100,
            'outliers_3sigma': ((self.returns_data.abs() > 3 * self.returns_data.std()).sum() / 
                               len(self.returns_data)).to_dict(),
            'data_completeness': (1 - self.returns_data.isnull().sum() / len(self.returns_data)).to_dict()
        }
        
        # Determine overall quality status
        if quality_metrics['missing_percentage'] < 5 and quality_metrics['total_observations'] > 100:
            quality_metrics['status'] = 'Excellent'
        elif quality_metrics['missing_percentage'] < 10 and quality_metrics['total_observations'] > 50:
            quality_metrics['status'] = 'Good'
        else:
            quality_metrics['status'] = 'Poor'
        
        return quality_metrics
