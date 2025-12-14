"""
Data Loader Module
==================
Handles OHLC data acquisition from various sources.

Supports:
- Yahoo Finance (yfinance)
- Local CSV files
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Data loader class for fetching and managing OHLC data.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory to store/load data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_from_yahoo(
        self,
        ticker: str,
        start_date: str = "2015-01-01",
        end_date: str = None,
        interval: str = "1d",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Download OHLC data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            interval: Data interval ('1d', '1h', '5m', etc.)
            save: Whether to save data to CSV

        Returns:
            DataFrame with OHLC data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Downloading {ticker} data from {start_date} to {end_date}...")

        # Download data
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure standard column names
        df = df.rename(columns={
            'Adj Close': 'Adj_Close'
        })

        # Reset index to have Date as a column, then set it back
        df.index.name = 'Date'
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # Sort by date
        df = df.sort_index()

        print(f"Downloaded {len(df)} records")

        # Save to CSV if requested
        if save:
            filepath = self.data_dir / f"{ticker}_ohlc.csv"
            df.to_csv(filepath)
            print(f"Data saved to {filepath}")

        return df

    def download_multiple_tickers(
        self,
        tickers: list,
        start_date: str = "2015-01-01",
        end_date: str = None,
        interval: str = "1d",
        save: bool = True
    ) -> dict:
        """
        Download OHLC data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            save: Whether to save data

        Returns:
            Dictionary of DataFrames keyed by ticker
        """
        data = {}
        for ticker in tickers:
            try:
                df = self.download_from_yahoo(
                    ticker, start_date, end_date, interval, save
                )
                data[ticker] = df
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")

        return data

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLC data from a CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with OHLC data
        """
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
        df = df.sort_index()
        print(f"Loaded {len(df)} records from {filepath}")
        return df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate OHLC data quality.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        validation = {
            'total_rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'missing_values': {},
            'has_required_columns': True,
            'issues': []
        }

        # Check required columns
        for col in required_cols:
            if col not in df.columns:
                validation['has_required_columns'] = False
                validation['issues'].append(f"Missing column: {col}")

        # Check for missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                validation['missing_values'][col] = missing
                validation['issues'].append(f"{col}: {missing} missing values")

        # Check for duplicate indices
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            validation['issues'].append(f"{duplicates} duplicate dates found")

        # Check OHLC consistency
        if validation['has_required_columns']:
            invalid_ohlc = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close'])
            ).sum()

            if invalid_ohlc > 0:
                validation['issues'].append(f"{invalid_ohlc} rows with invalid OHLC relationships")

        # Check for negative values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    validation['issues'].append(f"{col}: {neg_count} negative values")

        return validation

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLC data by handling missing values and anomalies.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        # Forward fill missing values (appropriate for time series)
        df = df.ffill()

        # Backward fill any remaining missing values at the start
        df = df.bfill()

        # Remove rows where Volume is 0 (likely non-trading days)
        if 'Volume' in df.columns:
            df = df[df['Volume'] > 0]

        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for the data.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_days': len(df),
            'start_date': str(df.index.min().date()),
            'end_date': str(df.index.max().date()),
            'trading_years': round((df.index.max() - df.index.min()).days / 365, 2),
            'price_stats': {
                'min_close': round(df['Close'].min(), 2),
                'max_close': round(df['Close'].max(), 2),
                'avg_close': round(df['Close'].mean(), 2),
                'std_close': round(df['Close'].std(), 2)
            },
            'volume_stats': {
                'avg_volume': int(df['Volume'].mean()),
                'max_volume': int(df['Volume'].max())
            },
            'returns_stats': {
                'total_return': round((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100, 2),
                'avg_daily_return': round(df['Close'].pct_change().mean() * 100, 4),
                'volatility_annual': round(df['Close'].pct_change().std() * np.sqrt(252) * 100, 2)
            }
        }

        return summary


def main():
    """Test the data loader module."""
    loader = DataLoader(data_dir="data")

    # Download SPY data
    df = loader.download_from_yahoo(
        ticker="SPY",
        start_date="2015-01-01",
        end_date="2024-12-01"
    )

    # Validate data
    validation = loader.validate_data(df)
    print("\nData Validation:")
    print(f"  Total rows: {validation['total_rows']}")
    print(f"  Date range: {validation['date_range']}")
    if validation['issues']:
        print("  Issues found:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    else:
        print("  No issues found!")

    # Get summary
    summary = loader.get_data_summary(df)
    print("\nData Summary:")
    print(f"  Trading years: {summary['trading_years']}")
    print(f"  Total return: {summary['returns_stats']['total_return']}%")
    print(f"  Annual volatility: {summary['returns_stats']['volatility_annual']}%")

    return df


if __name__ == "__main__":
    df = main()
