"""
Backtester Module
=================
Implements backtesting framework for evaluating trading strategies.

Supports:
- VectorBT (recommended for speed)
- Custom backtester (for more control)

Key Features:
- Realistic transaction costs
- Position sizing
- Performance metrics (Sharpe, Sortino, Max Drawdown, etc.)
- Benchmark comparisons
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import vectorbt
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    print("VectorBT not available. Using custom backtester only.")


class StrategyBacktester:
    """
    Custom backtester with detailed metrics and trade logging.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost per trade (0.001 = 0.1%)
            slippage: Slippage per trade (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.results = None
        self.trades = []

    def run(
        self,
        prices: pd.Series,
        signals: np.ndarray,
        position_size: float = 0.95
    ) -> dict:
        """
        Run backtest simulation.

        Args:
            prices: Series of closing prices with DatetimeIndex
            signals: Array of signals (1=buy, -1=sell, 0=hold)
            position_size: Fraction of capital to use per trade (0.95 = 95%)

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        position = 0
        shares = 0

        portfolio_values = []
        self.trades = []

        # Ensure signals is same length as prices
        if len(signals) < len(prices):
            # Pad signals at the beginning
            padding = len(prices) - len(signals)
            signals = np.concatenate([np.zeros(padding), signals])

        for i in range(len(prices)):
            price = prices.iloc[i]
            signal = signals[i] if i < len(signals) else 0

            # Apply slippage to price
            if signal == 1:
                execution_price = price * (1 + self.slippage)
            elif signal == -1:
                execution_price = price * (1 - self.slippage)
            else:
                execution_price = price

            # Execute trades
            if signal == 1 and position == 0:  # Buy
                trade_capital = capital * position_size
                shares = trade_capital / execution_price
                cost = shares * execution_price * self.transaction_cost
                capital -= (shares * execution_price + cost)
                position = 1

                self.trades.append({
                    'date': prices.index[i],
                    'type': 'BUY',
                    'price': execution_price,
                    'shares': shares,
                    'cost': cost,
                    'capital_after': capital
                })

            elif signal == -1 and position == 1:  # Sell
                proceeds = shares * execution_price
                cost = proceeds * self.transaction_cost
                capital += (proceeds - cost)

                # Calculate trade P&L
                entry_price = self.trades[-1]['price'] if self.trades else 0
                pnl = (execution_price - entry_price) * shares - cost - self.trades[-1].get('cost', 0)

                self.trades.append({
                    'date': prices.index[i],
                    'type': 'SELL',
                    'price': execution_price,
                    'shares': shares,
                    'proceeds': proceeds,
                    'cost': cost,
                    'pnl': pnl,
                    'capital_after': capital
                })

                shares = 0
                position = 0

            # Calculate portfolio value
            portfolio_value = capital + (shares * price)
            portfolio_values.append(portfolio_value)

        # Create results
        portfolio_values = pd.Series(portfolio_values, index=prices.index)
        returns = portfolio_values.pct_change().dropna()

        self.results = {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'trades': self.trades,
            'final_value': portfolio_values.iloc[-1],
            'total_return': (portfolio_values.iloc[-1] / self.initial_capital - 1) * 100,
            'num_trades': len(self.trades)
        }

        return self.results

    def calculate_metrics(self) -> dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        returns = self.results['returns']
        portfolio_values = self.results['portfolio_values']
        n_years = len(returns) / 252

        # Basic returns
        total_return = self.results['total_return']
        annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # Sharpe Ratio (assuming risk-free rate of 0)
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calmar Ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win Rate
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        winning_trades = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
        win_rate = (winning_trades / len(sell_trades) * 100) if sell_trades else 0

        # Average trade P&L
        avg_pnl = np.mean([t.get('pnl', 0) for t in sell_trades]) if sell_trades else 0

        # Profit Factor
        gross_profit = sum(t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Annual Return (%)': round(annual_return, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Sortino Ratio': round(sortino, 3),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Calmar Ratio': round(calmar, 3),
            'Win Rate (%)': round(win_rate, 2),
            'Profit Factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'Total Trades': len(self.trades),
            'Avg Trade P&L': round(avg_pnl, 2)
        }

        return metrics

    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)


class VectorBTBacktester:
    """
    VectorBT-based backtester (faster for large datasets).
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        fees: float = 0.001
    ):
        """
        Initialize VectorBT backtester.

        Args:
            initial_capital: Starting capital
            fees: Transaction fees (0.001 = 0.1%)
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not installed")

        self.initial_capital = initial_capital
        self.fees = fees
        self.portfolio = None

    def run(
        self,
        prices: pd.Series,
        signals: np.ndarray
    ) -> dict:
        """
        Run backtest using VectorBT.

        Args:
            prices: Series of closing prices
            signals: Array of signals (1=buy, -1=sell, 0=hold)

        Returns:
            Dictionary with backtest results
        """
        # Ensure signals is aligned with prices
        if len(signals) < len(prices):
            padding = len(prices) - len(signals)
            signals = np.concatenate([np.zeros(padding), signals])

        # Create entries and exits
        entries = pd.Series(signals == 1, index=prices.index)
        exits = pd.Series(signals == -1, index=prices.index)

        # Create portfolio
        self.portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            freq='1D'
        )

        # Get statistics
        stats = self.portfolio.stats()

        return {
            'portfolio': self.portfolio,
            'stats': stats
        }

    def calculate_metrics(self) -> dict:
        """Get performance metrics from VectorBT portfolio."""
        if self.portfolio is None:
            raise ValueError("Run backtest first")

        stats = self.portfolio.stats()

        metrics = {
            'Total Return (%)': round(stats.get('Total Return [%]', 0), 2),
            'Annual Return (%)': round(stats.get('Total Return [%]', 0) / (len(self.portfolio.returns()) / 252), 2),
            'Sharpe Ratio': round(stats.get('Sharpe Ratio', 0), 3),
            'Sortino Ratio': round(stats.get('Sortino Ratio', 0), 3),
            'Max Drawdown (%)': round(stats.get('Max Drawdown [%]', 0), 2),
            'Win Rate (%)': round(stats.get('Win Rate [%]', 0), 2),
            'Profit Factor': round(stats.get('Profit Factor', 0), 2),
            'Total Trades': int(stats.get('Total Trades', 0))
        }

        return metrics


def calculate_buy_and_hold(
    prices: pd.Series,
    initial_capital: float = 100000
) -> dict:
    """
    Calculate Buy & Hold benchmark performance.

    Args:
        prices: Series of closing prices
        initial_capital: Starting capital

    Returns:
        Dictionary of metrics
    """
    shares = initial_capital / prices.iloc[0]
    portfolio_values = shares * prices
    returns = portfolio_values.pct_change().dropna()

    final_value = portfolio_values.iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    n_years = len(returns) / 252

    # Sharpe Ratio
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    # Sortino Ratio
    negative_returns = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / negative_returns.std() if len(negative_returns) > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    return {
        'Strategy': 'Buy & Hold',
        'Total Return (%)': round(total_return, 2),
        'Annual Return (%)': round(((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Sortino Ratio': round(sortino, 3),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Win Rate (%)': 'N/A',
        'Total Trades': 1
    }


def single_factor_benchmark(
    prices: pd.Series,
    factor_signal: np.ndarray,
    factor_name: str,
    initial_capital: float = 100000
) -> dict:
    """
    Run single-factor strategy benchmark.

    Args:
        prices: Series of closing prices
        factor_signal: Array of signals
        factor_name: Name of the factor
        initial_capital: Starting capital

    Returns:
        Dictionary of metrics
    """
    backtester = StrategyBacktester(initial_capital=initial_capital)
    backtester.run(prices, factor_signal)
    metrics = backtester.calculate_metrics()
    metrics['Strategy'] = f'Single Factor: {factor_name}'

    return metrics


def generate_comparison_report(
    strategy_metrics: dict,
    benchmark_metrics_list: list
) -> pd.DataFrame:
    """
    Generate comprehensive comparison report.

    Args:
        strategy_metrics: Metrics from main strategy
        benchmark_metrics_list: List of benchmark metrics

    Returns:
        Comparison DataFrame
    """
    all_metrics = [strategy_metrics] + benchmark_metrics_list

    comparison_df = pd.DataFrame(all_metrics)

    if 'Strategy' in comparison_df.columns:
        comparison_df = comparison_df.set_index('Strategy')

    # Add ranking for key metrics
    for col in ['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)']:
        if col in comparison_df.columns:
            numeric_col = pd.to_numeric(comparison_df[col], errors='coerce')
            comparison_df[f'{col} Rank'] = numeric_col.rank(ascending=False)

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON REPORT")
    print("=" * 80)
    print(comparison_df.to_string())
    print("=" * 80)

    return comparison_df


def run_backtest(
    prices: pd.Series,
    signals: np.ndarray,
    initial_capital: float = 100000,
    use_vectorbt: bool = True
) -> tuple:
    """
    Run backtest with optional VectorBT.

    Args:
        prices: Series of closing prices
        signals: Array of signals
        initial_capital: Starting capital
        use_vectorbt: Whether to use VectorBT (if available)

    Returns:
        Tuple of (portfolio_values, metrics)
    """
    if use_vectorbt and VECTORBT_AVAILABLE:
        backtester = VectorBTBacktester(initial_capital=initial_capital)
        results = backtester.run(prices, signals)
        metrics = backtester.calculate_metrics()
        portfolio_values = backtester.portfolio.value()
    else:
        backtester = StrategyBacktester(initial_capital=initial_capital)
        results = backtester.run(prices, signals)
        metrics = backtester.calculate_metrics()
        portfolio_values = results['portfolio_values']

    return portfolio_values, metrics


def main():
    """Test the backtester module."""
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer

    # Load data
    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo("SPY", start_date="2020-01-01", end_date="2024-12-01")

    # Engineer features
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    df = engineer.generate_rule_based_signals(df)
    df = df.dropna()

    # Get prices and signals
    prices = df['Close']
    signals = df['signal'].values

    print("\n" + "="*60)
    print("BACKTESTING RULE-BASED STRATEGY")
    print("="*60)

    # Run backtest
    backtester = StrategyBacktester(initial_capital=100000)
    backtester.run(prices, signals)
    strategy_metrics = backtester.calculate_metrics()
    strategy_metrics['Strategy'] = 'AI Multi-Factor'

    # Calculate benchmarks
    buy_hold = calculate_buy_and_hold(prices)

    # RSI-only benchmark
    rsi_signals = np.where(df['rsi_14'] < 30, 1, np.where(df['rsi_14'] > 70, -1, 0))
    rsi_metrics = single_factor_benchmark(prices, rsi_signals, 'RSI Only')

    # Momentum-only benchmark
    momentum_signals = np.where(df['momentum_21d'] > 0.02, 1, np.where(df['momentum_21d'] < -0.02, -1, 0))
    momentum_metrics = single_factor_benchmark(prices, momentum_signals, 'Momentum Only')

    # Generate comparison
    comparison = generate_comparison_report(
        strategy_metrics,
        [buy_hold, rsi_metrics, momentum_metrics]
    )

    # Save comparison
    comparison.to_csv('reports/strategy_comparison.csv')
    print("\nComparison saved to reports/strategy_comparison.csv")

    return comparison


if __name__ == "__main__":
    comparison = main()
