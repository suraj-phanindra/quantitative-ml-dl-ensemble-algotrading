"""
Advanced Backtester V2
======================
Improvements over V1:
1. Stop-loss and take-profit orders
2. Trailing stops
3. Position sizing based on volatility (Kelly/ATR-based)
4. Regime-aware trading (reduce exposure in high vol)
5. Maximum drawdown circuit breaker
6. Detailed trade analytics
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RiskManager:
    """
    Risk management system for position sizing and stop-losses.
    """

    def __init__(
        self,
        max_position_size: float = 0.95,
        stop_loss_pct: float = 0.03,  # 3% stop-loss
        take_profit_pct: float = 0.06,  # 6% take-profit (2:1 ratio)
        trailing_stop_pct: float = 0.02,  # 2% trailing stop
        max_drawdown_limit: float = 0.15,  # 15% max drawdown circuit breaker
        volatility_scaling: bool = True,
        target_volatility: float = 0.15  # 15% annual target vol
    ):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.volatility_scaling = volatility_scaling
        self.target_volatility = target_volatility

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: float = None,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on volatility and confidence.

        Uses volatility targeting: if vol is high, reduce position.
        """
        base_size = self.max_position_size

        # Volatility adjustment
        if self.volatility_scaling and volatility is not None and volatility > 0:
            vol_scalar = min(self.target_volatility / volatility, 1.5)
            vol_scalar = max(vol_scalar, 0.3)  # Don't go below 30%
            base_size *= vol_scalar

        # Confidence adjustment
        base_size *= confidence

        # Calculate shares
        position_capital = capital * base_size
        shares = position_capital / price

        return shares, base_size

    def check_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        highest_price: float = None
    ) -> tuple:
        """
        Check if stop-loss or take-profit is triggered.

        Returns: (triggered, trigger_type)
        """
        pnl_pct = (current_price - entry_price) / entry_price

        # Stop-loss check
        if pnl_pct <= -self.stop_loss_pct:
            return True, 'stop_loss'

        # Take-profit check
        if pnl_pct >= self.take_profit_pct:
            return True, 'take_profit'

        # Trailing stop check
        if highest_price is not None and self.trailing_stop_pct > 0:
            trailing_stop_price = highest_price * (1 - self.trailing_stop_pct)
            if current_price <= trailing_stop_price and current_price > entry_price:
                return True, 'trailing_stop'

        return False, None

    def check_drawdown_limit(
        self,
        current_value: float,
        peak_value: float
    ) -> bool:
        """Check if max drawdown limit is breached."""
        drawdown = (peak_value - current_value) / peak_value
        return drawdown >= self.max_drawdown_limit


class AdvancedBacktester:
    """
    Advanced backtester with risk management.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        risk_manager: RiskManager = None
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_manager = risk_manager or RiskManager()

        self.results = None
        self.trades = []
        self.daily_stats = []

    def run(
        self,
        prices: pd.Series,
        signals: np.ndarray,
        probabilities: np.ndarray = None,
        volatility: pd.Series = None,
        regime: pd.Series = None
    ) -> dict:
        """
        Run backtest with risk management.

        Args:
            prices: Close prices
            signals: Trading signals (1=buy, -1=sell, 0=hold)
            probabilities: Model confidence (0-1)
            volatility: Annualized volatility series
            regime: Market regime (1=bull, 0=neutral, -1=bear)
        """
        capital = self.initial_capital
        position = 0
        shares = 0
        entry_price = 0
        highest_price_since_entry = 0

        portfolio_values = []
        self.trades = []
        peak_value = self.initial_capital
        trading_halted = False

        # Ensure signals match prices length
        if len(signals) < len(prices):
            padding = len(prices) - len(signals)
            signals = np.concatenate([np.zeros(padding), signals])
        if probabilities is not None and len(probabilities) < len(prices):
            padding = len(prices) - len(probabilities)
            probabilities = np.concatenate([np.ones(padding) * 0.5, probabilities])

        for i in range(len(prices)):
            price = prices.iloc[i]
            date = prices.index[i]
            signal = signals[i] if i < len(signals) else 0
            prob = probabilities[i] if probabilities is not None else 0.5
            vol = volatility.iloc[i] if volatility is not None and i < len(volatility) else 0.15

            # Check regime filter (don't trade in extreme bear)
            if regime is not None and i < len(regime):
                current_regime = regime.iloc[i]
                if current_regime == -1 and signal == 1:  # Bear market, skip buy
                    signal = 0

            # Calculate current portfolio value
            current_value = capital + (shares * price)
            portfolio_values.append(current_value)

            # Update peak value
            if current_value > peak_value:
                peak_value = current_value

            # Check drawdown circuit breaker
            if self.risk_manager.check_drawdown_limit(current_value, peak_value):
                if not trading_halted:
                    print(f"Circuit breaker triggered at {date}: DD = {(peak_value - current_value) / peak_value * 100:.1f}%")
                    trading_halted = True
                    # Force close position
                    if position == 1:
                        signal = -1

            # If trading halted, only allow sells
            if trading_halted and signal == 1:
                signal = 0

            # Check stop-loss/take-profit for existing position
            if position == 1:
                highest_price_since_entry = max(highest_price_since_entry, price)
                triggered, trigger_type = self.risk_manager.check_stop_loss(
                    price, entry_price, highest_price_since_entry
                )
                if triggered:
                    signal = -1  # Force sell

            # Execute trades
            if signal == 1 and position == 0:  # Buy
                # Calculate confidence from probability
                confidence = abs(prob - 0.5) * 2  # Scale to 0-1

                # Position sizing
                shares, position_pct = self.risk_manager.calculate_position_size(
                    capital, price * (1 + self.slippage), vol, confidence
                )

                execution_price = price * (1 + self.slippage)
                cost = shares * execution_price * self.transaction_cost
                capital -= (shares * execution_price + cost)
                position = 1
                entry_price = execution_price
                highest_price_since_entry = price

                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': execution_price,
                    'shares': shares,
                    'cost': cost,
                    'position_pct': position_pct,
                    'confidence': confidence,
                    'volatility': vol
                })

            elif signal == -1 and position == 1:  # Sell
                execution_price = price * (1 - self.slippage)
                proceeds = shares * execution_price
                cost = proceeds * self.transaction_cost
                capital += (proceeds - cost)

                # Calculate trade P&L
                pnl = (execution_price - entry_price) * shares - cost - self.trades[-1]['cost']
                pnl_pct = (execution_price - entry_price) / entry_price * 100
                holding_days = (date - self.trades[-1]['date']).days

                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': execution_price,
                    'shares': shares,
                    'proceeds': proceeds,
                    'cost': cost,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_days': holding_days,
                    'exit_reason': trigger_type if 'trigger_type' in dir() and triggered else 'signal'
                })

                shares = 0
                position = 0
                entry_price = 0

        # Final portfolio value
        portfolio_values = pd.Series(portfolio_values, index=prices.index)
        returns = portfolio_values.pct_change().dropna()

        self.results = {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'trades': self.trades,
            'final_value': portfolio_values.iloc[-1],
            'total_return': (portfolio_values.iloc[-1] / self.initial_capital - 1) * 100
        }

        return self.results

    def calculate_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if self.results is None:
            raise ValueError("Run backtest first")

        returns = self.results['returns']
        portfolio_values = self.results['portfolio_values']
        n_days = len(returns)
        n_years = n_days / 252

        # Basic returns
        total_return = self.results['total_return']
        annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # Risk metrics
        daily_std = returns.std()
        sharpe = np.sqrt(252) * returns.mean() / daily_std if daily_std > 0 else 0

        negative_returns = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / negative_returns.std() if len(negative_returns) > 0 else 0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade analysis
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']

        if sell_trades:
            winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]

            win_rate = len(winning_trades) / len(sell_trades) * 100

            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0

            profit_factor = sum(t['pnl'] for t in winning_trades) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')

            avg_holding = np.mean([t.get('holding_days', 0) for t in sell_trades])

            # Exit reason breakdown
            stop_loss_exits = len([t for t in sell_trades if t.get('exit_reason') == 'stop_loss'])
            take_profit_exits = len([t for t in sell_trades if t.get('exit_reason') == 'take_profit'])
            trailing_stop_exits = len([t for t in sell_trades if t.get('exit_reason') == 'trailing_stop'])
            signal_exits = len([t for t in sell_trades if t.get('exit_reason') == 'signal'])
        else:
            win_rate = 0
            avg_win = avg_loss = profit_factor = avg_holding = 0
            stop_loss_exits = take_profit_exits = trailing_stop_exits = signal_exits = 0

        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Annual Return (%)': round(annual_return, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Sortino Ratio': round(sortino, 3),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Calmar Ratio': round(calmar, 3),
            'Win Rate (%)': round(win_rate, 2),
            'Profit Factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'Avg Win ($)': round(avg_win, 2),
            'Avg Loss ($)': round(avg_loss, 2),
            'Avg Holding Days': round(avg_holding, 1),
            'Total Trades': len(self.trades),
            'Round Trips': len(sell_trades),
            'Stop Loss Exits': stop_loss_exits,
            'Take Profit Exits': take_profit_exits,
            'Trailing Stop Exits': trailing_stop_exits,
            'Signal Exits': signal_exits
        }

        return metrics

    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_monthly_returns(self) -> pd.DataFrame:
        """Get monthly returns breakdown."""
        returns = self.results['returns']
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        return monthly


def calculate_benchmark_v2(
    prices: pd.Series,
    initial_capital: float = 100000
) -> dict:
    """Enhanced Buy & Hold benchmark."""
    shares = initial_capital / prices.iloc[0]
    portfolio_values = shares * prices
    returns = portfolio_values.pct_change().dropna()

    total_return = (portfolio_values.iloc[-1] / initial_capital - 1) * 100
    n_years = len(returns) / 252

    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    return {
        'Strategy': 'Buy & Hold',
        'Total Return (%)': round(total_return, 2),
        'Annual Return (%)': round(((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Win Rate (%)': 'N/A',
        'Total Trades': 1
    }


def generate_comparison_report_v2(
    strategy_metrics: dict,
    benchmark_metrics_list: list,
    save_path: str = None
) -> pd.DataFrame:
    """Generate enhanced comparison report."""
    all_metrics = [strategy_metrics] + benchmark_metrics_list

    comparison_df = pd.DataFrame(all_metrics)
    if 'Strategy' in comparison_df.columns:
        comparison_df = comparison_df.set_index('Strategy')

    print("\n" + "=" * 90)
    print("STRATEGY PERFORMANCE COMPARISON V2")
    print("=" * 90)
    print(comparison_df.to_string())
    print("=" * 90)

    if save_path:
        comparison_df.to_csv(save_path)
        print(f"Saved to {save_path}")

    return comparison_df


def main():
    """Test advanced backtester."""
    import sys
    sys.path.insert(0, '.')
    from data_loader import DataLoader
    from feature_engineering_v2 import EnhancedFeatureEngineer

    # Load data
    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo("SPY", start_date="2020-01-01", end_date="2024-12-01")

    # Features
    engineer = EnhancedFeatureEngineer()
    df = engineer.create_all_features(df)
    df = df.dropna()

    # Simulate signals based on RSI + MACD (rule-based for testing)
    signals = np.where(
        (df['rsi_14'] < 35) & (df['macd_histogram'] > df['macd_histogram'].shift(1)), 1,
        np.where((df['rsi_14'] > 65) & (df['macd_histogram'] < df['macd_histogram'].shift(1)), -1, 0)
    )

    # Simulate probabilities
    proba = np.where(signals == 1, 0.7, np.where(signals == -1, 0.3, 0.5))

    # Run backtest
    risk_manager = RiskManager(
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        trailing_stop_pct=0.02,
        max_drawdown_limit=0.15
    )

    backtester = AdvancedBacktester(
        initial_capital=100000,
        risk_manager=risk_manager
    )

    results = backtester.run(
        df['Close'],
        signals,
        proba,
        df['volatility_20d']
    )

    metrics = backtester.calculate_metrics()

    print("\n" + "=" * 60)
    print("BACKTEST V2 RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Compare with buy & hold
    bh = calculate_benchmark_v2(df['Close'])
    print(f"\nBuy & Hold Return: {bh['Total Return (%)']}%")

    return metrics


if __name__ == "__main__":
    metrics = main()
