"""
Backtester V3 - Kelly Criterion Position Sizing and Dynamic Risk Management
Implements:
- Kelly Criterion for optimal position sizing
- Dynamic stop-loss based on ATR/volatility
- Regime-aware position sizing
- Max drawdown circuit breaker
- Detailed trade analytics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    position_pct: float
    confidence: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: float
    regime: str
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_days: Optional[int] = None


class KellyCriterion:
    """
    Kelly Criterion position sizing with safety adjustments
    """

    def __init__(self, max_kelly_fraction=0.5, min_position=0.1, max_position=0.9):
        """
        Args:
            max_kelly_fraction: Maximum fraction of Kelly to use (0.5 = half-Kelly)
            min_position: Minimum position size as fraction of capital
            max_position: Maximum position size as fraction of capital
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_position = min_position
        self.max_position = max_position
        self.historical_wins = []
        self.historical_losses = []

    def update_history(self, pnl_pct: float, is_win: bool):
        """Update win/loss history for Kelly calculation"""
        if is_win:
            self.historical_wins.append(pnl_pct)
        else:
            self.historical_losses.append(abs(pnl_pct))

    def calculate_kelly(self, win_rate: float = None, avg_win: float = None,
                        avg_loss: float = None) -> float:
        """
        Calculate Kelly Criterion optimal fraction

        Kelly % = W - [(1-W)/R]
        Where:
            W = Win probability
            R = Win/Loss ratio (avg win / avg loss)
        """
        # Use historical data if not provided
        if win_rate is None:
            total = len(self.historical_wins) + len(self.historical_losses)
            if total < 10:
                return self.min_position  # Not enough data
            win_rate = len(self.historical_wins) / total

        if avg_win is None:
            avg_win = np.mean(self.historical_wins) if self.historical_wins else 0.02

        if avg_loss is None:
            avg_loss = np.mean(self.historical_losses) if self.historical_losses else 0.02

        # Avoid division by zero
        if avg_loss == 0:
            return self.min_position

        # Calculate Kelly
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply safety fraction
        kelly = kelly * self.max_kelly_fraction

        # Clamp to min/max
        kelly = max(self.min_position, min(self.max_position, kelly))

        return kelly

    def get_position_size(self, confidence: float, volatility: float,
                          regime: str, current_drawdown: float) -> float:
        """
        Calculate position size based on Kelly and adjustments

        Args:
            confidence: Model confidence (0-1)
            volatility: Current market volatility
            regime: Current market regime ('bull', 'bear', 'neutral')
            current_drawdown: Current portfolio drawdown (negative value)

        Returns:
            Position size as fraction of capital
        """
        # Base Kelly position
        base_kelly = self.calculate_kelly()

        # Confidence adjustment (higher confidence = larger position)
        confidence_mult = 0.5 + confidence  # 0.5 to 1.5

        # Volatility adjustment (higher vol = smaller position)
        vol_adjustment = min(1.0, 0.15 / max(volatility, 0.05))

        # Regime adjustment
        regime_mult = {
            'bull': 1.1,
            'neutral': 1.0,
            'bear': 0.7
        }.get(regime, 1.0)

        # Drawdown adjustment (reduce position if in drawdown)
        dd_mult = 1.0
        if current_drawdown < -0.05:
            dd_mult = 0.8
        if current_drawdown < -0.10:
            dd_mult = 0.5
        if current_drawdown < -0.15:
            dd_mult = 0.25

        # Combined position size
        position = base_kelly * confidence_mult * vol_adjustment * regime_mult * dd_mult

        # Clamp to bounds
        position = max(self.min_position, min(self.max_position, position))

        return position


class DynamicRiskManager:
    """
    Dynamic risk management with ATR-based stops
    """

    def __init__(self,
                 base_stop_loss_atr=2.0,
                 base_take_profit_atr=4.0,
                 trailing_stop_atr=1.5,
                 max_drawdown_pct=0.15):
        """
        Args:
            base_stop_loss_atr: Stop loss as multiple of ATR
            base_take_profit_atr: Take profit as multiple of ATR
            trailing_stop_atr: Trailing stop as multiple of ATR
            max_drawdown_pct: Max portfolio drawdown before stopping
        """
        self.base_stop_loss_atr = base_stop_loss_atr
        self.base_take_profit_atr = base_take_profit_atr
        self.trailing_stop_atr = trailing_stop_atr
        self.max_drawdown_pct = max_drawdown_pct

    def calculate_stops(self, entry_price: float, atr: float,
                        regime: str, confidence: float) -> Tuple[float, float, float]:
        """
        Calculate dynamic stop levels

        Returns:
            (stop_loss_price, take_profit_price, trailing_stop_distance)
        """
        # Adjust ATR multipliers based on regime and confidence
        if regime == 'bull':
            sl_mult = self.base_stop_loss_atr * 1.2  # Wider stops in bull
            tp_mult = self.base_take_profit_atr * 1.3
        elif regime == 'bear':
            sl_mult = self.base_stop_loss_atr * 0.8  # Tighter stops in bear
            tp_mult = self.base_take_profit_atr * 0.8
        else:
            sl_mult = self.base_stop_loss_atr
            tp_mult = self.base_take_profit_atr

        # Confidence adjustment (higher confidence = wider stops)
        confidence_adj = 0.8 + 0.4 * confidence  # 0.8 to 1.2
        sl_mult *= confidence_adj
        tp_mult *= confidence_adj

        # Calculate prices
        stop_loss_price = entry_price - (sl_mult * atr)
        take_profit_price = entry_price + (tp_mult * atr)
        trailing_stop_distance = self.trailing_stop_atr * atr

        return stop_loss_price, take_profit_price, trailing_stop_distance

    def check_exit(self, entry_price: float, current_price: float,
                   high_since_entry: float, stop_loss: float,
                   take_profit: float, trailing_distance: float) -> Tuple[bool, str]:
        """
        Check if any exit condition is met

        Returns:
            (should_exit, reason)
        """
        # Stop loss
        if current_price <= stop_loss:
            return True, 'stop_loss'

        # Take profit
        if current_price >= take_profit:
            return True, 'take_profit'

        # Trailing stop
        trailing_stop_price = high_since_entry - trailing_distance
        if current_price <= trailing_stop_price and high_since_entry > entry_price * 1.01:
            return True, 'trailing_stop'

        return False, 'hold'


class BacktesterV3:
    """
    V3 Backtester with Kelly Criterion and Dynamic Risk Management
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 confidence_threshold: float = 0.55,
                 max_kelly_fraction: float = 0.5,
                 max_position: float = 0.85):
        """
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            confidence_threshold: Minimum confidence to trade
            max_kelly_fraction: Maximum Kelly fraction to use
            max_position: Maximum position size
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.confidence_threshold = confidence_threshold

        self.kelly = KellyCriterion(
            max_kelly_fraction=max_kelly_fraction,
            max_position=max_position
        )
        self.risk_manager = DynamicRiskManager()

        # State
        self.capital = initial_capital
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.current_position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.daily_returns: List[float] = []
        self.portfolio_history: List[Dict] = []

    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.current_position = None
        self.trades = []
        self.daily_returns = []
        self.portfolio_history = []

    def _get_regime(self, row: pd.Series) -> str:
        """Determine market regime from features"""
        if 'bull_regime' in row and row['bull_regime'] == 1:
            return 'bull'
        elif 'bear_regime' in row and row['bear_regime'] == 1:
            return 'bear'
        return 'neutral'

    def _get_volatility(self, row: pd.Series) -> float:
        """Get current volatility from features"""
        if 'volatility_20d' in row:
            return row['volatility_20d']
        elif 'realized_vol_21d' in row:
            return row['realized_vol_21d']
        return 0.15  # Default

    def _get_atr(self, row: pd.Series) -> float:
        """Get ATR from features"""
        if 'atr' in row:
            return row['atr']
        elif 'atr_percent' in row:
            return row['atr_percent'] * row['Close'] / 100
        return row['Close'] * 0.02  # Default 2% of price

    def run(self, df: pd.DataFrame, signals: np.ndarray,
            probabilities: np.ndarray, confidence: np.ndarray) -> Dict:
        """
        Run backtest

        Args:
            df: DataFrame with OHLC and features
            signals: Binary signals (1 = buy, 0 = sell/hold, -1 = no signal)
            probabilities: Model probabilities
            confidence: Confidence scores

        Returns:
            Dictionary with performance metrics
        """
        self.reset()

        # Ensure we have dates as index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        high_since_entry = 0
        circuit_breaker_active = False

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]
            signal = signals[i]
            prob = probabilities[i]
            conf = confidence[i]

            current_price = row['Close']
            regime = self._get_regime(row)
            volatility = self._get_volatility(row)
            atr = self._get_atr(row)

            # Calculate current drawdown
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            current_drawdown = (self.portfolio_value - self.peak_value) / self.peak_value

            # Check circuit breaker
            if current_drawdown < -self.risk_manager.max_drawdown_pct:
                if not circuit_breaker_active:
                    print(f"Circuit breaker activated at {date}: Drawdown = {current_drawdown:.2%}")
                    circuit_breaker_active = True

                # Force close any position
                if self.current_position:
                    self._close_position(date, current_price, 'circuit_breaker')

            # If we have a position
            if self.current_position:
                # Update high since entry for trailing stop
                if current_price > high_since_entry:
                    high_since_entry = current_price

                # Check exit conditions
                should_exit, exit_reason = self.risk_manager.check_exit(
                    self.current_position.entry_price,
                    current_price,
                    high_since_entry,
                    self.current_position.stop_loss_price,
                    self.current_position.take_profit_price,
                    self.current_position.trailing_stop_price
                )

                if should_exit:
                    self._close_position(date, current_price, exit_reason)
                elif signal == 0 and conf >= self.confidence_threshold:
                    # Model says sell with high confidence
                    self._close_position(date, current_price, 'signal')

            # If no position and signal to buy
            elif signal == 1 and conf >= self.confidence_threshold and not circuit_breaker_active:
                # Calculate position size
                position_pct = self.kelly.get_position_size(
                    conf, volatility, regime, current_drawdown
                )

                # Calculate stop levels
                sl_price, tp_price, trailing_dist = self.risk_manager.calculate_stops(
                    current_price, atr, regime, conf
                )

                # Open position
                self._open_position(
                    date, current_price, position_pct, conf,
                    sl_price, tp_price, trailing_dist, regime
                )
                high_since_entry = current_price

            # Reset circuit breaker if drawdown recovers
            if circuit_breaker_active and current_drawdown > -0.10:
                circuit_breaker_active = False
                print(f"Circuit breaker reset at {date}")

            # Update portfolio value
            if self.current_position:
                position_value = self.current_position.shares * current_price
                self.portfolio_value = self.capital + position_value
            else:
                self.portfolio_value = self.capital

            # Record daily state
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'capital': self.capital,
                'in_position': self.current_position is not None,
                'drawdown': current_drawdown,
                'regime': regime
            })

            # Calculate daily return
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2]['portfolio_value']
                daily_ret = (self.portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_ret)

        # Close any remaining position
        if self.current_position:
            self._close_position(df.index[-1], df.iloc[-1]['Close'], 'end_of_period')

        return self._calculate_metrics()

    def _open_position(self, date, price, position_pct, confidence,
                       stop_loss, take_profit, trailing_dist, regime):
        """Open a new position"""
        # Apply slippage
        entry_price = price * (1 + self.slippage)

        # Calculate shares
        investment = self.capital * position_pct
        cost_after_fees = investment * (1 - self.transaction_cost)
        shares = cost_after_fees / entry_price

        # Update capital
        self.capital -= investment

        # Create trade
        self.current_position = Trade(
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            position_pct=position_pct,
            confidence=confidence,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            trailing_stop_price=trailing_dist,  # This is distance, updated dynamically
            regime=regime
        )

    def _close_position(self, date, price, reason):
        """Close current position"""
        if not self.current_position:
            return

        # Apply slippage
        exit_price = price * (1 - self.slippage)

        # Calculate proceeds
        gross_proceeds = self.current_position.shares * exit_price
        net_proceeds = gross_proceeds * (1 - self.transaction_cost)

        # Calculate P&L
        entry_value = self.current_position.shares * self.current_position.entry_price
        pnl = net_proceeds - entry_value
        pnl_pct = pnl / entry_value

        # Update trade record
        self.current_position.exit_date = date
        self.current_position.exit_price = exit_price
        self.current_position.exit_reason = reason
        self.current_position.pnl = pnl
        self.current_position.pnl_pct = pnl_pct
        self.current_position.holding_days = (date - self.current_position.entry_date).days

        # Update Kelly history
        self.kelly.update_history(pnl_pct, pnl > 0)

        # Store trade
        self.trades.append(self.current_position)

        # Update capital
        self.capital += net_proceeds

        # Clear position
        self.current_position = None

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_return_pct': 0,
                'annual_return_pct': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown_pct': 0,
                'calmar_ratio': 0,
                'win_rate_pct': 0,
                'profit_factor': 0,
                'total_trades': 0,
            }

        # Basic returns
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        trading_days = len(self.daily_returns)
        years = trading_days / 252 if trading_days > 0 else 1
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

        # Risk metrics
        daily_ret_array = np.array(self.daily_returns)

        if len(daily_ret_array) > 0 and daily_ret_array.std() > 0:
            sharpe = daily_ret_array.mean() / daily_ret_array.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino (downside deviation)
        downside_returns = daily_ret_array[daily_ret_array < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = daily_ret_array.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = 0

        # Max drawdown
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        wins = [t for t in self.trades if t.pnl and t.pnl > 0]
        losses = [t for t in self.trades if t.pnl and t.pnl <= 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
        avg_holding = np.mean([t.holding_days for t in self.trades if t.holding_days])

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            reason = t.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        return {
            'total_return_pct': round(total_return * 100, 2),
            'annual_return_pct': round(annual_return * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'calmar_ratio': round(calmar, 3),
            'win_rate_pct': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_holding_days': round(avg_holding, 1) if avg_holding else 0,
            'total_trades': len(self.trades),
            'round_trips': len(self.trades),
            'final_capital': round(self.portfolio_value, 2),
            'exit_reasons': exit_reasons,
            'kelly_final': round(self.kelly.calculate_kelly(), 3)
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log as DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': round(t.entry_price, 2),
                'exit_price': round(t.exit_price, 2) if t.exit_price else None,
                'shares': round(t.shares, 2),
                'position_pct': round(t.position_pct * 100, 1),
                'confidence': round(t.confidence, 3),
                'regime': t.regime,
                'pnl': round(t.pnl, 2) if t.pnl else None,
                'pnl_pct': round(t.pnl_pct * 100, 2) if t.pnl_pct else None,
                'holding_days': t.holding_days,
                'exit_reason': t.exit_reason,
                'stop_loss': round(t.stop_loss_price, 2),
                'take_profit': round(t.take_profit_price, 2)
            })

        return pd.DataFrame(records)

    def get_portfolio_history(self) -> pd.DataFrame:
        """Get daily portfolio history"""
        return pd.DataFrame(self.portfolio_history)


def run_backtest_with_benchmark(df: pd.DataFrame, signals: np.ndarray,
                                 probabilities: np.ndarray, confidence: np.ndarray,
                                 ticker: str = 'SPY') -> Tuple[Dict, pd.DataFrame]:
    """
    Run backtest and compare with buy-and-hold benchmark

    Returns:
        (metrics_dict, comparison_df)
    """
    # V3 Strategy backtest
    backtester = BacktesterV3(
        initial_capital=100000,
        confidence_threshold=0.55,
        max_kelly_fraction=0.5
    )

    strategy_metrics = backtester.run(df, signals, probabilities, confidence)
    strategy_metrics['strategy'] = f'AI Multi-Factor V3 ({ticker})'

    # Buy and hold benchmark
    start_price = df.iloc[0]['Close']
    end_price = df.iloc[-1]['Close']
    bh_return = (end_price - start_price) / start_price

    # Calculate buy & hold Sharpe
    bh_daily_returns = df['Close'].pct_change().dropna()
    bh_sharpe = bh_daily_returns.mean() / bh_daily_returns.std() * np.sqrt(252) if bh_daily_returns.std() > 0 else 0

    # Buy & hold max drawdown
    cumulative = (1 + bh_daily_returns).cumprod()
    peak = cumulative.expanding().max()
    bh_drawdown = ((cumulative - peak) / peak).min()

    benchmark_metrics = {
        'strategy': f'Buy & Hold ({ticker})',
        'total_return_pct': round(bh_return * 100, 2),
        'annual_return_pct': round(((1 + bh_return) ** (252/len(df)) - 1) * 100, 2),
        'sharpe_ratio': round(bh_sharpe, 3),
        'sortino_ratio': None,
        'max_drawdown_pct': round(bh_drawdown * 100, 2),
        'win_rate_pct': None,
        'profit_factor': None,
        'total_trades': 1,
    }

    # Create comparison DataFrame
    comparison = pd.DataFrame([strategy_metrics, benchmark_metrics])

    return strategy_metrics, comparison, backtester.get_trade_log()


if __name__ == "__main__":
    print("V3 Backtester - Testing...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    prices = 100 * np.cumprod(1 + np.random.randn(252) * 0.02)

    df = pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 252),
        'volatility_20d': np.abs(np.random.randn(252)) * 0.2,
        'atr': prices * 0.02,
        'bull_regime': np.random.randint(0, 2, 252),
        'bear_regime': 0
    }, index=dates)

    # Sample signals
    signals = np.random.choice([0, 1, -1], 252, p=[0.4, 0.3, 0.3])
    probabilities = np.random.rand(252)
    confidence = np.abs(probabilities - 0.5) * 2

    # Run backtest
    backtester = BacktesterV3()
    metrics = backtester.run(df, signals, probabilities, confidence)

    print("\nBacktest Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nV3 Backtester test complete!")
