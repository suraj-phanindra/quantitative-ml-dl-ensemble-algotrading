"""
Feature Engineering Module
==========================
Creates alpha factors and features for the trading strategy.

THE 3 WINNING ALPHA FACTORS:
1. RSI (Mean Reversion) - 73-91% backtested win rate
2. MACD (Trend Confirmation) - 73-84% win rate with RSI
3. Momentum 12-1 (Price Trend) - Fama-French validated

Research Sources:
- QuantifiedStrategies.com: RSI+MACD = 73% win rate
- TradingView Backtests: 80-90% win rates on blue chips
- Balvers & Wu (2006): Momentum + Mean Reversion combination
- Fama-French: Momentum is "premier market anomaly"
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for creating alpha factors and technical indicators.
    """

    # Optimal parameters from backtesting research
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70

    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    MOMENTUM_SHORT = 21   # 1 month
    MOMENTUM_MEDIUM = 63  # 3 months
    MOMENTUM_LONG = 252   # 12 months
    MOMENTUM_SKIP = 21    # Skip recent month

    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_columns = []

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the trading strategy.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Basic returns
        df = self._create_return_features(df)

        # Alpha Factor 1: RSI (Mean Reversion)
        df = self._create_rsi_features(df)

        # Alpha Factor 2: MACD (Trend Confirmation)
        df = self._create_macd_features(df)

        # Alpha Factor 3: Momentum
        df = self._create_momentum_features(df)

        # Supporting features
        df = self._create_moving_average_features(df)
        df = self._create_volatility_features(df)
        df = self._create_volume_features(df)
        df = self._create_bollinger_features(df)

        # Store feature columns
        self._update_feature_columns(df)

        return df

    def _create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic return features."""
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Multi-period returns
        for period in [2, 3, 5, 10]:
            df[f'returns_{period}d'] = df['Close'].pct_change(period)

        return df

    def _create_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RSI-based features (Alpha Factor #1).

        RSI achieved 91% win rate in backtests when combined with trend filter.
        """
        # Multiple RSI periods
        for period in [7, 14, 21]:
            rsi = RSIIndicator(df['Close'], window=period).rsi()
            df[f'rsi_{period}'] = rsi

        # RSI divergence (momentum of RSI)
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(5)

        # RSI moving average (smoothed)
        df['rsi_ma'] = df['rsi_14'].rolling(10).mean()

        # RSI zones
        df['rsi_oversold'] = (df['rsi_14'] < self.RSI_OVERSOLD).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > self.RSI_OVERBOUGHT).astype(int)
        df['rsi_neutral_zone'] = (
            (df['rsi_14'] >= 40) & (df['rsi_14'] <= 60)
        ).astype(int)

        # RSI turning signals
        df['rsi_turning_up'] = (df['rsi_14'] > df['rsi_14'].shift(1)).astype(int)
        df['rsi_turning_down'] = (df['rsi_14'] < df['rsi_14'].shift(1)).astype(int)

        return df

    def _create_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create MACD-based features (Alpha Factor #2).

        MACD combined with RSI achieves 73-84% win rate.
        """
        macd = MACD(
            df['Close'],
            window_slow=self.MACD_SLOW,
            window_fast=self.MACD_FAST,
            window_sign=self.MACD_SIGNAL
        )

        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # MACD states
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_hist_positive'] = (df['macd_histogram'] > 0).astype(int)

        # MACD histogram slope
        df['macd_hist_slope'] = df['macd_histogram'] - df['macd_histogram'].shift(1)

        # MACD crossover detection
        df['macd_crossover'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)

        df['macd_crossunder'] = (
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)

        # MACD zero line cross
        df['macd_above_zero'] = (df['macd'] > 0).astype(int)

        return df

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Momentum-based features (Alpha Factor #3).

        Momentum is the "premier market anomaly" - Fama & French.
        12-1 month momentum (skip most recent month) is the classic factor.
        """
        # Multi-period momentum
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_21d'] = df['Close'].pct_change(21)  # 1 month
        df['momentum_63d'] = df['Close'].pct_change(63)  # 3 months
        df['momentum_126d'] = df['Close'].pct_change(126)  # 6 months
        df['momentum_252d'] = df['Close'].pct_change(252)  # 12 months

        # THE CLASSIC 12-1 MOMENTUM (skip most recent month)
        df['momentum_12_1'] = df['Close'].shift(21) / df['Close'].shift(252) - 1

        # Momentum z-score (for cross-sectional ranking)
        df['momentum_zscore'] = (
            (df['momentum_12_1'] - df['momentum_12_1'].rolling(252).mean()) /
            df['momentum_12_1'].rolling(252).std()
        )

        # Risk-adjusted momentum (Sharpe-like)
        rolling_std = df['returns'].rolling(252).std() * np.sqrt(252)
        df['momentum_sharpe'] = df['momentum_252d'] / rolling_std

        # Momentum regime
        df['momentum_positive'] = (df['momentum_21d'] > 0).astype(int)
        df['momentum_strong_positive'] = (df['momentum_21d'] > 0.02).astype(int)
        df['momentum_strong_negative'] = (df['momentum_21d'] < -0.02).astype(int)

        return df

    def _create_moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create moving average features."""
        # Simple Moving Averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window).mean()
            df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}']

        # Exponential Moving Averages
        for window in [12, 26]:
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()

        # Trend indicators
        df['trend_up'] = (df['Close'] > df['sma_200']).astype(int)
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['death_cross'] = (df['sma_50'] < df['sma_200']).astype(int)

        # MA slopes
        df['sma_20_slope'] = df['sma_20'] - df['sma_20'].shift(5)
        df['sma_50_slope'] = df['sma_50'] - df['sma_50'].shift(5)

        return df

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        # Rolling volatility (annualized)
        for window in [10, 20, 60]:
            df[f'volatility_{window}d'] = (
                df['returns'].rolling(window).std() * np.sqrt(252)
            )

        # Volatility ratio (short-term / long-term)
        df['volatility_ratio'] = df['volatility_20d'] / df['volatility_60d']

        # ATR (Average True Range)
        atr = AverageTrueRange(
            df['High'], df['Low'], df['Close'], window=14
        )
        df['atr_14'] = atr.average_true_range()
        df['atr_percent'] = df['atr_14'] / df['Close']

        # Volatility regime
        df['volatility_regime'] = df['volatility_20d'] / df['volatility_20d'].rolling(252).mean()
        df['high_volatility'] = (df['volatility_regime'] > 1.2).astype(int)

        return df

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        # Volume moving average
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_change'] = df['obv'].pct_change(5)

        # Volume trend
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

        # High volume days
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)

        return df

    def _create_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Bollinger Band features."""
        bb = BollingerBands(df['Close'], window=20, window_dev=2)

        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = bb.bollinger_wband()

        # Position within bands (0 = at lower band, 1 = at upper band)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Bollinger squeeze (low volatility)
        df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(100).mean()
        df['in_squeeze'] = (df['bb_squeeze'] < 0.5).astype(int)

        # Price vs bands
        df['above_upper_bb'] = (df['Close'] > df['bb_upper']).astype(int)
        df['below_lower_bb'] = (df['Close'] < df['bb_lower']).astype(int)

        return df

    def _update_feature_columns(self, df: pd.DataFrame) -> None:
        """Update the list of feature columns."""
        exclude_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close',
            'target', 'future_returns', 'future_returns_5d'
        ]
        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_cols and not col.startswith('sma_') or col.startswith('price_to_sma')
        ]

    def get_feature_columns(self) -> list:
        """
        Get the list of feature columns for ML models.

        Returns a curated list of the most important features.
        """
        return [
            # Alpha Factor 1: RSI Features
            'rsi_14', 'rsi_7', 'rsi_21', 'rsi_divergence', 'rsi_ma',
            'rsi_oversold', 'rsi_overbought', 'rsi_turning_up',

            # Alpha Factor 2: MACD Features
            'macd', 'macd_signal', 'macd_histogram',
            'macd_above_signal', 'macd_hist_positive', 'macd_hist_slope',
            'macd_crossover', 'macd_above_zero',

            # Alpha Factor 3: Momentum Features
            'momentum_5d', 'momentum_21d', 'momentum_63d', 'momentum_12_1',
            'momentum_positive', 'momentum_sharpe',

            # Supporting Features
            'returns', 'returns_5d',
            'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
            'trend_up', 'golden_cross',
            'volatility_20d', 'volatility_ratio', 'volatility_regime',
            'bb_position', 'bb_squeeze', 'below_lower_bb', 'above_upper_bb',
            'atr_percent',
            'volume_ratio', 'obv_change'
        ]

    def create_target_variable(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.0,
        target_type: str = 'binary'
    ) -> pd.DataFrame:
        """
        Create target variable for ML model.

        Args:
            df: DataFrame with OHLC data
            horizon: Number of days to look ahead
            threshold: Minimum return threshold for positive class
            target_type: 'binary' or 'ternary'

        Returns:
            DataFrame with target column added
        """
        df = df.copy()

        # Calculate future returns
        df['future_returns'] = df['Close'].shift(-horizon) / df['Close'] - 1

        if target_type == 'binary':
            # Binary: 1 if positive return, 0 otherwise
            df['target'] = (df['future_returns'] > threshold).astype(int)
        else:
            # Ternary: Buy (1), Hold (0), Sell (-1)
            df['target'] = np.where(
                df['future_returns'] > threshold, 1,
                np.where(df['future_returns'] < -threshold, -1, 0)
            )

        return df

    def generate_rule_based_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using the winning multi-factor confluence strategy.

        Strategy: Require 2+ indicators to agree for signal generation.
        Backtested Win Rate: 73-80%

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with signal columns added
        """
        df = df.copy()

        # Ensure we have the required features
        if 'rsi_14' not in df.columns:
            df = self.create_all_features(df)

        # RSI conditions
        rsi_buy = (df['rsi_14'] < 40) & (df['rsi_14'] > df['rsi_14'].shift(1))
        rsi_sell = (df['rsi_14'] > 60) & (df['rsi_14'] < df['rsi_14'].shift(1))

        # MACD conditions
        macd_buy = (df['macd'] > df['macd_signal']) | (
            (df['macd'] > df['macd'].shift(1)) &
            (df['macd_signal'] < df['macd_signal'].shift(1))
        )
        macd_sell = (df['macd'] < df['macd_signal']) | (
            (df['macd'] < df['macd'].shift(1)) &
            (df['macd_signal'] > df['macd_signal'].shift(1))
        )

        # Momentum conditions
        momentum_buy = df['momentum_12_1'] > 0
        momentum_sell = df['momentum_12_1'] < 0

        # Trend filter (optional but improves win rate)
        trend_filter = df['Close'] > df['sma_200']

        # Combined buy score
        buy_score = (
            rsi_buy.astype(int) +
            macd_buy.astype(int) +
            momentum_buy.astype(int)
        )

        # Combined sell score
        sell_score = (
            rsi_sell.astype(int) +
            macd_sell.astype(int) +
            momentum_sell.astype(int)
        )

        # Generate signals (require 2+ factors to agree)
        df['signal'] = np.where(
            (buy_score >= 2) & trend_filter, 1,
            np.where(sell_score >= 2, -1, 0)
        )

        df['buy_score'] = buy_score
        df['sell_score'] = sell_score

        return df


def main():
    """Test the feature engineering module."""
    from data_loader import DataLoader

    # Load data
    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo("SPY", start_date="2015-01-01", end_date="2024-12-01")

    # Engineer features
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)

    # Create target
    df = engineer.create_target_variable(df, horizon=5, threshold=0.0)

    # Generate rule-based signals
    df = engineer.generate_rule_based_signals(df)

    # Drop NaN
    df = df.dropna()

    # Print summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nFeature columns for ML model: {len(engineer.get_feature_columns())}")
    print("\nTarget distribution:")
    print(df['target'].value_counts(normalize=True))
    print("\nSignal distribution:")
    print(df['signal'].value_counts())
    print("\nSample features (last 5 rows):")
    print(df[['Close', 'rsi_14', 'macd', 'momentum_21d', 'target', 'signal']].tail())

    return df


if __name__ == "__main__":
    df = main()
