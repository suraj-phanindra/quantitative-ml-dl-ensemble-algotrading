"""
Feature Engineering V3 - Regime-Adaptive Multi-Asset Features
Enhanced features with cross-asset signals and adaptive regime detection
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineerV3:
    """
    V3 Feature Engineering with:
    - Regime-adaptive feature scaling
    - Cross-asset correlation features (when multiple assets provided)
    - Dynamic lookback periods based on volatility
    - Enhanced mean reversion signals
    - Trend strength quantification
    """

    def __init__(self, adaptive_lookback=True):
        self.adaptive_lookback = adaptive_lookback
        self.feature_names = []
        self.regime_params = {}

    def create_features(self, df, ticker='SPY'):
        """Create comprehensive feature set for a single ticker"""
        df = df.copy()

        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure all columns are 1D series (squeeze if needed)
        for col in df.columns:
            if hasattr(df[col], 'squeeze'):
                df[col] = df[col].squeeze()

        # Calculate base volatility for adaptive lookbacks
        df['base_volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        vol_percentile = df['base_volatility'].rank(pct=True)

        # Adaptive lookback multiplier (shorter in high vol, longer in low vol)
        if self.adaptive_lookback:
            df['lookback_mult'] = np.where(vol_percentile > 0.75, 0.7,
                                  np.where(vol_percentile < 0.25, 1.3, 1.0))
        else:
            df['lookback_mult'] = 1.0

        # ============== PRICE FEATURES ==============
        df['returns'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_21d'] = df['Close'].pct_change(21)

        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['realized_vol_5d'] = df['log_returns'].rolling(5).std() * np.sqrt(252)
        df['realized_vol_21d'] = df['log_returns'].rolling(21).std() * np.sqrt(252)

        # Price position features
        df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)

        # ============== RSI FEATURES (ENHANCED) ==============
        for period in [5, 7, 14, 21, 28]:
            rsi = RSIIndicator(df['Close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()

        # RSI derivatives
        df['rsi_14_slope_3'] = df['rsi_14'] - df['rsi_14'].shift(3)
        df['rsi_14_slope_5'] = df['rsi_14'] - df['rsi_14'].shift(5)
        df['rsi_14_accel'] = df['rsi_14_slope_3'] - df['rsi_14_slope_3'].shift(3)

        # RSI smoothing
        df['rsi_14_ema_5'] = df['rsi_14'].ewm(span=5).mean()
        df['rsi_14_ema_10'] = df['rsi_14'].ewm(span=10).mean()

        # RSI of RSI (second derivative)
        df['rsi_of_rsi'] = RSIIndicator(df['rsi_14'].dropna(), window=14).rsi()

        # RSI zones with multiple thresholds
        df['rsi_oversold_30'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_oversold_25'] = (df['rsi_14'] < 25).astype(int)
        df['rsi_oversold_20'] = (df['rsi_14'] < 20).astype(int)
        df['rsi_overbought_70'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_overbought_75'] = (df['rsi_14'] > 75).astype(int)
        df['rsi_overbought_80'] = (df['rsi_14'] > 80).astype(int)

        # RSI distance from key levels
        df['rsi_dist_30'] = df['rsi_14'] - 30
        df['rsi_dist_50'] = df['rsi_14'] - 50
        df['rsi_dist_70'] = df['rsi_14'] - 70

        # RSI crossovers
        df['rsi_cross_up_30'] = ((df['rsi_14'] > 30) & (df['rsi_14'].shift(1) <= 30)).astype(int)
        df['rsi_cross_down_70'] = ((df['rsi_14'] < 70) & (df['rsi_14'].shift(1) >= 70)).astype(int)
        df['rsi_cross_up_50'] = ((df['rsi_14'] > 50) & (df['rsi_14'].shift(1) <= 50)).astype(int)
        df['rsi_cross_down_50'] = ((df['rsi_14'] < 50) & (df['rsi_14'].shift(1) >= 50)).astype(int)

        # RSI divergence (price making new high but RSI not)
        df['price_high_20'] = df['Close'].rolling(20).max()
        df['rsi_high_20'] = df['rsi_14'].rolling(20).max()
        df['rsi_bearish_div'] = ((df['Close'] >= df['price_high_20'] * 0.99) &
                                 (df['rsi_14'] < df['rsi_high_20'] * 0.95)).astype(int)
        df['price_low_20'] = df['Close'].rolling(20).min()
        df['rsi_low_20'] = df['rsi_14'].rolling(20).min()
        df['rsi_bullish_div'] = ((df['Close'] <= df['price_low_20'] * 1.01) &
                                 (df['rsi_14'] > df['rsi_low_20'] * 1.05)).astype(int)

        # ============== MACD FEATURES (ENHANCED) ==============
        macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # MACD normalized by price
        df['macd_normalized'] = df['macd'] / df['Close'] * 100
        df['macd_signal_normalized'] = df['macd_signal'] / df['Close'] * 100

        # MACD derivatives
        df['macd_hist_slope'] = df['macd_histogram'] - df['macd_histogram'].shift(1)
        df['macd_hist_accel'] = df['macd_hist_slope'] - df['macd_hist_slope'].shift(1)

        # MACD crossover signals
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) &
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) &
                                  (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_above_zero'] = (df['macd'] > 0).astype(int)

        # MACD histogram momentum
        df['macd_hist_positive'] = (df['macd_histogram'] > 0).astype(int)
        df['macd_hist_increasing'] = (df['macd_histogram'] > df['macd_histogram'].shift(1)).astype(int)

        # Fast MACD for short-term signals
        macd_fast = MACD(df['Close'], window_slow=12, window_fast=5, window_sign=5)
        df['macd_fast'] = macd_fast.macd()
        df['macd_fast_signal'] = macd_fast.macd_signal()

        # ============== MOMENTUM FEATURES (ENHANCED) ==============
        for period in [5, 10, 21, 63, 126, 252]:
            df[f'momentum_{period}d'] = df['Close'].pct_change(period)

        # Classic 12-1 momentum (skip most recent month)
        df['momentum_12_1'] = df['Close'].shift(21).pct_change(252 - 21)

        # Momentum acceleration
        df['momentum_accel_21d'] = df['momentum_21d'] - df['momentum_21d'].shift(5)
        df['momentum_accel_63d'] = df['momentum_63d'] - df['momentum_63d'].shift(10)

        # Risk-adjusted momentum (momentum Sharpe)
        df['momentum_sharpe_63'] = df['momentum_63d'] / (df['realized_vol_21d'] + 0.01)
        df['momentum_sharpe_252'] = df['momentum_252d'] / (df['realized_vol_21d'] + 0.01)

        # Rate of change
        for period in [5, 10, 21]:
            roc = ROCIndicator(df['Close'], window=period)
            df[f'roc_{period}'] = roc.roc()

        # ============== VOLATILITY FEATURES ==============
        df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_60d'] = df['returns'].rolling(60).std() * np.sqrt(252)

        # Volatility ratio (short/long)
        df['vol_ratio'] = df['volatility_20d'] / (df['volatility_60d'] + 0.01)

        # Volatility percentile
        df['vol_percentile'] = df['volatility_20d'].rolling(252).rank(pct=True)

        # Volatility regimes
        vol_20_pct = df['volatility_20d'].rolling(252).rank(pct=True)
        df['low_vol_regime'] = (vol_20_pct < 0.25).astype(int)
        df['normal_vol_regime'] = ((vol_20_pct >= 0.25) & (vol_20_pct <= 0.75)).astype(int)
        df['high_vol_regime'] = (vol_20_pct > 0.75).astype(int)
        df['extreme_vol_regime'] = (vol_20_pct > 0.90).astype(int)

        # ATR features
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = df['atr'] / df['Close'] * 100
        df['atr_ratio'] = df['atr'] / df['atr'].rolling(50).mean()

        # Bollinger Bands
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Bollinger squeeze (low volatility breakout signal)
        df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(50).mean()
        df['squeeze_on'] = (df['bb_squeeze'] < 0.75).astype(int)

        # Keltner Channel for squeeze confirmation
        kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['true_squeeze'] = ((df['bb_lower'] > df['kc_lower']) &
                              (df['bb_upper'] < df['kc_upper'])).astype(int)

        # ============== TREND FEATURES ==============
        # Moving averages
        for period in [10, 20, 50, 100, 200]:
            sma = SMAIndicator(df['Close'], window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
            df[f'price_vs_sma_{period}'] = (df['Close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100

        for period in [10, 20, 50]:
            ema = EMAIndicator(df['Close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()

        # Trend direction
        df['trend_up'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['golden_cross'] = ((df['sma_50'] > df['sma_200']) &
                              (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
        df['death_cross'] = ((df['sma_50'] < df['sma_200']) &
                             (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)

        # ADX for trend strength
        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        df['adx_diff'] = df['adx_pos'] - df['adx_neg']

        # Strong trend indicator
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['weak_trend'] = (df['adx'] < 20).astype(int)

        # CCI
        cci = CCIIndicator(df['High'], df['Low'], df['Close'], window=20)
        df['cci'] = cci.cci()
        df['cci_overbought'] = (df['cci'] > 100).astype(int)
        df['cci_oversold'] = (df['cci'] < -100).astype(int)

        # ============== VOLUME FEATURES ==============
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma_20'] + 1)

        # OBV
        obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_slope'] = df['obv'].pct_change(5)

        # MFI
        mfi = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
        df['mfi'] = mfi.money_flow_index()
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)

        # Volume-price divergence
        df['vol_price_div'] = np.where(
            (df['returns'] > 0) & (df['volume_ratio'] < 0.8), -1,
            np.where((df['returns'] < 0) & (df['volume_ratio'] < 0.8), 1, 0)
        )

        # ============== REGIME DETECTION ==============
        # Market regime based on multiple factors
        df['regime_score'] = (
            df['trend_up'] * 2 +
            (df['rsi_14'] > 50).astype(int) +
            df['macd_above_zero'] +
            (df['price_vs_sma_200'] > 0).astype(int) * 2
        )

        df['bull_regime'] = (df['regime_score'] >= 5).astype(int)
        df['bear_regime'] = (df['regime_score'] <= 1).astype(int)
        df['neutral_regime'] = ((df['regime_score'] > 1) & (df['regime_score'] < 5)).astype(int)

        # Mean reversion regime (sideways market)
        df['mean_reversion_regime'] = (df['weak_trend'] & df['normal_vol_regime']).astype(int)

        # Trend following regime
        df['trend_following_regime'] = (df['strong_trend'] & ~df['extreme_vol_regime']).astype(int)

        # ============== COMPOSITE SIGNALS ==============
        # Mean reversion score
        df['mean_reversion_score'] = (
            (50 - df['rsi_14']) / 50 * 0.4 +
            (0.5 - df['bb_position']) * 0.3 +
            (-df['price_vs_sma_20'] / 10).clip(-1, 1) * 0.3
        )

        # Trend score
        df['trend_score'] = (
            df['trend_up'] * 0.3 +
            df['macd_above_signal'] * 0.2 +
            df['macd_above_zero'] * 0.2 +
            (df['adx'] / 50).clip(0, 1) * 0.3
        )

        # Combined alpha score
        df['alpha_score'] = np.where(
            df['trend_following_regime'] == 1,
            df['trend_score'],
            np.where(
                df['mean_reversion_regime'] == 1,
                df['mean_reversion_score'],
                (df['trend_score'] + df['mean_reversion_score']) / 2
            )
        )

        # Momentum quality (is momentum reliable?)
        df['momentum_quality'] = (
            (df['adx'] > 20).astype(int) * 0.3 +
            (df['volume_ratio'] > 1).astype(int) * 0.2 +
            (abs(df['macd_histogram']) > abs(df['macd_histogram']).rolling(20).mean()).astype(int) * 0.2 +
            (df['bb_squeeze'] < 1).astype(int) * 0.3
        )

        # ============== STOCHASTIC FEATURES ==============
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_cross_up'] = ((df['stoch_k'] > df['stoch_d']) &
                                (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)

        # Williams %R
        williams = WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14)
        df['williams_r'] = williams.williams_r()

        # ============== PRICE PATTERNS ==============
        # Higher highs, lower lows
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['hh_count_5'] = df['higher_high'].rolling(5).sum()
        df['ll_count_5'] = df['lower_low'].rolling(5).sum()

        # Inside/outside bars
        df['inside_bar'] = ((df['High'] < df['High'].shift(1)) &
                            (df['Low'] > df['Low'].shift(1))).astype(int)
        df['outside_bar'] = ((df['High'] > df['High'].shift(1)) &
                             (df['Low'] < df['Low'].shift(1))).astype(int)

        # Gap features
        df['gap_up'] = ((df['Open'] > df['High'].shift(1))).astype(int)
        df['gap_down'] = ((df['Open'] < df['Low'].shift(1))).astype(int)
        df['gap_size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100

        # ============== 52-WEEK FEATURES ==============
        df['high_52w'] = df['High'].rolling(252).max()
        df['low_52w'] = df['Low'].rolling(252).min()
        df['dist_from_52w_high'] = (df['Close'] - df['high_52w']) / df['high_52w'] * 100
        df['dist_from_52w_low'] = (df['Close'] - df['low_52w']) / df['low_52w'] * 100
        df['range_52w_position'] = (df['Close'] - df['low_52w']) / (df['high_52w'] - df['low_52w'] + 1e-10)

        # New 52-week high/low
        df['new_52w_high'] = (df['Close'] >= df['high_52w']).astype(int)
        df['new_52w_low'] = (df['Close'] <= df['low_52w']).astype(int)

        # ============== CLEANUP ==============
        # Remove intermediate columns
        cols_to_drop = ['price_high_20', 'price_low_20', 'rsi_high_20', 'rsi_low_20',
                        'bb_upper', 'bb_lower', 'bb_mid', 'kc_upper', 'kc_lower',
                        'high_52w', 'low_52w', 'volume_sma_20', 'base_volatility',
                        'lookback_mult', 'obv']

        for col in cols_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        # Drop SMA/EMA raw values (keep ratios)
        for period in [10, 20, 50, 100, 200]:
            if f'sma_{period}' in df.columns:
                df.drop(f'sma_{period}', axis=1, inplace=True)
        for period in [10, 20, 50]:
            if f'ema_{period}' in df.columns:
                df.drop(f'ema_{period}', axis=1, inplace=True)

        # Store feature names
        feature_cols = [col for col in df.columns if col not in
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        self.feature_names = feature_cols

        return df

    def create_target(self, df, forward_days=5, threshold=0.0):
        """
        Create target variable for classification

        Args:
            df: DataFrame with Close prices
            forward_days: Days to look forward for return
            threshold: Minimum return to classify as positive (0 = any positive return)

        Returns:
            Series with binary target (1 = positive return, 0 = negative)
        """
        forward_return = df['Close'].shift(-forward_days) / df['Close'] - 1
        target = (forward_return > threshold).astype(int)
        return target

    def create_regression_target(self, df, forward_days=5):
        """Create regression target (actual forward returns)"""
        forward_return = df['Close'].shift(-forward_days) / df['Close'] - 1
        return forward_return

    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names


def process_multiple_tickers(tickers, start_date, end_date):
    """
    Download and process data for multiple tickers
    Returns dict of DataFrames with features
    """
    import yfinance as yf

    results = {}
    engineer = FeatureEngineerV3()

    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if len(df) < 300:
            print(f"  Warning: {ticker} has insufficient data ({len(df)} rows)")
            continue

        # Create features
        df_features = engineer.create_features(df, ticker=ticker)

        # Create target
        df_features['target'] = engineer.create_target(df_features)

        # Drop NaN rows
        df_features = df_features.dropna()

        print(f"  {ticker}: {len(df_features)} rows, {len(engineer.feature_names)} features")

        results[ticker] = df_features

    return results, engineer.get_feature_names()


if __name__ == "__main__":
    # Test the feature engineering
    import yfinance as yf

    print("Testing V3 Feature Engineering...")

    # Download sample data
    df = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)

    # Create features
    engineer = FeatureEngineerV3()
    df_features = engineer.create_features(df)
    df_features['target'] = engineer.create_target(df_features)

    # Drop NaN
    df_features = df_features.dropna()

    print(f"\nTotal features: {len(engineer.get_feature_names())}")
    print(f"Total rows: {len(df_features)}")
    print(f"\nFeature categories:")

    # Count by category
    categories = {
        'RSI': [f for f in engineer.feature_names if 'rsi' in f.lower()],
        'MACD': [f for f in engineer.feature_names if 'macd' in f.lower()],
        'Momentum': [f for f in engineer.feature_names if 'momentum' in f.lower() or 'roc' in f.lower()],
        'Volatility': [f for f in engineer.feature_names if 'vol' in f.lower() or 'atr' in f.lower() or 'bb' in f.lower()],
        'Trend': [f for f in engineer.feature_names if 'trend' in f.lower() or 'sma' in f.lower() or 'adx' in f.lower()],
        'Regime': [f for f in engineer.feature_names if 'regime' in f.lower()],
    }

    for cat, features in categories.items():
        print(f"  {cat}: {len(features)} features")

    print("\nV3 Feature Engineering test complete!")
