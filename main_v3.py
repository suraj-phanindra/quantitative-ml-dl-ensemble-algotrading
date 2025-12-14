"""
Main V3 Pipeline - Multi-Ticker AI Trading Strategy
Implements all V3 improvements:
- Stacking ensemble with XGBoost, LightGBM, (CatBoost optional)
- Sharpe-ratio optimized hyperparameters
- Kelly Criterion position sizing
- Dynamic ATR-based stops
- Regime-adaptive trading
- Multi-ticker support
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering_v3 import FeatureEngineerV3
from models_v3 import StackingEnsembleV3, evaluate_model_comprehensive
from backtester_v3 import BacktesterV3, run_backtest_with_benchmark
from explainability import SHAPExplainer


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance"""
    print(f"\nDownloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if len(df) == 0:
        raise ValueError(f"No data downloaded for {ticker}")

    print(f"  Downloaded {len(df)} rows")
    return df


def run_pipeline_for_ticker(ticker: str, start: str, end: str,
                            n_trials: int = 30, confidence_threshold: float = 0.55,
                            save_reports: bool = True, report_dir: str = None) -> dict:
    """
    Run complete V3 pipeline for a single ticker

    Args:
        ticker: Stock ticker symbol
        start: Start date
        end: End date
        n_trials: Optuna trials for hyperparameter optimization
        confidence_threshold: Minimum confidence to trade
        save_reports: Whether to save reports
        report_dir: Directory for reports

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*70)
    print(f"V3 PIPELINE: {ticker}")
    print("="*70)

    results = {'ticker': ticker, 'start': start, 'end': end}

    # ========== STEP 1: Download Data ==========
    try:
        df = download_data(ticker, start, end)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return {'ticker': ticker, 'error': str(e)}

    # ========== STEP 2: Feature Engineering ==========
    print("\n[Step 2] Creating features...")
    # Use adaptive_lookback=False to get exactly 131 features (same as Sharpe > 1.0 run)
    engineer = FeatureEngineerV3(adaptive_lookback=False)
    df_features = engineer.create_features(df, ticker=ticker)

    # Create targets
    df_features['target'] = engineer.create_target(df_features, forward_days=5)
    df_features['forward_return'] = engineer.create_regression_target(df_features, forward_days=5)

    # Drop NaN
    df_features = df_features.dropna()

    feature_names = engineer.get_feature_names()
    print(f"  Created {len(feature_names)} features")
    print(f"  Total rows after cleaning: {len(df_features)}")

    # ========== STEP 3: Train/Test Split ==========
    print("\n[Step 3] Splitting data...")
    split_idx = int(len(df_features) * 0.8)

    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]

    # Prepare arrays
    feature_cols = [c for c in feature_names if c in df_features.columns and
                    c not in ['target', 'forward_return', 'Open', 'High', 'Low', 'Close', 'Volume']]

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    returns_train = train_df['forward_return'].values
    volatility_train = train_df['volatility_20d'].values if 'volatility_20d' in train_df.columns else np.ones(len(train_df)) * 0.15

    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    returns_test = test_df['forward_return'].values

    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    print(f"  Features: {len(feature_cols)}")

    results['data'] = {
        'total_rows': len(df_features),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'n_features': len(feature_cols),
        'test_start': str(test_df.index[0].date()),
        'test_end': str(test_df.index[-1].date())
    }

    # ========== STEP 4: Train Stacking Ensemble ==========
    print("\n[Step 4] Training Stacking Ensemble...")

    # Try to use CatBoost if available
    try:
        from catboost import CatBoostClassifier
        use_catboost = True
        print("  CatBoost available - using 3-model ensemble")
    except ImportError:
        use_catboost = False
        print("  CatBoost not available - using 2-model ensemble")

    ensemble = StackingEnsembleV3(
        confidence_threshold=confidence_threshold,
        use_catboost=use_catboost
    )

    # Train with validation set
    val_split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]
    ret_tr, ret_val = returns_train[:val_split], returns_train[val_split:]
    vol_tr = volatility_train[:val_split]

    train_metrics = ensemble.fit(
        X_tr, y_tr, X_val, y_val,
        ret_tr, ret_val, vol_tr,
        n_trials=n_trials
    )

    results['train_metrics'] = train_metrics

    # ========== STEP 5: Generate Predictions ==========
    print("\n[Step 5] Generating predictions...")

    signals, probabilities, confidence, high_conf_mask = ensemble.predict_with_confidence(X_test)

    # Evaluate on test set
    test_metrics = evaluate_model_comprehensive(
        y_test, probabilities, returns_test, ticker=ticker
    )

    results['test_metrics'] = test_metrics

    print(f"\nTest Set Metrics ({ticker}):")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  Precision:       {test_metrics['precision']:.4f}")
    print(f"  Recall:          {test_metrics['recall']:.4f}")
    print(f"  ROC AUC:         {test_metrics['roc_auc']:.4f}")
    print(f"  Sharpe (model):  {test_metrics['sharpe']:.4f}")
    print(f"  High Conf Acc:   {test_metrics['high_conf_accuracy']:.4f} ({test_metrics['high_conf_count']} trades)")

    # ========== STEP 6: Backtesting ==========
    print("\n[Step 6] Running backtest...")

    backtester = BacktesterV3(
        initial_capital=100000,
        confidence_threshold=confidence_threshold,
        max_kelly_fraction=0.5,
        max_position=0.85
    )

    # Prepare test dataframe for backtest
    test_df_bt = test_df.copy()

    backtest_metrics = backtester.run(
        test_df_bt, signals, probabilities, confidence
    )

    results['backtest_metrics'] = backtest_metrics

    print(f"\nBacktest Results ({ticker}):")
    print(f"  Total Return:    {backtest_metrics['total_return_pct']:.2f}%")
    print(f"  Annual Return:   {backtest_metrics['annual_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:    {backtest_metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:   {backtest_metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:    {backtest_metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:        {backtest_metrics['win_rate_pct']:.2f}%")
    print(f"  Profit Factor:   {backtest_metrics['profit_factor']:.2f}")
    print(f"  Total Trades:    {backtest_metrics['total_trades']}")
    print(f"  Kelly Criterion: {backtest_metrics['kelly_final']:.3f}")

    if 'exit_reasons' in backtest_metrics:
        print(f"  Exit Reasons:    {backtest_metrics['exit_reasons']}")

    # ========== STEP 7: Buy & Hold Comparison ==========
    print("\n[Step 7] Calculating benchmarks...")

    # Buy & Hold
    start_price = test_df.iloc[0]['Close']
    end_price = test_df.iloc[-1]['Close']
    bh_return = (end_price - start_price) / start_price * 100

    bh_daily_returns = test_df['Close'].pct_change().dropna()
    bh_sharpe = bh_daily_returns.mean() / bh_daily_returns.std() * np.sqrt(252) if bh_daily_returns.std() > 0 else 0

    cumulative = (1 + bh_daily_returns).cumprod()
    peak = cumulative.expanding().max()
    bh_drawdown = ((cumulative - peak) / peak).min() * 100

    results['benchmark'] = {
        'buy_hold_return': round(bh_return, 2),
        'buy_hold_sharpe': round(bh_sharpe, 3),
        'buy_hold_max_dd': round(bh_drawdown, 2)
    }

    print(f"\nBuy & Hold ({ticker}):")
    print(f"  Total Return:    {bh_return:.2f}%")
    print(f"  Sharpe Ratio:    {bh_sharpe:.3f}")
    print(f"  Max Drawdown:    {bh_drawdown:.2f}%")

    # ========== STEP 8: Save Reports ==========
    if save_reports and report_dir:
        print(f"\n[Step 8] Saving reports to {report_dir}...")

        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(os.path.join(report_dir, 'shap_analysis'), exist_ok=True)

        # Save predictions
        pred_df = pd.DataFrame({
            'date': test_df.index,
            'close': test_df['Close'].values,
            'signal': signals,
            'probability': probabilities,
            'confidence': confidence,
            'actual': y_test,
            'forward_return': returns_test
        })
        pred_df.to_csv(os.path.join(report_dir, f'predictions_{ticker}.csv'), index=False)

        # Save trade log
        trade_log = backtester.get_trade_log()
        if not trade_log.empty:
            trade_log.to_csv(os.path.join(report_dir, f'trade_log_{ticker}.csv'), index=False)

        # Save summary
        summary = {
            'ticker': ticker,
            'version': 'V3',
            'period': {'start': start, 'end': end},
            'test_period': results['data'],
            'hyperparameters': {
                'n_trials': n_trials,
                'confidence_threshold': confidence_threshold,
                'max_kelly_fraction': 0.5
            },
            'model_metrics': test_metrics,
            'backtest_metrics': backtest_metrics,
            'benchmark': results['benchmark']
        }

        with open(os.path.join(report_dir, f'summary_{ticker}.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save model (in root ai_trading_strategy/models_v3/)
        model_dir = 'models_v3'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'ensemble_{ticker}.pkl')
        try:
            ensemble.save(model_path)
            print(f"  Model saved to {model_path}")
        except Exception as e:
            print(f"  Warning: Model save failed: {e}")

        # SHAP Analysis
        print("  Generating SHAP analysis...")
        try:
            # Use XGBoost base model for SHAP (most interpretable)
            # Need to pass scaled data since ensemble uses scaler
            X_test_scaled = ensemble.scaler.transform(X_test)

            explainer = SHAPExplainer(
                ensemble.xgb_model.model,
                feature_names=feature_cols
            )

            # Generate comprehensive SHAP reports
            # Limit to 300 samples for faster generation
            n_samples = min(300, len(X_test_scaled))
            explainer.generate_reports(
                X_test_scaled[-n_samples:],  # Most recent samples
                output_dir=os.path.join(report_dir, 'shap_analysis'),
                prefix=ticker,
                n_waterfall=5,
                top_features_for_dependence=5
            )
        except Exception as e:
            print(f"  Warning: SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()

        print("  Reports saved!")

    return results


def run_multi_ticker_analysis(tickers: list, start: str, end: str,
                               n_trials: int = 30, confidence: float = 0.55) -> pd.DataFrame:
    """
    Run V3 pipeline on multiple tickers and create comparison

    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        n_trials: Optuna trials
        confidence: Confidence threshold

    Returns:
        DataFrame with comparison across all tickers
    """
    print("\n" + "="*70)
    print("V3 MULTI-TICKER ANALYSIS")
    print(f"Tickers: {', '.join(tickers)}")
    print("="*70)

    all_results = []

    for ticker in tickers:
        report_dir = os.path.join('reports_v3', ticker)

        result = run_pipeline_for_ticker(
            ticker=ticker,
            start=start,
            end=end,
            n_trials=n_trials,
            confidence_threshold=confidence,
            save_reports=True,
            report_dir=report_dir
        )

        if 'error' not in result:
            row = {
                'Ticker': ticker,
                'Total Return (%)': result['backtest_metrics']['total_return_pct'],
                'Annual Return (%)': result['backtest_metrics']['annual_return_pct'],
                'Sharpe Ratio': result['backtest_metrics']['sharpe_ratio'],
                'Sortino Ratio': result['backtest_metrics']['sortino_ratio'],
                'Max Drawdown (%)': result['backtest_metrics']['max_drawdown_pct'],
                'Calmar Ratio': result['backtest_metrics']['calmar_ratio'],
                'Win Rate (%)': result['backtest_metrics']['win_rate_pct'],
                'Profit Factor': result['backtest_metrics']['profit_factor'],
                'Total Trades': result['backtest_metrics']['total_trades'],
                'Kelly Final': result['backtest_metrics']['kelly_final'],
                'B&H Return (%)': result['benchmark']['buy_hold_return'],
                'B&H Sharpe': result['benchmark']['buy_hold_sharpe'],
                'B&H Max DD (%)': result['benchmark']['buy_hold_max_dd'],
                'Model Sharpe': result['test_metrics']['sharpe'],
                'High Conf Acc': result['test_metrics']['high_conf_accuracy']
            }
            all_results.append(row)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results)

    # Add rankings
    if len(comparison_df) > 1:
        comparison_df['Sharpe Rank'] = comparison_df['Sharpe Ratio'].rank(ascending=False)
        comparison_df['Return Rank'] = comparison_df['Total Return (%)'].rank(ascending=False)
        comparison_df['Win Rate Rank'] = comparison_df['Win Rate (%)'].rank(ascending=False)

    # Save comparison
    os.makedirs('reports_v3', exist_ok=True)
    comparison_df.to_csv('reports_v3/multi_ticker_comparison.csv', index=False)

    # Create summary
    print("\n" + "="*70)
    print("MULTI-TICKER COMPARISON SUMMARY")
    print("="*70)
    print(comparison_df.to_string(index=False))

    # Calculate aggregates
    print("\n" + "-"*70)
    print("AGGREGATE STATISTICS")
    print("-"*70)
    print(f"Average Sharpe Ratio:    {comparison_df['Sharpe Ratio'].mean():.3f}")
    print(f"Average Win Rate:        {comparison_df['Win Rate (%)'].mean():.2f}%")
    print(f"Average Max Drawdown:    {comparison_df['Max Drawdown (%)'].mean():.2f}%")
    print(f"Average Total Return:    {comparison_df['Total Return (%)'].mean():.2f}%")
    print(f"Best Performing Ticker:  {comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Ticker']}")

    # Compare to benchmarks
    avg_strategy_sharpe = comparison_df['Sharpe Ratio'].mean()
    avg_bh_sharpe = comparison_df['B&H Sharpe'].mean()
    print(f"\nStrategy vs Buy&Hold Sharpe: {avg_strategy_sharpe:.3f} vs {avg_bh_sharpe:.3f}")

    avg_strategy_dd = comparison_df['Max Drawdown (%)'].mean()
    avg_bh_dd = comparison_df['B&H Max DD (%)'].mean()
    print(f"Strategy vs Buy&Hold Max DD: {avg_strategy_dd:.2f}% vs {avg_bh_dd:.2f}%")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='V3 AI Trading Strategy Pipeline')
    parser.add_argument('--tickers', type=str, default='SPY,AAPL,NVDA,IBM',
                        help='Comma-separated list of tickers')
    parser.add_argument('--start', type=str, default='2018-01-01',
                        help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date YYYY-MM-DD')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna optimization trials per model')
    parser.add_argument('--confidence', type=float, default=0.52,
                        help='Confidence threshold for trading')
    parser.add_argument('--single', type=str, default=None,
                        help='Run single ticker only')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("AI-DRIVEN MULTI-FACTOR TRADING STRATEGY V3")
    print("="*70)
    print(f"Start Date:           {args.start}")
    print(f"End Date:             {args.end}")
    print(f"Optimization Trials:  {args.n_trials}")
    print(f"Confidence Threshold: {args.confidence}")

    if args.single:
        # Run single ticker
        tickers = [args.single]
    else:
        # Run multiple tickers
        tickers = [t.strip() for t in args.tickers.split(',')]

    print(f"Tickers:              {', '.join(tickers)}")

    # Run analysis
    if len(tickers) == 1:
        result = run_pipeline_for_ticker(
            ticker=tickers[0],
            start=args.start,
            end=args.end,
            n_trials=args.n_trials,
            confidence_threshold=args.confidence,
            save_reports=True,
            report_dir=f'reports_v3/{tickers[0]}'
        )
    else:
        comparison = run_multi_ticker_analysis(
            tickers=tickers,
            start=args.start,
            end=args.end,
            n_trials=args.n_trials,
            confidence=args.confidence
        )

    print("\n" + "="*70)
    print("V3 PIPELINE COMPLETE")
    print("="*70)
    print(f"Reports saved to: reports_v3/")


if __name__ == '__main__':
    main()
