"""
AI-Driven Multi-Factor Trading Strategy
========================================
CPAIT Batch 2025 Capstone Project

Complete Implementation Pipeline

WINNING STRATEGY SUMMARY:
- Alpha 1: RSI (Mean Reversion) - 73-91% win rate
- Alpha 2: MACD (Trend Confirmation) - 73-84% win rate
- Alpha 3: Momentum 12-1 (Price Trend) - Fama-French validated

Combined Win Rate Target: 65-80%
Sharpe Ratio Target: > 1.0

Usage:
    python main.py                    # Run full pipeline
    python main.py --ticker AAPL      # Use different ticker
    python main.py --start 2018-01-01 # Custom start date
"""

import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import HybridEnsemble, train_and_evaluate, XGBoostModel
from backtester import (
    StrategyBacktester, calculate_buy_and_hold,
    single_factor_benchmark, generate_comparison_report
)
from explainability import SHAPExplainer, generate_full_shap_report


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     AI-DRIVEN MULTI-FACTOR TRADING STRATEGY                         ║
║     with Explainability                                              ║
║                                                                      ║
║     CPAIT Batch 2025 Capstone Project                               ║
║                                                                      ║
║     Alpha Factors:                                                   ║
║     1. RSI (Mean Reversion)     - 73-91% win rate                   ║
║     2. MACD (Trend Confirmation) - 73-84% win rate                  ║
║     3. Momentum 12-1 (Price Trend) - Fama-French validated          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_pipeline(
    ticker: str = "SPY",
    start_date: str = "2015-01-01",
    end_date: str = None,
    initial_capital: float = 100000,
    test_size: float = 0.2,
    xgb_weight: float = 0.6,
    save_models: bool = True,
    generate_reports: bool = True
) -> dict:
    """
    Run the complete trading strategy pipeline.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data (defaults to today)
        initial_capital: Starting capital for backtest
        test_size: Fraction of data for testing
        xgb_weight: Weight for XGBoost in ensemble
        save_models: Whether to save trained models
        generate_reports: Whether to generate SHAP reports

    Returns:
        Dictionary with all results
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    results = {}

    # ========================================
    # STEP 1: DATA ACQUISITION
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: DATA ACQUISITION")
    print("="*60)

    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        save=True
    )

    # Validate data
    validation = loader.validate_data(df)
    print(f"\nData validation: {len(validation['issues'])} issues found")

    # Clean data if needed
    if validation['issues']:
        df = loader.clean_data(df)

    # Get summary
    summary = loader.get_data_summary(df)
    print(f"Ticker: {ticker}")
    print(f"Period: {summary['start_date']} to {summary['end_date']} ({summary['trading_years']} years)")
    print(f"Total records: {summary['total_days']}")

    results['data_summary'] = summary

    # ========================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)

    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)

    feature_columns = engineer.get_feature_columns()
    print(f"Created {len(df.columns)} total columns")
    print(f"Selected {len(feature_columns)} features for ML model")

    # ========================================
    # STEP 3: TARGET VARIABLE
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: TARGET VARIABLE CREATION")
    print("="*60)

    df = engineer.create_target_variable(df, horizon=5, threshold=0.0)

    # Drop NaN values
    df_clean = df.dropna(subset=feature_columns + ['target'])
    print(f"Clean samples: {len(df_clean)} (dropped {len(df) - len(df_clean)} NaN rows)")

    print("\nTarget distribution:")
    target_dist = df_clean['target'].value_counts(normalize=True)
    print(f"  Positive (1): {target_dist.get(1, 0)*100:.1f}%")
    print(f"  Negative (0): {target_dist.get(0, 0)*100:.1f}%")

    # ========================================
    # STEP 4: TRAIN/TEST SPLIT
    # ========================================
    print("\n" + "="*60)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("="*60)

    split_idx = int(len(df_clean) * (1 - test_size))
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    print(f"Training set: {len(train_df)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Test set: {len(test_df)} samples ({test_size*100:.0f}%)")
    print(f"Train period: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"Test period: {test_df.index[0].date()} to {test_df.index[-1].date()}")

    # Prepare data
    X_train = train_df[feature_columns].values
    y_train = train_df['target'].values
    X_test = test_df[feature_columns].values
    y_test = test_df['target'].values

    # ========================================
    # STEP 5: MODEL TRAINING
    # ========================================
    print("\n" + "="*60)
    print("STEP 5: MODEL TRAINING")
    print("="*60)

    # Create validation split
    val_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    # Train hybrid ensemble
    print(f"XGBoost weight: {xgb_weight}, LSTM weight: {1-xgb_weight}")

    ensemble = HybridEnsemble(xgb_weight=xgb_weight)
    ensemble.fit(
        X_tr, y_tr,
        X_val, y_val,
        feature_names=feature_columns,
        epochs=50,
        verbose=True
    )

    # Evaluate
    metrics = ensemble.evaluate(X_test, y_test)

    print("\nModel Performance on Test Set:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    results['model_metrics'] = metrics

    # Get predictions
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)

    # Save models
    if save_models:
        ensemble.save("models")

    # ========================================
    # STEP 6: SHAP EXPLAINABILITY
    # ========================================
    if generate_reports:
        print("\n" + "="*60)
        print("STEP 6: SHAP EXPLAINABILITY")
        print("="*60)

        # Align test data with predictions (for LSTM offset)
        sequence_length = 60
        X_test_aligned = X_test[-len(predictions):]
        test_dates = test_df.index[-len(predictions):]

        shap_results = generate_full_shap_report(
            ensemble.xgb_model,
            X_test_aligned,
            feature_columns,
            predictions=probabilities,
            dates=test_dates,
            output_dir="reports/shap_analysis"
        )

        results['feature_importance'] = shap_results['feature_importance'].to_dict('records')

        # Print top features
        print("\nTop 10 Most Important Features:")
        for i, row in shap_results['feature_importance'].head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

    # ========================================
    # STEP 7: BACKTESTING
    # ========================================
    print("\n" + "="*60)
    print("STEP 7: BACKTESTING")
    print("="*60)

    # Align prices with predictions
    test_prices = test_df['Close'].iloc[-len(predictions):]

    # Convert probabilities to signals
    buy_threshold = 0.55
    sell_threshold = 0.45
    signals = np.where(
        probabilities > buy_threshold, 1,
        np.where(probabilities < sell_threshold, -1, 0)
    )

    print(f"Signal distribution: Buy={np.sum(signals==1)}, Sell={np.sum(signals==-1)}, Hold={np.sum(signals==0)}")

    # Run backtest
    backtester = StrategyBacktester(
        initial_capital=initial_capital,
        transaction_cost=0.001,
        slippage=0.0005
    )
    backtest_results = backtester.run(test_prices, signals)
    strategy_metrics = backtester.calculate_metrics()
    strategy_metrics['Strategy'] = 'AI Multi-Factor'

    print("\nStrategy Performance:")
    for metric, value in strategy_metrics.items():
        if metric != 'Strategy':
            print(f"  {metric}: {value}")

    results['backtest_metrics'] = strategy_metrics

    # Save trade log
    trade_log = backtester.get_trade_log()
    if not trade_log.empty:
        trade_log.to_csv('reports/trade_log.csv', index=False)
        print(f"\nTrade log saved: {len(trade_log)} trades")

    # ========================================
    # STEP 8: BENCHMARK COMPARISON
    # ========================================
    print("\n" + "="*60)
    print("STEP 8: BENCHMARK COMPARISON")
    print("="*60)

    # Buy & Hold benchmark
    buy_hold_metrics = calculate_buy_and_hold(test_prices, initial_capital)

    # RSI-only benchmark
    test_df_aligned = test_df.iloc[-len(predictions):]
    rsi_signals = np.where(
        test_df_aligned['rsi_14'] < 30, 1,
        np.where(test_df_aligned['rsi_14'] > 70, -1, 0)
    )
    rsi_metrics = single_factor_benchmark(
        test_prices, rsi_signals, 'RSI Only', initial_capital
    )

    # Momentum-only benchmark
    momentum_signals = np.where(
        test_df_aligned['momentum_21d'] > 0.02, 1,
        np.where(test_df_aligned['momentum_21d'] < -0.02, -1, 0)
    )
    momentum_metrics = single_factor_benchmark(
        test_prices, momentum_signals, 'Momentum Only', initial_capital
    )

    # Generate comparison report
    comparison = generate_comparison_report(
        strategy_metrics,
        [buy_hold_metrics, rsi_metrics, momentum_metrics]
    )

    # Save comparison
    comparison.to_csv('reports/strategy_comparison.csv')
    results['comparison'] = comparison.to_dict()

    # ========================================
    # STEP 9: SAVE FINAL RESULTS
    # ========================================
    print("\n" + "="*60)
    print("STEP 9: SAVING RESULTS")
    print("="*60)

    # Save OHLC data
    df.to_csv('data/ohlc_data.csv')
    print("OHLC data saved: data/ohlc_data.csv")

    # Save predictions
    predictions_df = pd.DataFrame({
        'Date': test_prices.index,
        'Close': test_prices.values,
        'Signal': signals,
        'Probability': probabilities
    })
    predictions_df.to_csv('reports/predictions.csv', index=False)
    print("Predictions saved: reports/predictions.csv")

    # Save summary JSON
    summary_results = {
        'ticker': ticker,
        'period': {
            'start': start_date,
            'end': end_date
        },
        'data_summary': summary,
        'model_metrics': metrics,
        'backtest_metrics': {k: v for k, v in strategy_metrics.items() if k != 'Strategy'},
        'benchmark_comparison': {
            'ai_strategy': strategy_metrics.get('Total Return (%)', 0),
            'buy_hold': buy_hold_metrics.get('Total Return (%)', 0),
            'rsi_only': rsi_metrics.get('Total Return (%)', 0),
            'momentum_only': momentum_metrics.get('Total Return (%)', 0)
        }
    }

    with open('reports/summary.json', 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
    print("Summary saved: reports/summary.json")

    # ========================================
    # FINAL REPORT
    # ========================================
    print("\n" + "="*60)
    print("FINAL RESULTS - AI MULTI-FACTOR STRATEGY")
    print("="*60)

    print(f"\nTicker: {ticker}")
    print(f"Test Period: {test_prices.index[0].date()} to {test_prices.index[-1].date()}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${backtest_results['final_value']:,.2f}")

    print("\nKey Metrics:")
    print(f"  Total Return: {strategy_metrics['Total Return (%)']}%")
    print(f"  Annual Return: {strategy_metrics['Annual Return (%)']}%")
    print(f"  Sharpe Ratio: {strategy_metrics['Sharpe Ratio']}")
    print(f"  Max Drawdown: {strategy_metrics['Max Drawdown (%)']}%")
    print(f"  Win Rate: {strategy_metrics['Win Rate (%)']}%")
    print(f"  Total Trades: {strategy_metrics['Total Trades']}")

    print("\nBenchmark Comparison:")
    print(f"  AI Strategy:    {strategy_metrics['Total Return (%)']}%")
    print(f"  Buy & Hold:     {buy_hold_metrics['Total Return (%)']}%")
    print(f"  RSI Only:       {rsi_metrics['Total Return (%)']}%")
    print(f"  Momentum Only:  {momentum_metrics['Total Return (%)']}%")

    print("\n" + "="*60)
    print("IMPLEMENTATION COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  - data/ohlc_data.csv (OHLC dataset)")
    print("  - models/ (trained XGBoost + LSTM models)")
    print("  - reports/predictions.csv (trade signals)")
    print("  - reports/trade_log.csv (trade history)")
    print("  - reports/strategy_comparison.csv (benchmark comparison)")
    print("  - reports/summary.json (results summary)")
    print("  - reports/shap_analysis/ (SHAP explainability)")
    print("    - shap_global_importance.png")
    print("    - shap_beeswarm.png")
    print("    - shap_waterfall_*.png")
    print("    - trade_justifications.json")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI-Driven Multi-Factor Trading Strategy'
    )
    parser.add_argument(
        '--ticker', type=str, default='SPY',
        help='Stock ticker symbol (default: SPY)'
    )
    parser.add_argument(
        '--start', type=str, default='2015-01-01',
        help='Start date YYYY-MM-DD (default: 2015-01-01)'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='End date YYYY-MM-DD (default: today)'
    )
    parser.add_argument(
        '--capital', type=float, default=100000,
        help='Initial capital (default: 100000)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Test set size fraction (default: 0.2)'
    )
    parser.add_argument(
        '--xgb-weight', type=float, default=0.6,
        help='XGBoost weight in ensemble (default: 0.6)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save models'
    )
    parser.add_argument(
        '--no-reports', action='store_true',
        help='Skip SHAP report generation'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    print_banner()

    # Parse arguments
    args = parse_args()

    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('reports/shap_analysis').mkdir(parents=True, exist_ok=True)

    # Run pipeline
    results = run_pipeline(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        test_size=args.test_size,
        xgb_weight=args.xgb_weight,
        save_models=not args.no_save,
        generate_reports=not args.no_reports
    )

    return results


if __name__ == "__main__":
    results = main()
