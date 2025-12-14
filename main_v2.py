"""
AI-Driven Multi-Factor Trading Strategy V2
==========================================
CPAIT Batch 2025 Capstone Project

IMPROVEMENTS OVER V1:
1. Enhanced RSI-focused feature engineering (was underutilized)
2. Market regime detection (avoid trading in bad conditions)
3. Optuna hyperparameter optimization for XGBoost + LightGBM
4. Advanced LSTM with attention mechanism (GPU accelerated)
5. Confidence-based trading (skip low-confidence trades)
6. Stop-loss, take-profit, and trailing stops
7. Volatility-based position sizing
8. Walk-forward validation

Usage:
    python main_v2.py                    # Run full V2 pipeline
    python main_v2.py --ticker AAPL      # Different ticker
    python main_v2.py --n-trials 50      # More optimization trials
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
from feature_engineering_v2 import EnhancedFeatureEngineer
from models_v2 import AdvancedEnsembleV2, walk_forward_validation
from backtester_v2 import AdvancedBacktester, RiskManager, calculate_benchmark_v2, generate_comparison_report_v2
from explainability import SHAPExplainer, generate_full_shap_report


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     AI-DRIVEN MULTI-FACTOR TRADING STRATEGY V2                              ║
║     Enhanced with Hyperparameter Optimization & Risk Management             ║
║                                                                              ║
║     IMPROVEMENTS:                                                            ║
║     + Enhanced RSI features (primary alpha)                                  ║
║     + Market regime detection                                                ║
║     + Optuna hyperparameter tuning (XGBoost + LightGBM)                     ║
║     + Stop-loss, take-profit, trailing stops                                ║
║     + Confidence-based trading                                               ║
║     + Volatility-adjusted position sizing                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_pipeline_v2(
    ticker: str = "SPY",
    start_date: str = "2015-01-01",
    end_date: str = None,
    initial_capital: float = 100000,
    test_size: float = 0.2,
    n_trials: int = 30,
    use_lstm: bool = True,
    stop_loss: float = 0.03,
    take_profit: float = 0.06,
    confidence_threshold: float = 0.55,
    save_models: bool = True,
    generate_reports: bool = True
) -> dict:
    """
    Run the enhanced V2 pipeline.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    results = {}

    # ========================================
    # STEP 1: DATA ACQUISITION
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 70)

    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        save=True
    )

    summary = loader.get_data_summary(df)
    print(f"Ticker: {ticker}")
    print(f"Period: {summary['start_date']} to {summary['end_date']} ({summary['trading_years']} years)")
    print(f"Total records: {summary['total_days']}")

    results['data_summary'] = summary

    # ========================================
    # STEP 2: ENHANCED FEATURE ENGINEERING
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2: ENHANCED FEATURE ENGINEERING (V2)")
    print("=" * 70)

    engineer = EnhancedFeatureEngineer()
    df = engineer.create_all_features(df)

    feature_columns = engineer.get_feature_columns()
    # Filter to columns that exist
    feature_columns = [c for c in feature_columns if c in df.columns]

    print(f"Total features created: {len(df.columns)}")
    print(f"Selected for ML: {len(feature_columns)}")

    # Feature breakdown
    rsi_features = [f for f in feature_columns if 'rsi' in f.lower()]
    macd_features = [f for f in feature_columns if 'macd' in f.lower()]
    momentum_features = [f for f in feature_columns if 'momentum' in f.lower() or 'roc' in f.lower()]
    regime_features = [f for f in feature_columns if 'regime' in f.lower() or 'trend' in f.lower()]

    print(f"\nFeature breakdown:")
    print(f"  RSI-based: {len(rsi_features)} (ENHANCED)")
    print(f"  MACD-based: {len(macd_features)}")
    print(f"  Momentum: {len(momentum_features)}")
    print(f"  Regime: {len(regime_features)} (NEW)")

    # ========================================
    # STEP 3: TARGET VARIABLE
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 3: TARGET VARIABLE (Risk-Adjusted)")
    print("=" * 70)

    # Use slightly higher threshold for better signal quality
    df = engineer.create_target_variable(df, horizon=5, threshold=0.003, use_risk_adjusted=True)

    df_clean = df.dropna(subset=feature_columns + ['target'])
    print(f"Clean samples: {len(df_clean)}")

    target_dist = df_clean['target'].value_counts(normalize=True)
    print(f"\nTarget distribution:")
    print(f"  Positive (1): {target_dist.get(1, 0)*100:.1f}%")
    print(f"  Negative (0): {target_dist.get(0, 0)*100:.1f}%")

    # ========================================
    # STEP 4: TRAIN/TEST SPLIT
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("=" * 70)

    split_idx = int(len(df_clean) * (1 - test_size))
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    # Further split train for validation
    val_split = int(len(train_df) * 0.8)
    train_df_tr = train_df.iloc[:val_split]
    train_df_val = train_df.iloc[val_split:]

    X_train = train_df_tr[feature_columns].values
    y_train = train_df_tr['target'].values
    X_val = train_df_val[feature_columns].values
    y_val = train_df_val['target'].values
    X_test = test_df[feature_columns].values
    y_test = test_df['target'].values

    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Train period: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"Test period: {test_df.index[0].date()} to {test_df.index[-1].date()}")

    # ========================================
    # STEP 5: MODEL TRAINING WITH OPTIMIZATION
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 5: HYPERPARAMETER OPTIMIZATION & TRAINING")
    print("=" * 70)
    print(f"Optuna trials: {n_trials}")
    print(f"Using LSTM: {use_lstm}")

    ensemble = AdvancedEnsembleV2(
        n_trials=n_trials,
        sequence_length=30,
        use_lstm=use_lstm,
        confidence_threshold=confidence_threshold
    )

    ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_columns,
        epochs=100 if use_lstm else 0
    )

    # Evaluate
    metrics = ensemble.evaluate(X_test, y_test)

    print("\n" + "-" * 50)
    print("MODEL PERFORMANCE (Test Set)")
    print("-" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    results['model_metrics'] = metrics

    # Get predictions with confidence filtering
    signals, probabilities, confidence = ensemble.predict_with_confidence(X_test)

    print(f"\nSignal distribution (confidence threshold: {confidence_threshold}):")
    print(f"  Buy signals: {int((signals == 1).sum())}")
    print(f"  Sell signals: {int((signals == -1).sum())}")
    print(f"  Hold (low confidence): {int((signals == 0).sum())}")

    # Save models
    if save_models:
        ensemble.save("models_v2")

    # ========================================
    # STEP 6: SHAP EXPLAINABILITY
    # ========================================
    if generate_reports:
        print("\n" + "=" * 70)
        print("STEP 6: SHAP EXPLAINABILITY")
        print("=" * 70)

        # Align test data with predictions
        X_test_aligned = X_test[-len(signals):]
        test_dates = test_df.index[-len(signals):]

        try:
            shap_results = generate_full_shap_report(
                ensemble.xgb_model,
                X_test_aligned,
                feature_columns,
                predictions=probabilities,
                dates=test_dates,
                output_dir="reports_v2/shap_analysis"
            )

            results['feature_importance'] = shap_results['feature_importance'].to_dict('records')

            print("\nTop 10 Most Important Features (V2):")
            for i, row in shap_results['feature_importance'].head(10).iterrows():
                print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            print(f"SHAP analysis error: {e}")

    # ========================================
    # STEP 7: BACKTESTING WITH RISK MANAGEMENT
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 7: BACKTESTING WITH RISK MANAGEMENT (V2)")
    print("=" * 70)
    print(f"Stop-loss: {stop_loss*100}%")
    print(f"Take-profit: {take_profit*100}%")
    print(f"Trailing stop: 2%")

    # Align data
    test_prices = test_df['Close'].iloc[-len(signals):]
    test_volatility = test_df['volatility_20d'].iloc[-len(signals):] if 'volatility_20d' in test_df.columns else None

    # Create regime series if available
    if 'trend_bear' in test_df.columns:
        regime = test_df['trend_bear'].iloc[-len(signals):].apply(lambda x: -1 if x else 1)
    else:
        regime = None

    # Risk manager
    risk_manager = RiskManager(
        max_position_size=0.90,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        trailing_stop_pct=0.02,
        max_drawdown_limit=0.15,
        volatility_scaling=True,
        target_volatility=0.15
    )

    # Run backtest
    backtester = AdvancedBacktester(
        initial_capital=initial_capital,
        transaction_cost=0.001,
        slippage=0.0005,
        risk_manager=risk_manager
    )

    backtest_results = backtester.run(
        test_prices,
        signals,
        probabilities,
        test_volatility,
        regime
    )

    strategy_metrics = backtester.calculate_metrics()
    strategy_metrics['Strategy'] = 'AI Multi-Factor V2'

    print("\n" + "-" * 50)
    print("STRATEGY PERFORMANCE (V2)")
    print("-" * 50)
    for metric, value in strategy_metrics.items():
        if metric != 'Strategy':
            print(f"  {metric}: {value}")

    results['backtest_metrics'] = strategy_metrics

    # Save trade log
    trade_log = backtester.get_trade_log()
    if not trade_log.empty:
        Path('reports_v2').mkdir(exist_ok=True)
        trade_log.to_csv('reports_v2/trade_log_v2.csv', index=False)
        print(f"\nTrade log saved: {len(trade_log)} entries")

    # ========================================
    # STEP 8: BENCHMARK COMPARISON
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 8: BENCHMARK COMPARISON")
    print("=" * 70)

    # Buy & Hold
    buy_hold = calculate_benchmark_v2(test_prices, initial_capital)

    # RSI-only (improved rules)
    test_df_aligned = test_df.iloc[-len(signals):]
    rsi_signals = np.where(
        (test_df_aligned['rsi_14'] < 30) & (test_df_aligned['rsi_14'] > test_df_aligned['rsi_14'].shift(1)), 1,
        np.where((test_df_aligned['rsi_14'] > 70) & (test_df_aligned['rsi_14'] < test_df_aligned['rsi_14'].shift(1)), -1, 0)
    )

    rsi_backtester = AdvancedBacktester(initial_capital=initial_capital, risk_manager=risk_manager)
    rsi_backtester.run(test_prices, rsi_signals)
    rsi_metrics = rsi_backtester.calculate_metrics()
    rsi_metrics['Strategy'] = 'RSI Only (with stops)'

    # Momentum-only
    momentum_signals = np.where(
        test_df_aligned['momentum_21d'] > 0.02, 1,
        np.where(test_df_aligned['momentum_21d'] < -0.02, -1, 0)
    )

    mom_backtester = AdvancedBacktester(initial_capital=initial_capital, risk_manager=risk_manager)
    mom_backtester.run(test_prices, momentum_signals)
    mom_metrics = mom_backtester.calculate_metrics()
    mom_metrics['Strategy'] = 'Momentum Only (with stops)'

    # Generate comparison
    comparison = generate_comparison_report_v2(
        strategy_metrics,
        [buy_hold, rsi_metrics, mom_metrics],
        save_path='reports_v2/strategy_comparison_v2.csv'
    )
    results['comparison'] = comparison.to_dict()

    # ========================================
    # STEP 9: SAVE RESULTS
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 9: SAVING RESULTS")
    print("=" * 70)

    Path('reports_v2').mkdir(exist_ok=True)

    # Save predictions
    predictions_df = pd.DataFrame({
        'Date': test_prices.index,
        'Close': test_prices.values,
        'Signal': signals,
        'Probability': probabilities,
        'Confidence': confidence
    })
    predictions_df.to_csv('reports_v2/predictions_v2.csv', index=False)

    # Save summary
    summary_results = {
        'version': 'V2',
        'ticker': ticker,
        'period': {'start': start_date, 'end': end_date},
        'hyperparameters': {
            'n_trials': n_trials,
            'confidence_threshold': confidence_threshold,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        },
        'model_metrics': metrics,
        'backtest_metrics': {k: v for k, v in strategy_metrics.items() if k != 'Strategy'},
        'improvements_over_v1': {
            'enhanced_rsi_features': True,
            'regime_detection': True,
            'hyperparameter_optimization': True,
            'risk_management': True,
            'confidence_filtering': True
        }
    }

    with open('reports_v2/summary_v2.json', 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)

    # ========================================
    # FINAL REPORT
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - AI MULTI-FACTOR STRATEGY V2")
    print("=" * 70)

    print(f"\nTicker: {ticker}")
    print(f"Test Period: {test_prices.index[0].date()} to {test_prices.index[-1].date()}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${backtest_results['final_value']:,.2f}")

    print("\nKey Metrics (V2):")
    print(f"  Total Return: {strategy_metrics['Total Return (%)']}%")
    print(f"  Annual Return: {strategy_metrics['Annual Return (%)']}%")
    print(f"  Sharpe Ratio: {strategy_metrics['Sharpe Ratio']}")
    print(f"  Max Drawdown: {strategy_metrics['Max Drawdown (%)']}%")
    print(f"  Win Rate: {strategy_metrics['Win Rate (%)']}%")
    print(f"  Profit Factor: {strategy_metrics['Profit Factor']}")
    print(f"  Round Trips: {strategy_metrics['Round Trips']}")

    print("\nRisk Management Stats:")
    print(f"  Stop Loss Exits: {strategy_metrics['Stop Loss Exits']}")
    print(f"  Take Profit Exits: {strategy_metrics['Take Profit Exits']}")
    print(f"  Trailing Stop Exits: {strategy_metrics['Trailing Stop Exits']}")

    print("\nBenchmark Comparison:")
    print(f"  AI Strategy V2:  {strategy_metrics['Total Return (%)']}%")
    print(f"  Buy & Hold:      {buy_hold['Total Return (%)']}%")
    print(f"  RSI Only:        {rsi_metrics['Total Return (%)']}%")
    print(f"  Momentum Only:   {mom_metrics['Total Return (%)']}%")

    print("\n" + "=" * 70)
    print("V2 IMPLEMENTATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files (V2):")
    print("  - models_v2/ (optimized XGBoost + LightGBM + LSTM)")
    print("  - reports_v2/predictions_v2.csv")
    print("  - reports_v2/trade_log_v2.csv")
    print("  - reports_v2/strategy_comparison_v2.csv")
    print("  - reports_v2/summary_v2.json")
    print("  - reports_v2/shap_analysis/")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='AI Trading Strategy V2')
    parser.add_argument('--ticker', type=str, default='SPY')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--n-trials', type=int, default=30, help='Optuna optimization trials')
    parser.add_argument('--no-lstm', action='store_true', help='Skip LSTM training')
    parser.add_argument('--stop-loss', type=float, default=0.03)
    parser.add_argument('--take-profit', type=float, default=0.06)
    parser.add_argument('--confidence', type=float, default=0.55, help='Min confidence to trade')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-reports', action='store_true')
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    # Create directories
    Path('data').mkdir(exist_ok=True)
    Path('models_v2').mkdir(exist_ok=True)
    Path('reports_v2/shap_analysis').mkdir(parents=True, exist_ok=True)

    results = run_pipeline_v2(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        test_size=args.test_size,
        n_trials=args.n_trials,
        use_lstm=not args.no_lstm,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        confidence_threshold=args.confidence,
        save_models=not args.no_save,
        generate_reports=not args.no_reports
    )

    return results


if __name__ == "__main__":
    results = main()
