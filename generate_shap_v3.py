"""
Generate SHAP analysis for existing V3 models
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Fix matplotlib backend to avoid tkinter threading errors
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from explainability import SHAPExplainer
from feature_engineering_v3 import FeatureEngineerV3
from models_v3 import StackingEnsembleV3

def generate_shap_for_ticker(ticker: str, reports_dir: str = 'reports_v3'):
    """Generate SHAP analysis for a single ticker"""

    print(f"\n{'='*70}")
    print(f"Generating SHAP Analysis for {ticker}")
    print('='*70)

    # Paths
    model_path = os.path.join('models_v3', f'ensemble_{ticker}.pkl')
    report_dir = os.path.join(reports_dir, ticker)
    shap_dir = os.path.join(report_dir, 'shap_analysis')

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"  Error: Model not found at {model_path}")
        return False

    print(f"  Loading model from {model_path}...")
    try:
        ensemble = StackingEnsembleV3.load(model_path)
    except Exception as e:
        print(f"  Error loading model: {e}")
        return False

    # Load data
    print(f"  Loading data...")
    try:
        # Load predictions file to get test indices
        pred_file = os.path.join(report_dir, f'predictions_{ticker}.csv')
        if not os.path.exists(pred_file):
            print(f"  Error: Predictions file not found at {pred_file}")
            return False

        pred_df = pd.read_csv(pred_file)

        # We need to reload the raw data and recreate features
        import yfinance as yf
        from datetime import datetime

        print(f"  Downloading {ticker} data...")
        df = yf.download(ticker, start='2018-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)

        if df.empty:
            print(f"  Error: No data downloaded for {ticker}")
            return False

        print(f"  Creating features...")
        # Use same settings as training: adaptive_lookback=False for 131 features
        engineer = FeatureEngineerV3(adaptive_lookback=False)
        df_features = engineer.create_features(df, ticker=ticker)
        df_features = df_features.dropna()

        # Get feature columns (exact same as main_v3.py)
        feature_names = engineer.get_feature_names()
        feature_cols = [c for c in feature_names if c in df_features.columns and
                        c not in ['target', 'forward_return', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Split data (same as training - 80/20 split)
        split_idx = int(len(df_features) * 0.8)
        X_test = df_features[feature_cols].iloc[split_idx:].values

        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(feature_cols)}")

    except Exception as e:
        print(f"  Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Generate SHAP analysis
    print(f"  Generating SHAP analysis...")
    try:
        # Scale data using ensemble's scaler
        X_test_scaled = ensemble.scaler.transform(X_test)

        # Create SHAP explainer using XGBoost base model
        explainer = SHAPExplainer(
            ensemble.xgb_model.model,
            feature_names=feature_cols
        )

        # Generate reports (use last 300 samples)
        n_samples = min(300, len(X_test_scaled))
        print(f"  Analyzing last {n_samples} samples...")

        explainer.generate_reports(
            X_test_scaled[-n_samples:],
            output_dir=shap_dir,
            prefix=ticker,
            n_waterfall=5,
            top_features_for_dependence=5
        )

        print(f"  ✅ SHAP analysis complete for {ticker}!")
        return True

    except Exception as e:
        print(f"  ❌ SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Generate SHAP analysis for all V3 tickers"""

    tickers = ['SPY', 'AAPL', 'NVDA', 'IBM']

    print("\n" + "="*70)
    print("V3 SHAP ANALYSIS GENERATION")
    print("="*70)
    print(f"Tickers: {', '.join(tickers)}")

    results = {}
    for ticker in tickers:
        success = generate_shap_for_ticker(ticker)
        results[ticker] = "✅ Success" if success else "❌ Failed"

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for ticker, status in results.items():
        print(f"  {ticker}: {status}")
    print("="*70)


if __name__ == '__main__':
    main()
