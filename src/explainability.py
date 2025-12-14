"""
Explainability Module
=====================
Implements SHAP-based explainability for trade justification.

SHAP (SHapley Additive exPlanations) provides:
- Global feature importance (which factors matter most overall)
- Local explanations (why specific trades were made)
- Feature interactions (how factors work together)

"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP explainer for XGBoost trading models.
    """

    def __init__(self, model, feature_names: list = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained XGBoost model (or XGBoostModel wrapper)
            feature_names: List of feature names
        """
        # Handle our XGBoostModel wrapper
        if hasattr(model, 'model'):
            self.model = model.model
            self.scaler = model.scaler
            self.feature_names = model.feature_names or feature_names
        else:
            self.model = model
            self.scaler = None
            self.feature_names = feature_names

        self.explainer = None
        self.shap_values = None

    def fit(self, X_background: np.ndarray = None) -> 'SHAPExplainer':
        """
        Fit the SHAP explainer.

        Args:
            X_background: Background data for SHAP (optional for tree models)

        Returns:
            Self
        """
        # For tree-based models, TreeExplainer is fast and exact
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for given data.

        Args:
            X: Feature matrix to explain

        Returns:
            Array of SHAP values
        """
        # Scale if we have a scaler
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        self.shap_values = self.explainer.shap_values(X_scaled)

        return self.shap_values

    def plot_global_importance(
        self,
        X: np.ndarray,
        save_path: str = None,
        max_display: int = 20
    ) -> None:
        """
        Plot global feature importance (bar chart).

        Args:
            X: Feature matrix
            save_path: Path to save the plot
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.explain(X)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title("Global Feature Importance (SHAP)", fontsize=14)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Global importance plot saved to {save_path}")

        plt.close()

    def plot_beeswarm(
        self,
        X: np.ndarray,
        save_path: str = None,
        max_display: int = 20
    ) -> None:
        """
        Plot beeswarm (feature impact distribution).

        Args:
            X: Feature matrix
            save_path: Path to save the plot
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.explain(X)

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("Feature Impact Distribution (SHAP)", fontsize=14)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Beeswarm plot saved to {save_path}")

        plt.close()

    def plot_waterfall(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        save_path: str = None
    ) -> None:
        """
        Plot waterfall for a single prediction.

        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.explain(X)

        # Create explanation object
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]

        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=base_value,
            data=X[sample_idx],
            feature_names=self.feature_names
        )

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"Trade Explanation (Sample {sample_idx})", fontsize=14)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")

        plt.close()

    def plot_force(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        save_path: str = None
    ) -> None:
        """
        Plot force plot for a single prediction.

        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.explain(X)

        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]

        plt.figure(figsize=(20, 4))
        shap.force_plot(
            base_value,
            self.shap_values[sample_idx],
            X[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Force plot saved to {save_path}")

        plt.close()

    def plot_dependence(
        self,
        X: np.ndarray,
        feature: str,
        interaction_feature: str = None,
        save_path: str = None
    ) -> None:
        """
        Plot dependence plot for a feature.

        Args:
            X: Feature matrix
            feature: Feature name to plot
            interaction_feature: Feature to color by (optional)
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.explain(X)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f"SHAP Dependence: {feature}", fontsize=14)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Dependence plot saved to {save_path}")

        plt.close()

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.

        Returns:
            DataFrame with feature importances
        """
        if self.shap_values is None:
            raise ValueError("Call explain() first")

        importance = np.abs(self.shap_values).mean(axis=0)
        names = self.feature_names or [f"f{i}" for i in range(len(importance))]

        return pd.DataFrame({
            'feature': names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def explain_single_trade(
        self,
        X_single: np.ndarray,
        prediction: float,
        top_n: int = 5
    ) -> dict:
        """
        Generate explanation for a single trade.

        Args:
            X_single: Feature vector for single sample
            prediction: Model prediction probability
            top_n: Number of top factors to include

        Returns:
            Dictionary with trade explanation
        """
        # Ensure 2D
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        # Get SHAP values
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_single)
        else:
            X_scaled = X_single

        shap_values = self.explainer.shap_values(X_scaled)[0]

        # Get feature names
        names = self.feature_names or [f"f{i}" for i in range(len(shap_values))]

        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            'feature': names,
            'value': X_single[0],
            'shap_value': shap_values
        }).sort_values('shap_value', key=abs, ascending=False)

        # Build report
        signal = 'BUY' if prediction > 0.5 else 'HOLD/SELL'

        # Handle expected_value which might be array or scalar
        base_val = self.explainer.expected_value
        if hasattr(base_val, '__len__') and len(base_val) > 0:
            base_val = base_val[0] if isinstance(base_val, np.ndarray) else base_val

        report = {
            'signal': signal,
            'confidence': round(float(prediction) * 100, 1),
            'base_value': round(float(base_val), 4),
            'top_factors': []
        }

        for _, row in explanation_df.head(top_n).iterrows():
            direction = 'bullish' if row['shap_value'] > 0 else 'bearish'
            report['top_factors'].append({
                'factor': str(row['feature']),
                'value': round(float(row['value']), 4),
                'shap_impact': round(float(row['shap_value']), 4),
                'direction': direction
            })

        return report

    def generate_reports(
        self,
        X: np.ndarray,
        output_dir: str,
        prefix: str = "",
        n_waterfall: int = 5,
        top_features_for_dependence: int = 3
    ) -> None:
        """
        Generate comprehensive SHAP analysis reports.

        Args:
            X: Feature matrix to analyze
            output_dir: Directory to save reports
            prefix: Prefix for file names (e.g., ticker symbol)
            n_waterfall: Number of waterfall plots to generate
            top_features_for_dependence: Number of top features for dependence plots
        """
        print(f"  Generating SHAP reports for {len(X)} samples...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Fit explainer and calculate SHAP values
        self.fit()
        self.explain(X)

        # 1. Global feature importance (bar plot)
        global_path = output_path / f"{prefix}_shap_global_importance.png"
        self.plot_global_importance(X, save_path=str(global_path))

        # 2. Beeswarm plot (feature impact distribution)
        beeswarm_path = output_path / f"{prefix}_shap_beeswarm.png"
        self.plot_beeswarm(X, save_path=str(beeswarm_path))

        # 3. Feature importance CSV
        importance_df = self.get_feature_importance()
        csv_path = output_path / f"{prefix}_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        print(f"  Feature importance saved to {csv_path}")

        # 4. Waterfall plots for recent samples (most recent n_waterfall)
        recent_indices = list(range(max(0, len(X) - n_waterfall), len(X)))
        for idx in recent_indices:
            waterfall_path = output_path / f"{prefix}_shap_waterfall_{idx}.png"
            try:
                self.plot_waterfall(X, sample_idx=idx, save_path=str(waterfall_path))
            except Exception as e:
                print(f"  Warning: Waterfall plot for sample {idx} failed: {e}")

        # 5. Dependence plots for top features
        top_features = importance_df['feature'].head(top_features_for_dependence).tolist()
        for feature in top_features:
            dep_path = output_path / f"{prefix}_shap_dependence_{feature}.png"
            try:
                self.plot_dependence(X, feature=feature, save_path=str(dep_path))
            except Exception as e:
                print(f"  Warning: Dependence plot for {feature} failed: {e}")

        print(f"  SHAP analysis complete! Reports saved to {output_dir}")


def generate_trade_justification_report(
    explainer: SHAPExplainer,
    X: np.ndarray,
    predictions: np.ndarray,
    dates: pd.DatetimeIndex = None,
    n_trades: int = 10,
    save_path: str = None
) -> list:
    """
    Generate trade justification reports for multiple trades.

    Args:
        explainer: Fitted SHAP explainer
        X: Feature matrix
        predictions: Model predictions
        dates: Dates for the trades
        n_trades: Number of trades to explain
        save_path: Path to save JSON report

    Returns:
        List of trade reports
    """
    reports = []

    # Get the last n_trades
    start_idx = max(0, len(X) - n_trades)

    for i in range(start_idx, len(X)):
        report = explainer.explain_single_trade(
            X[i],
            predictions[i - start_idx] if len(predictions) > (i - start_idx) else 0.5
        )

        # Add date if available
        if dates is not None and i < len(dates):
            report['date'] = str(dates[i])

        report['trade_id'] = i - start_idx

        reports.append(report)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(reports, f, indent=2)
        print(f"Trade justifications saved to {save_path}")

    return reports


def generate_full_shap_report(
    model,
    X_test: np.ndarray,
    feature_names: list,
    predictions: np.ndarray = None,
    dates: pd.DatetimeIndex = None,
    output_dir: str = "reports/shap_analysis"
) -> dict:
    """
    Generate complete SHAP analysis report with all visualizations.

    Args:
        model: Trained model (XGBoostModel or xgb.XGBClassifier)
        X_test: Test feature matrix
        feature_names: List of feature names
        predictions: Model predictions
        dates: Dates for test data
        output_dir: Output directory for reports

    Returns:
        Dictionary with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING SHAP EXPLAINABILITY REPORT")
    print("="*60)

    # Create explainer
    explainer = SHAPExplainer(model, feature_names)
    explainer.fit()

    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.explain(X_test)

    # 1. Global Feature Importance
    print("Generating global importance plot...")
    explainer.plot_global_importance(
        X_test,
        save_path=str(output_dir / "shap_global_importance.png")
    )

    # 2. Beeswarm Plot
    print("Generating beeswarm plot...")
    explainer.plot_beeswarm(
        X_test,
        save_path=str(output_dir / "shap_beeswarm.png")
    )

    # 3. Waterfall plots for sample trades
    print("Generating waterfall plots...")
    for i in [0, len(X_test) // 2, -1]:
        if i < 0:
            i = len(X_test) + i
        explainer.plot_waterfall(
            X_test,
            sample_idx=i,
            save_path=str(output_dir / f"shap_waterfall_sample_{i}.png")
        )

    # 4. Force plots
    print("Generating force plots...")
    for i in [0, len(X_test) // 2]:
        try:
            explainer.plot_force(
                X_test,
                sample_idx=i,
                save_path=str(output_dir / f"shap_force_sample_{i}.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate force plot for sample {i}: {e}")

    # 5. Dependence plots for key features
    print("Generating dependence plots...")
    key_features = ['rsi_14', 'macd', 'momentum_21d']
    for feature in key_features:
        if feature in feature_names:
            try:
                explainer.plot_dependence(
                    X_test,
                    feature=feature,
                    save_path=str(output_dir / f"shap_dependence_{feature}.png")
                )
            except Exception as e:
                print(f"Warning: Could not generate dependence plot for {feature}: {e}")

    # 6. Feature importance table
    importance_df = explainer.get_feature_importance()
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # 7. Trade justifications
    if predictions is not None:
        print("Generating trade justifications...")
        reports = generate_trade_justification_report(
            explainer,
            X_test,
            predictions,
            dates=dates,
            n_trades=10,
            save_path=str(output_dir / "trade_justifications.json")
        )
    else:
        reports = []

    print("\n" + "="*60)
    print("SHAP REPORT COMPLETE")
    print("="*60)
    print(f"Reports saved to: {output_dir}")

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_importance': importance_df,
        'trade_reports': reports
    }


def main():
    """Test the explainability module."""
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from models import XGBoostModel

    # Load and prepare data
    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo("SPY", start_date="2018-01-01", end_date="2024-12-01")

    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    df = engineer.create_target_variable(df, horizon=5)
    df = df.dropna()

    feature_columns = engineer.get_feature_columns()

    # Split data
    split_idx = int(len(df) * 0.8)
    X_train = df[feature_columns].values[:split_idx]
    y_train = df['target'].values[:split_idx]
    X_test = df[feature_columns].values[split_idx:]
    y_test = df['target'].values[split_idx:]
    test_dates = df.index[split_idx:]

    # Train model
    print("Training XGBoost model...")
    model = XGBoostModel()
    model.fit(X_train, y_train, feature_names=feature_columns)

    # Get predictions
    predictions = model.predict_proba(X_test)[:, 1]

    # Generate full SHAP report
    results = generate_full_shap_report(
        model,
        X_test,
        feature_columns,
        predictions=predictions,
        dates=test_dates,
        output_dir="reports/shap_analysis"
    )

    # Print sample trade justification
    if results['trade_reports']:
        print("\nSample Trade Justification:")
        report = results['trade_reports'][0]
        print(f"  Signal: {report['signal']}")
        print(f"  Confidence: {report['confidence']}%")
        print("  Top factors:")
        for factor in report['top_factors']:
            print(f"    - {factor['factor']}: {factor['value']:.4f} ({factor['direction']}, SHAP: {factor['shap_impact']:.4f})")

    return results


if __name__ == "__main__":
    results = main()
