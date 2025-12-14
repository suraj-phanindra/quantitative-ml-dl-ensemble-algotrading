"""
Models Module
=============
Implements ML/DL models for the trading strategy.

Architecture: XGBoost + LSTM Hybrid Ensemble
- XGBoost: Best for tabular data, captures non-linear relationships
- LSTM: Captures temporal dependencies and sequential patterns
- Ensemble: Combines strengths, reduces overfitting

Research shows XGBoost often performs better on financial data,
so default ensemble weights give it higher weight (0.6).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM model will be disabled.")


class XGBoostModel:
    """
    XGBoost classifier for trading signal prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model with optimal parameters.

        Parameters are tuned based on financial data characteristics.
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
        early_stopping_rounds: int = 20
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features
            early_stopping_rounds: Early stopping patience

        Returns:
            Self
        """
        self.feature_names = feature_names

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]

            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        importance = self.model.feature_importances_
        names = self.feature_names if self.feature_names else [f"f{i}" for i in range(len(importance))]

        return pd.DataFrame({
            'feature': names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, filepath: str) -> None:
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        print(f"XGBoost model saved to {filepath}")

    def load(self, filepath: str) -> 'XGBoostModel':
        """Load model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = True
        print(f"XGBoost model loaded from {filepath}")
        return self


class LSTMModel:
    """
    LSTM model for sequential trading signal prediction.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        lstm_units: list = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.

        Args:
            sequence_length: Number of time steps in each sequence
            lstm_units: List of units for each LSTM layer [64, 32, 16]
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units or [64, 32, 16]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def _build_model(self, n_features: int) -> Sequential:
        """Build the LSTM architecture."""
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, n_features)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        # Second LSTM layer
        if len(self.lstm_units) > 1:
            model.add(LSTM(self.lstm_units[1], return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))

        # Third LSTM layer
        if len(self.lstm_units) > 2:
            model.add(LSTM(self.lstm_units[2], return_sequences=False))
            model.add(Dropout(self.dropout_rate))

        # Dense layers
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> tuple:
        """
        Prepare sequences for LSTM input.

        Args:
            X: Feature matrix
            y: Target array (optional)

        Returns:
            Tuple of (X_sequences, y_sequences) or just X_sequences
        """
        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])

        X_seq = np.array(X_seq)

        if y is not None:
            return X_seq, np.array(y_seq)
        return X_seq

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0
    ) -> 'LSTMModel':
        """
        Train the LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            verbose: Verbosity level

        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y_train)

        # Build model
        n_features = X_train.shape[1]
        self.model = self._build_model(n_features)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]

        # Train
        self.history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        X_seq = self._prepare_sequences(X_scaled)
        return self.model.predict(X_seq, verbose=0)

    def save(self, filepath: str) -> None:
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        # Save scaler separately
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"LSTM model saved to {filepath}")

    def load(self, filepath: str) -> 'LSTMModel':
        """Load model from file."""
        self.model = load_model(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True
        print(f"LSTM model loaded from {filepath}")
        return self


class HybridEnsemble:
    """
    Hybrid ensemble combining XGBoost and LSTM predictions.

    Research shows this combination captures both:
    - Non-linear feature relationships (XGBoost)
    - Temporal dependencies (LSTM)
    """

    def __init__(
        self,
        xgb_weight: float = 0.6,
        sequence_length: int = 60
    ):
        """
        Initialize hybrid ensemble.

        Args:
            xgb_weight: Weight for XGBoost predictions (LSTM gets 1 - xgb_weight)
            sequence_length: Sequence length for LSTM
        """
        self.xgb_weight = xgb_weight
        self.lstm_weight = 1 - xgb_weight
        self.sequence_length = sequence_length

        self.xgb_model = XGBoostModel()
        self.lstm_model = LSTMModel(sequence_length=sequence_length) if TENSORFLOW_AVAILABLE else None

        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
        epochs: int = 50,
        verbose: bool = True
    ) -> 'HybridEnsemble':
        """
        Train both models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
            epochs: LSTM training epochs
            verbose: Print progress

        Returns:
            Self
        """
        if verbose:
            print("Training XGBoost model...")

        self.xgb_model.fit(
            X_train, y_train,
            X_val, y_val,
            feature_names=feature_names
        )

        if self.lstm_model is not None:
            if verbose:
                print("Training LSTM model...")
            self.lstm_model.fit(
                X_train, y_train,
                epochs=epochs,
                verbose=0
            )
        else:
            if verbose:
                print("LSTM model skipped (TensorFlow not available)")
            self.xgb_weight = 1.0
            self.lstm_weight = 0.0

        self.is_fitted = True

        if verbose:
            print("Hybrid ensemble training complete!")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using ensemble."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted ensemble.

        Returns:
            Array of probabilities (aligned to shorter length)
        """
        # XGBoost predictions
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]

        if self.lstm_model is not None and self.lstm_model.is_fitted:
            # LSTM predictions
            lstm_proba = self.lstm_model.predict_proba(X).flatten()

            # Align lengths (LSTM has fewer predictions due to sequence requirement)
            min_len = min(len(xgb_proba), len(lstm_proba))
            xgb_proba_aligned = xgb_proba[-min_len:]
            lstm_proba_aligned = lstm_proba[-min_len:]

            # Weighted ensemble
            ensemble_proba = (
                self.xgb_weight * xgb_proba_aligned +
                self.lstm_weight * lstm_proba_aligned
            )
        else:
            # Only XGBoost
            ensemble_proba = xgb_proba

        return ensemble_proba

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Evaluate ensemble performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        # Get predictions
        proba = self.predict_proba(X_test)
        predictions = (proba > 0.5).astype(int)

        # Align y_test with predictions (for LSTM offset)
        y_aligned = y_test[-len(predictions):]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_aligned, predictions),
            'precision': precision_score(y_aligned, predictions, zero_division=0),
            'recall': recall_score(y_aligned, predictions, zero_division=0),
            'f1_score': f1_score(y_aligned, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y_aligned, proba) if len(np.unique(y_aligned)) > 1 else 0,
            'win_rate': (predictions == y_aligned).mean() * 100
        }

        return metrics

    def save(self, model_dir: str) -> None:
        """Save both models."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.xgb_model.save(str(model_dir / 'xgboost_model.pkl'))

        if self.lstm_model is not None and self.lstm_model.is_fitted:
            self.lstm_model.save(str(model_dir / 'lstm_model.keras'))

        # Save ensemble config
        config = {
            'xgb_weight': self.xgb_weight,
            'lstm_weight': self.lstm_weight,
            'sequence_length': self.sequence_length
        }
        joblib.dump(config, model_dir / 'ensemble_config.pkl')

        print(f"Ensemble saved to {model_dir}")

    def load(self, model_dir: str) -> 'HybridEnsemble':
        """Load both models."""
        model_dir = Path(model_dir)

        self.xgb_model.load(str(model_dir / 'xgboost_model.pkl'))

        lstm_path = model_dir / 'lstm_model.keras'
        if lstm_path.exists() and self.lstm_model is not None:
            self.lstm_model.load(str(lstm_path))

        # Load config
        config = joblib.load(model_dir / 'ensemble_config.pkl')
        self.xgb_weight = config['xgb_weight']
        self.lstm_weight = config['lstm_weight']
        self.sequence_length = config['sequence_length']

        self.is_fitted = True
        print(f"Ensemble loaded from {model_dir}")

        return self


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = 'target',
    test_size: float = 0.2,
    xgb_weight: float = 0.6
) -> tuple:
    """
    Complete training and evaluation pipeline.

    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column
        test_size: Fraction of data for testing
        xgb_weight: Weight for XGBoost in ensemble

    Returns:
        Tuple of (ensemble_model, predictions, metrics)
    """
    # Prepare data
    df_clean = df.dropna(subset=feature_columns + [target_column])

    X = df_clean[feature_columns].values
    y = df_clean[target_column].values

    # Time series split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Further split for validation
    val_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train ensemble
    ensemble = HybridEnsemble(xgb_weight=xgb_weight)
    ensemble.fit(
        X_tr, y_tr,
        X_val, y_val,
        feature_names=feature_columns,
        verbose=True
    )

    # Evaluate
    metrics = ensemble.evaluate(X_test, y_test)

    # Get predictions
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    print("="*50)

    return ensemble, predictions, probabilities, metrics


def main():
    """Test the models module."""
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer

    # Load and prepare data
    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo("SPY", start_date="2015-01-01", end_date="2024-12-01")

    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    df = engineer.create_target_variable(df, horizon=5)
    df = df.dropna()

    feature_columns = engineer.get_feature_columns()

    # Train and evaluate
    ensemble, predictions, probabilities, metrics = train_and_evaluate(
        df, feature_columns, 'target'
    )

    # Save model
    ensemble.save("models")

    return ensemble, metrics


if __name__ == "__main__":
    ensemble, metrics = main()
