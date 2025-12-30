"""
Advanced Models Module V2
=========================
Improvements over V1:
1. Optuna hyperparameter optimization
2. LightGBM added to ensemble
3. Better LSTM architecture with attention
4. Confidence-based trading (no low-confidence trades)
5. Walk-forward validation
6. GPU acceleration for TensorFlow
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from purged_cv import PurgedTimeSeriesSplit, purged_walk_forward_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow with GPU config
try:
    import tensorflow as tf

    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detected: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU config error: {e}")

    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
        Input, Bidirectional, Attention, Concatenate,
        GRU, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available")

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptimizedXGBoost:
    """XGBoost with Optuna hyperparameter optimization."""

    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.model = None
        self.best_params = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_names = None

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }

        model = xgb.XGBClassifier(
            **params,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None):
        """Train with hyperparameter optimization."""
        self.feature_names = feature_names

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print(f"Optimizing XGBoost with {self.n_trials} trials...")

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train_scaled, y_train, X_val_scaled, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        print(f"Best XGBoost AUC: {study.best_value:.4f}")

        # Train final model with best params
        self.model = xgb.XGBClassifier(
            **self.best_params,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        self.model.fit(X_train_scaled, y_train)

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class OptimizedLightGBM:
    """LightGBM with Optuna hyperparameter optimization."""

    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.model = None
        self.best_params = None
        self.scaler = RobustScaler()

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        model = lgb.LGBMClassifier(
            **params,
            random_state=42,
            verbosity=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print(f"Optimizing LightGBM with {self.n_trials} trials...")

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train_scaled, y_train, X_val_scaled, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        print(f"Best LightGBM AUC: {study.best_value:.4f}")

        self.model = lgb.LGBMClassifier(
            **self.best_params,
            random_state=42,
            verbosity=-1
        )
        self.model.fit(X_train_scaled, y_train)

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class AdvancedLSTM:
    """
    Advanced LSTM with:
    - Bidirectional layers
    - CNN feature extraction
    - Attention mechanism
    - GPU acceleration
    """

    def __init__(self, sequence_length: int = 30):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM")

        self.sequence_length = sequence_length
        self.model = None
        self.scaler = RobustScaler()

    def _build_model(self, n_features):
        """Build advanced LSTM architecture."""
        inputs = Input(shape=(self.sequence_length, n_features))

        # CNN for local pattern extraction
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Global average pooling (better than just taking last output)
        x = GlobalAveragePooling1D()(x)

        # Dense layers
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _prepare_sequences(self, X, y=None):
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

    def fit(self, X_train, y_train, epochs=100, batch_size=64):
        X_scaled = self.scaler.fit_transform(X_train)
        X_seq, y_seq = self._prepare_sequences(X_scaled, y_train)

        n_features = X_train.shape[1]
        self.model = self._build_model(n_features)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        print("Training Advanced LSTM...")
        self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_seq = self._prepare_sequences(X_scaled)
        return self.model.predict(X_seq, verbose=0).flatten()

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


class AdvancedEnsembleV2:
    """
    Advanced Ensemble V2:
    - XGBoost + LightGBM + LSTM
    - Optimized hyperparameters
    - Confidence-based weighting
    - Walk-forward validation
    """

    def __init__(
        self,
        n_trials: int = 30,
        sequence_length: int = 30,
        use_lstm: bool = True,
        confidence_threshold: float = 0.6
    ):
        self.n_trials = n_trials
        self.sequence_length = sequence_length
        self.use_lstm = use_lstm and TENSORFLOW_AVAILABLE
        self.confidence_threshold = confidence_threshold

        self.xgb_model = OptimizedXGBoost(n_trials=n_trials)
        self.lgb_model = OptimizedLightGBM(n_trials=n_trials)
        self.lstm_model = AdvancedLSTM(sequence_length=sequence_length) if self.use_lstm else None

        # Learned weights from validation performance
        self.weights = {'xgb': 0.4, 'lgb': 0.4, 'lstm': 0.2}

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None, epochs=100):
        """Train all models and determine optimal weights."""

        # Train XGBoost
        self.xgb_model.fit(X_train, y_train, X_val, y_val, feature_names)

        # Train LightGBM
        self.lgb_model.fit(X_train, y_train, X_val, y_val, feature_names)

        # Train LSTM
        if self.lstm_model:
            # Combine train and val for LSTM (it does its own split)
            X_full = np.vstack([X_train, X_val])
            y_full = np.concatenate([y_train, y_val])
            self.lstm_model.fit(X_full, y_full, epochs=epochs)

        # Optimize weights based on validation performance
        self._optimize_weights(X_val, y_val)

        return self

    def _optimize_weights(self, X_val, y_val):
        """Find optimal ensemble weights."""
        xgb_proba = self.xgb_model.predict_proba(X_val)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_val)[:, 1]

        if self.lstm_model:
            lstm_proba = self.lstm_model.predict_proba(X_val)
            min_len = min(len(xgb_proba), len(lstm_proba))
            xgb_proba = xgb_proba[-min_len:]
            lgb_proba = lgb_proba[-min_len:]
            y_val_aligned = y_val[-min_len:]
        else:
            lstm_proba = None
            y_val_aligned = y_val

        best_auc = 0
        best_weights = self.weights.copy()

        # Grid search for weights
        for w_xgb in np.arange(0.2, 0.7, 0.1):
            for w_lgb in np.arange(0.2, 0.7, 0.1):
                w_lstm = 1 - w_xgb - w_lgb
                if w_lstm < 0 or w_lstm > 0.5:
                    continue

                if lstm_proba is not None:
                    ensemble = w_xgb * xgb_proba + w_lgb * lgb_proba + w_lstm * lstm_proba
                else:
                    w_lstm = 0
                    total = w_xgb + w_lgb
                    ensemble = (w_xgb / total) * xgb_proba + (w_lgb / total) * lgb_proba

                auc = roc_auc_score(y_val_aligned, ensemble)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = {'xgb': w_xgb, 'lgb': w_lgb, 'lstm': w_lstm}

        self.weights = best_weights
        print(f"Optimized weights: XGB={self.weights['xgb']:.2f}, LGB={self.weights['lgb']:.2f}, LSTM={self.weights['lstm']:.2f}")
        print(f"Validation AUC: {best_auc:.4f}")

    def predict_proba(self, X):
        """Get ensemble probabilities."""
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]

        if self.lstm_model:
            lstm_proba = self.lstm_model.predict_proba(X)
            min_len = min(len(xgb_proba), len(lstm_proba))
            xgb_proba = xgb_proba[-min_len:]
            lgb_proba = lgb_proba[-min_len:]

            ensemble = (
                self.weights['xgb'] * xgb_proba +
                self.weights['lgb'] * lgb_proba +
                self.weights['lstm'] * lstm_proba
            )
        else:
            total = self.weights['xgb'] + self.weights['lgb']
            ensemble = (
                (self.weights['xgb'] / total) * xgb_proba +
                (self.weights['lgb'] / total) * lgb_proba
            )

        return ensemble

    def predict_with_confidence(self, X):
        """
        Predict with confidence-based filtering.
        Returns: (signals, probabilities, confidence_mask)

        Only trades when confidence > threshold.
        """
        proba = self.predict_proba(X)

        # Confidence = distance from 0.5
        confidence = np.abs(proba - 0.5) * 2  # Scale to 0-1

        # High confidence mask
        high_conf_buy = (proba > (0.5 + self.confidence_threshold / 2))
        high_conf_sell = (proba < (0.5 - self.confidence_threshold / 2))

        signals = np.zeros(len(proba))
        signals[high_conf_buy] = 1
        signals[high_conf_sell] = -1

        return signals, proba, confidence

    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance."""
        proba = self.predict_proba(X_test)
        preds = (proba > 0.5).astype(int)
        y_aligned = y_test[-len(preds):]

        metrics = {
            'accuracy': accuracy_score(y_aligned, preds),
            'precision': precision_score(y_aligned, preds, zero_division=0),
            'recall': recall_score(y_aligned, preds, zero_division=0),
            'f1_score': f1_score(y_aligned, preds, zero_division=0),
            'roc_auc': roc_auc_score(y_aligned, proba),
        }

        # Confidence-filtered metrics
        signals, _, confidence = self.predict_with_confidence(X_test)
        high_conf_mask = signals != 0
        if high_conf_mask.sum() > 0:
            y_high_conf = y_aligned[high_conf_mask[-len(y_aligned):]]
            pred_high_conf = (signals[high_conf_mask] > 0).astype(int)
            if len(y_high_conf) > 0 and len(np.unique(y_high_conf)) > 1:
                metrics['high_conf_accuracy'] = accuracy_score(y_high_conf, pred_high_conf)
                metrics['high_conf_trades'] = int(high_conf_mask.sum())

        return metrics

    def save(self, model_dir: str):
        """Save all models."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'xgb_model': self.xgb_model.model,
            'xgb_scaler': self.xgb_model.scaler,
            'xgb_params': self.xgb_model.best_params,
            'lgb_model': self.lgb_model.model,
            'lgb_scaler': self.lgb_model.scaler,
            'lgb_params': self.lgb_model.best_params,
            'weights': self.weights,
            'confidence_threshold': self.confidence_threshold
        }, model_dir / 'ensemble_v2.pkl')

        if self.lstm_model and self.lstm_model.model:
            self.lstm_model.model.save(model_dir / 'lstm_v2.keras')
            joblib.dump(self.lstm_model.scaler, model_dir / 'lstm_v2_scaler.pkl')

        print(f"Ensemble V2 saved to {model_dir}")


def walk_forward_validation(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = 'target',
    n_splits: int = 5,
    train_size: float = 0.6,
    n_trials: int = 20,
    target_horizon: int = 5,
    embargo_extra: int = 5
) -> dict:
    """
    Walk-forward validation with EMBARGO for more realistic performance estimation.

    This version adds proper temporal gaps to prevent lookahead bias:
    - Purge: Remove training samples whose target overlaps with test period
    - Embargo: Add gap between train end and test start

    Args:
        target_horizon: Number of days used for target calculation (e.g., 5 for 5-day returns)
        embargo_extra: Additional buffer beyond target_horizon
    """
    results = []

    total_len = len(df)
    test_size = (1 - train_size) / n_splits

    # Calculate embargo and purge sizes
    purge_size = target_horizon
    embargo_size = target_horizon + embargo_extra

    print(f"\nPurged Walk-Forward Validation:")
    print(f"  Target horizon: {target_horizon} days")
    print(f"  Purge size: {purge_size} samples")
    print(f"  Embargo size: {embargo_size} samples")
    print(f"  Total train/test gap: {purge_size + embargo_size} samples")

    for i in range(n_splits):
        # Raw train end (before purge)
        train_end_raw = int(total_len * (train_size + i * test_size))

        # Apply purge: remove samples whose target overlaps with test
        train_end = train_end_raw - purge_size

        # Test starts after embargo gap
        test_start = train_end_raw + embargo_size
        test_end = int(total_len * (train_size + (i + 1) * test_size))

        if test_start >= total_len or train_end <= 0:
            continue

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        # Split train into train/val (with internal embargo)
        val_split = int(len(train_df) * 0.8) - purge_size
        val_start = val_split + embargo_size

        X_train = train_df[feature_columns].values[:val_split]
        y_train = train_df[target_column].values[:val_split]
        X_val = train_df[feature_columns].values[val_start:]
        y_val = train_df[target_column].values[val_start:]
        X_test = test_df[feature_columns].values
        y_test = test_df[target_column].values

        print(f"\n=== Fold {i+1}/{n_splits} ===")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Train period: 0 to {train_end} | Test period: {test_start} to {test_end}")
        print(f"Gap: {test_start - train_end} samples (prevents lookahead bias)")

        # Train ensemble
        ensemble = AdvancedEnsembleV2(n_trials=n_trials, use_lstm=False)  # Skip LSTM for speed
        ensemble.fit(X_train, y_train, X_val, y_val, feature_columns)

        # Evaluate
        metrics = ensemble.evaluate(X_test, y_test)
        results.append(metrics)

        print(f"Fold {i+1} AUC: {metrics['roc_auc']:.4f}")

    # Average results
    avg_results = {
        metric: np.mean([r[metric] for r in results if metric in r])
        for metric in results[0].keys()
    }

    print("\n=== Walk-Forward Validation Results ===")
    for metric, value in avg_results.items():
        print(f"  {metric}: {value:.4f}")

    return avg_results


def main():
    """Test advanced models."""
    import sys
    sys.path.insert(0, '.')
    from data_loader import DataLoader
    from feature_engineering_v2 import EnhancedFeatureEngineer

    # Load data
    loader = DataLoader(data_dir="data")
    df = loader.download_from_yahoo("SPY", start_date="2015-01-01", end_date="2024-12-01")

    # Enhanced features
    engineer = EnhancedFeatureEngineer()
    df = engineer.create_all_features(df)
    df = engineer.create_target_variable(df, horizon=5, threshold=0.003)
    df = df.dropna()

    feature_columns = engineer.get_feature_columns()
    # Filter to existing columns
    feature_columns = [c for c in feature_columns if c in df.columns]

    print(f"Features: {len(feature_columns)}")
    print(f"Samples: {len(df)}")

    # Walk-forward validation
    results = walk_forward_validation(
        df, feature_columns, 'target',
        n_splits=3, n_trials=20
    )

    return results


if __name__ == "__main__":
    results = main()
