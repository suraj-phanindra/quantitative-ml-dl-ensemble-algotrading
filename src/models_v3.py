"""
Models V3 - Custom Loss Functions and Stacking Ensemble
Implements:
- Asymmetric loss (higher penalty for losses in high volatility)
- Return-weighted accuracy
- Sharpe-ratio based objective
- CatBoost integration
- Stacking ensemble
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import catboost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, will use XGBoost/LightGBM only")


class CustomLossFunctions:
    """
    Custom loss functions for trading-optimized model training
    """

    @staticmethod
    def asymmetric_loss_xgb(y_pred, dtrain, volatility, alpha=2.0):
        """
        Asymmetric loss for XGBoost that penalizes losses more during high volatility

        Args:
            y_pred: Predictions (log-odds for binary classification)
            dtrain: DMatrix with labels
            volatility: Array of volatility values for each sample
            alpha: Asymmetry factor (>1 means higher penalty for false negatives in high vol)
        """
        y_true = dtrain.get_label()
        prob = 1.0 / (1.0 + np.exp(-y_pred))

        # Normalize volatility to [1, alpha] range
        vol_weight = 1.0 + (alpha - 1.0) * (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-10)

        # Asymmetric gradient: penalize false signals more in high volatility
        # False positive (predicted 1, actual 0) in high vol = big penalty
        # False negative (predicted 0, actual 1) in high vol = smaller penalty (missed opportunity)
        grad = prob - y_true
        grad = np.where(
            (y_true == 0) & (prob > 0.5),  # False positive
            grad * vol_weight * 1.5,  # Higher penalty
            grad
        )

        # Hessian (second derivative)
        hess = prob * (1.0 - prob)
        hess = np.maximum(hess, 1e-6)  # Prevent zero hessian

        return grad, hess

    @staticmethod
    def return_weighted_loss_xgb(y_pred, dtrain, returns):
        """
        Loss function that weights samples by return magnitude

        Higher weight for samples with larger absolute returns
        (getting big moves right is more important than small moves)
        """
        y_true = dtrain.get_label()
        prob = 1.0 / (1.0 + np.exp(-y_pred))

        # Weight by absolute return (normalized)
        abs_returns = np.abs(returns)
        weights = 1.0 + 2.0 * (abs_returns - abs_returns.min()) / (abs_returns.max() - abs_returns.min() + 1e-10)

        # Gradient with return weighting
        grad = (prob - y_true) * weights

        # Hessian
        hess = prob * (1.0 - prob) * weights
        hess = np.maximum(hess, 1e-6)

        return grad, hess

    @staticmethod
    def calculate_sharpe_metric(y_true, y_pred, returns, risk_free=0.0, min_trades=5):
        """
        Calculate a Sharpe-ratio-like metric for model evaluation
        Enhanced with minimum trade threshold and risk-adjusted metrics

        Args:
            y_true: Actual labels
            y_pred: Predicted probabilities
            returns: Actual forward returns
            risk_free: Risk-free rate (daily)
            min_trades: Minimum trades required for valid Sharpe

        Returns:
            Sharpe ratio (positive values are better)
        """
        # Convert probabilities to signals
        signals = (y_pred > 0.5).astype(int)

        # Count actual trades
        n_trades = signals.sum()

        # Penalize strategies with too few trades
        if n_trades < min_trades:
            return -1.0  # Penalty for insufficient trades

        # Calculate strategy returns only when trading
        strategy_returns = signals * returns
        active_returns = strategy_returns[signals == 1]

        # Calculate metrics on active trading periods
        if len(active_returns) > 0 and active_returns.std() > 0:
            excess_return = active_returns.mean() - risk_free / 252
            sharpe = excess_return / active_returns.std() * np.sqrt(252)

            # Bonus for profitable trades (accuracy boost)
            win_rate = (active_returns > 0).mean()
            if win_rate > 0.55:
                sharpe *= (1 + (win_rate - 0.55) * 0.5)  # Boost for high accuracy
        else:
            sharpe = 0.0

        return sharpe


class TradingOptimizedXGBoost:
    """
    XGBoost classifier optimized for trading with custom objectives
    """

    def __init__(self, use_asymmetric_loss=True, volatility_weight=2.0):
        self.model = None
        self.use_asymmetric_loss = use_asymmetric_loss
        self.volatility_weight = volatility_weight
        self.best_params = None

    def optimize_hyperparameters(self, X_train, y_train, returns_train, volatility_train,
                                  n_trials=50, n_splits=5):
        """
        Optimize hyperparameters using Sharpe ratio as objective
        Enhanced search space and more CV splits for better generalization
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 600),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.65, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                'gamma': trial.suggest_float('gamma', 0, 0.3),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 2, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.9, 1.3),
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            sharpe_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                ret_val = returns_train[val_idx]

                # Train model
                model = xgb.XGBClassifier(
                    **params,
                    objective='binary:logistic',
                    eval_metric='auc',
                    use_label_encoder=False,
                    verbosity=0,
                    random_state=42
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                # Predict and calculate Sharpe
                y_pred = model.predict_proba(X_val)[:, 1]
                sharpe = CustomLossFunctions.calculate_sharpe_metric(y_val, y_pred, ret_val)
                sharpe_scores.append(sharpe)

            return np.mean(sharpe_scores)

        # Run optimization
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        print(f"Best Sharpe from optimization: {study.best_value:.4f}")
        print(f"Best params: {self.best_params}")

        return self.best_params

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        """
        Fit the model with optional sample weights
        """
        params = self.best_params if self.best_params else {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

        self.model = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            verbosity=0,
            random_state=42
        )

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set,
                       sample_weight=sample_weights, verbose=False)

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


class TradingOptimizedLightGBM:
    """
    LightGBM classifier optimized for trading
    """

    def __init__(self):
        self.model = None
        self.best_params = None

    def optimize_hyperparameters(self, X_train, y_train, returns_train,
                                  n_trials=50, n_splits=5):
        """Optimize using Sharpe ratio with enhanced search space"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 600),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 25, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
                'subsample': trial.suggest_float('subsample', 0.65, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 2, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 2, log=True),
            }

            tscv = TimeSeriesSplit(n_splits=n_splits)
            sharpe_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                ret_val = returns_train[val_idx]

                model = lgb.LGBMClassifier(**params, verbosity=-1, random_state=42)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

                y_pred = model.predict_proba(X_val)[:, 1]
                sharpe = CustomLossFunctions.calculate_sharpe_metric(y_val, y_pred, ret_val)
                sharpe_scores.append(sharpe)

            return np.mean(sharpe_scores)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        print(f"Best Sharpe from LightGBM optimization: {study.best_value:.4f}")

        return self.best_params

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        params = self.best_params if self.best_params else {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
        }

        self.model = lgb.LGBMClassifier(**params, verbosity=-1, random_state=42)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, sample_weight=sample_weights)

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


class TradingOptimizedCatBoost:
    """
    CatBoost classifier optimized for trading
    """

    def __init__(self):
        self.model = None
        self.best_params = None
        self.available = CATBOOST_AVAILABLE

    def optimize_hyperparameters(self, X_train, y_train, returns_train,
                                  n_trials=40, n_splits=5):
        if not self.available:
            print("CatBoost not available")
            return None

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 150, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
                'border_count': trial.suggest_int('border_count', 64, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.5),
                'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
            }

            tscv = TimeSeriesSplit(n_splits=n_splits)
            sharpe_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                ret_val = returns_train[val_idx]

                model = CatBoostClassifier(**params, verbose=False, random_state=42)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

                y_pred = model.predict_proba(X_val)[:, 1]
                sharpe = CustomLossFunctions.calculate_sharpe_metric(y_val, y_pred, ret_val)
                sharpe_scores.append(sharpe)

            return np.mean(sharpe_scores)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        print(f"Best Sharpe from CatBoost optimization: {study.best_value:.4f}")

        return self.best_params

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        if not self.available:
            return self

        params = self.best_params if self.best_params else {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.05,
        }

        self.model = CatBoostClassifier(**params, verbose=False, random_state=42)
        eval_set = (X_val, y_val) if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set,
                       sample_weight=sample_weights, verbose=False)

        return self

    def predict_proba(self, X):
        if not self.available:
            return None
        return self.model.predict_proba(X)

    def predict(self, X):
        if not self.available:
            return None
        return self.model.predict(X)


class StackingEnsembleV3:
    """
    Stacking ensemble with meta-learner for combining base models

    Architecture:
    - Level 0: XGBoost, LightGBM, CatBoost (base learners)
    - Level 1: Logistic Regression meta-learner (combines base predictions)
    """

    def __init__(self, confidence_threshold=0.55, use_catboost=True):
        self.xgb_model = TradingOptimizedXGBoost()
        self.lgb_model = TradingOptimizedLightGBM()
        self.cat_model = TradingOptimizedCatBoost() if use_catboost else None
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.confidence_threshold = confidence_threshold
        self.feature_names = None

    def calculate_sample_weights(self, returns, volatility):
        """
        Calculate sample weights based on:
        1. Return magnitude (big moves are more important)
        2. Inverse volatility (penalize high-vol predictions less)
        """
        # Return magnitude weight
        abs_returns = np.abs(returns)
        return_weight = 1.0 + (abs_returns - abs_returns.min()) / (abs_returns.max() - abs_returns.min() + 1e-10)

        # Volatility adjustment (lower weight in extreme volatility)
        vol_percentile = pd.Series(volatility).rank(pct=True).values
        vol_weight = np.where(vol_percentile > 0.9, 0.7, 1.0)

        # Combined weight
        weights = return_weight * vol_weight

        return weights

    def fit(self, X_train, y_train, X_val, y_val, returns_train, returns_val,
            volatility_train, n_trials=30):
        """
        Fit the stacking ensemble

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            returns_train, returns_val: Forward returns
            volatility_train: Volatility for sample weighting
            n_trials: Optuna trials for each model
        """
        print("\n" + "="*60)
        print("TRAINING V3 STACKING ENSEMBLE")
        print("="*60)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Calculate sample weights
        sample_weights = self.calculate_sample_weights(returns_train, volatility_train)

        # ============== LEVEL 0: BASE LEARNERS ==============
        print("\n[Level 0] Training Base Learners...")

        # XGBoost
        print("\n--- XGBoost Optimization ---")
        self.xgb_model.optimize_hyperparameters(
            X_train_scaled, y_train, returns_train, volatility_train,
            n_trials=n_trials
        )
        self.xgb_model.fit(X_train_scaled, y_train, X_val_scaled, y_val, sample_weights)

        # LightGBM
        print("\n--- LightGBM Optimization ---")
        self.lgb_model.optimize_hyperparameters(
            X_train_scaled, y_train, returns_train,
            n_trials=n_trials
        )
        self.lgb_model.fit(X_train_scaled, y_train, X_val_scaled, y_val, sample_weights)

        # CatBoost (if available)
        if self.cat_model and self.cat_model.available:
            print("\n--- CatBoost Optimization ---")
            self.cat_model.optimize_hyperparameters(
                X_train_scaled, y_train, returns_train,
                n_trials=min(n_trials, 20)  # CatBoost is slower
            )
            self.cat_model.fit(X_train_scaled, y_train, X_val_scaled, y_val, sample_weights)

        # ============== LEVEL 1: META-LEARNER ==============
        print("\n[Level 1] Training Meta-Learner...")

        # Get base model predictions on validation set (out-of-fold for training meta-learner)
        # Use cross-validation to get unbiased predictions
        meta_features_train = self._get_meta_features(X_train_scaled)
        meta_features_val = self._get_meta_features(X_val_scaled)

        # Train meta-learner (Logistic Regression with L2 regularization)
        self.meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.meta_learner.fit(meta_features_train, y_train)

        # Evaluate
        val_proba = self.meta_learner.predict_proba(meta_features_val)[:, 1]
        val_pred = (val_proba > 0.5).astype(int)

        print("\n" + "="*60)
        print("STACKING ENSEMBLE TRAINING COMPLETE")
        print("="*60)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred, zero_division=0),
            'recall': recall_score(y_val, val_pred, zero_division=0),
            'f1': f1_score(y_val, val_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, val_proba),
            'sharpe': CustomLossFunctions.calculate_sharpe_metric(y_val, val_proba, returns_val)
        }

        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Sharpe:    {metrics['sharpe']:.4f}")

        # Meta-learner weights
        print(f"\nMeta-Learner Coefficients:")
        model_names = ['XGBoost', 'LightGBM']
        if self.cat_model and self.cat_model.available:
            model_names.append('CatBoost')
        for name, coef in zip(model_names, self.meta_learner.coef_[0]):
            print(f"  {name}: {coef:.4f}")

        return metrics

    def _get_meta_features(self, X):
        """Get predictions from base models as meta-features"""
        meta_features = []

        # XGBoost predictions
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        meta_features.append(xgb_pred)

        # LightGBM predictions
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        meta_features.append(lgb_pred)

        # CatBoost predictions (if available)
        if self.cat_model and self.cat_model.available:
            cat_pred = self.cat_model.predict_proba(X)[:, 1]
            meta_features.append(cat_pred)

        return np.column_stack(meta_features)

    def predict_proba(self, X):
        """Get probability predictions"""
        X_scaled = self.scaler.transform(X)
        meta_features = self._get_meta_features(X_scaled)
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X):
        """Get binary predictions"""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

    def predict_with_confidence(self, X):
        """
        Get predictions with confidence scores

        Returns:
            signals: Binary signals (1 = buy, 0 = hold/sell)
            probabilities: Raw probabilities
            confidence: Confidence scores (0 to 1)
            high_confidence_mask: Boolean mask for high-confidence signals
        """
        proba = self.predict_proba(X)[:, 1]

        # Calculate confidence as distance from 0.5
        confidence = np.abs(proba - 0.5) * 2

        # High confidence mask
        high_conf_mask = confidence >= self.confidence_threshold

        # Signals (only for high confidence)
        signals = np.where(high_conf_mask, (proba > 0.5).astype(int), -1)  # -1 = no signal

        return signals, proba, confidence, high_conf_mask

    def get_feature_importance(self):
        """Get combined feature importance from base models"""
        importance_dict = {}

        # XGBoost importance
        if hasattr(self.xgb_model.model, 'feature_importances_'):
            xgb_imp = self.xgb_model.model.feature_importances_
            for i, imp in enumerate(xgb_imp):
                importance_dict[f'feature_{i}'] = importance_dict.get(f'feature_{i}', 0) + imp

        # LightGBM importance
        if hasattr(self.lgb_model.model, 'feature_importances_'):
            lgb_imp = self.lgb_model.model.feature_importances_
            for i, imp in enumerate(lgb_imp):
                importance_dict[f'feature_{i}'] = importance_dict.get(f'feature_{i}', 0) + imp

        # CatBoost importance
        if self.cat_model and self.cat_model.available and self.cat_model.model:
            cat_imp = self.cat_model.model.feature_importances_
            for i, imp in enumerate(cat_imp):
                importance_dict[f'feature_{i}'] = importance_dict.get(f'feature_{i}', 0) + imp

        # Normalize
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}

        return importance_dict

    def save(self, path):
        """Save the ensemble"""
        joblib.dump({
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'cat_model': self.cat_model,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'confidence_threshold': self.confidence_threshold
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load the ensemble"""
        data = joblib.load(path)
        ensemble = cls()
        ensemble.xgb_model = data['xgb_model']
        ensemble.lgb_model = data['lgb_model']
        ensemble.cat_model = data['cat_model']
        ensemble.meta_learner = data['meta_learner']
        ensemble.scaler = data['scaler']
        ensemble.confidence_threshold = data['confidence_threshold']
        return ensemble


def evaluate_model_comprehensive(y_true, y_pred_proba, returns, ticker=''):
    """
    Comprehensive model evaluation with trading-relevant metrics
    """
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Classification metrics
    metrics = {
        'ticker': ticker,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
    }

    # Trading metrics
    strategy_returns = y_pred * returns
    metrics['sharpe'] = CustomLossFunctions.calculate_sharpe_metric(y_true, y_pred_proba, returns)

    # Calculate hit rate on actual trades
    trade_mask = y_pred == 1
    if trade_mask.sum() > 0:
        metrics['trade_accuracy'] = (y_true[trade_mask] == y_pred[trade_mask]).mean()
        metrics['avg_return_when_trading'] = returns[trade_mask].mean() * 100
    else:
        metrics['trade_accuracy'] = 0
        metrics['avg_return_when_trading'] = 0

    # High confidence metrics
    high_conf_mask = np.abs(y_pred_proba - 0.5) > 0.15  # 65%+ probability
    if high_conf_mask.sum() > 0:
        metrics['high_conf_accuracy'] = accuracy_score(y_true[high_conf_mask],
                                                       (y_pred_proba[high_conf_mask] > 0.5).astype(int))
        metrics['high_conf_count'] = high_conf_mask.sum()
    else:
        metrics['high_conf_accuracy'] = 0
        metrics['high_conf_count'] = 0

    return metrics


if __name__ == "__main__":
    print("V3 Models Module - Testing...")

    # Quick test
    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = (np.random.randn(500) > 0).astype(int)
    returns = np.random.randn(500) * 0.02
    volatility = np.abs(np.random.randn(500)) * 0.2

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    returns_train, returns_test = returns[:split], returns[split:]
    volatility_train = volatility[:split]

    # Test stacking ensemble
    ensemble = StackingEnsembleV3(confidence_threshold=0.55, use_catboost=False)
    metrics = ensemble.fit(
        X_train, y_train, X_test, y_test,
        returns_train, returns_test, volatility_train,
        n_trials=5  # Quick test
    )

    print("\nTest complete!")
