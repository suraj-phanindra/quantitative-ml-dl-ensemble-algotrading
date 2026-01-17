"""
Purged Cross-Validation Module
==============================
Implements proper temporal cross-validation for financial ML to prevent lookahead bias.

Based on:
- De Prado, M. L. (2018). Advances in Financial Machine Learning. Wiley.
- Chapter 7: Cross-Validation in Finance

Key concepts:
1. EMBARGO: Gap between train and test to prevent information leakage from
   autocorrelated samples
2. PURGING: Remove training samples whose label period overlaps with test samples

Why this matters for tree-based models:
- XGBoost/LightGBM/CatBoost can memorize patterns at train/test boundaries
- Standard TimeSeriesSplit gives overly optimistic results (CV Sharpe 3-8 vs actual 0.5-1)
- Adjacent financial samples are highly correlated (autocorrelation)
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional, List
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for time series with embargo.

    This addresses the key issues with standard TimeSeriesSplit:
    1. Adds embargo gap between train and test sets
    2. Purges training samples that have label overlap with test samples

    Parameters
    ----------
    n_splits : int
        Number of folds (default: 5)
    embargo_pct : float
        Percentage of total samples to use as embargo gap (default: 0.01 = 1%)
    purge_pct : float
        Percentage of samples to purge from training near test boundary (default: 0.0)
        Set this to your target horizon / total_samples if using forward-looking targets

    Example
    -------
    >>> cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
    >>> for train_idx, test_idx in cv.split(X):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with embargo and purging.

        Yields
        ------
        train_idx : ndarray
            Training set indices (with purging applied)
        test_idx : ndarray
            Test set indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate embargo and purge sizes
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        # Size of each test fold
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test indices for this fold
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]

            # Training indices: everything before test_start minus embargo and purge
            # Plus everything after test_end plus embargo

            # Before test set (with purge)
            train_before_end = max(0, test_start - embargo_size - purge_size)
            train_before = indices[:train_before_end]

            # After test set (with embargo)
            train_after_start = min(n_samples, test_end + embargo_size)
            train_after = indices[train_after_start:]

            train_idx = np.concatenate([train_before, train_after])

            if len(train_idx) == 0:
                continue

            yield train_idx, test_idx


class PurgedWalkForwardCV(BaseCrossValidator):
    """
    Walk-Forward Cross-Validation with Embargo (Expanding Window).

    More realistic for trading: train on all past data, test on future.
    Each fold uses more training data than the previous.

    Timeline:
    Fold 1: [====TRAIN====]--embargo--[TEST]
    Fold 2: [======TRAIN======]--embargo--[TEST]
    Fold 3: [========TRAIN========]--embargo--[TEST]

    Parameters
    ----------
    n_splits : int
        Number of test periods
    train_pct : float
        Minimum percentage of data for first training set (default: 0.5)
    embargo_pct : float
        Gap between train end and test start as percentage (default: 0.02)
    purge_pct : float
        Samples to remove from train end (default: 0.0, set to horizon/n_samples)
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_pct: float = 0.5,
        embargo_pct: float = 0.02,
        purge_pct: float = 0.0
    ):
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate expanding window train/test splits with embargo.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        # Reserve test data
        test_total = n_samples - int(n_samples * self.train_pct)
        test_size = test_total // self.n_splits

        for i in range(self.n_splits):
            # Training ends here (expanding)
            train_end_raw = int(n_samples * self.train_pct) + i * test_size

            # Apply purge (remove samples too close to test)
            train_end = train_end_raw - purge_size

            # Test starts after embargo
            test_start = train_end_raw + embargo_size
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples or train_end <= 0:
                continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation (CPCV).

    From De Prado: generates multiple train/test combinations while
    maintaining temporal order and applying embargo/purge.

    More robust than standard k-fold for small datasets.

    Parameters
    ----------
    n_splits : int
        Number of groups to divide data into
    n_test_groups : int
        Number of groups to use for testing in each fold
    embargo_pct : float
        Embargo as percentage of data
    purge_pct : float
        Purge as percentage of data
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0
    ):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_groups)

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate CPCV splits."""
        from itertools import combinations

        n_samples = len(X)
        indices = np.arange(n_samples)

        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        # Divide into groups
        group_size = n_samples // self.n_splits
        group_indices = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            group_indices.append(indices[start:end])

        # Generate all combinations of test groups
        for test_groups in combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([group_indices[g] for g in test_groups])

            # Training groups (non-test)
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]

            # Apply embargo and purge
            train_idx_list = []
            for g in train_groups:
                group_idx = group_indices[g]

                # Check if this group is adjacent to any test group
                is_before_test = any(g + 1 == tg for tg in test_groups)
                is_after_test = any(g - 1 == tg for tg in test_groups)

                if is_before_test:
                    # Remove end of this group (purge + embargo)
                    cutoff = max(0, len(group_idx) - purge_size - embargo_size)
                    group_idx = group_idx[:cutoff]
                elif is_after_test:
                    # Remove start of this group (embargo)
                    group_idx = group_idx[embargo_size:]

                if len(group_idx) > 0:
                    train_idx_list.append(group_idx)

            if len(train_idx_list) > 0:
                train_idx = np.concatenate(train_idx_list)
                yield train_idx, test_idx


def calculate_embargo_size(
    n_samples: int,
    target_horizon: int,
    feature_lookback: int = 0
) -> int:
    """
    Calculate appropriate embargo size based on target and features.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    target_horizon : int
        Number of periods used for target (e.g., 5 for 5-day returns)
    feature_lookback : int
        Maximum lookback of any feature (e.g., 50 for SMA_50)

    Returns
    -------
    int
        Recommended embargo size in samples
    """
    # Embargo should be at least target_horizon to prevent label leakage
    # Plus some buffer for feature autocorrelation
    embargo = target_horizon + max(feature_lookback // 10, 5)
    return embargo


def calculate_purge_size(target_horizon: int) -> int:
    """
    Calculate purge size based on target horizon.

    Training samples whose labels overlap with test period should be removed.

    Parameters
    ----------
    target_horizon : int
        Number of periods used for target calculation

    Returns
    -------
    int
        Number of samples to purge from end of training set
    """
    return target_horizon


class PurgedTimeSeriesSplit:
    """
    Drop-in replacement for sklearn's TimeSeriesSplit with embargo/purge.

    Use this as a direct replacement in existing code:

    Before:
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)

    After:
        from purged_cv import PurgedTimeSeriesSplit
        tscv = PurgedTimeSeriesSplit(n_splits=5, embargo_td=5, purge_td=5)

    Parameters
    ----------
    n_splits : int
        Number of splits
    embargo_td : int
        Embargo period in number of samples (e.g., 5 for 5-day target)
    purge_td : int
        Purge period in number of samples (typically = target horizon)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_td: int = 5,
        purge_td: int = 5
    ):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
        self.purge_td = purge_td

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits with embargo and purge.

        Timeline for each fold:
        [...TRAIN...][PURGE][EMBARGO][...TEST...]
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        indices = np.arange(n_samples)

        # Minimum training size
        min_train = n_samples // (self.n_splits + 1)
        test_size = (n_samples - min_train) // self.n_splits

        for i in range(self.n_splits):
            # Test indices
            test_start = min_train + i * test_size
            test_end = test_start + test_size
            if i == self.n_splits - 1:
                test_end = n_samples

            test_idx = indices[test_start:test_end]

            # Train indices: everything before test_start minus purge and embargo
            train_end = max(0, test_start - self.embargo_td - self.purge_td)
            train_idx = indices[:train_end]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def purged_walk_forward_validation(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = 'target',
    n_splits: int = 5,
    train_pct: float = 0.6,
    target_horizon: int = 5,
    embargo_extra: int = 5,
    model_fn: callable = None,
    verbose: bool = True
) -> dict:
    """
    Proper walk-forward validation with embargo and purge.

    This is the recommended validation approach for trading models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target
    feature_columns : list
        List of feature column names
    target_column : str
        Name of target column
    n_splits : int
        Number of test folds
    train_pct : float
        Minimum training data percentage
    target_horizon : int
        Number of periods used for target (for purge calculation)
    embargo_extra : int
        Additional embargo beyond target horizon
    model_fn : callable
        Function that returns a model instance. If None, uses XGBoost.
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Dictionary with metrics averaged across folds
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    results = []
    n_samples = len(df)

    # Calculate embargo and purge sizes
    purge_size = target_horizon
    embargo_size = target_horizon + embargo_extra

    if verbose:
        print(f"Purged Walk-Forward CV Settings:")
        print(f"  - Total samples: {n_samples}")
        print(f"  - Target horizon: {target_horizon} days")
        print(f"  - Purge size: {purge_size} samples")
        print(f"  - Embargo size: {embargo_size} samples")
        print(f"  - Train/test gap: {purge_size + embargo_size} samples")

    # Calculate test periods
    test_total = n_samples - int(n_samples * train_pct)
    test_size = test_total // n_splits

    for i in range(n_splits):
        # Training ends here
        train_end_raw = int(n_samples * train_pct) + i * test_size

        # Apply purge (remove samples whose target overlaps with test)
        train_end = train_end_raw - purge_size

        # Test starts after embargo gap
        test_start = train_end_raw + embargo_size
        test_end = min(test_start + test_size, n_samples)

        if test_start >= n_samples or train_end <= 0:
            continue

        # Get data splits
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        # Further split train into train/val (with internal embargo)
        val_split = int(len(train_df) * 0.8) - purge_size

        X_train = train_df[feature_columns].values[:val_split]
        y_train = train_df[target_column].values[:val_split]
        X_val = train_df[feature_columns].values[val_split + embargo_size:]
        y_val = train_df[target_column].values[val_split + embargo_size:]
        X_test = test_df[feature_columns].values
        y_test = test_df[target_column].values

        if verbose:
            print(f"\n=== Fold {i+1}/{n_splits} ===")
            print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
            print(f"Train period: 0 to {train_end}")
            print(f"Test period: {test_start} to {test_end}")
            print(f"Gap (purge+embargo): {test_start - train_end} samples")

        # Train model
        if model_fn is None:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            model = model_fn()

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        fold_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
        }
        results.append(fold_metrics)

        if verbose:
            print(f"Fold {i+1} - Accuracy: {fold_metrics['accuracy']:.4f}, AUC: {fold_metrics['roc_auc']:.4f}")

    # Average results
    avg_results = {
        metric: np.mean([r[metric] for r in results])
        for metric in results[0].keys()
    }
    avg_results['std_auc'] = np.std([r['roc_auc'] for r in results])

    if verbose:
        print("\n" + "="*50)
        print("PURGED WALK-FORWARD RESULTS (Realistic Estimates)")
        print("="*50)
        for metric, value in avg_results.items():
            print(f"  {metric}: {value:.4f}")

    return avg_results


# Convenience function for quick comparison
def compare_cv_methods(
    X: np.ndarray,
    y: np.ndarray,
    target_horizon: int = 5,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Compare standard vs purged CV to show the performance gap.

    This demonstrates why standard TimeSeriesSplit gives inflated results.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    results = []

    # Standard TimeSeriesSplit (biased)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    standard_aucs = []
    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBClassifier(n_estimators=100, verbosity=0, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1]
        standard_aucs.append(roc_auc_score(y[test_idx], y_proba))

    results.append({
        'method': 'Standard TimeSeriesSplit',
        'mean_auc': np.mean(standard_aucs),
        'std_auc': np.std(standard_aucs),
        'bias': 'HIGH (no embargo)'
    })

    # Purged TimeSeriesSplit (unbiased)
    ptscv = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        embargo_td=target_horizon,
        purge_td=target_horizon
    )
    purged_aucs = []
    for train_idx, test_idx in ptscv.split(X):
        model = xgb.XGBClassifier(n_estimators=100, verbosity=0, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1]
        purged_aucs.append(roc_auc_score(y[test_idx], y_proba))

    results.append({
        'method': 'Purged TimeSeriesSplit',
        'mean_auc': np.mean(purged_aucs),
        'std_auc': np.std(purged_aucs),
        'bias': 'LOW (with embargo)'
    })

    df = pd.DataFrame(results)
    print("\nCV Method Comparison:")
    print(df.to_string(index=False))
    print(f"\nPerformance inflation from standard CV: {(np.mean(standard_aucs) - np.mean(purged_aucs)) / np.mean(purged_aucs) * 100:.1f}%")

    return df
