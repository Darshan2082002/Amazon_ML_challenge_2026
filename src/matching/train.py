import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple
from sklearn.metrics import fbeta_score

def calculate_binary_f05(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate binary F0.5 score on pair predictions."""
    return float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))

def train_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray
) -> Tuple[lgb.Booster, float]:
    """Trains LightGBM classifier with early stopping and tunes threshold for F0.5 optimization on validation set."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": 42
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False)
        ]
    )

    val_probs = model.predict(X_val)

    best_thresh = 0.50
    best_f05 = -1.0

    # Search wider range with finer steps to optimize F0.5 Precision-heavy trade-off
    for thresh in np.arange(0.10, 0.96, 0.01):
        preds = (val_probs >= thresh).astype(int)
        score = calculate_binary_f05(y_val, preds)

        if score > best_f05:
            best_f05 = score
            best_thresh = float(thresh)

    print(f"Optimal threshold: {best_thresh:.2f} | Validation F0.5: {best_f05:.4f}")

    return model, best_thresh