import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> dict:
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f05 = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))

    return {
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
    }


def compute_macro_f05(pred_df: pd.DataFrame, beta: float = 0.5) -> dict:
    """Ultra-fast vectorized macro-F0.5 computation per entity group."""
    print("   Computing overall micro metrics...")
    overall_metrics = evaluate_predictions(
        pred_df["true_label"].values, pred_df["predicted_match"].values, beta=beta
    )

    print("   Computing fast vectorized macro metrics across groups...")
    y_true = pred_df["true_label"].values
    y_pred = pred_df["predicted_match"].values

    # Calculate True Positives, False Positives, False Negatives vectorially per row
    pred_df["tp"] = (y_true == 1) & (y_pred == 1)
    pred_df["fp"] = (y_true == 0) & (y_pred == 1)
    pred_df["fn"] = (y_true == 1) & (y_pred == 0)

    # Group-level aggregation
    grouped = pred_df.groupby("source1_entity_id")[["tp", "fp", "fn"]].sum()

    tp = grouped["tp"].values
    fp = grouped["fp"].values
    fn = grouped["fn"].values

    # Vectorized Precision & Recall per group with zero-division handling
    denom_p = tp + fp
    denom_r = tp + fn

    p_group = np.where(denom_p > 0, tp / denom_p, 0.0)
    r_group = np.where(denom_r > 0, tp / denom_r, 0.0)

    # Vectorized F0.5 per group
    beta_sq = beta ** 2
    denom_f = (beta_sq * p_group) + r_group
    f_group = np.where(
        denom_f > 0,
        (1 + beta_sq) * (p_group * r_group) / denom_f,
        0.0
    )

    return {
        "micro_precision": overall_metrics["precision"],
        "micro_recall": overall_metrics["recall"],
        "micro_f0.5": overall_metrics["f0.5"],
        "macro_precision": float(np.mean(p_group)),
        "macro_recall": float(np.mean(r_group)),
        "macro_f0.5": float(np.mean(f_group)),
    }