import os
import glob
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)


def load_tsv_or_csv(path):
    """
    Load TSV or CSV file automatically.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if path.lower().endswith(".tsv"):
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path)


def find_column(df, possible_names):
    """
    Find a column using several possible names.
    """

    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def load_prediction_details(path):
    """
    Load Member 2 prediction_details.tsv.
    """

    df = load_tsv_or_csv(path)

    print("\nPrediction file:")
    print(path)

    print("\nColumns:")
    print(list(df.columns))

    s1_col = find_column(
        df,
        [
            "source1_entity_id",
            "source_1_entity_id",
            "s1_id",
            "source1_id"
        ]
    )

    candidate_col = find_column(
        df,
        [
            "candidate_entity_id",
            "candidate_id",
            "entity_id"
        ]
    )

    probability_col = find_column(
        df,
        [
            "match_probability",
            "probability",
            "score",
            "match_score"
        ]
    )

    decision_col = find_column(
        df,
        [
            "decision",
            "prediction",
            "predicted_label",
            "match"
        ]
    )

    if s1_col is None:
        raise ValueError(
            "Could not find Source 1 ID column."
        )

    if candidate_col is None:
        raise ValueError(
            "Could not find candidate ID column."
        )

    if decision_col is None:
        raise ValueError(
            "Could not find prediction/decision column."
        )

    rename_map = {
        s1_col: "source1_entity_id",
        candidate_col: "candidate_entity_id",
        decision_col: "prediction"
    }

    if probability_col is not None:
        rename_map[probability_col] = "match_probability"

    df = df.rename(
        columns=rename_map
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce"
    ).fillna(0).astype(int)

    if "match_probability" in df.columns:
        df["match_probability"] = pd.to_numeric(
            df["match_probability"],
            errors="coerce"
        )

    return df


def find_ground_truth(data_directory):
    """
    Search for likely ground-truth files.

    This does NOT assume a specific filename.
    """

    patterns = [
        "**/*ground*truth*.csv",
        "**/*ground*truth*.tsv",
        "**/*label*.csv",
        "**/*label*.tsv",
        "**/*match*.csv",
        "**/*match*.tsv",
        "**/*truth*.csv",
        "**/*truth*.tsv"
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(
            glob.glob(
                os.path.join(
                    data_directory,
                    pattern
                ),
                recursive=True
            )
        )

    # Remove duplicates
    candidates = list(dict.fromkeys(candidates))

    return candidates


def load_ground_truth(path):
    """
    Load ground truth and normalize its columns.
    """

    df = load_tsv_or_csv(path)

    print("\nGround-truth file:")
    print(path)

    print("\nColumns:")
    print(list(df.columns))

    s1_col = find_column(
        df,
        [
            "source1_entity_id",
            "source_1_entity_id",
            "s1_id",
            "source1_id"
        ]
    )

    candidate_col = find_column(
        df,
        [
            "candidate_entity_id",
            "candidate_id",
            "entity_id"
        ]
    )

    label_col = find_column(
        df,
        [
            "true_label",
            "label",
            "ground_truth",
            "is_match",
            "match",
            "target"
        ]
    )

    if s1_col is None or candidate_col is None:
        raise ValueError(
            "Ground truth must contain Source 1 ID "
            "and candidate ID columns."
        )

    if label_col is None:
        raise ValueError(
            "Could not find ground-truth label column."
        )

    df = df.rename(
        columns={
            s1_col: "source1_entity_id",
            candidate_col: "candidate_entity_id",
            label_col: "true_label"
        }
    )

    df["true_label"] = pd.to_numeric(
        df["true_label"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["true_label"]
    )

    df["true_label"] = (
        df["true_label"]
        .astype(int)
    )

    return df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "true_label"
        ]
    ]


def calculate_metrics(df):
    """
    Calculate classification metrics.
    """

    y_true = df["true_label"]
    y_pred = df["prediction"]

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f0.5": fbeta_score(
            y_true,
            y_pred,
            beta=0.5,
            zero_division=0
        )
    }

    return metrics


def evaluate_predictions(
    prediction_path,
    ground_truth_path,
    output_directory
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    predictions = load_prediction_details(
        prediction_path
    )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    evaluation_df = predictions.merge(
        ground_truth,
        on=[
            "source1_entity_id",
            "candidate_entity_id"
        ],
        how="inner"
    )

    if evaluation_df.empty:
        raise ValueError(
            "No matching rows were found between "
            "predictions and ground truth."
        )

    print(
        f"\nNumber of evaluated candidate pairs: "
        f"{len(evaluation_df)}"
    )

    metrics = calculate_metrics(
        evaluation_df
    )

    print("\n========== MODEL EVALUATION ==========")

    for name, value in metrics.items():
        print(
            f"{name.upper():10s}: {value:.4f}"
        )

    # Save metrics
    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        os.path.join(
            output_directory,
            "evaluation_metrics.csv"
        ),
        index=False
    )

    # Confusion matrix
    cm = confusion_matrix(
        evaluation_df["true_label"],
        evaluation_df["prediction"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual_NonMatch",
            "Actual_Match"
        ],
        columns=[
            "Predicted_NonMatch",
            "Predicted_Match"
        ]
    )

    cm_df.to_csv(
        os.path.join(
            output_directory,
            "confusion_matrix.csv"
        )
    )

    # False positives
    false_positives = evaluation_df[
        (evaluation_df["true_label"] == 0)
        &
        (evaluation_df["prediction"] == 1)
    ]

    false_positives.to_csv(
        os.path.join(
            output_directory,
            "false_positives.csv"
        ),
        index=False
    )

    # False negatives
    false_negatives = evaluation_df[
        (evaluation_df["true_label"] == 1)
        &
        (evaluation_df["prediction"] == 0)
    ]

    false_negatives.to_csv(
        os.path.join(
            output_directory,
            "false_negatives.csv"
        ),
        index=False
    )

    # Save complete evaluation data
    evaluation_df.to_csv(
        os.path.join(
            output_directory,
            "evaluation_details.csv"
        ),
        index=False
    )

    print(
        "\nFalse positives:",
        len(false_positives)
    )

    print(
        "False negatives:",
        len(false_negatives)
    )

    print(
        "\nEvaluation results saved to:",
        output_directory
    )

    return evaluation_df, metrics