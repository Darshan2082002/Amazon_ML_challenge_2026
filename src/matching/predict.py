import os
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple

ID_COLUMNS = ["source1_entity_id", "candidate_entity_id"]

def predict_matches(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    threshold: float,
    all_source1_ids: List[str]
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Generates final predictions using the trained booster and aligns features strictly.
    Returns:
        - Dictionary of matches per S1 ID
        - DataFrame containing prediction details (probabilities & decisions)
    """
    matches = {s1_id: [] for s1_id in all_source1_ids}

    if features_df.empty:
        empty_details = pd.DataFrame(columns=ID_COLUMNS + ["match_probability", "decision"])
        return matches, empty_details

    # Align feature columns with model training parameters
    feature_cols = model.feature_name()
    X = features_df[feature_cols]

    probabilities = model.predict(X)

    prediction_df = features_df[ID_COLUMNS].copy()
    prediction_df["match_probability"] = probabilities
    prediction_df["decision"] = (prediction_df["match_probability"] >= threshold).astype(int)

    matched_rows = prediction_df[prediction_df["decision"] == 1]

    for s1_id, group in matched_rows.groupby("source1_entity_id"):
        matches[s1_id] = sorted(group["candidate_entity_id"].unique().tolist())

    return matches, prediction_df

def save_prediction_details(prediction_df: pd.DataFrame, output_path: str):
    """Saves detailed inference metadata for downstream error analysis by Member 3."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    prediction_df.to_csv(output_path, sep="\t", index=False)
    print(f"Prediction details saved to: {output_path}")