import pandas as pd
from typing import Dict, List

def run_heuristic_baseline(
    features_df: pd.DataFrame,
    all_source1_ids: List[str],
    min_name_score: float = 0.85,
    min_addr_score: float = 0.70
) -> Dict[str, List[str]]:
    """
    Rule-based baseline matcher ensuring all Source 1 entities are preserved in output.
    """
    required_columns = {"source1_entity_id", "candidate_entity_id", "name_token_set", "addr_token_set"}
    missing = required_columns - set(features_df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    # Initialize all IDs with empty lists to preserve full S1 key set
    matched = {s1_id: [] for s1_id in all_source1_ids}

    if features_df.empty:
        return matched

    matches = features_df[
        (features_df["name_token_set"] >= min_name_score) &
        (features_df["addr_token_set"] >= min_addr_score)
    ]

    for s1_id, group in matches.groupby("source1_entity_id"):
        matched[s1_id] = sorted(group["candidate_entity_id"].unique().tolist())

    return matched