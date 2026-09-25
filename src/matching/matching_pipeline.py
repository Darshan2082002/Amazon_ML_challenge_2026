import os
import pandas as pd
from typing import Dict, List
import lightgbm as lgb

from src.matching.features import build_pair_features
from src.matching.predict import predict_matches, save_prediction_details

def export_matching_results(matches: Dict[str, List[str]], output_path: str):
    """Exports final matching_results.tsv according to contest specification."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lines = ["source1_entity_id\tmatched_entity_ids"]

    for s1_id in sorted(matches.keys()):
        cand_ids = matches[s1_id]
        cand_str = ",".join(sorted(cand_ids))
        lines.append(f"{s1_id}\t{cand_str}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Final matching output saved to: {output_path}")

def execute_matching_stage(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    candidate_pairs: Dict[str, List[str]],
    model: lgb.Booster,
    threshold: float,
    output_matching_file: str = "output/matching_results.tsv",
    output_details_file: str = "output/prediction_details.tsv"
):
    """
    Main orchestration step linking Member 1 outputs through Member 2 processing.
    """
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    s23_dict = s23_df.set_index("entity_id").to_dict("index")
    all_s1_ids = list(s1_df["entity_id"].unique())

    print("Building pair features...")
    features_df = build_pair_features(s1_dict, s23_dict, candidate_pairs)

    print("Running inference and collecting prediction probabilities...")
    matches, prediction_df = predict_matches(model, features_df, threshold, all_s1_ids)

    print("Saving outputs...")
    export_matching_results(matches, output_matching_file)
    save_prediction_details(prediction_df, output_details_file)