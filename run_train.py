import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import extract_pair_features

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


def detect_id_column(df: pd.DataFrame) -> str:
    candidates = ["entity_id", "source1_entity_id", "source23_entity_id", "s1_id", "s23_id", "id"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def main():
    print("============================================")
    print("     MEMBER 2 - TRAINING & FEATURE PIPELINE ")
    print("============================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s1_path = DATA_DIR / "source1_clean.csv"
    s23_path = DATA_DIR / "source23_clean.csv"
    pairs_path = DATA_DIR / "candidate_pairs.json"

    if not s1_path.exists() or not s23_path.exists() or not pairs_path.exists():
        print(f"[ERROR] Clean data or candidate pairs missing from {DATA_DIR}!")
        sys.exit(1)

    print("Loading cleaned datasets...")
    s1_df = pd.read_csv(s1_path, dtype=str).fillna("")
    s23_df = pd.read_csv(s23_path, dtype=str).fillna("")

    with open(pairs_path, "r", encoding="utf-8") as f:
        candidate_pairs = json.load(f)

    s1_id_col = detect_id_column(s1_df)
    s23_id_col = detect_id_column(s23_df)

    s1_df[s1_id_col] = s1_df[s1_id_col].astype(str).str.strip()
    s23_df[s23_id_col] = s23_df[s23_id_col].astype(str).str.strip()

    s1_dict = s1_df.set_index(s1_id_col).to_dict(orient="index")
    s23_dict = s23_df.set_index(s23_id_col).to_dict(orient="index")

    print("Extracting feature vectors for candidate pairs...")
    records = []

    for raw_s1_id, cand_ids in candidate_pairs.items():
        s1_id = str(raw_s1_id).strip()
        if s1_id not in s1_dict:
            continue
        s1_row = s1_dict[s1_id]

        for raw_cand_id in cand_ids:
            cand_id = str(raw_cand_id).strip()
            if cand_id not in s23_dict:
                continue
            s23_row = s23_dict[cand_id]

            feat = extract_pair_features(s1_row, s23_row)
            feat["source1_entity_id"] = s1_id
            feat["source23_entity_id"] = cand_id
            feat["true_label"] = 1 if feat["exact_match"] == 1.0 or feat["lev_ratio"] > 0.85 else 0
            records.append(feat)

    feature_df = pd.DataFrame(records)
    feature_cols = ["exact_match", "jaccard_tok", "jaccard_3gram", "lev_dist", "lev_ratio", "len_diff", "len_ratio"]

    X = feature_df[feature_cols]
    y = feature_df["true_label"]

    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
        X, y, feature_df, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    print("\nTraining LightGBM Classifier...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {"objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt", "learning_rate": 0.05, "num_leaves": 31, "verbose": -1}
    model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])

    model.save_model(str(OUTPUT_DIR / "lgbm_matching_model.txt"))

    # Generate full dataset predictions
    feature_df["match_probability"] = model.predict(X)
    feature_df["predicted_match"] = (feature_df["match_probability"] >= 0.5).astype(int)

    # Save prediction details TSV for Member 3 evaluation
    pred_path = OUTPUT_DIR / "prediction_details.tsv"
    feature_df[["source1_entity_id", "source23_entity_id", "true_label", "predicted_match", "match_probability"]].to_csv(pred_path, sep="\t", index=False)

    # FORMAT EXACT LEADERBOARD SUBMISSION: output/matching_results.tsv
    print("\nFormatting output/matching_results.tsv...")
    matched_only = feature_df[feature_df["predicted_match"] == 1]
    
    # Group matched entity_ids by source1_entity_id
    grouped_matches = (
        matched_only.groupby("source1_entity_id")["source23_entity_id"]
        .apply(lambda ids: ",".join(ids))
        .reset_index()
        .rename(columns={"source23_entity_id": "matched_entity_ids"})
    )

    # Outer join to ensure every source1_entity_id is present
    all_s1_ids = pd.DataFrame({"source1_entity_id": list(candidate_pairs.keys())})
    submission_df = pd.merge(all_s1_ids, grouped_matches, on="source1_entity_id", how="left").fillna("")

    matching_tsv_path = OUTPUT_DIR / "matching_results.tsv"
    submission_df.to_csv(matching_tsv_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)

    print(f"Saved required submission files to:\n - {matching_tsv_path}\n - {OUTPUT_DIR / 'candidate_pairs.tsv'}")
    print("\n============================================")
    print("        MEMBER 2 TRAINING COMPLETE         ")
    print("============================================\n")


if __name__ == "__main__":
    main()