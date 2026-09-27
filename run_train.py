import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.matching.features import build_pair_features
from src.matching.train import train_classifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_TRAIN_DIR = DATA_DIR / "student_resource" / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "output"

def load_and_prepare_data():
    """
    Loads Member 1 outputs, ensures string-consistent keys,
    merges ground truth labels, and splits into train/validation sets.
    """
    print("Loading cleaned datasets from Member 1...")
    s1_clean_path = DATA_DIR / "source1_clean.csv"
    s23_clean_path = DATA_DIR / "source23_clean.csv"
    cand_path = DATA_DIR / "candidate_pairs.json"
    gt_path = RAW_TRAIN_DIR / "train_ground_truth.tsv"

    # Verify input files exist
    required_files = [s1_clean_path, s23_clean_path, cand_path, gt_path]
    for file_path in required_files:
        if not file_path.exists():
            print(f"\n[ERROR] Missing required file: {file_path}")
            sys.exit(1)

    # 1. Load DataFrames with strict string IDs
    s1_df = pd.read_csv(s1_clean_path, dtype=str).fillna("")
    s23_df = pd.read_csv(s23_clean_path, dtype=str).fillna("")

    # Automatically identify ID column name if 'entity_id' is missing
    s1_id_col = "entity_id" if "entity_id" in s1_df.columns else s1_df.columns[0]
    s23_id_col = "entity_id" if "entity_id" in s23_df.columns else s23_df.columns[0]

    s1_df["entity_id"] = s1_df[s1_id_col].str.strip()
    s23_df["entity_id"] = s23_df[s23_id_col].str.strip()

    # Convert DataFrames to dicts with clean string keys
    s1_dict = {str(k).strip(): v for k, v in s1_df.set_index("entity_id").to_dict("index").items()}
    s23_dict = {str(k).strip(): v for k, v in s23_df.set_index("entity_id").to_dict("index").items()}

    # 2. Load candidate pairs & ensure string key normalization
    print("Loading candidate pairs...")
    with open(cand_path, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)

    all_candidates = {
        str(s1_id).strip(): [str(cand).strip() for cand in cands]
        for s1_id, cands in raw_candidates.items()
    }

    # 3. Load ground truth labels
    print("Loading ground truth labels...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    
    s1_gt_col = next((c for c in ["source1_entity_id", "s1_id", "source1_id"] if c in gt_df.columns), gt_df.columns[0])
    match_gt_col = next((c for c in ["candidate_entity_id", "match_id", "matched_entity_id"] if c in gt_df.columns), gt_df.columns[1])
    
    gt_df[s1_gt_col] = gt_df[s1_gt_col].str.strip()
    gt_df[match_gt_col] = gt_df[match_gt_col].str.strip()

    # Build set of true positive matches
    positive_matches = set(zip(gt_df[s1_gt_col], gt_df[match_gt_col]))

    # Print Diagnostic Summary
    print("\n--- Diagnostic Check ---")
    print(f"Source 1 records in dict: {len(s1_dict)}")
    print(f"Source 2/3 records in dict: {len(s23_dict)}")
    print(f"Candidate pairs S1 keys: {len(all_candidates)}")
    print(f"Ground truth match pairs: {len(positive_matches)}")
    print("------------------------\n")

    # 4. Perform 80/20 train/validation split
    print("Splitting dataset into 80% train and 20% validation sets...")
    s1_ids = list(all_candidates.keys())
    train_s1_ids, val_s1_ids = train_test_split(s1_ids, test_size=0.2, random_state=42)

    cand_train = {s1_id: all_candidates[s1_id] for s1_id in train_s1_ids}
    cand_val = {s1_id: all_candidates[s1_id] for s1_id in val_s1_ids}

    return s1_dict, s23_dict, cand_train, cand_val, positive_matches


def extract_labels_for_pairs(pair_df, positive_matches):
    """
    Assigns binary label y = 1 for ground truth matches, 0 for negative candidates.
    """
    labels = []
    for _, row in pair_df.iterrows():
        pair = (str(row["source1_entity_id"]).strip(), str(row["candidate_entity_id"]).strip())
        labels.append(1 if pair in positive_matches else 0)
    return np.array(labels)


def main():
    s1_dict, s23_dict, cand_train, cand_val, positive_matches = load_and_prepare_data()

    # Feature Engineering - Train Set
    print("Building pair features for Training set...")
    X_train_df = build_pair_features(s1_dict, s23_dict, cand_train)
    
    if len(X_train_df) == 0:
        print("\n[ERROR] Train feature DataFrame is empty. Check entity ID key matching between s1_dict and candidate_pairs.json.")
        sys.exit(1)

    y_train = extract_labels_for_pairs(X_train_df, positive_matches)
    print(f"Train pairs built: {len(X_train_df)} (Positives: {sum(y_train)})")

    # Feature Engineering - Validation Set
    print("\nBuilding pair features for Validation set...")
    X_val_df = build_pair_features(s1_dict, s23_dict, cand_val)
    y_val = extract_labels_for_pairs(X_val_df, positive_matches)
    print(f"Validation pairs built: {len(X_val_df)} (Positives: {sum(y_val)})")

    # Separate metadata columns from feature columns
    id_cols = ["source1_entity_id", "candidate_entity_id"]
    X_train_features = X_train_df.drop(columns=[c for c in id_cols if c in X_train_df.columns])
    X_val_features = X_val_df.drop(columns=[c for c in id_cols if c in X_val_df.columns])

    # Model Training & Threshold Search
    print("\nTraining LightGBM model & optimizing F0.5 threshold...")
    model, best_threshold = train_classifier(X_train_features, y_train, X_val_features, y_val)

    # Save model artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "lgbm_matching_model.txt"
    model.save_model(str(model_path))

    config_path = OUTPUT_DIR / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": float(best_threshold)}, f, indent=2)

    print("\n============================================")
    print("           TRAINING COMPLETE                ")
    print("============================================")
    print(f"Model saved to: {model_path}")
    print(f"Optimal F0.5 Threshold: {best_threshold:.4f}")
    print("============================================\n")


if __name__ == "__main__":
    main()