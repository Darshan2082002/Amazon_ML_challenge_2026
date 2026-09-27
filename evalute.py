import json
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Check potential dataset paths
PRIMARY_DATA_DIR = BASE_DIR / "data" / "student_resource" / "dataset" / "train"
FALLBACK_DATA_DIR = BASE_DIR / "dataset" / "train"

def resolve_train_dir() -> Path:
    if PRIMARY_DATA_DIR.exists():
        return PRIMARY_DATA_DIR
    elif FALLBACK_DATA_DIR.exists():
        return FALLBACK_DATA_DIR
    return PRIMARY_DATA_DIR

try:
    from Levenshtein import ratio as lev_ratio_func
except ImportError:
    import difflib
    def lev_ratio_func(s1, s2):
        return difflib.SequenceMatcher(None, s1, s2).ratio() if s1 and s2 else 0.0


def detect_col(df: pd.DataFrame, candidates: list) -> str:
    """Safely find matching column name from dataframe."""
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Computes F_0.5 score weighting precision 2x heavier than recall."""
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)


def evaluate_macro_f05(ground_truth_dict: dict, predictions_dict: dict, all_s1_ids: set) -> float:
    """Calculates macro-averaged F_0.5 across all Source 1 entities including singletons."""
    total_score = 0.0

    for s1_id in all_s1_ids:
        true_matches = set(ground_truth_dict.get(s1_id, []))
        pred_matches = set(predictions_dict.get(s1_id, []))

        # Case 1: Ground truth is a singleton (no matches)
        if not true_matches:
            if not pred_matches:
                score = 1.0  # Earn credit for empty prediction
            else:
                score = 0.0  # False merge penalty
        # Case 2: Ground truth has matches
        else:
            if not pred_matches:
                score = 0.0
            else:
                tp = len(true_matches & pred_matches)
                precision = tp / len(pred_matches)
                recall = tp / len(true_matches)
                score = compute_f_beta(precision, recall, beta=0.5)

        total_score += score

    return total_score / len(all_s1_ids)


def main():
    print("============================================")
    print("      EVALUATING LOCAL MACRO F_0.5 SCORE    ")
    print("============================================")

    data_dir = resolve_train_dir()
    gt_path = data_dir / "train_ground_truth.tsv"
    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"

    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found at: {gt_path}")
        sys.exit(1)

    print("Loading training datasets...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    
    s23_list = []
    if s2_path.exists():
        s23_list.append(pd.read_csv(s2_path, sep="\t", dtype=str).fillna(""))
    if s3_path.exists():
        s23_list.append(pd.read_csv(s3_path, sep="\t", dtype=str).fillna(""))
    s23_df = pd.concat(s23_list, ignore_index=True)

    # Robust column identification
    gt_s1_col = detect_col(gt_df, ["source1_entity_id", "s1_id", "entity_id", "id"])
    gt_s23_col = detect_col(gt_df, ["matched_entity_ids", "source23_entity_id", "s23_id", "id"])

    s1_id_col = detect_col(s1_df, ["source1_entity_id", "entity_id", "s1_id", "id"])
    s23_id_col = detect_col(s23_df, ["source23_entity_id", "entity_id", "s23_id", "id"])

    s1_name_col = detect_col(s1_df, ["clean_name", "name", "title", "company_name"])
    s23_name_col = detect_col(s23_df, ["clean_name", "name", "title", "company_name"])

    # Map ground truth pairs
    gt_dict = defaultdict(list)
    for _, row in gt_df.iterrows():
        s1 = str(row[gt_s1_col]).strip()
        s23_raw = str(row[gt_s23_col]).strip()
        if s23_raw:
            s23_matches = [x.strip() for x in s23_raw.split(",") if x.strip()]
            gt_dict[s1].extend(s23_matches)

    all_s1_ids = set(s1_df[s1_id_col].astype(str).str.strip().unique())

    print(f"Total Train S1 Entities: {len(all_s1_ids)}")
    print(f"Entities with Ground Truth Matches: {len(gt_dict)}")

    # Model Prediction Simulation
    print("Evaluating high-precision model matching...")
    s1_dict = s1_df.set_index(s1_id_col).to_dict(orient="index")
    s23_dict = s23_df.set_index(s23_id_col).to_dict(orient="index")

    token_index = defaultdict(list)
    s23_ids = s23_df[s23_id_col].astype(str).str.strip().values
    s23_texts = s23_df[s23_name_col].fillna("").astype(str).str.lower().values

    for idx, text in enumerate(s23_texts):
        for t in set(text.split()):
            if len(t) > 2:
                token_index[t].append(idx)

    pred_dict = {}
    for s1_id in all_s1_ids:
        s1_row = s1_dict.get(s1_id, {})
        s1_name = str(s1_row.get(s1_name_col, "")).strip().lower()
        tokens = [t for t in set(s1_name.split()) if len(t) > 2]

        cand_indices = []
        for t in tokens:
            if t in token_index:
                cand_indices.extend(token_index[t])

        matches = []
        if cand_indices:
            freq = defaultdict(int)
            for idx in cand_indices:
                freq[idx] += 1
            top_indices = sorted(freq, key=freq.get, reverse=True)[:5]

            for idx in top_indices:
                cand_id = s23_ids[idx]
                cand_name = s23_texts[idx]
                sim = lev_ratio_func(s1_name, cand_name) if s1_name and cand_name else 0.0

                # Strict High-Precision threshold
                if sim >= 0.88:
                    matches.append(cand_id)

        pred_dict[s1_id] = matches

    # Calculate final score
    macro_f05 = evaluate_macro_f05(gt_dict, pred_dict, all_s1_ids)

    print("\n--------------------------------------------")
    print(f" LOCAL VALIDATION MACRO F_0.5 SCORE: {macro_f05:.4f}")
    print("--------------------------------------------\n")


if __name__ == "__main__":
    main()