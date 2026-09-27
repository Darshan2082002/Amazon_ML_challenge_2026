import csv
import json
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from Levenshtein import ratio as lev_ratio_func
except ImportError:
    def lev_ratio_func(s1, s2):
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        import difflib
        return difflib.SequenceMatcher(None, s1, s2).ratio()

OUTPUT_DIR = BASE_DIR / "output"


def extract_field(row_dict: dict, keys: list) -> str:
    """Extract and normalize text fields."""
    for col in keys:
        if col in row_dict and row_dict[col]:
            return str(row_dict[col]).strip().lower()
    return ""


def jaccard_similarity(str1: str, str2: str) -> float:
    set1, set2 = set(str1.split()), set(str2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def main():
    print("============================================")
    print("  MEMBER 2 - HIGH-PRECISION INFERENCE (F_0.5)")
    print("============================================")

    test_s1_path = OUTPUT_DIR / "test_s1_clean.csv"
    test_s23_path = OUTPUT_DIR / "test_s23_clean.csv"
    pairs_path = OUTPUT_DIR / "candidate_pairs_test.json"

    if not test_s1_path.exists() or not pairs_path.exists():
        print("[ERROR] Missing input files! Run run_member1.py first.")
        sys.exit(1)

    print("Loading preprocessed test datasets...")
    s1_df = pd.read_csv(test_s1_path, dtype=str).fillna("")
    s23_df = pd.read_csv(test_s23_path, dtype=str).fillna("")

    with open(pairs_path, "r", encoding="utf-8") as f:
        candidate_pairs = json.load(f)

    s1_id_col = "source1_entity_id" if "source1_entity_id" in s1_df.columns else s1_df.columns[0]
    s23_id_col = "source23_entity_id" if "source23_entity_id" in s23_df.columns else s23_df.columns[0]

    s1_df[s1_id_col] = s1_df[s1_id_col].astype(str).str.strip()
    s23_df[s23_id_col] = s23_df[s23_id_col].astype(str).str.strip()

    all_s1_ids_df = pd.DataFrame({"source1_entity_id": s1_df[s1_id_col].unique()})

    s1_dict = s1_df.set_index(s1_id_col).to_dict(orient="index")
    s23_dict = s23_df.set_index(s23_id_col).to_dict(orient="index")

    print("Computing high-precision similarity metrics...")
    records = []
    
    name_keys = ["clean_name", "name", "title", "company_name"]
    addr_keys = ["clean_addr", "address", "location"]

    for s1_id, cand_ids in candidate_pairs.items():
        if s1_id not in s1_dict or not cand_ids:
            continue
        
        s1_row = s1_dict[s1_id]
        s1_name = extract_field(s1_row, name_keys)
        s1_addr = extract_field(s1_row, addr_keys)

        for cand_id in cand_ids:
            if cand_id not in s23_dict:
                continue
            
            s23_row = s23_dict[cand_id]
            s23_name = extract_field(s23_row, name_keys)
            s23_addr = extract_field(s23_row, addr_keys)

            # 1. Name Levenshtein ratio
            name_lev = lev_ratio_func(s1_name, s23_name) if s1_name and s23_name else 0.0
            
            # 2. Token Jaccard overlap
            name_jaccard = jaccard_similarity(s1_name, s23_name)

            # 3. Address similarity (if present)
            addr_lev = lev_ratio_func(s1_addr, s23_addr) if s1_addr and s23_addr else 0.0

            # Composite High-Precision Score
            if s1_addr and s23_addr:
                final_score = 0.75 * name_lev + 0.25 * addr_lev
            else:
                final_score = 0.8 * name_lev + 0.2 * name_jaccard

            records.append({
                "source1_entity_id": s1_id,
                "source23_entity_id": cand_id,
                "score": final_score,
                "name_lev": name_lev
            })

    if not records:
        submission_df = all_s1_ids_df.copy()
        submission_df["matched_entity_ids"] = ""
    else:
        feature_df = pd.DataFrame(records)

        # High-Precision Threshold (0.88) to maximize F_0.5 score
        feature_df["predicted_match"] = (
            (feature_df["score"] >= 0.88) | (feature_df["name_lev"] >= 0.92)
        ).astype(int)

        matched_only = feature_df[feature_df["predicted_match"] == 1]
        print(f"Predicted {len(matched_only)} high-precision matches.")

        # Aggregate unique matched entity IDs per S1 ID
        grouped_matches = (
            matched_only.groupby("source1_entity_id")["source23_entity_id"]
            .apply(lambda ids: ",".join(sorted(set(ids))))
            .reset_index()
            .rename(columns={"source23_entity_id": "matched_entity_ids"})
        )

        submission_df = pd.merge(all_s1_ids_df, grouped_matches, on="source1_entity_id", how="left").fillna("")

    matching_tsv_path = OUTPUT_DIR / "matching_results.tsv"
    submission_df.to_csv(matching_tsv_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)

    print(f"\nSUCCESS: Updated high-precision submission file saved to:\n - {matching_tsv_path}")
    print("============================================\n")


if __name__ == "__main__":
    main()