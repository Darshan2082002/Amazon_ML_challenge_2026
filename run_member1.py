import csv
import json
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing.cleaning import preprocess_dataframe
from src.preprocessing.blocking import generate_candidate_pairs

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def main():
    print("============================================")
    print("     MEMBER 1 - DATA CLEANING & BLOCKING   ")
    print("============================================")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s1_raw_path = DATA_DIR / "source1.csv"
    s23_raw_path = DATA_DIR / "source23.csv"

    if not s1_raw_path.exists():
        s1_raw_path = DATA_DIR / "source1_clean.csv"
    if not s23_raw_path.exists():
        s23_raw_path = DATA_DIR / "source23_clean.csv"

    if not s1_raw_path.exists() or not s23_raw_path.exists():
        print(f"[ERROR] Source files missing in {DATA_DIR}!")
        sys.exit(1)

    print(f"Loading raw datasets from:\n - {s1_raw_path}\n - {s23_raw_path}")
    s1_df = pd.read_csv(s1_raw_path, dtype=str, on_bad_lines="skip", engine="python").fillna("")
    s23_df = pd.read_csv(s23_raw_path, dtype=str, on_bad_lines="skip", engine="python").fillna("")

    s1_id_col = detect_column(s1_df, ["entity_id", "source1_entity_id", "s1_id", "id"])
    s23_id_col = detect_column(s23_df, ["entity_id", "source23_entity_id", "s23_id", "id"])

    name_cols = ["name", "title", "company_name", "clean_name"]
    s1_name_col = detect_column(s1_df, name_cols)
    s23_name_col = detect_column(s23_df, name_cols)

    s1_df[s1_id_col] = s1_df[s1_id_col].astype(str).str.strip()
    s23_df[s23_id_col] = s23_df[s23_id_col].astype(str).str.strip()

    print("Cleaning text fields...")
    s1_clean = preprocess_dataframe(s1_df, text_cols=[c for c in s1_df.columns if c != s1_id_col])
    s23_clean = preprocess_dataframe(s23_df, text_cols=[c for c in s23_df.columns if c != s23_id_col])

    s1_clean.to_csv(DATA_DIR / "source1_clean.csv", index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    s23_clean.to_csv(DATA_DIR / "source23_clean.csv", index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    print("\nGenerating candidate pairs via Blocking/TF-IDF...")
    candidate_pairs = generate_candidate_pairs(
        s1_clean, s23_clean, s1_id_col, s23_id_col, s1_name_col, s23_name_col, top_k=5
    )

    # Save JSON for training script
    pairs_json_path = DATA_DIR / "candidate_pairs.json"
    with open(pairs_json_path, "w", encoding="utf-8") as f:
        json.dump(candidate_pairs, f, indent=2)

    # Save output/candidate_pairs.tsv as required by challenge guidelines
    candidate_tsv_rows = []
    for s1_id, c_list in candidate_pairs.items():
        candidate_tsv_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(c_list)
        })

    cand_df = pd.DataFrame(candidate_tsv_rows)
    cand_tsv_path = OUTPUT_DIR / "candidate_pairs.tsv"
    cand_df.to_csv(cand_tsv_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)

    print(f"Saved {len(candidate_pairs)} candidate pair mappings to:\n - {pairs_json_path}\n - {cand_tsv_path}")
    print("\n============================================")
    print("        MEMBER 1 PIPELINE COMPLETE         ")
    print("============================================\n")


if __name__ == "__main__":
    main()