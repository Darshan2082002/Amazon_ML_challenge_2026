import json
import os
import pandas as pd

from src.preprocessing.blocking import run_candidate_blocking
from src.preprocessing.cleaning import preprocess_dataframe
from src.preprocessing.eda import generate_source_profile, print_eda_summary


def load_tsv(file_path: str) -> pd.DataFrame:
    """Helper to read TSV files reliably."""
    return pd.read_csv(file_path, sep="\t", on_bad_lines="skip")


def execute_member1_pipeline(
    s1_raw_path: str,
    s2_raw_path: str,
    s3_raw_path: str,
    output_dir: str,
    id_col: str = "record_id",
    name_col: str = "name",
    addr_col: str = "address",
    country_col: str = "country",
):
    """Executes data cleaning, combining, profiling, and blocking."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading raw TSV files...")
    s1_raw = load_tsv(s1_raw_path)
    s2_raw = load_tsv(s2_raw_path)
    s3_raw = load_tsv(s3_raw_path)

    # Automatically identify fallback column names if needed
    def detect_col(df, target, fallbacks):
        if target in df.columns:
            return target
        for fb in fallbacks:
            if fb in df.columns:
                return fb
        return df.columns[0]  # Fallback to first column

    # Combine Source 2 and Source 3 into a single candidate set
    s23_raw = pd.concat([s2_raw, s3_raw], ignore_index=True)

    print("Preprocessing & Normalizing datasets...")
    id_c1 = detect_col(s1_raw, id_col, ["id", "S1_ID", "record_id"])
    id_c23 = detect_col(s23_raw, id_col, ["id", "S2_ID", "S3_ID", "record_id"])

    name_c1 = detect_col(s1_raw, name_col, ["title", "entity_name"])
    name_c23 = detect_col(s23_raw, name_col, ["title", "entity_name"])

    addr_c1 = detect_col(s1_raw, addr_col, ["location", "street_address"])
    addr_c23 = detect_col(s23_raw, addr_col, ["location", "street_address"])

    country_c1 = (
        detect_col(s1_raw, country_col, ["country_code"])
        if country_col in s1_raw.columns
        else None
    )
    country_c23 = (
        detect_col(s23_raw, country_col, ["country_code"])
        if country_col in s23_raw.columns
        else None
    )

    s1_clean = preprocess_dataframe(s1_raw, id_c1, name_c1, addr_c1, country_c1)
    s23_clean = preprocess_dataframe(
        s23_raw, id_c23, name_c23, addr_c23, country_c23
    )

    # Save cleaned outputs required by Member 2
    s1_clean_path = os.path.join(output_dir, "source1_clean.csv")
    s23_clean_path = os.path.join(output_dir, "source23_clean.csv")
    s1_clean.to_csv(s1_clean_path, index=False)
    s23_clean.to_csv(s23_clean_path, index=False)
    print(f"Clean datasets saved to '{output_dir}'.")

    # Run EDA
    print("Running Exploratory Data Analysis...")
    profiles = [
        generate_source_profile(s1_clean, "Source 1"),
        generate_source_profile(s23_clean, "Source 2 & 3 Combined"),
    ]
    print_eda_summary(profiles)

    # Execute Candidate Blocking
    print("Executing Candidate Blocking...")
    candidate_pairs, blocking_stats = run_candidate_blocking(
        s1_clean, s23_clean
    )

    # Save candidate outputs
    json_path = os.path.join(output_dir, "candidate_pairs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(candidate_pairs, f, indent=2)

    tsv_path = os.path.join(output_dir, "candidate_pairs.tsv")
    lines = ["source1_entity_id\tcandidate_entity_id"]
    for s1_id, cands in candidate_pairs.items():
        for cand_id in cands:
            lines.append(f"{s1_id}\t{cand_id}")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n============================================")
    print("        BLOCKING PERFORMANCE STATS          ")
    print("============================================")
    for stat_name, val in blocking_stats.items():
        print(f"{stat_name}: {val}")
    print("============================================\n")

    print(f"Member 1 complete. Deliverables stored in '{output_dir}'.")