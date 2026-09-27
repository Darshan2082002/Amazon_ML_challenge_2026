import pandas as pd
from typing import Dict, List


def generate_source_profile(df: pd.DataFrame, source_name: str) -> Dict[str, float]:
    """
    Generates basic summary statistics for dataset profiling.
    """
    total_records = len(df)

    # Check missing clean values
    missing_names = df["clean_name"].eq("").sum() if "clean_name" in df.columns else 0
    missing_addrs = df["clean_addr"].eq("").sum() if "clean_addr" in df.columns else 0

    # Count duplicate records based on normalized fields
    if "clean_name" in df.columns and "clean_addr" in df.columns:
        duplicates_exact = df.duplicated(subset=["clean_name", "clean_addr"]).sum()
    else:
        duplicates_exact = 0

    unique_entities = total_records - duplicates_exact

    profile = {
        "source": source_name,
        "total_records": total_records,
        "unique_entities": unique_entities,
        "duplicate_records": duplicates_exact,
        "missing_name_pct": round((missing_names / total_records) * 100, 2) if total_records else 0.0,
        "missing_addr_pct": round((missing_addrs / total_records) * 100, 2) if total_records else 0.0,
    }

    return profile


def print_eda_summary(profiles: List[Dict]):
    """
    Outputs a structured summary table of dataset profiles.
    """
    eda_df = pd.DataFrame(profiles)
    print("\n============================================")
    print("      EXPLORATORY DATA ANALYSIS (EDA)       ")
    print("============================================")
    print(eda_df.to_string(index=False))
    print("============================================\n")