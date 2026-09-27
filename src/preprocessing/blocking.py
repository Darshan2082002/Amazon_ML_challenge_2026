from typing import Dict, List
import pandas as pd


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    s1_id_col: str,
    s23_id_col: str,
    s1_text_col: str,
    s23_text_col: str,
    top_k: int = 5
) -> Dict[str, List[str]]:
    """C-accelerated vectorized inverted index blocking with explicit column renaming."""
    print("   Preparing token dataframes...")
    
    # 1. Extract IDs and split text fields into clean word tokens
    s1_sub = s1_df[[s1_id_col, s1_text_col]].dropna().copy()
    s23_sub = s23_df[[s23_id_col, s23_text_col]].dropna().copy()

    # Rename ID columns explicitly to prevent merge collision (_x / _y)
    s1_sub = s1_sub.rename(columns={s1_id_col: "s1_id"})
    s23_sub = s23_sub.rename(columns={s23_id_col: "s23_id"})

    s1_sub["token"] = s1_sub[s1_text_col].astype(str).str.split()
    s23_sub["token"] = s23_sub[s23_text_col].astype(str).str.split()

    # 2. Explode token arrays into rows
    print("   Exploding token lists for Source 1 & Source 2/3...")
    s1_exploded = s1_sub.explode("token")
    s23_exploded = s23_sub.explode("token")

    # Filter out empty strings and short noise words (< 3 chars)
    s1_exploded = s1_exploded[s1_exploded["token"].str.len() > 2][["s1_id", "token"]].drop_duplicates()
    s23_exploded = s23_exploded[s23_exploded["token"].str.len() > 2][["s23_id", "token"]].drop_duplicates()

    # 3. Vectorized merge on token equality
    print("   Merging records on token overlaps...")
    matches = pd.merge(s1_exploded, s23_exploded, on="token", how="inner")

    # 4. Count token overlaps per pair and extract Top-K candidates
    print("   Aggregating candidate counts...")
    pair_counts = (
        matches.groupby(["s1_id", "s23_id"], sort=False)
        .size()
        .reset_index(name="overlap_count")
    )

    # Sort by overlap and retain top K candidates per s1_id
    pair_counts = pair_counts.sort_values(["s1_id", "overlap_count"], ascending=[True, False])
    top_candidates = pair_counts.groupby("s1_id").head(top_k)

    # Group candidate IDs into list format
    candidate_pairs_map = (
        top_candidates.groupby("s1_id")["s23_id"]
        .apply(list)
        .to_dict()
    )

    # Fill empty candidate lists for S1 IDs with no token matches
    print("   Finalizing candidate mapping...")
    all_s1_ids = s1_df[s1_id_col].astype(str).str.strip().unique()
    candidate_pairs = {s1_id: candidate_pairs_map.get(s1_id, []) for s1_id in all_s1_ids}

    return candidate_pairs