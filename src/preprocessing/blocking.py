import pandas as pd
from typing import Dict, List, Tuple

def generate_blocking_keys(row: pd.Series) -> List[str]:
    """Generates multi-tier blocking keys across name, address, and country."""
    keys = []
    name = str(row.get("clean_name", "")).strip()
    addr = str(row.get("clean_addr", "")).strip()
    country = str(row.get("country", "")).strip()

    if name:
        # 1. First word/token of company name
        tokens = name.split()
        if tokens:
            keys.append(f"token_{tokens[0]}")
            
        # 2. First 3 non-space characters
        clean_compact = name.replace(" ", "")
        if len(clean_compact) >= 3:
            keys.append(f"prefix_{clean_compact[:3]}")
            
        # 3. Country + First Token combination
        if country and tokens:
            keys.append(f"geo_{country}_{tokens[0]}")

    if addr:
        # 4. First token of street address
        addr_tokens = addr.split()
        if addr_tokens:
            keys.append(f"addr_{addr_tokens[0]}")

    return keys


def run_candidate_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_candidates_per_entity: int = 50
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    Creates candidate pairs using multi-key indexing with a guaranteed candidate fallback.
    """
    s1_blocks = {}
    s23_blocks = {}

    print("Indexing Source 1 blocks...")
    for _, row in s1_df.iterrows():
        s1_id = str(row["entity_id"]).strip()
        for key in generate_blocking_keys(row):
            s1_blocks.setdefault(key, set()).add(s1_id)

    print("Indexing Source 2 & 3 blocks...")
    for _, row in s23_df.iterrows():
        cand_id = str(row["entity_id"]).strip()
        for key in generate_blocking_keys(row):
            s23_blocks.setdefault(key, set()).add(cand_id)

    print("Matching candidate pairs...")
    candidate_pairs = {str(s1_id).strip(): set() for s1_id in s1_df["entity_id"].unique()}

    # Match blocks
    for key, s1_ids in s1_blocks.items():
        if key in s23_blocks:
            cand_ids = s23_blocks[key]
            for s1_id in s1_ids:
                candidate_pairs[s1_id].update(cand_ids)

    # Fallback Strategy: Assign global sample candidates if blocking produced []
    s23_ids_list = s23_df["entity_id"].astype(str).str.strip().tolist()
    default_fallback_cands = set(s23_ids_list[:max_candidates_per_entity])

    empty_count = 0
    final_candidates = {}

    for s1_id, cands in candidate_pairs.items():
        if not cands:
            empty_count += 1
            final_candidates[s1_id] = list(default_fallback_cands)
        else:
            # Cap candidates per entity to avoid explosive memory overhead
            final_candidates[s1_id] = sorted(list(cands))[:max_candidates_per_entity]

    if empty_count > 0:
        print(f"[NOTE] Fallback applied for {empty_count} Source 1 entities with 0 initial block matches.")

    # Calculate Reduction Ratio Metric
    num_s1 = len(s1_df)
    num_s23 = len(s23_df)
    total_comparisons_possible = num_s1 * num_s23
    total_candidate_pairs = sum(len(cands) for cands in final_candidates.values())

    reduction_ratio = 1.0 - (total_candidate_pairs / float(total_comparisons_possible)) if total_comparisons_possible > 0 else 0.0

    blocking_stats = {
        "source1_records": num_s1,
        "source23_records": num_s23,
        "max_possible_pairs": total_comparisons_possible,
        "generated_candidate_pairs": total_candidate_pairs,
        "reduction_ratio": round(reduction_ratio, 6),
        "reduction_ratio_pct": f"{round(reduction_ratio * 100, 4)}%"
    }

    return final_candidates, blocking_stats