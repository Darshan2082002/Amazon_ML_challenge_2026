import pandas as pd
from typing import Dict, List
from src.matching.similarity import compute_string_similarities, compute_numeric_overlap

def safe_country_match(country1, country2) -> int:
    """Returns 1 only when both country values are present and equal after normalization."""
    if pd.isna(country1) or pd.isna(country2):
        return 0

    c1 = str(country1).strip().lower()
    c2 = str(country2).strip().lower()

    if not c1 or not c2:
        return 0

    return int(c1 == c2)

def build_pair_features(
    s1_records: Dict[str, dict],
    s23_records: Dict[str, dict],
    candidate_pairs: Dict[str, List[str]]
) -> pd.DataFrame:
    """Generates a tabular feature dataset for every (Source 1, Candidate) pair."""
    rows = []

    for s1_id, cand_ids in candidate_pairs.items():
        if s1_id not in s1_records:
            continue

        s1 = s1_records[s1_id]

        for cand_id in cand_ids:
            if cand_id not in s23_records:
                continue

            cand = s23_records[cand_id]

            s1_name = str(s1.get("clean_name", "")).strip()
            cand_name = str(cand.get("clean_name", "")).strip()
            s1_addr = str(s1.get("clean_addr", "")).strip()
            cand_addr = str(cand.get("clean_addr", "")).strip()

            name_sims = compute_string_similarities(s1_name, cand_name)
            addr_sims = compute_string_similarities(s1_addr, cand_addr)
            num_sim = compute_numeric_overlap(s1_addr, cand_addr)

            row = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,

                "same_country": safe_country_match(s1.get("country"), cand.get("country")),

                # Name features
                "name_ratio": name_sims["ratio"],
                "name_partial_ratio": name_sims["partial_ratio"],
                "name_token_sort": name_sims["token_sort_ratio"],
                "name_token_set": name_sims["token_set_ratio"],
                "name_jaro_winkler": name_sims["jaro_winkler"],

                # Address features
                "addr_ratio": addr_sims["ratio"],
                "addr_partial_ratio": addr_sims["partial_ratio"],
                "addr_token_sort": addr_sims["token_sort_ratio"],
                "addr_token_set": addr_sims["token_set_ratio"],
                "addr_jaro_winkler": addr_sims["jaro_winkler"],
                "addr_num_jaccard": num_sim,

                # Exact-match indicators
                "name_exact": int(s1_name.lower() == cand_name.lower() and s1_name != ""),
                "address_exact": int(s1_addr.lower() == cand_addr.lower() and s1_addr != ""),
            }

            rows.append(row)

    return pd.DataFrame(rows)