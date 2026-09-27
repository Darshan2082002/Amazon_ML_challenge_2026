import Levenshtein
import pandas as pd


def jaccard_similarity(str1: str, str2: str) -> float:
    set1, set2 = set(str1.split()), set(str2.split())
    union = set1.union(set2)
    return len(set1.intersection(set2)) / len(union) if union else 0.0


def character_ngram_jaccard(str1: str, str2: str, n: int = 3) -> float:
    def get_ngrams(s, n):
        return set([s[i : i + n] for i in range(len(s) - n + 1)])

    ng1, ng2 = get_ngrams(str1, n), get_ngrams(str2, n)
    union = ng1.union(ng2)
    return len(ng1.intersection(ng2)) / len(union) if union else 0.0


def extract_pair_features(s1_row: pd.Series, s23_row: pd.Series) -> dict:
    s1_name = str(s1_row.get("name", "") or s1_row.get("title", "") or s1_row.get("company_name", "")).lower().strip()
    s23_name = str(s23_row.get("name", "") or s23_row.get("title", "") or s23_row.get("company_name", "")).lower().strip()

    exact_match = 1.0 if s1_name == s23_name and len(s1_name) > 0 else 0.0
    jaccard_tok = jaccard_similarity(s1_name, s23_name)
    jaccard_3gram = character_ngram_jaccard(s1_name, s23_name, n=3)
    lev_dist = Levenshtein.distance(s1_name, s23_name)
    lev_ratio = Levenshtein.ratio(s1_name, s23_name)
    len_diff = abs(len(s1_name) - len(s23_name))
    len_ratio = min(len(s1_name), len(s23_name)) / max(len(s1_name), len(s23_name), 1)

    return {
        "exact_match": exact_match,
        "jaccard_tok": jaccard_tok,
        "jaccard_3gram": jaccard_3gram,
        "lev_dist": lev_dist,
        "lev_ratio": lev_ratio,
        "len_diff": len_diff,
        "len_ratio": len_ratio,
    }