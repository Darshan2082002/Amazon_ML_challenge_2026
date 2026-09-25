import re
from rapidfuzz import fuzz, distance

def compute_string_similarities(str1: str, str2: str) -> dict:
    """
    Compute multiple string similarity metrics between two strings.
    Returns values normalized in the range [0.0, 1.0].
    """
    str1 = str(str1).strip() if str1 is not None else ""
    str2 = str(str2).strip() if str2 is not None else ""

    if not str1 or not str2:
        return {
            "ratio": 0.0,
            "partial_ratio": 0.0,
            "token_sort_ratio": 0.0,
            "token_set_ratio": 0.0,
            "jaro_winkler": 0.0,
        }

    return {
        "ratio": fuzz.ratio(str1, str2) / 100.0,
        "partial_ratio": fuzz.partial_ratio(str1, str2) / 100.0,
        "token_sort_ratio": fuzz.token_sort_ratio(str1, str2) / 100.0,
        "token_set_ratio": fuzz.token_set_ratio(str1, str2) / 100.0,
        "jaro_winkler": distance.JaroWinkler.similarity(str1, str2),
    }

def compute_numeric_overlap(addr1: str, addr2: str) -> float:
    """
    Compute Jaccard similarity between numbers appearing in two addresses.
    Returns 0.0 if either address lacks numeric values to prevent false matches.
    """
    nums1 = set(re.findall(r"\d+", str(addr1 or "")))
    nums2 = set(re.findall(r"\d+", str(addr2 or "")))

    # Missing numeric information should NOT be treated as a perfect match (1.0)
    if not nums1 or not nums2:
        return 0.0

    return len(nums1 & nums2) / len(nums1 | nums2)