import json
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Primary and fallback paths for student resource directory structure
PRIMARY_DATA_DIR = BASE_DIR / "data" / "student_resource" / "dataset"
FALLBACK_DATA_DIR = BASE_DIR / "dataset"

OUTPUT_DIR = BASE_DIR / "output"


def resolve_data_dir() -> Path:
    if PRIMARY_DATA_DIR.exists():
        return PRIMARY_DATA_DIR
    elif FALLBACK_DATA_DIR.exists():
        return FALLBACK_DATA_DIR
    return PRIMARY_DATA_DIR


def load_and_combine_sources(s2_path: Path, s3_path: Path) -> pd.DataFrame:
    dfs = []
    if s2_path.exists():
        print(f"Loading Test Source 2: {s2_path}")
        dfs.append(pd.read_csv(s2_path, sep="\t", dtype=str).fillna(""))
    if s3_path.exists():
        print(f"Loading Test Source 3: {s3_path}")
        dfs.append(pd.read_csv(s3_path, sep="\t", dtype=str).fillna(""))

    if not dfs:
        raise FileNotFoundError(f"Neither {s2_path} nor {s3_path} was found!")

    return pd.concat(dfs, ignore_index=True)


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def main():
    print("============================================")
    print("   MEMBER 1 - CANDIDATE BLOCKING (TEST)     ")
    print("============================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir()

    test_s1_path = data_dir / "test" / "test_source1.tsv"
    test_s2_path = data_dir / "test" / "test_source2.tsv"
    test_s3_path = data_dir / "test" / "test_source3.tsv"

    if not test_s1_path.exists():
        print(f"[ERROR] Test dataset missing at: {test_s1_path}")
        sys.exit(1)

    print(f"Loading Test Source 1: {test_s1_path}")
    s1_df = pd.read_csv(test_s1_path, sep="\t", dtype=str).fillna("")

    s23_df = load_and_combine_sources(test_s2_path, test_s3_path)

    s1_id_col = detect_column(s1_df, ["source1_entity_id", "entity_id", "id"])
    s23_id_col = detect_column(s23_df, ["source23_entity_id", "entity_id", "id"])

    s1_text_col = detect_column(s1_df, ["name", "clean_name", "title", "address"])
    s23_text_col = detect_column(s23_df, ["name", "clean_name", "title", "address"])

    print(f"Total Test S1 Records: {len(s1_df)}")
    print(f"Total Test S2/3 Records: {len(s23_df)}")

    print("\nBuilding inverted token index for Source 2/3...")
    s23_ids = s23_df[s23_id_col].astype(str).str.strip().values
    s23_texts = s23_df[s23_text_col].astype(str).str.lower().values

    s1_ids = s1_df[s1_id_col].astype(str).str.strip().values
    s1_texts = s1_df[s1_text_col].astype(str).str.lower().values

    token_index = defaultdict(list)
    for idx, text in enumerate(s23_texts):
        for token in set(text.split()):
            if len(token) > 2:
                token_index[token].append(idx)

    print("Matching candidate pairs for Test Source 1...")
    candidate_pairs = {}
    top_k = 5

    for i, s1_id in enumerate(s1_ids):
        tokens = [t for t in set(s1_texts[i].split()) if len(t) > 2]

        matches = []
        for t in tokens:
            if t in token_index:
                matches.extend(token_index[t])

        if not matches:
            candidate_pairs[s1_id] = []
        else:
            freq = defaultdict(int)
            for m in matches:
                freq[m] += 1
            top_indices = sorted(freq, key=freq.get, reverse=True)[:top_k]
            candidate_pairs[s1_id] = [s23_ids[idx] for idx in top_indices]

    pairs_path = OUTPUT_DIR / "candidate_pairs_test.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(candidate_pairs, f)

    s1_df.to_csv(OUTPUT_DIR / "test_s1_clean.csv", index=False)
    s23_df.to_csv(OUTPUT_DIR / "test_s23_clean.csv", index=False)

    print(f"\nSaved test candidate pairs to: {pairs_path}")
    print("============================================\n")


if __name__ == "__main__":
    main()