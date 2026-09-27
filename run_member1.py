import sys
from pathlib import Path
from src.preprocessing.preprocessing_pipeline import execute_member1_pipeline

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_DIR = BASE_DIR / "data" / "student_resource" / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "data"

if __name__ == "__main__":
    s1_path = TRAIN_DATA_DIR / "train_source1.tsv"
    s2_path = TRAIN_DATA_DIR / "train_source2.tsv"
    s3_path = TRAIN_DATA_DIR / "train_source3.tsv"

    # Verify input files exist
    for file_path in [s1_path, s2_path, s3_path]:
        if not file_path.exists():
            print(f"[ERROR] Could not find required file: {file_path}")
            sys.exit(1)

    print(f"Processing Train Split Files:")
    print(f"  - Source 1: {s1_path.name}")
    print(f"  - Source 2: {s2_path.name}")
    print(f"  - Source 3: {s3_path.name}\n")

    execute_member1_pipeline(
        s1_raw_path=str(s1_path),
        s2_raw_path=str(s2_path),
        s3_raw_path=str(s3_path),
        output_dir=str(OUTPUT_DIR),
        id_col="id",
        name_col="name",
        addr_col="address",
        country_col="country",
    )