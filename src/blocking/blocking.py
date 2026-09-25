import pandas as pd
import os
import sys

# Ensure Python can find the preprocessing module when run from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess import preprocess_source_data

def generate_candidate_pairs(df_s1, df_s2, df_s3):
    """Generates a baseline block using Country + First 4 chars of the cleaned name."""
    
    # Filter out empty records before creating keys
    df_s1 = df_s1[(df_s1['name_clean'] != "") & (df_s1['country'] != "")]
    df_s2 = df_s2[(df_s2['name_clean'] != "") & (df_s2['country'] != "")]
    df_s3 = df_s3[(df_s3['name_clean'] != "") & (df_s3['country'] != "")]

    # Create baseline blocking keys
    df_s1['block_key'] = df_s1['country'] + "_" + df_s1['name_clean'].str[:4]
    df_s2['block_key'] = df_s2['country'] + "_" + df_s2['name_clean'].str[:4]
    df_s3['block_key'] = df_s3['country'] + "_" + df_s3['name_clean'].str[:4]
    
    # Merge candidates on the blocking key
    candidates_s2 = df_s1[['entity_id', 'block_key']].merge(
        df_s2[['entity_id', 'block_key']], on='block_key', suffixes=('_s1', '_cand')
    )
    candidates_s3 = df_s1[['entity_id', 'block_key']].merge(
        df_s3[['entity_id', 'block_key']], on='block_key', suffixes=('_s1', '_cand')
    )
    
    all_candidates = pd.concat([candidates_s2, candidates_s3])
    
    # Group and sort for reproducible output
    candidate_pairs = all_candidates.groupby('entity_id_s1')['entity_id_cand'].apply(
        lambda x: ','.join(sorted(set(x)))
    ).reset_index()
    candidate_pairs.columns = ['source1_entity_id', 'candidate_entity_ids']
    
    return candidate_pairs

if __name__ == "__main__":
    # Adjust these paths to match your local setup if necessary
    path_s1 = r"D:\Amazon_ML_challenge_2026\data\student_resource\dataset\train\train_source1.tsv"
    path_s2 = r"D:\Amazon_ML_challenge_2026\data\student_resource\dataset\train\train_source2.tsv"
    path_s3 = r"D:\Amazon_ML_challenge_2026\data\student_resource\dataset\train\train_source3.tsv"

    print("Loading data...")
    df_s1 = pd.read_csv(path_s1, sep="\t", dtype=str)
    df_s2 = pd.read_csv(path_s2, sep="\t", dtype=str)
    df_s3 = pd.read_csv(path_s3, sep="\t", dtype=str)

    print("Preprocessing data...")
    df_s1_clean = preprocess_source_data(df_s1)
    df_s2_clean = preprocess_source_data(df_s2)
    df_s3_clean = preprocess_source_data(df_s3)

    print("Generating candidate pairs...")
    candidate_pairs_tsv = generate_candidate_pairs(df_s1_clean, df_s2_clean, df_s3_clean)
    
    # Save to the root output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    candidate_pairs_tsv.to_csv(output_path, sep="\t", index=False)
    print(f"Blocking complete. Saved to: {output_path}")