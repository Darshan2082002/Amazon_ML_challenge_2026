import pandas as pd
import os

def evaluate_blocking_output(candidate_path, gt_path, total_s1, total_s2, total_s3):
    print("Loading ground truth and generated candidates...")
    
    # 1. Load Ground Truth
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    true_pairs = set()
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        for m in matches:
            if m.strip():
                true_pairs.add((s1_id, m.strip()))

    # 2. Load Generated Candidates
    cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str).fillna("")
    generated_pairs = set()
    for _, row in cand_df.iterrows():
        s1_id = row['source1_entity_id']
        candidates = str(row['candidate_entity_ids']).split(',')
        for c in candidates:
            if c.strip():
                generated_pairs.add((s1_id, c.strip()))

    # 3. Calculate Metrics
    total_true_links = len(true_pairs)
    true_links_found = len(true_pairs.intersection(generated_pairs))
    total_generated = len(generated_pairs)
    total_possible = total_s1 * (total_s2 + total_s3)
    
    recall = true_links_found / total_true_links if total_true_links > 0 else 0
    reduction_ratio = 1 - (total_generated / total_possible)
    
    print("\n--- Blocking Evaluation ---")
    print(f"Total True Matches in GT: {total_true_links}")
    print(f"True Matches Found by Blocking: {true_links_found}")
    print(f"Blocking Recall: {recall:.4f}")
    print(f"Candidate Pairs Generated: {total_generated}")
    print(f"Reduction Ratio: {reduction_ratio:.6f}")

if __name__ == "__main__":
    # You can update these counts dynamically later, but static counts work for the eval script
    TOTAL_S1 = 10000 
    TOTAL_S2 = 10000
    TOTAL_S3 = 10000 
    
    candidate_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'candidate_pairs.tsv'))
    gt_file = r"D:\Amazon_ML_challenge_2026\data\student_resource\dataset\train\train_ground_truth.tsv"
    
    evaluate_blocking_output(candidate_file, gt_file, TOTAL_S1, TOTAL_S2, TOTAL_S3)