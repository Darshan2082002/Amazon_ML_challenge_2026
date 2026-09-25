import pandas as pd
import os
import time

def evaluate_blocking_output(candidate_path, gt_path, total_s1, total_s2, total_s3):
    start_time = time.time()
    
    print("Loading and processing ground truth...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    
    # Fast vectorized expansion of comma-separated lists
    gt_exploded = gt_df.assign(match_id=gt_df['matched_entity_ids'].str.split(',')).explode('match_id')
    gt_exploded = gt_exploded[gt_exploded['match_id'].str.strip() != ""]
    true_pairs = set(zip(gt_exploded['source1_entity_id'], gt_exploded['match_id'].str.strip()))

    print("Loading and processing generated candidates...")
    cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str).fillna("")
    
    cand_exploded = cand_df.assign(cand_id=cand_df['candidate_entity_ids'].str.split(',')).explode('cand_id')
    cand_exploded = cand_exploded[cand_exploded['cand_id'].str.strip() != ""]
    generated_pairs = set(zip(cand_exploded['source1_entity_id'], cand_exploded['cand_id'].str.strip()))

    print("Calculating metrics...")
    total_true_links = len(true_pairs)
    true_links_found = len(true_pairs.intersection(generated_pairs))
    total_generated = len(generated_pairs)
    total_possible = total_s1 * (total_s2 + total_s3)
    
    recall = true_links_found / total_true_links if total_true_links > 0 else 0
    reduction_ratio = 1 - (total_generated / total_possible)
    
    print(f"\n--- Blocking Evaluation (Completed in {time.time() - start_time:.2f}s) ---")
    print(f"Total True Matches in GT: {total_true_links}")
    print(f"True Matches Found by Blocking: {true_links_found}")
    print(f"Blocking Recall: {recall:.4f}")
    print(f"Candidate Pairs Generated: {total_generated}")
    print(f"Reduction Ratio: {reduction_ratio:.6f}")

if __name__ == "__main__":
    # Ensure these are the exact lengths of your raw S1, S2, and S3 datasets
    # (Check your EDA report from Member 1 to plug in the exact numbers)
    TOTAL_S1 = 10000 
    TOTAL_S2 = 10000
    TOTAL_S3 = 10000 
    
    candidate_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'candidate_pairs.tsv'))
    gt_file = r"D:\Amazon_ML_challenge_2026\data\student_resource\dataset\train\train_ground_truth.tsv"
    
    evaluate_blocking_output(candidate_file, gt_file, TOTAL_S1, TOTAL_S2, TOTAL_S3)