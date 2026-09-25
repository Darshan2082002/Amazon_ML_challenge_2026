import pandas as pd
import lightgbm as lgb
from src.matching.matching_pipeline import execute_matching_stage

# Load datasets
s1_df = pd.read_csv("data/source1_clean.csv")
s23_df = pd.read_csv("data/source23_clean.csv")
# candidate_pairs = load_candidate_pairs_from_member1()

# Load trained LightGBM model
model = lgb.Booster(model_file="output/lgbm_matching_model.txt")
threshold = 0.75  # Use the optimal threshold from train.py

# Execute pipeline
execute_matching_stage(
    s1_df=s1_df,
    s23_df=s23_df,
    candidate_pairs=candidate_pairs,
    model=model,
    threshold=threshold,
    output_matching_file="output/matching_results.tsv",
    output_details_file="output/prediction_details.tsv"
)