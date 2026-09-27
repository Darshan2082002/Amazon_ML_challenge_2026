from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    s1_id_col: str,
    s23_id_col: str,
    s1_text_col: str,
    s23_text_col: str,
    top_k: int = 5
) -> Dict[str, List[str]]:
    """Ultra-fast candidate pair generation using sparse matrix operations without dense conversions."""
    print("   Extracting text lists...")
    s1_texts = s1_df[s1_text_col].fillna("").astype(str).tolist()
    s23_texts = s23_df[s23_text_col].fillna("").astype(str).tolist()

    print("   Fitting TF-IDF Vectorizer (max_features=3000)...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=3000,
        sublinear_tf=True
    )
    vectorizer.fit(s1_texts + s23_texts)

    X_s1 = vectorizer.transform(s1_texts)
    X_s23 = vectorizer.transform(s23_texts)

    s1_ids = s1_df[s1_id_col].astype(str).str.strip().tolist()
    s23_ids = s23_df[s23_id_col].astype(str).str.strip().tolist()

    candidate_pairs = {}
    batch_size = 5000
    total_rows = X_s1.shape[0]

    print(f"   Processing {total_rows} rows in batches of {batch_size}...")

    for i in range(0, total_rows, batch_size):
        end_i = min(i + batch_size, total_rows)
        batch_s1 = X_s1[i:end_i]

        # Fast matrix multiplication on sparse matrices
        sim_matrix: csr_matrix = batch_s1.dot(X_s23.T)

        for row_offset in range(end_i - i):
            global_idx = i + row_offset
            s1_id = s1_ids[global_idx]

            # Extract row non-zero elements fast
            row_start = sim_matrix.indptr[row_offset]
            row_end = sim_matrix.indptr[row_offset + 1]

            indices = sim_matrix.indices[row_start:row_end]
            data = sim_matrix.data[row_start:row_end]

            if len(data) == 0:
                # Fallback: assign first top_k IDs if no TF-IDF overlap
                candidate_pairs[s1_id] = s23_ids[:top_k]
            elif len(data) <= top_k:
                sorted_idx = indices[np.argsort(-data)]
                candidate_pairs[s1_id] = [s23_ids[idx] for idx in sorted_idx]
            else:
                top_part = np.argpartition(data, -top_k)[-top_k:]
                sorted_idx = indices[top_part[np.argsort(-data[top_part])]]
                candidate_pairs[s1_id] = [s23_ids[idx] for idx in sorted_idx]

        if (i + batch_size) % 25000 < batch_size or end_i == total_rows:
            print(f"   Processed {end_i}/{total_rows} rows...")

    return candidate_pairs