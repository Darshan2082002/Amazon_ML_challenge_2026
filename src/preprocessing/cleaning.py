import pandas as pd


def preprocess_dataframe(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Fast vectorized text cleaning using Pandas string methods."""
    cleaned_df = df.copy()
    for col in text_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = (
                cleaned_df[col]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.replace(r"[^\w\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    return cleaned_df