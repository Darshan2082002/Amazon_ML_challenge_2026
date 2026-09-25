import pandas as pd

def preprocess_source_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes names and addresses for S1, S2, and S3."""
    df.fillna("", inplace=True)
    
    # 1. Clean names
    df['name_clean'] = df['business_name'].astype(str).str.lower().str.strip()
    df['name_clean'] = df['name_clean'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['name_clean'] = df['name_clean'].str.replace(
        r'\b(incorporated|corporation|limited|private|company|inc|corp|ltd|pvt|co)\b', 
        '', regex=True
    )
    df['name_clean'] = df['name_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 2. Clean addresses (Added to prevent memory explosion)
    df['address_clean'] = df['business_address'].astype(str).str.lower().str.strip()
    df['address_clean'] = df['address_clean'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['address_clean'] = df['address_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 3. Clean country
    df['country'] = df['country'].astype(str).str.strip().str.lower()
    
    return df