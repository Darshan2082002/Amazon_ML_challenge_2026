import re
import pandas as pd

# Abbreviation replacement maps
LEGAL_ENTITIES = {
    r"\bincorporated\b": "inc",
    r"\bcorporation\b": "corp",
    r"\bcompany\b": "co",
    r"\blimited\b": "ltd",
    r"\bllc\b": "",
    r"\bplc\b": "",
}

ADDRESS_ABBRS = {
    r"\bstreet\b": "st",
    r"\broad\b": "rd",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bparkway\b": "pkwy",
    r"\bsuite\b": "ste",
    r"\bapartment\b": "apt",
}

def clean_text(text: str) -> str:
    """Standardize raw string values."""
    if pd.isna(text) or text is None:
        return ""
    
    # Cast, lower, remove non-alphanumeric except spaces
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Strip multi-spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_name(name: str) -> str:
    """Normalize company/business name."""
    clean = clean_text(name)
    if not clean:
        return ""

    for pattern, replacement in LEGAL_ENTITIES.items():
        clean = re.sub(pattern, replacement, clean)

    return re.sub(r"\s+", " ", clean).strip()

def normalize_address(address: str) -> str:
    """Normalize physical address strings."""
    clean = clean_text(address)
    if not clean:
        return ""

    for pattern, replacement in ADDRESS_ABBRS.items():
        clean = re.sub(pattern, replacement, clean)

    return re.sub(r"\s+", " ", clean).strip()

def preprocess_dataframe(df: pd.DataFrame, id_col: str, name_col: str, addr_col: str, country_col: str = None) -> pd.DataFrame:
    """Processes raw source DataFrames and extracts clean fields."""
    processed = pd.DataFrame()
    
    processed["entity_id"] = df[id_col].astype(str).str.strip()
    processed["clean_name"] = df[name_col].apply(normalize_name)
    processed["clean_addr"] = df[addr_col].apply(normalize_address)
    
    if country_col and country_col in df.columns:
        processed["country"] = df[country_col].apply(clean_text)
    else:
        processed["country"] = ""
        
    return processed