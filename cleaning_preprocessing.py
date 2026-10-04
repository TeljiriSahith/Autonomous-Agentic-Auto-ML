import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

# =====================================================================
# 1. IMPUTATION FUNCTIONS (Handling Missing Values)
# =====================================================================

def impute_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replaces missing values in a numeric column with its mean."""
    if col in df.columns:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    return df

def impute_median(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replaces missing values in a numeric column with its median (outlier resistant)."""
    if col in df.columns:
        med_val = df[col].median()
        df[col] = df[col].fillna(med_val)
    return df

def impute_mode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replaces missing values in a categorical/text column with the most frequent value."""
    if col in df.columns and not df[col].mode().empty:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    return df

def impute_constant(df: pd.DataFrame, col: str, fill_value="Missing") -> pd.DataFrame:
    """Replaces missing values with a designated constant placeholder."""
    if col in df.columns:
        df[col] = df[col].fillna(fill_value)
    return df

# =====================================================================
# 2. COLUMN REMOVAL & CLEANING FUNCTIONS
# =====================================================================

def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drops an irrelevant, ID, or constant column from the DataFrame."""
    if col in df.columns:
        df = df.drop(columns=[col])
    return df

# =====================================================================
# 3. ENCODING FUNCTIONS (Handling Categorical Data)
# =====================================================================

def encode_one_hot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Applies One-Hot Encoding to a nominal categorical column."""
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df

def encode_label(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Applies Label Encoding to an ordinal or binary categorical column."""
    if col in df.columns:
        le = LabelEncoder()
        # Convert to string to avoid mixed type exceptions
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# =====================================================================
# 4. SCALING FUNCTIONS (Handling Numerical Features)
# =====================================================================

def scale_standard(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Standardizes values to mean=0, std=1 (StandardScaler)."""
    if col in df.columns:
        scaler = StandardScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
    return df

def scale_minmax(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normalizes values into the range [0, 1] (MinMaxScaler)."""
    if col in df.columns:
        scaler = MinMaxScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
    return df

def scale_robust(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Scales using IQR, making it robust against extreme outliers."""
    if col in df.columns:
        scaler = RobustScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
    return df

# =====================================================================
# 5. FUNCTION REGISTRY MAP
# =====================================================================
# This dictionary maps the exact function name strings to their Python callables

OPERATIONS_MAP = {
    # Imputation operations
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_mode": impute_mode,
    "impute_constant": impute_constant,
    # Column operations
    "drop_column": drop_column,
    # Encoding operations
    "encode_one_hot": encode_one_hot,
    "encode_label": encode_label,
    # Scaling operations
    "scale_standard": scale_standard,
    "scale_minmax": scale_minmax,
    "scale_robust": scale_robust,
}

# =====================================================================
# 6. MASTER DISPATCHER / EXECUTION LOOP
# =====================================================================

def execute_cleaning_pipeline(df: pd.DataFrame, instruction_list: list) -> pd.DataFrame:
    """
    Iterates through instructions provided by LLM and executes mapped functions.
    
    Expected instruction format:
    [
        {"column": "customer_id", "operation": "drop_column"},
        {"column": "age", "operation": "impute_median"},
        {"column": "age", "operation": "scale_standard"},
        {"column": "country", "operation": "encode_one_hot"}
    ]
    """
    processed_df = df.copy()

    for step in instruction_list:
        col = step.get("column")
        op_name = step.get("operation")

        if op_name in OPERATIONS_MAP:
            print(f"-> Executing '{op_name}' on column '{col}'")
            func = OPERATIONS_MAP[op_name]
            processed_df = func(processed_df, col)
        else:
            print(f"[Warning] Operation '{op_name}' is not registered in OPERATIONS_MAP. Skipping.")

    return processed_df