import numpy as np
import pandas as pd

def clean_and_prepare_data(df):
    # Convert all columns to numeric, coercing errors
    df = df.apply(pd.to_numeric, errors='coerce')
    df.sort_index(inplace=True)

    # Handle NaNs
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill

    # Handle infinities
    if np.isinf(df.values).any():
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    return df