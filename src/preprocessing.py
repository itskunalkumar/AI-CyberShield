"""Cleaning utilities: infinite/missing value handling using training-only statistics."""
import numpy as np
import pandas as pd


def clean_frame(df: pd.DataFrame, features: list[str], medians: pd.Series) -> pd.DataFrame:
    """Select features in a fixed order, replace inf with NaN and fill with training medians."""
    x = df.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
    return x.fillna(medians).astype(float)


def fit_medians(df: pd.DataFrame, features: list[str]) -> pd.Series:
    medians = df[features].replace([np.inf, -np.inf], np.nan).median()
    return medians.fillna(0.0)
