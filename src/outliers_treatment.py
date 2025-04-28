"""
Utility functions for the treatment of outliers in dataframes.
"""

import numpy as np
import pandas as pd


def remove_outliers_iqr(df: pd.DataFrame, column_name: list[str]) -> pd.DataFrame:
    """
    Removes outliers based on the IQR method.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (list[str]): The list of column names.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """

    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return df_filtered


def remove_outliers_zscore(df: pd.DataFrame, column_name: list[str], threshold: float = 3) -> pd.DataFrame:
    """
    Removes outliers based on the z-score method.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (list[str]): The list of column names.
        threshold (): The z-score threshold to identify outliers.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """

    z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())

    df_filtered = df[z_scores < threshold]

    return df_filtered


def cap_outliers_iqr(df: pd.DataFrame, column_name: list[str]) -> pd.DataFrame:
    """
    Caps outliers based on the IQR method.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (list[str]): The list of column names.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """

    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])

    return df
