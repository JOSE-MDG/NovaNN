import numpy as np
import pandas as pd


def normalize(x_data: np.ndarray, x_mean: float, x_std: float) -> np.ndarray:
    """Normalize input data using provided mean and standard deviation.

    Args:
        x_data: Input data array to normalize.
        x_mean: Mean value for normalization.
        x_std: Standard deviation for normalization.
    Returns:
        Normalized data array.
    """
    return (x_data - x_mean) / x_std


def split_features_and_labels(
    df: pd.DataFrame, label_column: str = "label"
) -> tuple[np.ndarray, np.ndarray]:
    """Split a tabular dataset into features and integer labels.

    This helper is robust to whether the CSV includes an explicit "label"
    column header or not. If a "label" column exists, it is used; otherwise,
    the first column is treated as the label column.
    """
    if label_column in df.columns:
        y = df[label_column].to_numpy(dtype=np.int32)
        x = df.drop(columns=[label_column]).to_numpy(dtype=np.float32)
    else:
        # Assume the first column holds labels when no explicit header is present
        y = df.iloc[:, 0].to_numpy(dtype=np.int32)
        x = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return x, y
