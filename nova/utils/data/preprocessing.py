from __future__ import annotations
import pandas as pd
import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor


def normalize(
    x_data: ndarray | Tensor, x_mean: float | Tensor, x_std: float | Tensor
) -> ndarray | Tensor:
    """
    Normalize input data using provided mean and standard deviation.

    Args:
        x_data (ndarray | Tensor): Input data to normalize.
        x_mean (float): Mean value for normalization.
        x_std (float): Standard deviation for normalization.

    Returns:
        ndarray or Tensor: Normalized data.

    Examples:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> normalize(x, x_mean=2.0, x_std=1.0)
        array([-1.,  0.,  1.])
    """
    return (x_data - x_mean) / x_std


def split_features_and_labels(
    df: pd.DataFrame, label_column: str = "label"
) -> tuple[ndarray, ndarray]:
    """
    Split a tabular dataset into feature and label arrays.

    Args:
        df (pd.DataFrame): Input dataset.
        label_column (str): Name of the label column. Defaults to "label".

    Returns:
        tuple[ndarray, ndarray]: Features array and labels array (int32).

    Notes:
        - If `label_column` does not exist, the first column is assumed to be labels.
        - Features are returned as float32, labels as int32.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'label':[0,1], 'f1':[0.1,0.2], 'f2':[0.3,0.4]})
        >>> x, y = split_features_and_labels(data)
        >>> x
        array([[0.1, 0.3],
               [0.2, 0.4]], dtype=float32)
        >>> y
        array([0, 1], dtype=int32)
    """
    if label_column in df.columns:
        y = df[label_column].to_numpy(dtype=np.int32)
        x = df.drop(columns=[label_column]).to_numpy(dtype=np.float32)
    else:
        y = df.iloc[:, 0].to_numpy(dtype=np.int32)
        x = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return x, y
