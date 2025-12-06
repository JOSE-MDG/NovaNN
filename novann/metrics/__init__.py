"""Evaluation metrics for model performance assessment.

This module provides various metrics to quantify model performance during
training and evaluation, including accuracy measures for classification and
R² score for regression tasks.
"""

from .metrics import *

__all__ = ["accuracy", "binary_accuracy", "r2_score"]
