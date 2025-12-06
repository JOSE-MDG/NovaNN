"""Loss functions for training and evaluating neural networks.

This module provides a collection of loss functions for both classification
and regression tasks. All functions compute the discrepancy between predictions
and targets and provide gradients for backpropagation in a functional style.
"""

from .functional import CrossEntropyLoss, MSE, MAE, BinaryCrossEntropy

__all__ = ["CrossEntropyLoss", "MSE", "MAE", "BinaryCrossEntropy"]
