from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional
from numpy import ndarray
from nova.utils.decorators import no_inplace_op
from nova.autograd._ops.utils import unbroadcasting
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients, LossReduction
    from nova.autograd.engine import Context

__all__ = ["MSELoss", "BCELoss", "BCEWithLogitsLoss"]


def reduce(
    loss: ndarray,
    reduction_mode: LossReduction = "mean",
) -> ndarray:
    """
    Applies reduction to loss tensor based on specified mode.

    Args:
        loss: Unreduced loss array.
        reduction_mode: Type of reduction to apply.
            - 'none': No reduction, returns full loss tensor
            - 'sum': Sums all elements
            - 'mean': Averages all elements
    Returns:
        Reduced loss.

    Raises:
        ValueError: If reduction_mode is invalid.
    """
    if reduction_mode == "none":
        return loss
    elif reduction_mode == "sum":
        return np.sum(loss)
    elif reduction_mode == "mean":
        return np.mean(loss)
    else:
        raise ValueError(
            f"reduction expect ('sum','mean','none'), got '{reduction_mode}'"
        )


sigmoid = lambda input: 1.0 / (1.0 + np.exp(-input))  # noqa: E731


@no_inplace_op
class BCEWithLogitsLoss(Function):
    """
    Binary Cross Entropy Loss with Logits (numerically stable).

    Combines sigmoid activation and BCE in a single, stable operation:

        loss_i = max(x_i, 0) - x_i * y_i + log(1 + exp(-|x_i|))
        With pos_weight:
            loss_i = max(x_i, 0) - x_i * y_i + (1 + (pos_weight - 1) * y_i) * log(1 + exp(-|x_i|))

    Supports optional weighting, positive class weighting, and reduction.

    Forward:
        Applies numerically stable BCE computation on logits.

    Backward:
        ∂L/∂input = sigmoid(x) - target
        Scaled by pos_weight and optional element-wise weight.

    Reduction:
        - 'mean': average over elements
        - 'sum': sum over elements
        - 'none': element-wise
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        target: ndarray,
        weight: Optional[ndarray] = None,
        reduction: LossReduction = "mean",
        pos_weight: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Compute loss_i = max(x_i, 0) - x_i * y_i + log(1 + exp(-|x_i|))
        """
        ctx.reduction = reduction
        ctx.save_for_backward(input, target, weight, pos_weight)
        ctx.saved_shapes = input.shape

        # Pre-allocate loss array
        loss = np.empty_like(input)

        # max_val = max(x, 0)
        max_val = np.maximum(input, 0, out=loss)  # Reuse loss for max_val

        # loss = max_val - input * target
        np.multiply(input, target, out=loss)
        np.subtract(max_val, loss, out=loss)

        # log_term = log(1 + exp(-abs(input)))
        log_term = np.empty_like(input)
        np.abs(input, out=log_term)
        np.negative(log_term, out=log_term)
        np.exp(log_term, out=log_term)
        np.log1p(log_term, out=log_term)

        if pos_weight is not None:
            # log_weight = 1.0 + (pos_weight - 1.0) * target
            log_weight = np.empty_like(target)
            np.subtract(pos_weight, 1.0, out=log_weight)
            np.multiply(log_weight, target, out=log_weight)
            np.add(log_weight, 1.0, out=log_weight)
            np.multiply(log_weight, log_term, out=log_weight)
            loss += log_weight
        else:
            loss += log_term

        if weight is not None:
            if weight.shape != target.shape:
                raise ValueError(
                    f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
                )
            loss *= weight

        return reduce(loss, reduction)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for binary cross entropy with logits

        Gradient:
            ∂L/∂input = sigmoid(x) - target
        """
        input, target, weight, pos_weight = ctx.saved_tensors
        input_shape = ctx.saved_shapes

        # Pre-allocate grad_input
        grad_input = np.empty_like(input)

        # sigmoid(input) = 1 / (1 + exp(-input))
        np.negative(input, out=grad_input)
        np.exp(grad_input, out=grad_input)
        np.add(grad_input, 1.0, out=grad_input)
        np.reciprocal(grad_input, out=grad_input)  # Now grad_input = sigmoid(input)

        if pos_weight is not None:
            # grad = (sig * (1 + (pos - 1) * target)) - (pos * target)
            p_target = np.multiply(pos_weight, target)
            temp = np.empty_like(grad_input)
            np.subtract(p_target, target, out=temp)
            np.add(temp, 1.0, out=temp)
            np.multiply(grad_input, temp, out=grad_input)
            grad_input -= p_target
        else:
            grad_input -= target

        if weight is not None:
            if weight.shape != target.shape:
                raise ValueError(
                    f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
                )
            grad_input *= weight

        grad_input *= grad_output
        if ctx.reduction == "mean":
            grad_input /= input.size

        return (unbroadcasting(grad_input, input_shape), None)


@no_inplace_op
class BCELoss(Function):
    """
    Binary Cross Entropy (BCE) Loss operation.

    Computes element-wise binary cross entropy:

        L = -[target * log(input + eps) + (1 - target) * log(1 - input + eps)]

    Supports optional element-wise weighting and reduction modes.

    Forward:
        loss_i = - (y_i * log(x_i + ε) + (1 - y_i) * log(1 - x_i + ε))
        If weight is provided: loss_i = loss_i * weight_i

    Backward:
        ∂L/∂input = (sigmoid derivative) = [(1 - target)/(1 - input + eps)] - [target/(input + eps)]

    Reduction:
        - 'mean': Average over all elements
        - 'sum': Sum over all elements
        - 'none': Return element-wise loss
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        target: ndarray,
        weight: Optional[ndarray] = None,
        reduction: LossReduction = "mean",
    ) -> ndarray:
        """Compute BCE = -(target * log(input) + (1 - target) * log(1 - input))"""
        ctx.reduction = reduction
        ctx.save_for_backward(input, target, weight)
        ctx.saved_shapes = input.shape

        eps = 1e-8
        ctx.save_for_backward(input, target, weight)
        ctx.saved_shapes = input.shape

        # loss = -(target * log(input) + (1-target) * log(1-input))
        loss = -(
            target * np.log(input + eps) + (1.0 - target) * np.log(1.0 - input + eps)
        )

        if weight is not None:
            if weight.shape != target.shape:
                raise ValueError(
                    f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
                )
            loss *= weight
        return reduce(loss, reduction)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for Binary Cross Entropy (BCE) loss.

        Gradient:
            ∂L/∂input  = [(1 - target) / (1 - input + ε)]  -  [target / (input + ε)]
        """

        input, target, weight = ctx.saved_tensors
        input_shape = ctx.saved_shapes
        eps = 1e-8

        # Pre-allocate grad_input
        grad_input = np.empty_like(input)

        # denom = input * (1 - input)
        denom = np.empty_like(input)
        np.subtract(1.0, input, out=denom)
        np.multiply(input, denom, out=denom)

        # grad_input = (input - target) / denom
        np.subtract(input, target, out=grad_input)
        np.divide(grad_input, denom + eps, out=grad_input)

        if weight is not None:
            grad_input *= weight

        grad_input *= grad_output
        if ctx.reduction == "mean":
            grad_input /= input.size

        return (unbroadcasting(grad_input, input_shape), None)


@no_inplace_op
class MSELoss(Function):
    """
    Mean Squared Error (MSE) / L2 Loss operation.

    Computes the squared difference between each element in the input and target.
    This implementation is an 'atomic' operation, which is numerically more
    stable than computing (a - b) ** 2 through separate subtraction and
    power operations.

    Forward:
        L = (input - target)^2
        If weight is provided: L = weight * (input - target)^2

    Backward:
        ∂L/∂input  =  2 * (input - target) * grad_output

    Reduction:
        - 'mean': Gradients are scaled by 1/N (N = total number of elements).
        - 'sum':  Gradients are not scaled.
        - 'none': Gradients are returned element-wise.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        target: ndarray,
        reduction: LossReduction = "mean",
        weight: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Compute (input - target) ** 2.
        """
        ctx.reduction = reduction
        ctx.save_for_backward(input, target, weight)
        ctx.saved_shapes = input.shape

        diff = input - target
        loss = np.square(diff, out=diff)

        if weight is not None:
            if weight.shape != target.shape:
                raise ValueError(
                    f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
                )
            loss *= weight

        return reduce(loss, reduction)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for mse loss.

        Gradient:
            ∂L/∂input  =  2 * (input - target) * grad_output
        """
        input, target, weight = ctx.saved_tensors
        input_shape = ctx.saved_shapes

        # Pre-allocate and compute: grad_input = 2 * (input - target)
        grad_input = np.empty_like(input)
        np.subtract(input, target, out=grad_input)
        grad_input *= 2.0

        if weight is not None:
            grad_input *= weight

        if ctx.reduction == "mean":
            grad_input /= input.size

        grad_input *= grad_output
        return (unbroadcasting(grad_input, input_shape), None)
