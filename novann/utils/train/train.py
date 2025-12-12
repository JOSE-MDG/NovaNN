from novann.utils.log_config import logger
from novann.utils.decorators import chronometer
from novann.utils.data import DataLoader
from novann.model import Sequential
from novann._typing import Optimizer, LossFunc
from typing import Callable


@chronometer
def train(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    net: Sequential,
    optimizer: Optimizer,
    loss_fn: LossFunc,
    epochs: int,
    show_logs_every: int = 0,
    metric: Callable[[Sequential, DataLoader], float] = None,
    verbose: bool = True,
    get_model: bool = True,
):
    """
    Runs the full training loop for a given model, optimizer, loss function
    and dataloaders. Optionally logs progress and evaluates a metric on the
    validation dataset after each epoch.

    Args:
        train_loader (DataLoader): Dataloader providing training batches
            `(input, target)`.
        eval_loader (DataLoader): Dataloader used for validation evaluation.
        net (Sequential): Model to be trained. Must implement `forward`,
            `backward`, `train`, and `eval`.
        optimizer (Optimizer): Optimizer responsible for updating parameters.
        loss_fn (LossFunc): Loss function returning `(loss, grad)` where
            `grad` is the gradient of the loss w.r.t. model outputs.
        epochs (int): Number of training epochs.
        show_logs_every (int, optional): Frequency (in epochs) at which logs
            are printed. If `0`, logs are shown every epoch.
        metric (Callable[[Sequential, DataLoader], float], optional):
            Validation metric function. If provided, it is computed at the
            end of each epoch in evaluation mode.
        verbose (bool, optional): If True, enables logging.
        get_model (bool, optional): If True, returns the trained model.

    Returns:
        Sequential or None: Returns the trained model if `get_model=True`,
        otherwise returns nothing.

    Notes:
        - Gradients are reset to None before each backward pass.
        - The loss reported in logs corresponds to the last batch of the epoch.
        - The model is switched to training mode before each epoch and to
          evaluation mode during validation metric computation.
    """

    # Training mode
    net.train()

    # Training loop
    for epoch in range(epochs):
        for input, target in train_loader:
            # Set gradients to None
            optimizer.zero_grad()

            # Forward pass
            outputs = net(input)

            # Compute loss and gradients
            cost, grad = loss_fn(outputs, target)

            # Backward pass
            net.backward(grad)

            # Update parameters
            optimizer.step()

        # Validation result after each epoch
        if metric is not None:
            net.eval()
            result = metric(net, eval_loader)
            net.train()

        if verbose:
            if show_logs_every > 0:
                if (epoch + 1) % show_logs_every == 0:
                    net.train()
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs}, Loss: {cost:.4f}, "
                        f"Validation {metric.__name__}: {result:.4f}"
                    )
            else:
                net.train()
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {cost:.4f}, "
                    f"Validation {metric.__name__}: {result:.4f}"
                )

    if get_model:
        return net
