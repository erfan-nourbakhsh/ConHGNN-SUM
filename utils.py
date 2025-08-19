import torch
from tools.logger import *


def set_device(hps):
    """
    Set the device (CPU or GPU) for PyTorch based on hyperparameters and availability.

    Args:
        hps: Hyperparameter/config object that contains:
            - cuda: bool, whether GPU usage is requested
            - gpu: str or None, GPU ID
            - device: attribute to be set to torch.device

    Returns:
        hps: Updated hyperparameter object with `device` set.
    """
    # Check if CUDA is requested, a GPU ID is provided, and CUDA is available
    if hps.cuda and hps.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")  # Set device to first GPU
        logger.info("[INFO] Use cuda")  # Log the choice
    else:
        device = torch.device("cpu")  # Fallback to CPU
        logger.info("[INFO] Use CPU")  # Log the choice

    # Update the hps object with the selected device
    hps.device = device

    return hps  # Return updated hyperparameters
