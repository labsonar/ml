"""Module to unify torch device access
"""
import torch
import typing

def get_available_device() -> torch.device:
    """
    Get the available device for computation.

    Returns:
        torch.device: The available device, either 'cuda' (GPU) or 'cpu'.
    """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_available_device():
    """ Print the available device for computation. """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("No GPU available, using CPU.")


def load_ts_model(path: str, device: typing.Optional[torch.device] = None) -> \
    torch.jit.ScriptModule:
    """
    Load a TorchScript (.ts) model onto a device and set it to eval mode.
    """
    device = device or get_available_device()
    model = torch.jit.load(path)
    model.to(device)
    model.eval()
    return model
