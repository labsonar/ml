"""
General Utilities Module

This module provides utility functions
"""
import os
import random
import datetime
import shutil
import collections
import contextlib

import numpy as np

import torch

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig


def set_seed():
    """ Set random seed for reproducibility. """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def backup_folder(base_dir, time_str_format = "%Y%m%d-%H%M%S"):
    """Method to backup all files in a folder in a timestamp based folder

    Args:
        base_dir (_type_): Directory to backup
        time_str_format (str, optional): Time string format for the folder.
            Defaults to "%Y%m%d-%H%M%S".
    """
    backup_dir = os.path.join(base_dir, datetime.datetime.now().strftime(time_str_format))
    os.makedirs(backup_dir)

    contents = os.listdir(base_dir)
    for item in contents:
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path):
            try:
                datetime.datetime.strptime(item, time_str_format)
                continue
            except ValueError:
                pass
        shutil.move(item_path, backup_dir)

def format_header(size: int, title: str = ""):
    """ Create a string to use as a separator in a log. """
    if len(title) == 0:
        return f"{'='*60}"

    before = (size - len(title) - 2)//2
    after = size - before - len(title) - 2
    return f"{'='*before} {title} {'='*after}"

@contextlib.contextmanager
def evaluating(model):
    """
    Context manager that temporarily sets a PyTorch model to evaluation mode.

    This utility preserves the original training state of the model.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

def save_wav(data: np.ndarray | torch.Tensor,
             fs: int | lps_qty.Frequency,
             filename: str) -> None:
    """Export a .wav file

    Args:
        signal (np.ndarray, torch.Tensor): Signal to be normalized and exported
        fs (int, lps_qty.Frequency): Sample Frequency
        filename (str): Filename
    """

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    lps_sig.save_wav(data, fs, filename)

def save_convert_wav(data: np.ndarray | torch.Tensor,
             fs: int | lps_qty.Frequency,
             filename: str) -> None:
    """Export a .wav file

    Args:
        signal (np.ndarray, torch.Tensor): Signal to be normalized and exported
        fs (int, lps_qty.Frequency): Sample Frequency
        filename (str): Filename
    """

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    lps_sig.save_convert_wav(data, fs, filename)

def shortest_relative_path(paths: list[str]) -> list[tuple[str, str]]:
    """
        Returns a list of tuples (name, path) where name is the shortest unique relative path
        for each file in paths.
    """
    paths = [os.path.abspath(p) for p in paths]

    depth = 1

    while True:
        name_map = collections.defaultdict(list)

        for p in paths:
            parts = p.split(os.sep)
            suffix_parts = parts[-depth:]

            suffix_parts = list(suffix_parts)
            suffix_parts[-1] = os.path.splitext(suffix_parts[-1])[0]

            name = "/".join(suffix_parts)
            name_map[name].append(p)

        duplicates = {k: v for k, v in name_map.items() if len(v) > 1}

        if not duplicates:
            break

        depth += 1

    result = []
    for name, ps in name_map.items():
        for p in ps:
            result.append((name, p))

    result.sort(key=lambda x: x[0])

    return result
