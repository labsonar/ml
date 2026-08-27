"""
Metrics Module
"""
import typing

import pandas as pd
import sklearn.metrics as sk_metrics

import torch
import torch.utils.data as torch_data

import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_general

def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: torch_data.DataLoader,
    device: typing.Optional[torch.device] = None,
) -> typing.Tuple[float, float]:
    """
    Evaluate a binary or multiclass classifier and return
    (balanced_accuracy, macro_f1).

    Supports models whose forward() returns:
        - shape (B,)    -> binary logits/probabilities (threshold at 0.5)
        - shape (B, 1)  -> binary logits/probabilities (threshold at 0.5)
        - shape (B, C)  -> multiclass logits (argmax)

    Returns:
        Tuple[float, float]: (balanced_accuracy, macro_f1)
    """
    device = device or ml_device.get_available_device()
    model.to(device)

    with ml_general.evaluating(model):

        y_true = []
        y_pred = []

        with torch.inference_mode():
            for x, y in dataloader:

                if isinstance(x, list):
                    raise RuntimeError("evaluate_classifier expects a non-paired DataLoader ")

                x = x.to(device)
                y = y.to(device)

                output = model(x)

                if output.ndim == 1:
                    pred = (output >= 0.5).long()
                elif output.ndim == 2 and output.shape[1] == 1:
                    pred = (output[:, 0] >= 0.5).long()
                else:
                    pred = torch.argmax(output, dim=1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        balanced_accuracy = sk_metrics.balanced_accuracy_score(y_true, y_pred)
        macro_f1 = float(sk_metrics.f1_score(y_true, y_pred, average="macro"))

        return balanced_accuracy, macro_f1


def evaluate_splits(
    model: torch.nn.Module,
    loaders: typing.Dict[str, torch_data.DataLoader],
    device: typing.Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Evaluate a classifier across multiple named splits (e.g. "train"/"val"/"test") and return
    a DataFrame indexed by split name, with columns "balanced_accuracy" and "macro_f1".
    """

    rows = []
    for split_name, loader in loaders.items():
        balanced_accuracy, macro_f1 = evaluate_classifier(model, loader, device=device)
        rows.append({
            "split": split_name,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        })

    df = pd.DataFrame(rows)
    return df.set_index("split")
