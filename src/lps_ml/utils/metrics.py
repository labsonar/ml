"""
Metrics Module
"""
import typing
import enum
import os

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import scipy.linalg as sci_alg
import ot
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib as tikz

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

def save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        filename,
        title,
):
    """
    Save a confusion matrix as PNG and TikZ.
    """

    cm = sk_metrics.confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    fig.tight_layout()

    # Save PNG
    fig.savefig(filename, dpi=150)

    # Save TikZ using the same filename
    tikz_filename = os.path.splitext(filename)[0] + ".tikz"
    tikz.save(tikz_filename)

    plt.close(fig)

def distribution_statistics(
    values: typing.Union[np.ndarray, typing.Sequence[float]],
) -> typing.Dict[str, float]:
    """
    Basic distributional summary (mean/std/median/max/min) of an array of
    values.
    """
    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
    }

DistributionValues = np.ndarray | typing.Sequence[float]
NestedResults = dict[str, DistributionValues | "NestedResults"]
def nested_distribution_statistics(
        results: NestedResults,
        columns: list[str] | None = None
):
    """Summarize arbitrarily nested dictionaries into a DataFrame.

    summarize_nested({
        "train": {
            "real": {
                "S1": s1_train,
                "S2": s2_train,
            },
            "fake": {
                "S1": s1_fake,
                "S2": s2_fake,
            },
        },
    })

    """

    rows = []

    def _walk(data, path):
        if isinstance(data, dict):
            for key, value in data.items():
                _walk(value, path + [key])
            return

        row = dict(zip(columns or [f"level_{i}" for i in range(len(path))],
                       path))
        row.update(distribution_statistics(data))
        rows.append(row)

    _walk(results, [])

    return pd.DataFrame(rows)



def latent_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((x1 - x2) ** 2) + 1e-8).float()

def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """ Root-mean-square distance between two waveforms/arrays. """
    return float(np.sqrt(np.mean((x - y) ** 2)))

def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """ Pearson correlation between two flattened arrays. """
    return float(np.corrcoef(x.flatten(), y.flatten())[0, 1])

def wasserstein(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    2-Wasserstein distance between two empirical point clouds (uniform
    weights), via optimal transport. Requires the `pot` package.
    Used by test_ldm_eval.py.
    """
    n_a = len(points_a)
    n_b = len(points_b)

    a = np.ones((n_a,)) / n_a
    b = np.ones((n_b,)) / n_b

    cost = ot.dist(points_a, points_b, metric='sqeuclidean')
    wasserstein_sq = ot.emd2(a, b, cost)

    return float(np.sqrt(wasserstein_sq))

def latent_fid(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Frechet-distance (FID-style) between two Gaussian-fitted point clouds.
    Used by test_ldm_eval.py.
    """

    mu_a = np.mean(points_a, axis=0)
    mu_b = np.mean(points_b, axis=0)

    cov_a = np.cov(points_a, rowvar=False)
    cov_b = np.cov(points_b, rowvar=False)

    covmean = sci_alg.sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_a - mu_b

    fid = diff @ diff + np.trace(cov_a + cov_b - 2 * covmean)
    return float(fid)

class DistanceMetric(enum.Enum):
    """ Class to centralize distance metrics in latent space """
    LATENT = latent_distance
    L2 = l2_distance
    CORRELATION = correlation
    WASSERSTEIN = wasserstein
    LATENT_FID = latent_fid

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

def alignment_scores(
    c: typing.Union[np.ndarray, torch.Tensor],
    t: typing.Union[np.ndarray, torch.Tensor],
    g: typing.Union[np.ndarray, torch.Tensor],
    metric: DistanceMetric,
    eps: float = 1e-8,
) -> typing.Tuple[
    typing.Union[float, np.ndarray, torch.Tensor],
    typing.Union[float, np.ndarray, torch.Tensor],
]:
    """
    Compute S1/S2 alignment scores between conditioning, target and generated
    samples using the selected distance metric.

    Returns:
        Tuple containing (S1, S2).

    S1 = 1 - d(g, t) / d(c, t)

    S2 = d(c, t) / (d(c, g) + d(g, t))
    """
    d_ct = metric(c, t)
    d_gt = metric(g, t)
    d_cg = metric(c, g)

    s1 = 1.0 - (d_gt / (d_ct + eps))
    s2 = d_ct / (d_cg + d_gt + eps)

    return s1, s2
