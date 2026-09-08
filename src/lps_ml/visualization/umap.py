"""
UMAP Visualization Module
"""
import os
import pickle
import typing
import umap

import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as sk_manifold
import tikzplotlib as tikz

def umap_string(model_name: str, output_dir: str) -> str:
    aux_name = model_name.replace("/", "_")
    umap_path = os.path.join(output_dir, f"umap_{aux_name}.pkl")
    return umap_path

def plot_2d_embedding(
    groups: typing.Dict[str, np.ndarray],
    filename: str,
    figsize: typing.Tuple[int, int] = (8, 8),
    point_size: int = 8,
    alpha: float = 0.6,
    highlight: typing.Optional[str] = None,
    highlight_color: str = "black",
) -> None:
    """
    Scatter-plot 2D embedded points, one color per named group, and save
    to `filename`.
    """
    plt.figure(figsize=figsize)

    for label, points in groups.items():
        if label == highlight:
            continue
        plt.scatter(points[:, 0], points[:, 1], s=point_size, alpha=alpha, label=str(label))

    if highlight is not None and highlight in groups:
        points = groups[highlight]
        plt.scatter(points[:, 0], points[:, 1], s=point_size + 4,
                    color=highlight_color, label=str(highlight))

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    tikz_filename = os.path.splitext(filename)[0] + ".tikz"
    tikz.save(tikz_filename)
    plt.close()

def export_umap(
    data: np.ndarray,
    labels: np.ndarray,
    filename: str,
    model_filename: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
):
    """
    Fit a UMAP reducer on `data`, save the fitted reducer (pickle) and
    save a scatter plot colored by `labels`.
    """

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )

    embedding = reducer.fit_transform(data)

    with open(model_filename, "wb") as f:
        pickle.dump(reducer, f)

    groups = {str(label): embedding[labels == label] for label in np.unique(labels)}
    plot_2d_embedding(groups, filename)

    trustworthiness = sk_manifold.trustworthiness(
        X=data,
        X_embedded=embedding,
        n_neighbors=n_neighbors
    )

    return reducer, embedding, trustworthiness

def load_umap(model_path: str):
    """ Load a previously-fitted UMAP reducer saved via `export_umap`. """
    with open(model_path, "rb") as f:
        return pickle.load(f)
