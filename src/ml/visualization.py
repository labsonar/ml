"""
Module for data visualization.

This module provides functions to visualize high-dimensional data using t-distributed Stochastic
Neighbor Embedding (t-SNE), a dimensionality reduction technique particularly useful for visualizing
complex datasets in a lower-dimensional space.
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as sklearn

def export_tsne(data: np.ndarray, labels: np.ndarray, filename: str) -> None:
    """
    Export data visualized using t-SNE to a file.

    Parameters:
        data (np.ndarray): The data to visualize, shape (n_samples, n_features).
        labels (np.ndarray): Labels corresponding to each data point, shape (n_samples,).
        filename (str): The filename including path to save the plot.
    """
    tsne = sklearn.TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        plt.scatter(tsne_data[indices, 0], tsne_data[indices, 1],
                    label=str(label), cmap='jet', s=10)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')