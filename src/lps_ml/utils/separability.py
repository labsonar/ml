import abc
import enum
import typing
import numpy as np

import sklearn.metrics as sk_metrics

import torch
import torch.utils.data as torch_data


def dataloader_to_numpy(old_loader: torch_data.DataLoader, batch_size: int = 1024) -> np.ndarray:
    """ Converts DataLoader to NumPy using a new loader with a batch the size of the dataset."""

    use_sampler = old_loader.sampler is not None

    new_loader = torch_data.DataLoader(
        old_loader.dataset,
        batch_size=batch_size,
        shuffle=(not use_sampler) and getattr(old_loader, "shuffle", False),
        sampler=old_loader.sampler if use_sampler else None,
        num_workers=old_loader.num_workers,
        pin_memory=old_loader.pin_memory,
        collate_fn=old_loader.collate_fn,
        drop_last=False
    )
    new_loader.num_workers = 0

    for batch in new_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        return x

def to_numpy(data: typing.Union[np.ndarray, torch.Tensor, torch_data.DataLoader]) -> np.ndarray:
    " Unifies input data for np.ndarray "

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    if isinstance(data, torch_data.DataLoader):
        return dataloader_to_numpy(data)

    raise TypeError(f"Unsupported type: {type(data)}")

def flatten_if_needed(x: np.ndarray) -> np.ndarray:
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
        # return x.mean(axis=-1)
    return x


class SeparabilityMetric(abc.ABC):

    def apply(
        self,
        data_a: typing.Union[np.ndarray, torch.Tensor, torch_data.DataLoader],
        data_b: typing.Union[np.ndarray, torch.Tensor, torch_data.DataLoader],
    ) -> float:

        x_a = to_numpy(data_a)
        x_b = to_numpy(data_b)

        x = np.concatenate([x_a, x_b], axis=0)

        labels = np.concatenate([
            np.zeros(len(x_a)),
            np.ones(len(x_b))
        ])

        x = flatten_if_needed(x)

        return self.compute(x, labels)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abc.abstractmethod
    def compute(self, x: np.ndarray, labels: np.ndarray) -> float:
        pass

    @staticmethod
    def compare_dataloaders(
        data_a: typing.Union[np.ndarray, torch.Tensor, torch_data.DataLoader],
        data_b: typing.Union[np.ndarray, torch.Tensor, torch_data.DataLoader],
        metrics: typing.List['SeparabilityMetric']
    ) -> dict:

        results = {}

        for metric in metrics:
            value = metric.apply(data_a, data_b)
            results[str(metric)] = value

        return results

class SilhouetteScore(SeparabilityMetric):

    def compute(self, x: np.ndarray, labels: np.ndarray) -> float:
        return float(sk_metrics.silhouette_score(x, labels))

class DaviesBouldinIndex(SeparabilityMetric):

    def compute(self, x: np.ndarray, labels: np.ndarray) -> float:
        return float(sk_metrics.davies_bouldin_score(x, labels))

class Separability(enum.Enum):
    SILHOUETTE = SilhouetteScore
    DAVIES_BOULDIN = DaviesBouldinIndex

    def get(self) -> SeparabilityMetric:
        return self.value()

    def apply(self, data_a, data_b) -> float:
        return self.value().apply(data_a, data_b)
