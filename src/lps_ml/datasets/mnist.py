"""Module do manage acess to MNIST dataset
"""
import multiprocessing

import torch
import torch.utils.data as torch_data
import torchvision.datasets as torch_set
import torchvision.transforms as torch_trf

import lps_ml.core as ml_core

class MNIST(ml_core.BaseDataModule):
    """ Simple MNIST DataModule. """

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = None,
                 binary: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() // 2)
        self.binary = binary

        self.mnist_test = None
        self.mnist_predict = None
        self.mnist_train = None
        self.mnist_val = None

    def get_n_targets(self) -> int:
        return 1 if self.binary else 10

    def setup(self, stage: str):

        transform = torch_trf.transforms.Compose([
            torch_trf.transforms.ToTensor(),
            # torch_trf.transforms.Normalize((0.1307,), (0.3081,)), # original scale
            torch_trf.transforms.Lambda(lambda x: (x * 2) - 1) # scale to [-1,1] more like audio
        ])

        self.mnist_test = torch_set.MNIST(self.data_dir, train=False,
                                          download=True, transform=transform)
        self.mnist_predict = torch_set.MNIST(self.data_dir, train=False,
                                             download=True, transform=transform)

        mnist_full = torch_set.MNIST(self.data_dir, train=True,
                                        download=True, transform=transform)

        self.mnist_train, self.mnist_val = torch_data.random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

        if self.binary:

            def _to_binary(dataset):
                if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
                    dataset.dataset.targets = dataset.dataset.targets % 2
                elif hasattr(dataset, "targets"):
                    dataset.targets = dataset.targets % 2

            _to_binary(self.mnist_train)
            _to_binary(self.mnist_val)
            _to_binary(self.mnist_test)
            _to_binary(self.mnist_predict)

    def train_dataloader(self):
        return torch_data.DataLoader(self.mnist_train,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     shuffle=True)

    def val_dataloader(self):
        return torch_data.DataLoader(self.mnist_val,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers)

    def test_dataloader(self):
        return torch_data.DataLoader(self.mnist_test,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch_data.DataLoader(self.mnist_predict,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers)
