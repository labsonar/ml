"""DataModule

"""
import abc
import os
import typing
import json
import multiprocessing

import tqdm
import pandas as pd
import numpy as np

import torch
import torch.utils.data as torch_data
import lightning

import lps_ml.core.cv as ml_cv
import lps_utils.hashable as utils_hash
import lps_ml.core.loader as ml_loader
import lps_ml.core.processor as ml_proc

class BaseDataModule(lightning.LightningDataModule):
    """ Basic DataModule """

    def get_sample_shape(self, subset: str = "train") -> typing.List[int]:
        """
        Returns the shape of the dataset samples.
        """
        if not getattr(self, "_has_setup", False):
            self.prepare_data()
            self.setup("fit")

        if subset == "train":
            loader = self.train_dataloader()
        elif subset == "val":
            loader = self.val_dataloader()
        elif subset == "test":
            loader = self.test_dataloader()
        elif subset == "predict":
            loader = self.predict_dataloader()
        else:
            raise ValueError(f"invalid subset: {subset}")

        x, _ = next(iter(loader))
        return list(x.shape[1:])

    @abc.abstractmethod
    def get_n_targets(self) -> int:
        """ Return the number of targets in dataset. """

class ProcessedDataset(torch_data.Dataset):
    """ Simple dataset for processed data in .npy format """

    def __init__(self, dataframe: pd.DataFrame, processed_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fragment_path = os.path.join(self.processed_dir, f"{row['id_fragment']}.npy")
        fragment = np.load(fragment_path)
        if self.transform:
            fragment = self.transform(fragment)
        return torch.from_numpy(fragment).float(), row['target']

class AudioDataModule(BaseDataModule, utils_hash.Hashable):
    """ Basic DataModule for process and load audio datasets. """

    def __init__(self,
                 file_loader: ml_loader.AudioFileLoader,
                 file_processor: ml_proc.AudioProcessor,
                 description_df: pd.DataFrame,
                 processed_dir: str,
                 batch_size: int = 32,
                 num_workers: int = None,
                 cv: ml_cv.CrossValidator = None,
                 transform=None,
                 id_column: str = "ID",
                 target_column: str = "Target"):
        super().__init__()
        self.file_loader = file_loader
        self.file_processor = file_processor
        self.description_df = description_df
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() // 2)
        self.cv = cv or ml_cv.HoldOutCV()
        self.transform = transform
        self.id_column = id_column
        self.target_column = target_column

        self.file_ids = description_df[id_column].to_list()
        self.targets = description_df[target_column].to_list()

        self.dataframe = None
        self.folds = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.processed_dir = os.path.join(processed_dir, f"{hash(self)}")
        self.csv_file = os.path.join(self.processed_dir, "description.csv")

    def _get_params(self):
        return {
            "file_loader": self.file_loader.__get_hash_base__(),
            "file_processor": self.file_processor.__get_hash_base__(),
        }

    def prepare_data(self):

        os.makedirs(self.processed_dir, exist_ok=True)

        existing_df = None
        processed_ids = set()
        fragment_idx = 0

        if os.path.exists(self.csv_file):
            existing_df = pd.read_csv(self.csv_file)
            processed_ids = set(existing_df["file_id"].unique())
            fragment_idx = existing_df["id_fragment"].max() + 1 if not existing_df.empty else 0

        pending_ids = [fid for fid in self.file_ids if fid not in processed_ids]
        if not pending_ids:
            return

        new_records = []

        for file_id in tqdm.tqdm(pending_ids, desc="Processando arquivos", ncols=120):
            fs, data = self.file_loader.load(file_id)
            fragments = self.file_processor.process(fs, data)

            for frag in fragments:
                frag_path = os.path.join(self.processed_dir, f"{fragment_idx}.npy")
                np.save(frag_path, frag)
                new_records.append({
                    "id_fragment": fragment_idx,
                    "file_id": file_id
                })
                fragment_idx += 1

        df_new = pd.DataFrame(new_records)
        if existing_df is not None:
            df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_csv(self.csv_file, index=False)
        print(f"[prepare_data] Dataset processed with {len(df)} fragments at {self.csv_file}.")

        if existing_df is None:
            description = self.__get_hash_base__()
            desc_file = os.path.join(self.processed_dir, "description.json")
            with open(desc_file, "w", encoding="utf-8") as f:
                json.dump(description, f, indent=4, ensure_ascii=False)
            print(f"[prepare_data] Description saved to {desc_file}")

    def setup(self, stage=None):
        """
        Loads the metadata CSV and generates cross-validation folds.
        """
        if not os.path.exists(self.csv_file):
            self.prepare_data()

        df = pd.read_csv(self.csv_file)

        valid_ids = set(self.file_ids)
        df = df[df["file_id"].isin(valid_ids)].reset_index(drop=True)

        id_to_target = dict(zip(self.file_ids, self.targets))
        df["target"] = df["file_id"].map(id_to_target)

        self.dataframe = df
        self.folds = self.cv.apply(self.file_ids, self.targets)
        self.set_fold(0)

    def set_fold(self, fold_idx: int):
        """
        Sets the active fold for training/validation/test split.

        Args:
            fold_idx: Index of the fold to activate.
        """
        if self.folds is None:
            raise RuntimeError("Folds not initialized. Call setup() first.")

        if not 0 <= fold_idx < len(self.folds):
            raise ValueError(f"Invalid fold index {fold_idx} (max {len(self.folds)-1}).")

        fold_map = self.folds[fold_idx]

        df = self.dataframe.copy()
        df["group"] = df["file_id"].map(fold_map)

        self.train_df = df[df["group"] == ml_cv.FoldRole.TRAIN]
        self.val_df = df[df["group"] == ml_cv.FoldRole.VALIDATION]
        self.test_df = df[df["group"] == ml_cv.FoldRole.TEST]

    def train_dataloader(self):
        return torch_data.DataLoader(
                ProcessedDataset(self.train_df, self.processed_dir, self.transform),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

    def val_dataloader(self):
        return torch_data.DataLoader(
                ProcessedDataset(self.val_df, self.processed_dir, self.transform),
                batch_size=self.batch_size,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return torch_data.DataLoader(
                ProcessedDataset(self.test_df, self.processed_dir, self.transform),
                batch_size=self.batch_size,
                num_workers=self.num_workers)

    def all_dataloader(self, shuffle: bool = False):
        return torch_data.DataLoader(
            ProcessedDataset(self.dataframe, self.processed_dir, self.transform),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )

    def get_n_targets(self) -> int:
        return len(set(self.targets))

    def to_df(self) -> pd.DataFrame:
        """ Returns dataset information as a DataFrame. """
        return self.description_df

    def to_compile_df(self) -> pd.DataFrame:
        """ Returns dataset compiled information as a DataFrame. """
        return self.description_df.groupby(self.target_column).size().reset_index(name='Qty')
