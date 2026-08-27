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

    def verify_sample_shapes(self, other: "BaseDataModule") -> None:
        """
        Sanity check that two DataModules produce samples of the same shape
        """
        shape1 = self.get_sample_shape()
        shape2 = other.get_sample_shape()

        if shape1 != shape2:
            raise RuntimeError(
                "The two datamodules have different sample shapes:\n"
                f"\tself = {shape1}\n"
                f"\totherdm2 = {shape2}"
            )

class BaseProcessedDataset(torch_data.Dataset):
    """Base dataset with common fragment loading logic."""

    def __init__(self, dataframe: pd.DataFrame, processed_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_fragment(self, frag_id: int):
        path = os.path.join(self.processed_dir, f"{frag_id}.npy")
        x = np.load(path)

        if x.ndim == 1:
            x = x[np.newaxis, :]

        if self.transform:
            x = self.transform(x)

        return torch.from_numpy(x).float()

class ProcessedDataset(BaseProcessedDataset):
    """Dataset for single fragments with target."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = self._load_fragment(row["id_fragment"])
        y = row["Target"]

        return x, y

class PairedProcessedDataset(BaseProcessedDataset):
    """Dataset for paired fragments."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        y = row["Target"]

        fragment_cols = [c for c in row.index if c.startswith("id_fragment_")]
        fragment_ids = row[fragment_cols].tolist()

        data = []
        for fragment_id in fragment_ids:
            data.append(self._load_fragment(int(fragment_id)))

        return data, y

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
                 target_column: str = "Target",
                 group_column: str | None = None):
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
        self.group_column = group_column or id_column

        self.file_ids = description_df[id_column].to_list()
        self.targets = description_df[target_column].to_list()

        self.dataframe = None
        self.folds = []
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.id_to_target = None

        self.processed_dir = os.path.join(processed_dir, f"{hash(self)}")
        self.csv_file = os.path.join(self.processed_dir, "description.csv")

    def _get_params(self):
        return {
            "file_loader": self.file_loader.__get_hash_base__(),
            "file_processor": self.file_processor.__get_hash_base__(),
            "group_column": self.group_column,
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

        id_to_target = dict(zip(self.file_ids, self.targets))
        df[self.target_column] = df["file_id"].map(id_to_target)

        self.dataframe = df

        id_to_group = dict(zip(self.description_df[self.id_column],
                               self.description_df[self.group_column]))

        group_df = self.description_df.groupby(self.group_column).first().reset_index()
        unique_groups = group_df[self.group_column].tolist()
        group_targets = group_df[self.target_column].tolist()

        group_folds = self.cv.apply(unique_groups, group_targets)

        fold_records = {"file_id": self.file_ids}

        for idx, fold_map in enumerate(group_folds):
            file_fold_map = {}
            fold_column_roles = []

            for file_id in self.file_ids:
                group_val = id_to_group[file_id]
                role = fold_map[group_val]
                file_fold_map[file_id] = role

                fold_column_roles.append(str(role))

            self.folds.append(file_fold_map)

            fold_records[f"fold_{idx}"] = fold_column_roles

        folds_df = pd.DataFrame(fold_records)
        folds_csv_path = os.path.join(self.processed_dir, "folds_mapping.csv")
        folds_df.to_csv(folds_csv_path, index=False)

        self.set_fold(0)

    def set_fold(self, fold_idx: int):
        """
        Sets the active fold for training/validation/test split.

        Args:
            fold_idx: Index of the fold to activate.
        """
        if self.get_n_folds() == 0:
            raise RuntimeError("Folds not initialized. Call setup() first.")

        if not 0 <= fold_idx < len(self.folds):
            raise ValueError(f"Invalid fold index {fold_idx} (max {len(self.folds)-1}).")

        fold_map = self.folds[fold_idx]

        df = self.dataframe.copy()
        df["group"] = df["file_id"].map(fold_map)

        self.train_df = df[df["group"] == ml_cv.FoldRole.TRAIN]
        self.val_df = df[df["group"] == ml_cv.FoldRole.VALIDATION]
        self.test_df = df[df["group"] == ml_cv.FoldRole.TEST]

    def get_n_folds(self) -> int:
        """ Returns the number of folds generated by the cross-validator. """
        return len(self.folds)


    def _build_dataloader(self,
                          df: pd.DataFrame | None,
                          shuffle: bool) -> torch_data.DataLoader:

        if df is None:
            df = pd.DataFrame(columns=["id_fragment", "file_id", self.target_column])

        g = torch.Generator()
        g.manual_seed(42)

        return torch_data.DataLoader(
            ProcessedDataset(df, self.processed_dir, self.transform),
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=g,
            num_workers=self.num_workers
        )

    def train_dataloader(self) -> torch_data.DataLoader:
        """ Returns a torch_data.DataLoader for train subset. """
        return self._build_dataloader(self.train_df, True)

    def val_dataloader(self) -> torch_data.DataLoader:
        """ Returns a torch_data.DataLoader for validation subset. """
        return self._build_dataloader(self.val_df, False)

    def test_dataloader(self) -> torch_data.DataLoader:
        """ Returns a torch_data.DataLoader for test subset. """
        return self._build_dataloader(self.test_df, False)

    def all_dataloader(self) -> torch_data.DataLoader:
        """ Returns a torch_data.DataLoader for all subset. """
        return self._build_dataloader(self.dataframe, False)

    def _dataloader_dict(self,
                         df: pd.DataFrame | None,
                         shuffle: bool) -> typing.Dict[int, torch_data.DataLoader]:
        """ Returns a dictionary mapping target -> DataLoader. """

        if df is None:
            raise RuntimeError("df is not initialized. Call setup() first.")

        loaders = {}

        grouped = df.groupby(self.target_column)

        for target, df_target in grouped:

            dataset = ProcessedDataset(
                df_target,
                self.processed_dir,
                self.transform
            )

            g = torch.Generator()
            g.manual_seed(42)

            loader = torch_data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                generator=g,
                num_workers=0,
            )

            loaders[int(target)] = loader

        return loaders

    def train_dataloader_dict(self) -> typing.Dict[int, torch_data.DataLoader]:
        """ Returns a dictionary mapping target -> DataLoader. Using only train all data. """
        return self._dataloader_dict(self.train_df, shuffle=True)

    def val_dataloader_dict(self) -> typing.Dict[int, torch_data.DataLoader]:
        """ Returns a dictionary mapping target -> DataLoader. Using only val all data. """
        return self._dataloader_dict(self.val_df, shuffle=False)

    def test_dataloader_dict(self) -> typing.Dict[int, torch_data.DataLoader]:
        """ Returns a dictionary mapping target -> DataLoader. Using only test all data. """
        return self._dataloader_dict(self.test_df, shuffle=False)

    def all_dataloader_dict(self) -> typing.Dict[int, torch_data.DataLoader]:
        """ Returns a dictionary mapping target -> DataLoader. For all data. """
        return self._dataloader_dict(self.dataframe, shuffle=False)


    def get_n_targets(self) -> int:
        return len(set(self.targets))

    def to_df(self) -> pd.DataFrame:
        """ Returns dataset information as a DataFrame. """
        return self.description_df

    def to_compile_df(self) -> pd.DataFrame:
        """ Returns dataset compiled information as a DataFrame. """
        return self.description_df.groupby(self.target_column).size().reset_index(name='Qty')

    def get_dataloader_by_role(self, role: ml_cv.FoldRole | str | None = None) -> \
        torch_data.DataLoader:
        """ Returns the DataLoader for a given cv.FoldRole. """

        if role is None:
            return self.all_dataloader()

        if isinstance(role, str):
            role = ml_cv.FoldRole[role]

        if role == ml_cv.FoldRole.TRAIN:
            return self.train_dataloader()
        if role == ml_cv.FoldRole.VALIDATION:
            return self.val_dataloader()
        if role == ml_cv.FoldRole.TEST:
            return self.test_dataloader()

        raise ValueError(f"Unsupported fold role: {role}")

    def get_dataloader_dict_by_role(self, role: ml_cv.FoldRole | str | None = None) -> \
        typing.Dict[int, "torch_data.DataLoader"]:
        """ Returns the DataLoader dictionary for a given cv.FoldRole. """
        if role is None:
            return self.all_dataloader_dict()

        if isinstance(role, str):
            role = ml_cv.FoldRole[role]

        if role == ml_cv.FoldRole.TRAIN:
            return self.train_dataloader_dict()
        if role == ml_cv.FoldRole.VALIDATION:
            return self.val_dataloader_dict()
        if role == ml_cv.FoldRole.TEST:
            return self.test_dataloader_dict()

        raise ValueError(f"Unsupported fold role: {role}")

    def get_ids_by_role(self, role: ml_cv.FoldRole | str) -> typing.Set[int]:
        """
        Returns the set of file_ids belonging to a given cv.FoldRole.

        Requires `setup()` to have been called already.
        """

        if isinstance(role, str):
            role = ml_cv.FoldRole[role]

        if role == ml_cv.FoldRole.TRAIN:
            if self.train_df is None:
                raise RuntimeError("DataModule not set up. Call setup() first.")
            df = self.train_df

        elif role == ml_cv.FoldRole.VALIDATION:
            if self.val_df is None:
                raise RuntimeError("DataModule not set up. Call setup() first.")
            df = self.val_df

        elif role == ml_cv.FoldRole.TEST:
            if self.test_df is None:
                raise RuntimeError("DataModule not set up. Call setup() first.")
            df = self.test_df

        else:
            raise ValueError(f"Unsupported fold role: {role}")

        return set(df["file_id"].dropna().astype(int).unique().tolist())

class PairedAudioDataModule:
    _DEFAULT_PAIR_BUILDERS = {}

    def __class_getitem__(cls, base_class):

        class _Paired(base_class):

            def __init__(self, *args, pair_builder=None, **kwargs):
                super().__init__(*args, **kwargs)

                if pair_builder is not None:
                    self.pair_builder = pair_builder
                else:
                    if base_class not in cls._DEFAULT_PAIR_BUILDERS:
                        raise ValueError(
                            f"No default pair_builder registered for {base_class.__name__}"
                        )
                    self.pair_builder = cls._DEFAULT_PAIR_BUILDERS[base_class]

            def _expand_file_pairs_to_fragments(self,
                                                df_pairs: pd.DataFrame,
                                                df_frag: pd.DataFrame) -> pd.DataFrame:

                if df_pairs.empty:
                    return df_pairs

                pairs = []

                grouped = df_frag.groupby("file_id")

                for _, row in df_pairs.iterrows():

                    group_header = row.index[0]
                    group_id = row[group_header]

                    n_scenarios = len(row) - 1

                    dfs = []
                    n_frags = []
                    for i in range(n_scenarios):
                        fid = row[f"file_id_{i}"]
                        dfs.append(grouped.get_group(fid))
                        n_frags.append(len(dfs[-1]))

                    for k in range(min(n_frags)):
                          pairs.append({
                            group_header: group_id,
                            **{
                                f"id_fragment_{i}": df.iloc[k]["id_fragment"]
                                for i, df in enumerate(dfs)
                            }
                          })

                return pd.DataFrame(pairs)

            def _build_dataloader(self,
                                df: pd.DataFrame,
                                shuffle: bool) -> torch_data.DataLoader:

                if df is None:
                    raise RuntimeError("df is not initialized. Call setup() first.")

                file_ids = df["file_id"].unique()
                df_meta = self.description_df[
                    self.description_df[self.id_column].isin(file_ids)
                ]

                file_pairs = self.pair_builder(df_meta)

                pairs_df = self._expand_file_pairs_to_fragments(file_pairs, df)

                merge_key = pairs_df.columns[0]

                pairs_df = pairs_df.merge(
                    self.description_df[[merge_key, "Target"]].drop_duplicates(),
                    on=merge_key,
                    how="left"
                )

                return torch_data.DataLoader(
                    PairedProcessedDataset(
                        pairs_df,
                        self.processed_dir,
                        self.transform
                    ),
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers
                )

        return _Paired

    @classmethod
    def register_pair_builder(cls, base_class):
        def decorator(func):
            cls._DEFAULT_PAIR_BUILDERS[base_class] = func
            return func
        return decorator

class DomainDataset(torch_data.Dataset):
    """
    Wrap an existing dataset and replace its target with a fixed label. Use build_domain_dataloader
    to combine two dataloaders into a single domain classification dataloader.
    """

    def __init__(self, dataset: torch_data.Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = domain_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        y = torch.tensor(self.domain_label, dtype=torch.long)
        return x, y

    @staticmethod
    def build_domain_dataloader(
        loader1: torch_data.DataLoader,
        loader2: torch_data.DataLoader,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        label1: int = 0,
        label2: int = 1,
    ) -> torch_data.DataLoader:
        """
        Combine two existing DataLoaders into a single binary domain-classification
        DataLoader.
        """
        dataset1 = DomainDataset(loader1.dataset, domain_label=label1)
        dataset2 = DomainDataset(loader2.dataset, domain_label=label2)

        combined_dataset = torch_data.ConcatDataset([dataset1, dataset2])

        return torch_data.DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
