"""
4 classes Module
"""
import os

import pandas as pd

import lps_ml.core as ml_core
import lps_ml.datasets.selection as ml_sel

class FourClasses(ml_core.AudioDataModule):
    """ DataModule for 4 classes dataset. """

    @staticmethod
    def loader(data_base_dir: str) -> ml_core.AudioFileLoader:
        """ Get AudioFileLoader for FourClasses dataset. """
        return ml_core.AudioFileLoader(
            data_base_dir=data_base_dir,
            extract_id=lambda rel_path: int(
                os.path.splitext(os.path.basename(rel_path))[0][-2:]
            )
        )

    @staticmethod
    def as_df() -> pd.DataFrame:
        """ Returns dataset information as a DataFrame."""
        info_file = os.path.join(os.path.dirname(__file__),
                                 "dataset_info",
                                 "four_classes.csv")
        return pd.read_csv(info_file)

    def __init__(self,
                 file_processor: ml_core.AudioProcessor,
                 processed_dir: str = "/data/Processed_data/4classes",
                 data_dir: str = "/data/4classes",
                 batch_size: int = 32,
                 cv: ml_core.CrossValidator = None,
                 selection: ml_sel.Selector = None,
                 num_workers: int = None):

        df = FourClasses.as_df()

        if selection is None:
            unique_targets = sorted(df["Class"].unique())

            selection = ml_sel.Selector(
                ml_sel.LabelTarget(column="Class", values=unique_targets)
            )

        super().__init__(file_loader = FourClasses.loader(data_base_dir=data_dir),
                         file_processor = file_processor,
                         description_df = selection.apply(df),
                         processed_dir = processed_dir,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         cv = cv)
