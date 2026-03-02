"""
WavDirDataset Module

Dataset genérico baseado em estrutura de diretórios:

input_dir/
    classA/
        a.wav
        b.wav
    classB/
        1.wav
        2.wav
"""

import os
import pandas as pd

import lps_utils.utils as lps_utils
import lps_ml.core as ml_core
import lps_ml.datasets.selection as ml_sel

class AudioFolder(ml_core.AudioDataModule):
    """Generic DataModule for datasets organized by directory per class."""

    @staticmethod
    def _build_dataframe(input_dir: str) -> pd.DataFrame:
        """
        Scans directory and builds a dataframe with:
            ID, Target, Filepath, Class
        """
        wav_files = lps_utils.find_files(input_dir, extension=".wav")

        records = []
        class_to_target = {}
        next_target = 0
        next_id = 0

        for filepath in sorted(wav_files):
            class_name = os.path.basename(os.path.dirname(filepath))
            rel_path = os.path.relpath(filepath, input_dir)

            if class_name not in class_to_target:
                class_to_target[class_name] = next_target
                next_target += 1

            records.append({
                "ID": next_id,
                "Target": class_to_target[class_name],
                "Filepath": filepath,
                "RelPath": rel_path,
                "Class": class_name
            })

            next_id += 1

        return pd.DataFrame(records)

    @staticmethod
    def loader(input_dir: str,
            dataframe: pd.DataFrame) -> ml_core.AudioFileLoader:
        """
        Creates AudioFileLoader using rel_path → ID mapping directly
        from dataframe.
        """

        rel_to_id = dataframe.set_index("RelPath")["ID"]

        def extract_id(rel_path: str) -> int:
            try:
                return int(rel_to_id.loc[rel_path])
            except KeyError:
                raise ValueError(f"{rel_path} not found in dataframe.")

        return ml_core.AudioFileLoader(
            data_base_dir=input_dir,
            extract_id=extract_id
        )

    def __init__(self,
                 input_dir: str,
                 file_processor: ml_core.AudioProcessor,
                 processed_dir: str = "/data/Processed_data/audio_folder",
                 batch_size: int = 32,
                 cv: ml_core.CrossValidator = None,
                 selection: ml_sel.Selector = None,
                 num_workers: int = None):

        df = self._build_dataframe(input_dir)

        if selection is None:
            unique_targets = sorted(df["Target"].unique())

            selection = ml_sel.Selector(
                ml_sel.LabelTarget(column="Target", values=unique_targets)
            )

        df_selected = selection.apply(df)

        super().__init__(
            file_loader=self.loader(input_dir, df_selected),
            file_processor=file_processor,
            description_df=df_selected,
            processed_dir=processed_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            cv=cv
        )
