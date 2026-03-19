"""
Cross Validation Module

This module provides functionality to split datasets into training, validation, and testing parts.
"""
import abc
import enum
import typing

import numpy as np

import sklearn.model_selection as sk_selection


class FoldRole(enum.Enum):
    """Enum representing the role of a sample in a cross-validation fold."""
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

class CrossValidator(abc.ABC):
    """Abstract base class for all cross-validation strategies."""

    @abc.abstractmethod
    def apply(
        self,
        ids: typing.List[int],
        targets: typing.List[int],
        random_state: int = 42,
    ) -> typing.List[typing.Dict[int, FoldRole]]:
        """
        Apply the cross-validation strategy.

        Args:
            ids (List[int]): Unique identifiers of the samples.
            targets (List[int]): Corresponding target labels.
            random_state (int): Random seed for reproducibility.

        Returns:
            List[Dict[int, FoldRole]]:
                A list of folds, each as a mapping {sample_id: FoldRole}.
        """

class StratifiedKFold(CrossValidator):
    """Standard Stratified K-Fold cross-validation (train/validation)."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True):
        self.n_splits = n_splits
        self.shuffle = shuffle

    def apply(
        self,
        ids: typing.List[int],
        targets: typing.List[int],
        random_state: int = 42,
    ) -> typing.List[typing.Dict[int, FoldRole]]:
        kf = sk_selection.StratifiedKFold(n_splits=self.n_splits,
                                          shuffle=self.shuffle,
                                          random_state=random_state)
        folds = []

        for train_idx, val_idx in kf.split(ids, targets):
            mapping = {ids[i]: FoldRole.TRAIN for i in train_idx}
            mapping.update({ids[i]: FoldRole.VALIDATION for i in val_idx})
            folds.append(mapping)

        return folds

class HoldOutCV(CrossValidator):
    """Three-way Hold-Out validation (train/validation/test)."""

    def __init__(self, test_size: float = 0.2, val_size: float = 0.2):
        """
        Args:
            test_size (float): Proportion of samples used for testing.
            val_size (float): Proportion (of remaining data) used for validation.
        """
        self.test_size = test_size
        self.val_size = val_size

    def apply(
        self,
        ids: typing.List[int],
        targets: typing.List[int],
        random_state: int = 42,
    ) -> typing.List[typing.Dict[int, FoldRole]]:
        ids = np.array(ids)
        targets = np.array(targets)

        trainval_idx, test_idx = sk_selection.train_test_split(
            np.arange(len(ids)),
            test_size=self.test_size,
            stratify=targets,
            random_state=random_state,
        )

        train_idx, val_idx = sk_selection.train_test_split(
            trainval_idx,
            test_size=self.val_size,
            stratify=targets[trainval_idx],
            random_state=random_state,
        )

        mapping = {ids[i]: FoldRole.TRAIN for i in train_idx}
        mapping.update({ids[i]: FoldRole.VALIDATION for i in val_idx})
        mapping.update({ids[i]: FoldRole.TEST for i in test_idx})

        return [mapping]

class FiveByTwo(CrossValidator):
    """5x2 cross-validation (used in the 5x2 F-test)."""

    def __init__(self, shuffle: bool = True):
        self.shuffle = shuffle

    def apply(
        self,
        ids: typing.List[int],
        targets: typing.List[int],
        random_state: int = 42,
    ) -> typing.List[typing.Dict[int, FoldRole]]:

        folds = []
        for rep in range(5):

            split1, split2 = sk_selection.train_test_split(
                np.arange(len(ids)),
                test_size=0.5,
                stratify=targets,
                random_state=random_state + rep,
            )

            mapping1 = {ids[i]: FoldRole.TRAIN for i in split1}
            mapping1.update({ids[i]: FoldRole.VALIDATION for i in split2})
            folds.append(mapping1)

            mapping2 = {ids[i]: FoldRole.TRAIN for i in split2}
            mapping2.update({ids[i]: FoldRole.VALIDATION for i in split1})
            folds.append(mapping2)

        return folds

class OverfitCV(CrossValidator):
    """
    Forces the same samples into TRAIN, VALIDATION, and TEST.
    Ideal for debugging convergence.
    """

    def __init__(self, n_samples: int = 1):
        self.n_samples = n_samples

    def apply(self, ids, targets, random_state=42):

        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(ids), size=self.n_samples, replace=False)

        selected_ids = [ids[i] for i in idx]

        mapping = {}

        for sid in selected_ids:
            mapping[sid] = FoldRole.TRAIN  # todos iguais

        return [mapping]
