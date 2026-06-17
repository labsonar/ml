"""
Selection Module (Refactored)
"""
import typing
import abc
import pandas as pd

class Filter(abc.ABC):
    """Abstract base class representing a filter to apply on a collection."""

    @abc.abstractmethod
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the collection based on selected values present in a column.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be filtered.

        Returns:
            pd.DataFrame: A filtered DataFrame.
        """

class Constraint:
    """Define a single constraint for filtering or classification."""
    def __init__(self, header: str, values: typing.List[typing.Any]):
        self.header = header
        self.values = values

    def matches(self, item: pd.Series) -> bool:
        """Check if a given dataframe row matches this constraint."""
        return item[self.header] in self.values

    def mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return subset of rows matching this constraint."""
        return df[self.header].isin(self.values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constraint) and \
               self.header == other.header and \
               self.values == other.values

class ConstraintFilter(Filter):
    """Filter that selects rows matching one or more constraint groups."""

    def __init__(self,
                 constraints: typing.Union[
                     typing.List[
                         typing.Union[
                             typing.List[typing.Union[Constraint, typing.Dict[str, typing.Any]]],
                             Constraint,
                             typing.Dict[str, typing.Any]
                         ]
                     ],
                     Constraint,
                     typing.Dict[str, typing.Any]
                 ],
                 remove_elements_in: bool = True):
        self.remove_elements_in = remove_elements_in
        self.constraints: typing.List[Constraint] = []

        def flatten_and_convert(item):
            """Recursively flatten input and convert dicts to Constraint."""
            if isinstance(item, Constraint):
                return [item]
            elif isinstance(item, dict):
                return [Constraint(header=item["header"], values=item["value"])]
            elif isinstance(item, list):
                result = []
                for sub in item:
                    result.extend(flatten_and_convert(sub))
                return result
            else:
                raise TypeError(f"Unsupported constraint type: {type(item)}")

        self.constraints = flatten_and_convert(constraints)

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filtering operation."""
        mask = pd.Series(False, index=input_df.index)
        for c in self.constraints:
            mask |= c.mask(input_df)
        if self.remove_elements_in:
            mask = ~mask
        return input_df.loc[mask]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConstraintFilter) and self.constraints == other.constraints

class LabelFilter(ConstraintFilter):
    """Simplified filter for selecting rows where column values are in a list."""

    def __init__(self, header: str, values: typing.List[str], remove_elements_in: bool = False):
        super().__init__([Constraint(header=header, values=values)],
                         remove_elements_in=remove_elements_in)

class CallbackFilter(Filter):
    """Filter based on a boolean callback."""

    def __init__(self, function: typing.Callable[[pd.Series], bool]):
        self.function = function

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        mask = input_df.apply(self.function, axis=1)
        return input_df.loc[mask]

class MultiFilter(Filter):
    """Composer of Filters."""

    def __init__(self, filters: typing.Union[typing.List[Filter], Filter]):
        self.filters = filters if isinstance(filters, list) else [filters]

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df
        for f in self.filters:
            df = f.apply(df)
        return df

class Target(abc.ABC):
    """Abstract base class to generate labelled dataframes."""

    DEFAULT_TARGET_HEADER = 'Target'

    def __init__(self, n_targets: int, include_others: bool):
        self.n_targets = n_targets
        self.include_others = include_others

    def get_n_targets(self) -> int:
        """Return the number of targets."""
        return self.n_targets + (1 if self.include_others else 0)

    @abc.abstractmethod
    def label(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the labelling in a dataframe.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be classified.

        Returns:
            pd.DataFrame: A DataFrame with target column.
        """

    def grouped_column(self) -> str:
        """Name of the target column to use when grouping results."""
        return self.DEFAULT_TARGET_HEADER


class ConstraintTarget(Target):
    """Target generator based on a list of constraint groups."""

    def __init__(self,
                 constraints: typing.List[
                     typing.Union[
                         typing.List[typing.Union[Constraint, typing.Dict[str, typing.Any]]],
                         Constraint,
                         typing.Dict[str, typing.Any]
                     ]
                 ],
                 include_others: bool = False):
        """
        Parameters:
        - constraints: List where each element represents one class.
            Each class can be:
              - a list of constraints,
              - a single Constraint,
              - or a dict {"header": ..., "value": ...}.
        - include_others: Whether to include unmatched elements as an extra target.
        """

        self.constraints: typing.List[typing.List[Constraint]] = []
        for group in constraints:
            if isinstance(group, list):
                normalized_group = [
                    c if isinstance(c, Constraint) else Constraint(header=c["header"],
                                                                   values=c["value"])
                    for c in group
                ]
            else:
                normalized_group = [
                    group if isinstance(group, Constraint) else Constraint(header=group["header"],
                                                                           values=group["value"])
                ]

            self.constraints.append(normalized_group)

        super().__init__(n_targets=len(self.constraints), include_others=include_others)

    def label(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the constraint-based classification to the dataframe."""

        def selection(item: pd.Series) -> typing.Optional[int]:
            for index, constraint_group in enumerate(self.constraints):
                if any(c.matches(item) for c in constraint_group):
                    return index
            return None

        df = input_df.copy()
        df[self.DEFAULT_TARGET_HEADER] = df.apply(selection, axis=1)

        if self.include_others:
            df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].fillna(self.n_targets)
        else:
            df = df.dropna(subset=[self.DEFAULT_TARGET_HEADER])

        df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].astype(int)
        return df

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConstraintTarget) and \
               self.constraints == other.constraints and \
               self.include_others == other.include_others


class LabelTarget(ConstraintTarget):
    """Simplified target assigning labels based on column values."""

    def __init__(self, column: str, values: typing.List[str], include_others: bool = False):
        constraints = [[Constraint(header=column, values=[v])] for v in values]
        super().__init__(constraints=constraints, include_others=include_others)


class CallbackTarget(Target):
    """Target based on a callback function."""

    def __init__(self,
                 n_targets: int,
                 function: typing.Callable[[pd.Series], int],
                 include_others: bool = False):
        super().__init__(n_targets=n_targets, include_others=include_others)
        self.function = function

    def label(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df[self.DEFAULT_TARGET_HEADER] = df.apply(self.function, axis=1)

        if self.include_others:
            df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].fillna(self.n_targets)
        else:
            df = df.dropna(subset=[self.DEFAULT_TARGET_HEADER])

        df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].astype(int)
        return df

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CallbackTarget) and \
               self.n_targets == other.n_targets and \
               self.include_others == other.include_others

class CombinationTarget(Target):
    """
    Target generator based on unique combinations of one or more columns.
    Each unique combination is mapped to an integer label.
    """

    def __init__(self, columns: typing.Union[str, typing.List[str]]):

        if isinstance(columns, str):
            columns = [columns]

        self.columns = columns
        self._mapping: typing.Dict[typing.Tuple, int] = {}
        self._combinations: typing.List = []

        super().__init__(n_targets=0, include_others=False)

    def label(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        self.update_mapping(df)

        df[self.DEFAULT_TARGET_HEADER] = [
            self._mapping.get(comb, None) for comb in self._combinations
        ]

        df = df.dropna(subset=[self.DEFAULT_TARGET_HEADER])
        df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].astype(int)

        return df

    def update_mapping(self, input_df: pd.DataFrame):
        """Return the combination → label mapping."""
        if not self._mapping:
            self._combinations = list(zip(*(input_df[col] for col in self.columns)))

            unique_combinations = pd.unique(self._combinations)

            self._mapping = {
                comb: idx for idx, comb in enumerate(unique_combinations)
            }
            self.n_targets = len(self._mapping)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CombinationTarget) and \
               self.columns == other.columns

class ColumnTarget(Target):

    def __init__(self,
                 column: str,
                 map_values: bool = True):

        self.column = column
        self.map_values = map_values

        self._mapping: typing.Dict[typing.Any, int] = {}

        super().__init__(n_targets=0, include_others=False)

    def update_mapping(self, input_df: pd.DataFrame):

        if not self.map_values:
            return

        if not self._mapping:
            unique_values = pd.unique(input_df[self.column])

            self._mapping = {
                value: idx
                for idx, value in enumerate(unique_values)
            }

            self.n_targets = len(self._mapping)

    def label(self, input_df: pd.DataFrame) -> pd.DataFrame:

        df = input_df.copy()

        if self.map_values:

            self.update_mapping(df)

            df[self.DEFAULT_TARGET_HEADER] = (
                df[self.column]
                .map(self._mapping)
            )

            df = df.dropna(subset=[self.DEFAULT_TARGET_HEADER])

            df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].astype(int)

        else:

            df[self.DEFAULT_TARGET_HEADER] = df[self.column]

            df = df.dropna(subset=[self.DEFAULT_TARGET_HEADER])

            if pd.api.types.is_integer_dtype(df[self.column]):
                df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].astype(int)
            else:
                df[self.DEFAULT_TARGET_HEADER] = df[self.DEFAULT_TARGET_HEADER].astype(float)

        return df

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ColumnTarget)
            and self.column == other.column
            and self.map_values == other.map_values
        )

class Selector:
    """Class representing a filtered and labelled subset of data."""

    def __init__(self,
                 target: Target,
                 filters: typing.Union[typing.List[Filter], Filter, None] = None):

        self.target = target
        self.filters = [] if filters is None else (
            filters if isinstance(filters, list) else [filters]
        )

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply all filters, then label the resulting DataFrame."""
        df = input_df.copy()
        print("apply: ", len(df))
        for f in self.filters:
            df = f.apply(df)
            print("\t filt: ", len(df))
        df = self.target.label(df)
        print("target: ", len(df))
        return df

    def grouped_column(self) -> str:
        return self.target.grouped_column()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Selector) and \
               self.target == other.target and \
               self.filters == other.filters
