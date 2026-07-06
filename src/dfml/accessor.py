"""
dfml.accessor
=============

Registers a `.ml` accessor on pandas DataFrames, giving sklearn-style
preprocessing a pandas-native, chainable syntax.

Example
-------
>>> import pandas as pd
>>> import dfml  # noqa: F401  (side-effect import registers the accessor)
>>> df.ml.impute().ml.encode().ml.scale()
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

ScaleMethod = Literal["standard", "minmax", "robust"]
EncodeMethod = Literal["onehot", "label"]
ImputeStrategy = Literal["mean", "median", "most_frequent", "constant"]

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


@pd.api.extensions.register_dataframe_accessor("ml")
class MLAccessor:
    """ML preprocessing accessor for pandas DataFrames.

    Accessed via ``df.ml.<method>()``. Every method returns a new
    DataFrame (the original is never mutated), so calls can be chained:

    >>> df.ml.impute().ml.encode().ml.scale()

    Fitted transformers are cached on the accessor instance (keyed by
    method name) so that ``.ml.transform(new_df)`` can later apply the
    *same* fitted transformation to unseen data (e.g. a test set),
    mirroring the fit/transform contract sklearn users already expect.
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._df = pandas_obj
        # Cache of fitted transformers, keyed by the step name that
        # created them, so `.transform()` can replay them on new data.
        self._fitted: dict[str, dict] = {}

    @staticmethod
    def _validate(obj) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("The .ml accessor only works on DataFrames.")

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def scale(
        self,
        columns: Optional[Iterable[str]] = None,
        method: ScaleMethod = "standard",
    ) -> pd.DataFrame:
        """Scale numeric columns using a sklearn scaler.

        Parameters
        ----------
        columns : list of str, optional
            Columns to scale. Defaults to all numeric columns.
        method : {"standard", "minmax", "robust"}
            Which sklearn scaler to use.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the given columns scaled.
        """
        if method not in _SCALERS:
            raise ValueError(f"method must be one of {list(_SCALERS)}, got {method!r}")

        cols = list(columns) if columns is not None else list(
            self._df.select_dtypes(include="number").columns
        )
        if not cols:
            return self._df.copy()

        scaler = _SCALERS[method]()
        result = self._df.copy()
        result[cols] = scaler.fit_transform(result[cols])

        self._fitted["scale"] = {"transformer": scaler, "columns": cols}
        return result

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(
        self,
        columns: Optional[Iterable[str]] = None,
        method: EncodeMethod = "onehot",
        drop_first: bool = False,
    ) -> pd.DataFrame:
        """Encode categorical columns.

        Parameters
        ----------
        columns : list of str, optional
            Columns to encode. Defaults to all non-numeric columns.
        method : {"onehot", "label"}
            "onehot" expands each category into its own 0/1 column.
            "label" assigns each category an integer code in place.
        drop_first : bool
            For one-hot encoding, drop the first category level to
            avoid the dummy-variable trap. Ignored for "label".

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the given columns encoded.
        """
        cols = list(columns) if columns is not None else list(
            self._df.select_dtypes(exclude="number").columns
        )
        if not cols:
            return self._df.copy()

        result = self._df.copy()

        if method == "onehot":
            encoder = OneHotEncoder(sparse_output=False, drop="first" if drop_first else None)
            encoded_array = encoder.fit_transform(result[cols])
            encoded_cols = encoder.get_feature_names_out(cols)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=result.index)
            result = pd.concat([result.drop(columns=cols), encoded_df], axis=1)
            self._fitted["encode"] = {"transformer": encoder, "columns": cols}

        elif method == "label":
            encoders = {}
            for col in cols:
                le = LabelEncoder()
                result[col] = le.fit_transform(result[col])
                encoders[col] = le
            self._fitted["encode"] = {"transformer": encoders, "columns": cols}

        else:
            raise ValueError(f"method must be 'onehot' or 'label', got {method!r}")

        return result

    # ------------------------------------------------------------------
    # Imputing
    # ------------------------------------------------------------------
    def impute(
        self,
        columns: Optional[Iterable[str]] = None,
        strategy: ImputeStrategy = "mean",
        fill_value=None,
    ) -> pd.DataFrame:
        """Fill missing values using a sklearn SimpleImputer.

        Parameters
        ----------
        columns : list of str, optional
            Columns to impute. Defaults to all numeric columns for
            "mean"/"median", or all columns for "most_frequent"/"constant".
        strategy : {"mean", "median", "most_frequent", "constant"}
            Imputation strategy, passed through to sklearn.
        fill_value :
            Value to use when strategy="constant".

        Returns
        -------
        pd.DataFrame
            A new DataFrame with missing values filled.
        """
        if columns is not None:
            cols = list(columns)
        elif strategy in ("mean", "median"):
            cols = list(self._df.select_dtypes(include="number").columns)
        else:
            cols = list(self._df.columns)

        if not cols:
            return self._df.copy()

        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        result = self._df.copy()
        result[cols] = imputer.fit_transform(result[cols])

        self._fitted["impute"] = {"transformer": imputer, "columns": cols}
        return result

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def split(
        self,
        target: str,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify: bool = False,
    ):
        """Split into train/test features and target.

        Parameters
        ----------
        target : str
            Name of the target column.
        test_size : float
            Fraction of rows to hold out for testing.
        random_state : int, optional
            Seed for reproducibility.
        stratify : bool
            If True, stratify the split by the target column.

        Returns
        -------
        X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
        """
        from sklearn.model_selection import train_test_split

        if target not in self._df.columns:
            raise KeyError(f"Target column {target!r} not found in DataFrame.")

        X = self._df.drop(columns=[target])
        y = self._df[target]

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def fitted_(self) -> dict:
        """Dict of fitted transformers from the most recent calls, keyed by step name."""
        return self._fitted
