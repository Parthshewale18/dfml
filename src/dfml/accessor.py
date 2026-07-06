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

   