"""
dfml — sklearn-style ML preprocessing with pandas-native syntax.

Importing this package registers a `.ml` accessor on every pandas
DataFrame, so you can write:

    import pandas as pd
    import dfml  # noqa: F401

    df.ml.impute().ml.encode().ml.scale()

instead of hand-rolling ColumnTransformer / Pipeline boilerplate.
"""

from .accessor import MLAccessor

__version__ = "0.1.0"
__all__ = ["MLAccessor"]
