import numpy as np
import pandas as pd
import pytest

import dfml  # noqa: F401  (registers the .ml accessor)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "age": [25, 32, 47, np.nan, 51],
            "income": [50000, 64000, 120000, 82000, np.nan],
            "city": ["NY", "LA", "NY", "SF", "LA"],
            "target": [0, 1, 1, 0, 1],
        }
    )


class TestScale:
    def test_scale_default_numeric_columns(self, df):
        filled = df.ml.impute()
        scaled = filled.ml.scale()
        numeric_cols = ["age", "income"]
        # standard-scaled columns should have ~mean 0
        assert np.allclose(scaled[numeric_cols].mean(), 0, atol=1e-8)

    def test_scale_specific_columns(self, df):
        filled = df.ml.impute()
        scaled = filled.ml.scale(columns=["age"])
        assert np.isclose(scaled["age"].mean(), 0, atol=1e-8)
        # income untouched
        assert scaled["income"].equals(filled["income"])

    def test_scale_invalid_method_raises(self, df):
        with pytest.raises(ValueError):
            df.ml.scale(method="not-a-method")

    def test_scale_does_not_mutate_original(self, df):
        original = df.copy()
        df.ml.impute().ml.scale()
        pd.testing.assert_frame_equal(df, original)


class TestEncode:
    def test_onehot_encode_creates_dummy_columns(self, df):
        encoded = df.ml.encode(columns=["city"], method="onehot")
        assert "city" not in encoded.columns
        assert any(col.startswith("city_") for col in encoded.columns)

    def test_label_encode_returns_integers(self, df):
        encoded = df.ml.encode(columns=["city"], method="label")
        assert pd.api.types.is_integer_dtype(encoded["city"])

    def test_encode_invalid_method_raises(self, df):
        with pytest.raises(ValueError):
            df.ml.encode(columns=["city"], method="bad-method")

    def test_encode_no_categorical_columns_returns_copy(self):
        numeric_only = pd.DataFrame({"a": [1, 2, 3]})
        result = numeric_only.ml.encode()
        pd.testing.assert_frame_equal(result, numeric_only)


class TestImpute:
    def test_impute_fills_missing_numeric(self, df):
        result = df.ml.impute(strategy="mean")
        assert result[["age", "income"]].isna().sum().sum() == 0

    def test_impute_median_strategy(self, df):
        result = df.ml.impute(strategy="median")
        assert result["age"].isna().sum() == 0

    def test_impute_specific_columns(self, df):
        result = df.ml.impute(columns=["age"])
        assert result["age"].isna().sum() == 0
        # income still has its NaN since it wasn't targeted
        assert result["income"].isna().sum() == 1


class TestSplit:
    def test_split_returns_four_parts(self, df):
        filled = df.ml.impute()
        X_train, X_test, y_train, y_test = filled.ml.split(target="target", test_size=0.4, random_state=0)
        assert len(X_train) + len(X_test) == len(filled)
        assert "target" not in X_train.columns
        assert len(y_train) == len(X_train)

    def test_split_missing_target_raises(self, df):
        with pytest.raises(KeyError):
            df.ml.split(target="does_not_exist")


class TestChaining:
    def test_full_chain_runs_end_to_end(self, df):
        result = df.ml.impute().ml.encode(columns=["city"]).ml.scale(columns=["age", "income"])
        assert result.isna().sum().sum() == 0
        assert "city" not in result.columns


class TestFittedCache:
    def test_fitted_stores_transformer_after_scale(self, df):
        filled = df.ml.impute()
        scaled_accessor = filled.ml
        scaled_accessor.scale()
        assert "scale" in scaled_accessor.fitted_
