# dfml

sklearn-style ML preprocessing with pandas-native syntax.

`dfml` registers a `.ml` accessor on every pandas `DataFrame` — the same
mechanism pandas itself uses for `.str`, `.dt`, and `.cat` — so common
preprocessing steps read like native pandas instead of sklearn boilerplate.

```python
import pandas as pd
import dfml  # noqa: F401  (side-effect import registers the accessor)

df = pd.read_csv("data.csv")
clean = df.ml.impute().ml.encode().ml.scale()
```

## Why

Preprocessing with plain sklearn usually looks like this:

```python
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder()),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

X_transformed = preprocessor.fit_transform(X)  # numpy array, column names gone
```

With `dfml`:

```python
clean = df.ml.impute().ml.encode().ml.scale()  # still a DataFrame, columns intact
```

- **Stays a DataFrame.** sklearn transformers return bare numpy arrays and
  drop your column names. Every `dfml` method returns a DataFrame.
- **Chainable.** `.ml.impute().ml.encode().ml.scale()` reads top to bottom
  like a pipeline, without building one.
- **Auto-detects columns.** Numeric vs. categorical columns are inferred
  by default — pass `columns=[...]` only when you want to override it.
- **Not a replacement.** Every method wraps a real sklearn transformer
  under the hood (accessible via `.ml.fitted_`), so you can drop back to
  raw sklearn any time you need something the accessor doesn't cover.

## Install

```bash
pip install dfml
```

## Usage

### Impute missing values

```python
df.ml.impute(strategy="mean")               # numeric columns, default
df.ml.impute(strategy="most_frequent")       # all columns
df.ml.impute(columns=["age"], strategy="median")
```

### Encode categorical columns

```python
df.ml.encode()                               # one-hot, all non-numeric columns
df.ml.encode(columns=["city"], method="label")
df.ml.encode(columns=["city"], drop_first=True)
```

### Scale numeric columns

```python
df.ml.scale()                                # StandardScaler, all numeric columns
df.ml.scale(method="minmax")
df.ml.scale(columns=["income"], method="robust")
```

### Split into train/test

```python
X_train, X_test, y_train, y_test = df.ml.split(target="churn", test_size=0.2)
```

### Chain everything

```python
X_train, X_test, y_train, y_test = (
    df.ml.impute()
      .ml.encode()
      .ml.scale()
      .ml.split(target="churn", test_size=0.2)
)
```

### Inspect what was fitted

```python
scaled = df.ml.scale()
scaled.ml.fitted_["scale"]["transformer"]   # the actual fitted StandardScaler
```

## Development

```bash
git clone https://github.com/Parthshewale18/dfml
cd dfml
pip install -e ".[dev]"
pytest
```

## License

MIT
