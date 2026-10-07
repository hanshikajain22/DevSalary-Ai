# src/features.py
# Purpose: Build the preprocessing "recipe" (ColumnTransformer) used by every model.
# The recipe is only FITTED on training data, so test data never influences it.

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, LANGUAGE_FEATURES


def build_preprocessor():
    """Return an unfitted ColumnTransformer for the cleaned salary data."""

    # Numbers: fill blanks with the median, then scale (helps Linear Regression)
    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Text categories: fill blanks with the label "Missing", then one-hot encode.
    # - handle_unknown="ignore": a category never seen in training will not crash the app
    # - min_frequency=20: categories with fewer than 20 rows are grouped together
    #   (helps with the 160+ countries, many of which have very few rows)
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                  sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC_FEATURES),
        ("categorical", categorical_steps, CATEGORICAL_FEATURES),
        # language flags are already 0/1, so pass them through unchanged
        ("languages", "passthrough", LANGUAGE_FEATURES),
    ])