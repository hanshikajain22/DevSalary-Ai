# src/split_data.py
# Purpose: Split the cleaned data into train and test sets, then check that the
# preprocessing recipe works. Run from the project folder:
#     python src/split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

from features import build_preprocessor
from preprocessing import (CLEAN_DATA_PATH, TARGET_COLUMN, NUMERIC_FEATURES,
                           CATEGORICAL_FEATURES, LANGUAGE_FEATURES)

TEST_SIZE = 0.2        # 20% of rows are kept aside for the final test
RANDOM_STATE = 42      # fixed number so the split is the same every run

clean_data = pd.read_csv(CLEAN_DATA_PATH)

input_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + LANGUAGE_FEATURES
X = clean_data[input_columns]
y = clean_data[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print("Train salary median:", round(y_train.median()), "| Test salary median:", round(y_test.median()))

# Save the split so every later step uses exactly the same train/test rows
X_train.assign(**{TARGET_COLUMN: y_train}).to_csv("data/train.csv", index=False)
X_test.assign(**{TARGET_COLUMN: y_test}).to_csv("data/test.csv", index=False)

# Fit the recipe on TRAINING data only, then apply it to both sets
preprocessor = build_preprocessor()
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)

print("\nColumns before preprocessing:", X_train.shape[1])
print("Columns after preprocessing :", X_train_ready.shape[1])
print("Any missing values left in train?", pd.DataFrame(X_train_ready).isnull().any().any())
print("Any missing values left in test? ", pd.DataFrame(X_test_ready).isnull().any().any())

# Show, for each categorical feature, how many categories exist and how many were grouped as "rare"
encoder = preprocessor.named_transformers_["categorical"].named_steps["one_hot"]
print("\nCategory counts (seen in training / grouped as rare because < 20 rows):")
for column_name, categories, rare in zip(CATEGORICAL_FEATURES, encoder.categories_,
                                         encoder.infrequent_categories_):
    rare_count = 0 if rare is None else len(rare)
    print(f"  {column_name:12s} {len(categories):4d} seen / {rare_count:4d} grouped as rare")