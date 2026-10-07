# src/model_training.py
# Purpose: Train and compare salary prediction models

import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_preprocessor
from preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    LANGUAGE_FEATURES,
    TARGET_COLUMN
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

print("=" * 60)
print("DEVSalary AI - MODEL TRAINING")
print("=" * 60)

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

input_columns = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + LANGUAGE_FEATURES
)

X_train = train_data[input_columns]
y_train = train_data[TARGET_COLUMN]

X_test = test_data[input_columns]
y_test = test_data[TARGET_COLUMN]

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))
print("Input features:", len(input_columns))


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# --------------------------------------------------
# Train and Evaluate
# --------------------------------------------------

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)
    print("-" * 60)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[model_name] = pipeline

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))


# --------------------------------------------------
# Compare Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# --------------------------------------------------
# Select Best Model
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_model_path = os.path.join(
    MODEL_DIR,
    "best_salary_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)


# Save comparison results

results_path = os.path.join(
    MODEL_DIR,
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# --------------------------------------------------
# Save Individual Models
# --------------------------------------------------

for model_name, pipeline in trained_models.items():

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        pipeline,
        path
    )


# --------------------------------------------------
# Final Output
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Selected:", best_model_name)
print("Saved as:", best_model_path)

print("\nFiles saved:")
print("- best_salary_model.joblib")
print("- model_results.csv")
print("- linear_regression.joblib")
print("- random_forest.joblib")
print("- gradient_boosting.joblib")

print("\nMODEL TRAINING COMPLETED")
print("=" * 60)