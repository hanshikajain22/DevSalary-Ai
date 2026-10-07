# src/evaluate_model.py
# Purpose: Evaluate the best salary prediction model

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_PATH = "data/test.csv"
MODEL_PATH = "models/best_salary_model.joblib"
OUTPUT_DIR = "data/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load Model and Test Data
# --------------------------------------------------

print("=" * 60)
print("DEVSalary AI - MODEL EVALUATION")
print("=" * 60)

model = joblib.load(MODEL_PATH)
test_data = pd.read_csv(TEST_PATH)

print("\nTest rows:", len(test_data))

# --------------------------------------------------
# Prepare Features
# --------------------------------------------------

target_column = "ConvertedCompYearly"

feature_columns = [
    column for column in test_data.columns
    if column != target_column
]

X_test = test_data[feature_columns]
y_test = test_data[target_column]

# --------------------------------------------------
# Make Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(X_test)

# --------------------------------------------------
# Calculate Metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

print("\n" + "-" * 60)
print("MODEL PERFORMANCE")
print("-" * 60)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

# --------------------------------------------------
# Save Predictions
# --------------------------------------------------

prediction_data = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": predictions,
    "Absolute Error": abs(
        y_test.values - predictions
    )
})

prediction_path = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)

prediction_data.to_csv(
    prediction_path,
    index=False
)

print("\nPredictions saved:", prediction_path)

# --------------------------------------------------
# 1. Actual vs Predicted
# --------------------------------------------------

plt.figure(figsize=(9, 7))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")

# Perfect prediction reference line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()

actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(actual_predicted_path)
plt.close()

print("Actual vs predicted graph saved")

# --------------------------------------------------
# 2. Prediction Error Distribution
# --------------------------------------------------

errors = y_test.values - predictions

plt.figure(figsize=(9, 6))

plt.hist(
    errors,
    bins=50
)

plt.xlabel("Prediction Error")
plt.ylabel("Number of Developers")
plt.title("Distribution of Prediction Errors")

plt.tight_layout()

error_path = os.path.join(
    OUTPUT_DIR,
    "prediction_errors.png"
)

plt.savefig(error_path)
plt.close()

print("Prediction error graph saved")

# --------------------------------------------------
# 3. Error by Salary Range
# --------------------------------------------------

evaluation_data = prediction_data.copy()

evaluation_data["Salary Range"] = pd.cut(
    evaluation_data["Actual Salary"],
    bins=[
        0,
        30000,
        60000,
        100000,
        150000,
        250000,
        float("inf")
    ],
    labels=[
        "0-30K",
        "30K-60K",
        "60K-100K",
        "100K-150K",
        "150K-250K",
        "250K+"
    ]
)

range_error = (
    evaluation_data
    .groupby("Salary Range", observed=False)["Absolute Error"]
    .mean()
)

plt.figure(figsize=(9, 6))

range_error.plot(kind="bar")

plt.xlabel("Actual Salary Range")
plt.ylabel("Mean Absolute Error")
plt.title("Prediction Error by Salary Range")

plt.xticks(rotation=20)

plt.tight_layout()

range_error_path = os.path.join(
    OUTPUT_DIR,
    "error_by_salary_range.png"
)

plt.savefig(range_error_path)
plt.close()

print("Salary range error graph saved")

# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nFiles created:")

print("- predictions.csv")
print("- actual_vs_predicted.png")
print("- prediction_errors.png")
print("- error_by_salary_range.png")

print("\nBest model: Random Forest")
print("Evaluation completed successfully.")