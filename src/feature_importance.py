# src/feature_importance.py
# Purpose: Find the most important features used by the Random Forest model

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "models/best_salary_model.joblib"
OUTPUT_DIR = "data/feature_importance"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DEVSalary AI - FEATURE IMPORTANCE")
print("=" * 60)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("\nBest model loaded successfully.")

# --------------------------------------------------
# Get preprocessing and Random Forest model
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["model"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get importance values
importance_values = random_forest.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# --------------------------------------------------
# Display top 20 features
# --------------------------------------------------

top_features = importance_df.head(20)

print("\n" + "-" * 60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("-" * 60)

print(top_features.to_string(index=False))

# --------------------------------------------------
# Save all feature importance values
# --------------------------------------------------

csv_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)

print("\nAll feature importance values saved:")
print(csv_path)

# --------------------------------------------------
# Create graph
# --------------------------------------------------

plot_data = top_features.sort_values(
    by="Importance"
)

plt.figure(figsize=(11, 8))

plt.barh(
    plot_data["Feature"],
    plot_data["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Features Influencing Salary Prediction")

plt.tight_layout()

graph_path = os.path.join(
    OUTPUT_DIR,
    "top_features.png"
)

plt.savefig(graph_path)
plt.close()

print("Feature importance graph saved:")
print(graph_path)

# --------------------------------------------------
# Final output
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE COMPLETED")
print("=" * 60)