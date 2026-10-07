# src/eda.py
# Purpose: Exploratory Data Analysis for developer salary data

import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/train.csv"
OUTPUT_DIR = "data/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("EDA started")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# --------------------------------------------------
# 1. Salary Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df["ConvertedCompYearly"], bins=50)
plt.xlabel("Yearly Salary")
plt.ylabel("Number of Developers")
plt.title("Distribution of Developer Salaries")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/salary_distribution.png")
plt.close()

print("1. Salary distribution saved")

# --------------------------------------------------
# 2. Salary vs Work Experience
# --------------------------------------------------

experience_salary = (
    df.groupby("WorkExp")["ConvertedCompYearly"]
    .median()
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.plot(
    experience_salary["WorkExp"],
    experience_salary["ConvertedCompYearly"],
    marker="o"
)
plt.xlabel("Years of Work Experience")
plt.ylabel("Median Yearly Salary")
plt.title("Median Salary vs Work Experience")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/salary_vs_experience.png")
plt.close()

print("2. Salary vs experience saved")

# --------------------------------------------------
# 3. Salary by Education
# --------------------------------------------------

education_salary = (
    df.groupby("EdLevel")["ConvertedCompYearly"]
    .median()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
education_salary.plot(kind="bar")
plt.xlabel("Education Level")
plt.ylabel("Median Yearly Salary")
plt.title("Median Salary by Education Level")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/salary_by_education.png")
plt.close()

print("3. Salary by education saved")

# --------------------------------------------------
# 4. Salary by Employment Type
# --------------------------------------------------

employment_salary = (
    df.groupby("Employment")["ConvertedCompYearly"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))
employment_salary.plot(kind="bar")
plt.xlabel("Employment Type")
plt.ylabel("Median Yearly Salary")
plt.title("Median Salary by Employment Type")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/salary_by_employment.png")
plt.close()

print("4. Salary by employment saved")

# --------------------------------------------------
# 5. Salary by Remote Work
# --------------------------------------------------

remote_salary = (
    df.groupby("RemoteWork")["ConvertedCompYearly"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))
remote_salary.plot(kind="bar")
plt.xlabel("Remote Work Type")
plt.ylabel("Median Yearly Salary")
plt.title("Median Salary by Remote Work Type")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/salary_by_remote.png")
plt.close()

print("5. Salary by remote work saved")

# --------------------------------------------------
# 6. Top Countries by Median Salary
# --------------------------------------------------

country_data = (
    df.groupby("Country")
    .agg(
        median_salary=("ConvertedCompYearly", "median"),
        developer_count=("ConvertedCompYearly", "count")
    )
)

# Keep countries with at least 50 developers
country_data = country_data[
    country_data["developer_count"] >= 50
]

top_countries = (
    country_data["median_salary"]
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(11, 7))
top_countries.sort_values().plot(kind="barh")
plt.xlabel("Median Yearly Salary")
plt.ylabel("Country")
plt.title("Top Countries by Median Developer Salary")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_countries_salary.png")
plt.close()

print("6. Country salary graph saved")

# --------------------------------------------------
# 7. Salary by Developer Type
# --------------------------------------------------

devtype_salary = (
    df.groupby("DevType")["ConvertedCompYearly"]
    .median()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(11, 7))
devtype_salary.sort_values().plot(kind="barh")
plt.xlabel("Median Yearly Salary")
plt.ylabel("Developer Type")
plt.title("Top Developer Types by Median Salary")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/salary_by_devtype.png")
plt.close()

print("7. Developer type graph saved")

# --------------------------------------------------
# 8. Correlation Analysis
# --------------------------------------------------

numeric_columns = [
    "WorkExp",
    "YearsCode",
    "ConvertedCompYearly"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(7, 5))
plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Between Numeric Features")

for i in range(len(correlation)):
    for j in range(len(correlation)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation.png")
plt.close()

print("8. Correlation graph saved")

# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("EDA COMPLETED")
print("=" * 55)

print("Graphs saved in:", OUTPUT_DIR)
print("Total graphs created: 8")