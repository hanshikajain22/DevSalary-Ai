# src/inspect_features.py
# Purpose: Look deeper at the target and candidate features.
# This script does NOT modify any data.

import pandas as pd

pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 200)

df = pd.read_csv("data/survey_results_public.csv", low_memory=False)
schema = pd.read_csv("data/survey_results_schema.csv")

TARGET = "ConvertedCompYearly"
# Only look at rows that actually have a salary value
salary_rows = df[df[TARGET].notnull()]

print("=" * 60)
print("A. SCHEMA FILE COLUMNS")
print(schema.columns.tolist())

print("\nB. SCHEMA DESCRIPTIONS FOR KEY COLUMNS")
key_columns = ["WorkExp", "YearsCode", "Employment", "DevType", "OrgSize",
               "ICorPM", "Industry", "ConvertedCompYearly", "RemoteWork"]
for column_name in key_columns:
    # find schema rows where any cell exactly matches the column name
    mask = schema.apply(lambda row: row.astype(str).eq(column_name).any(), axis=1)
    print("\n---", column_name, "---")
    print(schema[mask].to_string(index=False))

print("\nC. TARGET PERCENTILES (rows with salary)")
print("Rows with salary:", len(salary_rows))
print(salary_rows[TARGET].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]))

print("\nD. EXTREME VALUE COUNTS")
for limit in [100, 1000, 10000]:
    print("Salary below", limit, ":", (salary_rows[TARGET] < limit).sum())
for limit in [500000, 1000000]:
    print("Salary above", limit, ":", (salary_rows[TARGET] > limit).sum())

print("\nE. TOP 15 COUNTRIES (rows with salary)")
print(salary_rows["Country"].value_counts().head(15))
india_rows = salary_rows["Country"].astype(str).str.contains("India", na=False)
print("Rows with salary where Country contains 'India':", india_rows.sum())

print("\nF. CANDIDATE FEATURES: dtype, % missing, unique values (rows with salary)")
candidate_columns = ["Age", "EdLevel", "Employment", "WorkExp", "YearsCode",
                     "DevType", "OrgSize", "ICorPM", "RemoteWork", "Industry",
                     "Country", "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
                     "PlatformHaveWorkedWith", "WebframeHaveWorkedWith"]
for column_name in candidate_columns:
    missing_pct = salary_rows[column_name].isnull().mean() * 100
    print(f"{column_name:28s} dtype={str(salary_rows[column_name].dtype):8s} "
          f"missing={missing_pct:5.1f}%  unique={salary_rows[column_name].nunique()}")

print("\nG. VALUE COUNTS FOR SMALL CATEGORY COLUMNS (rows with salary)")
for column_name in ["Age", "EdLevel", "OrgSize", "RemoteWork", "ICorPM", "Industry"]:
    print("\n---", column_name, "---")
    print(salary_rows[column_name].value_counts(dropna=False).head(15))

print("\nH. EXPERIENCE COLUMNS")
print("WorkExp summary:")
print(salary_rows["WorkExp"].describe())
print("YearsCode sample unique values:")
print(salary_rows["YearsCode"].dropna().unique()[:20])

print("\nI. EMPLOYMENT and DEVTYPE (top 10 each, rows with salary)")
print(salary_rows["Employment"].value_counts().head(10))
print(salary_rows["DevType"].value_counts().head(10))