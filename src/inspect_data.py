# src/inspect_data.py
# Purpose: Look at the REAL dataset. This script does NOT modify any data.

import pandas as pd

DATA_PATH = "data/survey_results_public.csv"

# low_memory=False avoids mixed-type warnings on large survey files
df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 60)
print("1. SHAPE (rows, columns)")
print(df.shape)

print("\n2. COLUMN NAMES")
for column_name in df.columns:
    print(column_name)

print("\n3. DATA TYPES (count of columns per type)")
print(df.dtypes.value_counts())

print("\n4. DUPLICATE ROWS")
print("Fully duplicated rows:", df.duplicated().sum())

print("\n5. MISSING VALUES (top 25 columns by % missing)")
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
print(missing_percent.head(25).round(1))

print("\n6. CANDIDATE SALARY COLUMNS (names containing 'comp' or 'salary')")
salary_candidates = [c for c in df.columns
                     if "comp" in c.lower() or "salary" in c.lower()]
print(salary_candidates)

print("\n7. SUMMARY OF EACH CANDIDATE SALARY COLUMN")
for column_name in salary_candidates:
    print("\n---", column_name, "---")
    print("Non-missing values:", df[column_name].notnull().sum())
    print("Data type:", df[column_name].dtype)
    if pd.api.types.is_numeric_dtype(df[column_name]):
        print(df[column_name].describe())

print("\n8. CANDIDATE EXPERIENCE / EDUCATION / LOCATION COLUMNS")
keywords = ["year", "ed", "country", "dev", "org", "remote", "employ"]
matches = [c for c in df.columns if any(k in c.lower() for k in keywords)]
print(matches)