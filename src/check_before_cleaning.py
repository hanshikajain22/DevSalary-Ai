# src/check_before_cleaning.py
# Purpose: Read-only checks BEFORE cleaning. Does NOT modify any data.
# Output is written to data_report3.txt (UTF-8) so special characters are safe.

import contextlib
import pandas as pd

df = pd.read_csv("data/survey_results_public.csv", low_memory=False)
TARGET = "ConvertedCompYearly"
salary_rows = df[df[TARGET].notnull()]

with open("data_report3.txt", "w", encoding="utf-8") as report_file:
    with contextlib.redirect_stdout(report_file):

        print("=" * 60)
        print("1. LOW SALARIES (< 1000) BY COUNTRY (top 10)")
        low_salary_rows = salary_rows[salary_rows[TARGET] < 1000]
        print(low_salary_rows["Country"].value_counts().head(10))
        print("\nLow-salary rows by Employment:")
        print(low_salary_rows["Employment"].value_counts())

        print("\n2. ENCODING CHECK for EdLevel (repr shows the real characters)")
        for value in salary_rows["EdLevel"].dropna().unique()[:4]:
            print(repr(value))

        print("\n3. WorkExp CHECKS (rows with salary)")
        work_exp = salary_rows["WorkExp"]
        print("Missing WorkExp:", work_exp.isnull().sum())
        print("WorkExp > 50:", (work_exp > 50).sum())
        print("WorkExp > 60:", (work_exp > 60).sum())
        print("WorkExp percentiles:")
        print(work_exp.quantile([0.5, 0.9, 0.95, 0.99]))
        print("Missing WorkExp by Employment:")
        print(salary_rows[work_exp.isnull()]["Employment"].value_counts())

        print("\n4. WorkExp vs YearsCode")
        both = salary_rows[["WorkExp", "YearsCode"]].dropna()
        print("Rows with both:", len(both))
        print("Correlation:", round(both["WorkExp"].corr(both["YearsCode"]), 3))
        print("WorkExp > YearsCode:", (both["WorkExp"] > both["YearsCode"]).sum())

        print("\n5. LANGUAGE COLUMN SAMPLES")
        for value in salary_rows["LanguageHaveWorkedWith"].dropna().head(5):
            print(repr(value))
        languages = (salary_rows["LanguageHaveWorkedWith"]
                     .dropna().str.split(";").explode().str.strip())
        print("\nTop 20 individual languages:")
        print(languages.value_counts().head(20))
        print("Total distinct languages:", languages.nunique())

        print("\n6. ROWS REMAINING UNDER PROPOSED FILTERS (counts only)")
        bad_employment = ["Student", "Not employed", "Retired", "I prefer not to say"]
        keep_rows = salary_rows[
            (salary_rows[TARGET] >= 1000)
            & (salary_rows[TARGET] <= 500000)
            & (~salary_rows["Employment"].isin(bad_employment))
        ]
        print("Rows after filters:", len(keep_rows))
        print("India rows after filters:",
              (keep_rows["Country"] == "India").sum())

print("Done. Open data_report3.txt")
