# src/preprocessing.py
# Purpose: Clean the raw Stack Overflow survey data for the salary model.
# Every rule here is based on what we saw in the real data (Phase 2 and 3.1).
#
# Run from the project folder (devsalary-ai) with:
#     python src/preprocessing.py

import pandas as pd

# ---------------------------------------------------------------
# Settings (kept in one place so they are easy to change/explain)
# ---------------------------------------------------------------
RAW_DATA_PATH = "data/survey_results_public.csv"
CLEAN_DATA_PATH = "data/clean_salary_data.csv"

TARGET_COLUMN = "ConvertedCompYearly"

MIN_SALARY = 1000          # below this is not a realistic yearly salary
MAX_SALARY = 500000        # above this is very rare and distorts error metrics
MAX_WORK_EXPERIENCE = 50   # 99th percentile was 43; values up to 100 are not credible

# Employment types where the salary value is ambiguous (not a current job)
EXCLUDED_EMPLOYMENT = ["Student", "Not employed", "Retired", "I prefer not to say"]

# Language name in the survey  ->  simple column name we will create (1 = uses it, 0 = does not)
LANGUAGE_COLUMNS = {
    "JavaScript": "lang_javascript",
    "HTML/CSS": "lang_html_css",
    "SQL": "lang_sql",
    "Python": "lang_python",
    "Bash/Shell (all shells)": "lang_bash_shell",
    "TypeScript": "lang_typescript",
    "C#": "lang_csharp",
    "Java": "lang_java",
    "PowerShell": "lang_powershell",
    "C++": "lang_cpp",
    "PHP": "lang_php",
    "C": "lang_c",
    "Go": "lang_go",
    "Rust": "lang_rust",
    "Kotlin": "lang_kotlin",
}

# Columns the model will use as input
NUMERIC_FEATURES = ["WorkExp", "YearsCode"]
CATEGORICAL_FEATURES = ["Country", "DevType", "EdLevel", "Employment",
                        "OrgSize", "RemoteWork", "ICorPM", "Industry"]
LANGUAGE_FEATURES = list(LANGUAGE_COLUMNS.values())


def has_language(language_text, language_name):
    """Return 1 if language_name is in the ';'-separated text, else 0.
    We split on ';' and compare whole names. Using text search instead
    would wrongly match 'C' inside 'C#' or 'C++'."""
    if pd.isnull(language_text):
        return 0
    language_list = [item.strip() for item in language_text.split(";")]
    return 1 if language_name in language_list else 0


def clean_survey_data(raw_df, verbose=True):
    """Apply the cleaning rules one by one and report how many rows remain."""

    def report(step_name, data):
        if verbose:
            print(f"{step_name:<55s} rows left: {len(data)}")

    df = raw_df.copy()          # never change the original DataFrame
    report("Start (raw data)", df)

    # Rule 1: a supervised model needs the correct answer, so keep rows with a salary
    df = df[df[TARGET_COLUMN].notnull()].copy()
    report("1. Keep rows that have a salary", df)

    # Rule 2: remove unrealistic salaries
    df = df[(df[TARGET_COLUMN] >= MIN_SALARY) & (df[TARGET_COLUMN] <= MAX_SALARY)].copy()
    report(f"2. Salary between {MIN_SALARY} and {MAX_SALARY}", df)

    # Rule 3: remove people who are not currently working
    df = df[~df["Employment"].isin(EXCLUDED_EMPLOYMENT)].copy()
    report("3. Remove student/not employed/retired/no answer", df)

    # Rule 4: the survey said "if your answer is 0, leave blank",
    # so a blank WorkExp is treated as 0 years (ASSUMPTION - state it in the report)
    blank_work_exp = df["WorkExp"].isnull().sum()
    df["WorkExp"] = df["WorkExp"].fillna(0)
    if verbose:
        print(f"   (blank WorkExp treated as 0 years: {blank_work_exp} rows)")
    df = df[df["WorkExp"] <= MAX_WORK_EXPERIENCE].copy()
    report(f"4. Remove WorkExp above {MAX_WORK_EXPERIENCE} years", df)

    # Rule 5: turn the multi-select language column into 1/0 columns.
    # A blank language answer becomes 0 for every language.
    for language_name, column_name in LANGUAGE_COLUMNS.items():
        df[column_name] = df["LanguageHaveWorkedWith"].apply(
            lambda text, name=language_name: has_language(text, name))

    # Rule 6: keep only the columns we need (everything else was reviewed and dropped)
    columns_to_keep = (NUMERIC_FEATURES + CATEGORICAL_FEATURES
                       + LANGUAGE_FEATURES + [TARGET_COLUMN])
    df = df[columns_to_keep].reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw_data = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    clean_data = clean_survey_data(raw_data)

    clean_data.to_csv(CLEAN_DATA_PATH, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("Saved:", CLEAN_DATA_PATH)
    print("Final shape (rows, columns):", clean_data.shape)

    print("\nMissing values per column (only columns that have any):")
    missing_counts = clean_data.isnull().sum()
    print(missing_counts[missing_counts > 0])

    print("\nTarget summary:")
    print(clean_data[TARGET_COLUMN].describe().round(0))

    print("\nRows per language flag (how many developers use each):")
    print(clean_data[LANGUAGE_FEATURES].sum())

    print("\nEmployment values left:")
    print(clean_data["Employment"].value_counts())
    print("\nRows from India:", (clean_data["Country"] == "India").sum())