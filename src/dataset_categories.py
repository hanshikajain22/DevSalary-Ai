import pandas as pd

TRAIN_PATH = "data/train.csv"

df = pd.read_csv(TRAIN_PATH)

print("=" * 70)
print("DEVSalary AI - DATASET CATEGORIES")
print("=" * 70)

columns = [
    "Country",
    "DevType",
    "EdLevel",
    "Employment",
    "OrgSize",
    "RemoteWork",
    "ICorPM",
    "Industry"
]

for column in columns:
    print("\n" + "-" * 70)
    print(f"{column}")
    print("-" * 70)

    values = sorted(
        df[column].dropna().astype(str).unique()
    )

    print(f"Total categories: {len(values)}")

    for value in values:
        print(value)

print("\n" + "=" * 70)
print("CATEGORY EXTRACTION COMPLETED")
print("=" * 70)