import joblib
import pandas as pd

MODEL_PATH = "models/best_salary_model.joblib"

print("=" * 60)
print("DEVSalary AI - SALARY PREDICTION")
print("=" * 60)

# Load model
model = joblib.load(MODEL_PATH)

print("\nModel loaded successfully.")

# --------------------------------------------------
# User Input
# --------------------------------------------------

print("\nEnter developer details:")

work_exp = float(input("Years of Work Experience: "))
years_code = float(input("Years of Coding Experience: "))

country = input("Country: ")
dev_type = input("Developer Type: ")
ed_level = input("Education Level: ")
employment = input("Employment Type: ")
org_size = input("Organization Size: ")
remote_work = input("Remote Work: ")
ic_or_pm = input("Individual Contributor or Manager: ")
industry = input("Industry: ")

# --------------------------------------------------
# Language Inputs
# --------------------------------------------------

print("\nEnter programming language usage.")
print("Type yes or no.")

languages = [
    "javascript",
    "html_css",
    "sql",
    "python",
    "bash_shell",
    "typescript",
    "csharp",
    "java",
    "powershell",
    "cpp",
    "php",
    "c",
    "go",
    "rust",
    "kotlin"
]

language_values = {}

for language in languages:
    answer = input(f"{language}: ").lower()
    language_values[f"lang_{language}"] = 1 if answer == "yes" else 0

# --------------------------------------------------
# Create input DataFrame
# --------------------------------------------------

data = {
    "WorkExp": [work_exp],
    "YearsCode": [years_code],
    "Country": [country],
    "DevType": [dev_type],
    "EdLevel": [ed_level],
    "Employment": [employment],
    "OrgSize": [org_size],
    "RemoteWork": [remote_work],
    "ICorPM": [ic_or_pm],
    "Industry": [industry]
}

data.update({
    key: [value]
    for key, value in language_values.items()
})

input_data = pd.DataFrame(data)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = model.predict(input_data)

salary = prediction[0]

print("\n" + "=" * 60)
print("SALARY PREDICTION")
print("=" * 60)

print(f"\nPredicted Annual Salary: ${salary:,.2f}")

print("\nPrediction completed successfully.")