import os
import pandas as pd
import numpy as np
from helper import data_audit

# Paths
RAW_DATA_PATH = "data/raw/survey.csv"
CLEAN_DATA_PATH = "data/clean"
OUTPUT_FILE = os.path.join(CLEAN_DATA_PATH, "survey.csv")

# Load
df = pd.read_csv(RAW_DATA_PATH)

# Checking for missing or inconsistencies in dataset
data_audit(df)

df = df.replace(r'^\s*$', np.nan, regex=True)

# Normalize Gender column: lowercase, strip whitespace
df['Gender_cleaned'] = df['Gender'].str.lower().str.strip()

# Define gender mappings
male_labels = {
    'male', 'm', 'male-ish', 'maile', 'cis male', 'male (cis)',
    'man', 'mal', 'make', 'msle', 'mail', 'malr', 'cis man'
}
female_labels = {
    'female', 'f', 'cis female', 'woman', 'femake', 'femail',
    'female (cis)', 'female (trans)', 'trans-female', 'trans woman'
}

# Map values to Male/Female/Other
def clean_gender(g):
    if g in male_labels:
        return 'Male'
    elif g in female_labels:
        return 'Female'
    else:
        return 'Other'

df['Gender_cleaned'] = df['Gender_cleaned'].apply(clean_gender)

employee_estimates = {
    '1-5': 3,
    '6-25': 15,
    '26-100': 63,
    '100-500': 300,
    '500-1000': 750,
    'More than 1000': 1500  # Or any large estimate
}
mapping_work = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Unknown": 4
}

mapping_yes_no = {
    "No" : 0,
    "Yes" : 1,
    "Some of them" : 2,
    "Not sure" : 3,
    "Don't know" : 3,
    "Maybe" : 3
}

mapping_leave = {
    "Very easy" : 1,
    "Somewhat easy" : 2,
    "Don't know" : 3,
    "Somewhat difficult" : 4,
    "Very difficult" : 5
}

df['Employees_estimate'] = df['no_employees'].map(employee_estimates)
df['work_interfere_cleaned'] = df['work_interfere'].map(mapping_work)
df['leave_cleaned'] = df['leave'].map(mapping_leave)

yes_no = ['self_employed', 'family_history', 'treatment', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program',
          'seek_help', 'anonymity', 'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
          'phys_health_interview', 'mental_vs_physical', 'obs_consequence']

for col in yes_no:
    df[col + '_cleaned'] = df[col].map(mapping_yes_no)

print(df.head())

# Make sure the clean folder exists
os.makedirs(CLEAN_DATA_PATH, exist_ok=True)

# Taking the clean subset of the df
keep_cols = ['Age', 'Country'] + list(df.columns[27:])
df_clean = df[keep_cols].copy()

# Imputing values
df_clean['self_employed_cleaned'] = df_clean['self_employed_cleaned'].fillna(0)
df_clean['work_interfere_cleaned'] = df_clean['work_interfere_cleaned'].fillna(4)

# Check Age distribution
print("Age Summary:")
print(df_clean['Age'].describe())
print("\nUnique Ages (lowest 10):", sorted(df_clean['Age'].unique())[:10])
print("Unique Ages (highest 10):", sorted(df_clean['Age'].unique())[-10:])

# Check for weird or extreme ages
weird_ages = df_clean[(df_clean['Age'] < 10) | (df_clean['Age'] > 100)]
print("\nWeird ages detected:")
print(weird_ages[['Age']].value_counts())

# Check Country values
print("\nNumber of unique countries:", df_clean['Country'].nunique())
print("Most common countries:\n", df_clean['Country'].value_counts().head(15))

# Look for messy country names (short ones, weird ones)
print("\nPotentially messy country entries:")
print([c for c in df_clean['Country'].unique() if len(str(c)) <= 3])

df_clean = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 100)]
print("New shape after dropping valid ages: ", df_clean.shape)

# Save cleaned data
df_clean.to_csv(OUTPUT_FILE, index=False)

print("Cleaned data saved to:", OUTPUT_FILE)