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

# Save cleaned data
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned data saved to:", OUTPUT_FILE)