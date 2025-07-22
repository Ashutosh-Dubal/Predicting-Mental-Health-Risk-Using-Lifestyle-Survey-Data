import os
import pandas as pd
from helper import data_audit

# Paths
RAW_DATA_PATH = "data/raw/survey.csv"
CLEAN_DATA_PATH = "data/clean"
OUTPUT_FILE = os.path.join(CLEAN_DATA_PATH, "survey.csv")

# Load
df = pd.read_csv(RAW_DATA_PATH)

# Checking for missing or inconsistencies in dataset
data_audit(df)

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

print(df.head())

# Make sure the clean folder exists
os.makedirs(CLEAN_DATA_PATH, exist_ok=True)

# Save cleaned data
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned data saved to:", OUTPUT_FILE)