import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

CLEAN_DATA_PATH = "data/clean/survey.csv"
df = pd.read_csv(CLEAN_DATA_PATH)

target = "seek_help_cleaned"
drop_cols = ['Country', 'state', 'Gender_cleaned']
X = df.drop(columns=[target] + drop_cols)
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confustion Matrix:")
print(confusion_matrix(y_test, y_pred))