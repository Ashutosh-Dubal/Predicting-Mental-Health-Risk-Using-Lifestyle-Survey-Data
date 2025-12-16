import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, confusion_matrix
from helper import extract_estimator_with_attr

save_folder = "visuals/model_training"
os.makedirs(save_folder, exist_ok=True)

# LOADING DATA

CLEAN_DATA_PATH = "data/clean/survey.csv"
df = pd.read_csv(CLEAN_DATA_PATH)
df = df.fillna(-1)

target = "seek_help_cleaned"
drop_cols = ['Country', 'Gender_cleaned']
X = df.drop(columns=[target] + drop_cols)
y = df[target]
# print(len(X)) - 21

# CATEGORICAL AND NUMERIC

categorical_cols = [col for col in X.columns if col.endswith('_cleaned')]
if target in categorical_cols:
    categorical_cols.remove(target)

numeric_cols = ['Age', 'Employees_estimate']

for col in categorical_cols:
    X[col] = X[col].astype('category')

# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# PREPROCESSING : ONE HOT ENCODING

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)],
    remainder='passthrough'
)

# DEFINE MODELS

models = {
    "RandomForest_Balanced" : RandomForestClassifier(
        n_estimators=200, class_weight= 'balanced', random_state=42
    ),
    "CategoricalNB": CategoricalNB(),
    "LogisticRegression": LogisticRegression(
        max_iter=5000, class_weight='balanced', random_state=42
    )
}

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# TRAIN AND SAVE MODELS

results = []

for name, model in models.items():
    clf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    # Train final model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 {name}")
    print(f"Average CV F1 (macro): {mean_cv:.3f} ± {std_cv:.3f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", cm)

    # Save model - Pipeline
    model_path = f"models/{name}.joblib"
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to: {model_path}")

    # Save model - Core
    if name in ("RandomForest_Balanced"):
        core = extract_estimator_with_attr(model, "feature_importances_")
        joblib.dump(core, os.path.join("models", f"{name}_core.joblib"))
    elif name in ("LogisticRegression"):
        core = extract_estimator_with_attr(model, "coef_")
        joblib.dump(core, os.path.join("models", f"{name}_core.joblib"))

    # Save results
    results.append({
        "Model": name,
        "CV_F1_mean": mean_cv,
        "CV_F1_std": std_cv,
        "Test_Accuracy": report["accuracy"],
        "Test_F1_macro": report["macro avg"]["f1-score"]
    })

# SAVE MODEL PERFORMANCE
results_df = pd.DataFrame(results)
results_df.to_csv("models/model_performance_summary.csv", index=False)
print("\nModel performance summary saved to models/model_performance_summary.csv")

print("\n=== VotingClassifier (RF + LR + CategoricalNB) ===")
voting_clf = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42
        )),
        ("lr", LogisticRegression(
            max_iter=5000, class_weight='balanced', random_state=42
        )),
        ("nb", CategoricalNB())
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print(classification_report(y_test, y_pred_voting))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_voting))

model_path = f"models/voting_clf.joblib"
joblib.dump(voting_clf, model_path)
print(f"✅ Model saved to: {model_path}")

# FEATURE IMPORTANCE PLOT

rf_model = joblib.load("models/RandomForest_Balanced.joblib")
rf_feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()
all_features = np.append(rf_feature_names, numeric_cols)

importances = rf_model.named_steps['model'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(15), importances[indices[:15]], align='center')
plt.xticks(range(15), [all_features[i] for i in indices[:15]], rotation=75)
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "RF_15_features.png"))
plt.close()

