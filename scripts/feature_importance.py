import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper import build_pretty_name_mapping

# Paths
model_path = "models"
save_folder = "visuals/feature_importance"
os.makedirs(save_folder, exist_ok=True)

# =========================
# RANDOM FOREST FEATURE IMPORTANCE
# =========================
rf_pipeline = joblib.load(os.path.join(model_path, "RandomForest_Balanced.joblib"))
rf_model = rf_pipeline.named_steps['model']
preprocessor_rf = rf_pipeline.named_steps['preprocessor']

rf_features = preprocessor_rf.get_feature_names_out()
rf_importances = rf_model.feature_importances_

# Indices of features sorted by importance (descending)
rf_indices = np.argsort(rf_importances)[::-1]

# Pretty name mapping
pretty_map_rf = build_pretty_name_mapping(preprocessor_rf)

# ---- TOP 15 ----
rf_top15_features = rf_features[rf_indices[:15]]
rf_top15_importances = rf_importances[rf_indices[:15]]

rf_top15 = pd.DataFrame({
    "Feature": rf_top15_features,
    "Pretty": [pretty_map_rf.get(f, f) for f in rf_top15_features],
    "Importance": rf_top15_importances
}).sort_values(by="Importance", ascending=False)

print("\n=== Random Forest Feature Importance (Top 15) ===")
print(rf_top15[["Pretty", "Importance"]].to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(rf_top15["Pretty"], rf_top15["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "RF_15_features_pretty.png"))
plt.close()

# ---- BOTTOM 15 (least important) ----
rf_bottom15_features = rf_features[rf_indices[-15:]]
rf_bottom15_importances = rf_importances[rf_indices[-15:]]

rf_bottom15 = pd.DataFrame({
    "Feature": rf_bottom15_features,
    "Pretty": [pretty_map_rf.get(f, f) for f in rf_bottom15_features],
    "Importance": rf_bottom15_importances
}).sort_values(by="Importance", ascending=True)

print("\n=== Random Forest Least Important Features (Bottom 15) ===")
print(rf_bottom15[["Pretty", "Importance"]].to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(rf_bottom15["Pretty"], rf_bottom15["Importance"])
plt.title("Least 15 Important Features (Random Forest)")
plt.xlabel("Importance (lower → less predictive power)")
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "RF_least15_features_pretty.png"))
plt.close()

# ---- Combined RF summary (Top + Bottom) ----
rf_top15_labeled = rf_top15.copy()
rf_top15_labeled["Group"] = "Top"

rf_bottom15_labeled = rf_bottom15.copy()
rf_bottom15_labeled["Group"] = "Bottom"

rf_summary = pd.concat([rf_top15_labeled, rf_bottom15_labeled], ignore_index=True)

# Save all tables to CSV
rf_top15.to_csv(os.path.join(save_folder, "rf_top15_importances.csv"), index=False)
rf_bottom15.to_csv(os.path.join(save_folder, "rf_bottom15_importances.csv"), index=False)
rf_summary.to_csv(os.path.join(save_folder, "rf_top_bottom_summary.csv"), index=False)

print("\n=== Random Forest Summary (Top + Bottom 15) ===")
print(rf_summary[["Group", "Pretty", "Importance"]].to_string(index=False))

# =========================
# LOGISTIC REGRESSION FEATURE IMPORTANCE
# (your LR Yes + No code goes here, unchanged)
# =========================

lr_pipeline = joblib.load(os.path.join(model_path, "LogisticRegression.joblib"))
lr_model = lr_pipeline.named_steps['model']
preprocessor_lr = lr_pipeline.named_steps['preprocessor']

lr_features = preprocessor_lr.get_feature_names_out()
pretty_map_lr = build_pretty_name_mapping(preprocessor_lr)

classes = lr_model.classes_

# ---- Class 1 = "Yes" ----
yes_class = 1
yes_idx = list(classes).index(yes_class)
coeffs_yes = lr_model.coef_[yes_idx]

lr_yes_importances = pd.DataFrame({
    "Feature": lr_features,
    "Pretty": [pretty_map_lr.get(f, f) for f in lr_features],
    "Coefficient": coeffs_yes,
    "Abs_Coefficient": np.abs(coeffs_yes)
}).sort_values(by="Abs_Coefficient", ascending=False).head(15)

print("\n=== Logistic Regression Coefficients (Class = Yes, Top 15) ===")
print(lr_yes_importances[["Pretty", "Coefficient"]].to_string(index=False))

lr_yes_importances.to_csv(os.path.join(save_folder, "lr_yes_top15_coeffs.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.barh(lr_yes_importances["Pretty"], lr_yes_importances["Coefficient"])
plt.gca().invert_yaxis()
plt.title("Top 15 Influential Features (Logistic Regression — Yes)")
plt.xlabel("Coefficient (positive → more likely to seek help)")
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "LR_15_features_yes_pretty.png"))
plt.close()

# ---- Class 0 = "No" ----
if 0 in classes:
    no_class = 0
    no_idx = list(classes).index(no_class)
    coeffs_no = lr_model.coef_[no_idx]

    lr_no_importances = pd.DataFrame({
        "Feature": lr_features,
        "Pretty": [pretty_map_lr.get(f, f) for f in lr_features],
        "Coefficient": coeffs_no,
        "Abs_Coefficient": np.abs(coeffs_no)
    }).sort_values(by="Abs_Coefficient", ascending=False).head(15)

    print("\n=== Logistic Regression Coefficients (Class = No, Top 15) ===")
    print(lr_no_importances[["Pretty", "Coefficient"]].to_string(index=False))

    lr_no_importances.to_csv(os.path.join(save_folder, "lr_no_top15_coeffs.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(lr_no_importances["Pretty"], lr_no_importances["Coefficient"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Influential Features (Logistic Regression — No)")
    plt.xlabel("Coefficient (positive → more likely to predict 'No')")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "LR_15_features_no_pretty.png"))
    plt.close()
else:
    print("\n[WARN] Class 0 ('No') not found in Logistic Regression classes:", classes)