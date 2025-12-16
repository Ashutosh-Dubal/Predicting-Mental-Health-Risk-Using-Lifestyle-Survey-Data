import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
CLEAN_DATA_PATH = "data/clean/survey.csv"
RAW_DATA_PATH = "data/raw/survey.csv"
save_folder = "visuals/eda"
os.makedirs(save_folder, exist_ok=True)

# Load
df_clean = pd.read_csv(CLEAN_DATA_PATH)

# Clean subset
# keep_cols = ['Age', 'Country'] + list(df.columns[27:])
# df_clean = df[keep_cols].copy()

# Define targets to analyze
targets = {
    "seek_help_cleaned": "Seek Help",
    "treatment_cleaned": "Treatment"
}

# Label maps
binary_map = {0: "No", 1: "Yes"}
seek_help_map = {0: "No", 1: "Yes", 3: "Uncertain"}
treatment_map = {0: "No", 1: "Yes"}
support_map = {0: "No", 1: "Yes", 2: "Some of them"}  # for coworkers/supervisor
work_map = {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Often", 4: "Unknown"}

target_label_map = {
    "seek_help_cleaned": seek_help_map,
    "treatment_cleaned": treatment_map
}

feature_label_map = {
    "family_history_cleaned": binary_map,
    "supervisor_cleaned": support_map,
    "coworkers_cleaned": support_map,
    "self_employed_cleaned": binary_map
}

social_factors = [
    ("family_history_cleaned", "Family History"),
    ("supervisor_cleaned", "Supervisor Support"),
    ("coworkers_cleaned", "Coworker Support"),
    ("self_employed_cleaned", "Self-Employed"),
]

# ---------- Demographics ----------
for target, title in targets.items():
    # Age groups
    df_clean['Age_group'] = pd.cut(
        df_clean['Age'],
        bins=[18, 25, 35, 50, 100],
        labels=["18-25", "26-35", "36-50", "50+"]
    )

    ct = pd.crosstab(df_clean['Age_group'], df_clean[target], normalize='index')
    ct.plot(kind="bar", stacked=True, colormap="tab10", figsize=(8,6))
    plt.title(f"{title} Responses by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Proportion")
    plt.legend(title=title, labels=["No", "Yes", "Uncertain"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"age_vs_{target}.png"))
    plt.close()

    # Gender
    ct = pd.crosstab(df_clean['Gender_cleaned'], df_clean[target], normalize='index')
    ct.plot(kind="bar", stacked=True, colormap="tab10", figsize=(8,6))
    plt.title(f"{title} Responses by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Proportion")
    plt.legend(title=title, labels=["No", "Yes", "Uncertain"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"gender_vs_{target}.png"))
    plt.close()

    # Country (Top 10)
    top_countries = df_clean['Country'].value_counts().head(10).index
    df_top_countries = df_clean[df_clean['Country'].isin(top_countries)]
    ct = pd.crosstab(df_top_countries['Country'], df_top_countries[target], normalize='index')
    ct.plot(kind="bar", stacked=True, colormap="tab10", figsize=(10,6))
    plt.title(f"{title} Responses by Top 10 Countries")
    plt.xlabel("Country")
    plt.ylabel("Proportion")
    plt.legend(title=title, labels=["No", "Yes", "Uncertain"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"country_vs_{target}.png"))
    plt.close()

    # Company size
    ct = pd.crosstab(df_clean['Employees_estimate'], df_clean[target], normalize='index')
    ct.plot(kind="bar", stacked=True, colormap="tab10", figsize=(8,6))
    plt.title(f"{title} Responses by Company Size")
    plt.xlabel("Company Size (Employees)")
    plt.ylabel("Proportion")
    plt.legend(title=title, labels=["No", "Yes", "Uncertain"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"company_size_vs_{target}.png"))
    plt.close()

    # ---------- Correlations ----------
    corr = df_clean.corr(numeric_only=True)
    top10_features = corr[target].drop(target).sort_values(ascending=False).head(10).index
    print(top10_features)
    subset_features = top10_features.tolist() + [target]
    top_corr_matrix = corr.loc[subset_features, subset_features]

    plt.figure(figsize=(10,8))
    sns.heatmap(top_corr_matrix, annot=True, cmap="coolwarm")
    plt.title(f"Top Correlations with {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"top_corr_heatmap_{target}.png"))
    plt.close()

    # Barplot of correlations
    top_corr = corr[target].drop(target).sort_values(ascending=False).head(10)
    sns.barplot(x=top_corr.values, y=top_corr.index, palette="coolwarm")
    plt.title(f"Top 10 Correlations with {title}")
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"top_corr_barplot_{target}.png"))
    plt.close()

# ---------- Social/Personal Context ----------

for target, title in targets.items():
    for col, label in social_factors:
        plot_df = df_clean.copy()

        # Apply mappings for both target and factor
        plot_df[target + "_plot"] = plot_df[target].map(target_label_map[target])
        plot_df[col + "_plot"] = plot_df[col].map(feature_label_map[col])

        # Countplot
        sns.countplot(x=col + "_plot", hue=target + "_plot", data=plot_df)
        plt.title(f"{label} vs {title}")
        plt.xlabel(label)
        plt.ylabel("Count")
        plt.legend(title=title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{col}_vs_{target}_count.png"))
        plt.close()

        # Proportion barplot
        sns.barplot(
            x=col + "_plot",
            y=target,
            data=plot_df,
            estimator=lambda x: sum(x)/len(x)
        )
        plt.title(f"{label} vs {title}")
        plt.xlabel(label)
        plt.ylabel(f"Proportion {title}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{col}_vs_{target}_barplot.png"))
        plt.close()

        # Stacked bar plot
        ct.plot(kind="bar", stacked=True, figsize=(8, 6), colormap="tab10")
        plt.title(f"{label} vs {title} (Stacked Proportions)")
        plt.xlabel(label)
        plt.ylabel("Proportion")
        plt.legend(title=title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{col}_vs_{target}_stacked_bar.png"))
        plt.close()

        # Heatmap (proportion table)
        ct = pd.crosstab(plot_df[col + "_plot"], plot_df[target + "_plot"], normalize='index')
        sns.heatmap(ct, annot=True, cmap="Blues", cbar=False, fmt=".2f")
        plt.title(f"Proportion of {title} by {label} (Heatmap)")
        plt.xlabel(title)
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{col}_vs_{target}_heatmap.png"))
        plt.close()