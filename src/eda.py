"""Exploratory Data Analysis. Saves plots to eda_outputs/."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "eda_outputs")


def run_eda(df: pd.DataFrame, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Missing value analysis
    missing = df.isnull().sum()
    missing.to_csv(os.path.join(out_dir, "missing_values.csv"))

    # Histograms
    df.hist(figsize=(14, 12), bins=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histograms.png"))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
    plt.close()

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Class distribution (0 = no disease, 1 = disease)")
    plt.savefig(os.path.join(out_dir, "class_distribution.png"))
    plt.close()

    # Feature relationship (example: age vs target)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="target", y="age", data=df)
    plt.title("Age by target class")
    plt.savefig(os.path.join(out_dir, "age_vs_target.png"))
    plt.close()

    # Ranked correlation-with-target bar chart — surfaces the strongest
    # linear predictors at a glance, rather than reading them off a heatmap.
    corr_with_target = (
        df.corr(numeric_only=True)["target"]
        .drop("target")
        .sort_values()
    )
    plt.figure(figsize=(8, 6))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in corr_with_target]
    plt.barh(corr_with_target.index, corr_with_target.values, color=colors)
    plt.title("Feature correlation with target (signed)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "target_correlation_ranked.png"))
    plt.close()

    # Engineered-feature relationship: heart-rate reserve by target, if present
    if "hr_reserve" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x="target", y="hr_reserve", data=df)
        plt.title("Heart-rate reserve (220 - age - thalach) by target class")
        plt.savefig(os.path.join(out_dir, "hr_reserve_vs_target.png"))
        plt.close()

    print(f"EDA artifacts written to {out_dir}")


if __name__ == "__main__":
    from preprocessing import load_data, clean_data
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
    run_eda(clean_data(load_data(csv_path)))
