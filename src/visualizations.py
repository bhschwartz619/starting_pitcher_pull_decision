from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------
decision_output_path = TABLES_DIR / "decision_framework_output.csv"
re_table_path = TABLES_DIR / "run_expectancy_table.csv"

df = pd.read_csv(decision_output_path)
re_df = pd.read_csv(re_table_path)


# --------------------------------------------------
# 1. Decision Value by Inning
# --------------------------------------------------
inning_df = df.groupby("inning")["decision_value"].mean()

plt.figure()
inning_df.plot(marker="o", linewidth=2)

plt.title("Managers Deviate More in Late Innings", fontsize=14)
plt.xlabel("Inning")
plt.ylabel("Runs Lost per Decision")

plt.xticks(range(1, 10))
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "decision_value_by_inning.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #1 saved")


# --------------------------------------------------
# 2. Decision Value by TTO
# --------------------------------------------------
tto_df = df.groupby("tto_bucket")["decision_value"].mean()

plt.figure()
tto_df.plot(kind="bar")

plt.title("Third Time Through the Order Drives Inefficiency", fontsize=14)
plt.xlabel("Times Through Order")
plt.ylabel("Runs Lost per Decision")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "decision_value_by_tto.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #2 saved")


# --------------------------------------------------
# 3. Pull Advantage Distribution
# --------------------------------------------------
plt.figure(figsize=(9, 5))

df["pull_advantage"].hist(bins=50)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Break-even",
    color="orange",
)

plt.axvline(
    0.03,
    linestyle="--",
    linewidth=2,
    label="Pull threshold",
    color="green",
)

plt.title("Distribution of Pull Advantage")

plt.xlabel(
    "Run Difference: Starter Expected Runs - Reliever Expected Runs"
)

plt.ylabel("Number of Situations")

plt.text(
    -0.24,
    50000,
    "Most situations favor\nkeeping the starter",
    fontsize=10,
)

plt.text(
    0.04,
    20000,
    "Pull recommended\nwhen advantage exceeds threshold",
    fontsize=10,
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "pull_advantage_distribution_annotated.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #3 saved")


# --------------------------------------------------
# 4. Run Expectancy Heatmap
# --------------------------------------------------
re_df["base_state"] = (
    re_df["base_state"]
    .astype(int)
    .astype(str)
    .str.zfill(3)
)

re_pivot = re_df.pivot(
    index="outs",
    columns="base_state",
    values="runs_from_state",
)

base_order = ["000", "100", "010", "110", "001", "101", "011", "111"]

re_pivot = re_pivot.reindex(columns=base_order)

plt.figure(figsize=(10, 4))

plt.imshow(re_pivot, aspect="auto")

plt.colorbar(label="Expected Runs")

plt.xticks(range(len(re_pivot.columns)), re_pivot.columns)
plt.yticks(range(len(re_pivot.index)), re_pivot.index)

plt.xlabel("Base State (1B, 2B, 3B)")
plt.ylabel("Outs")

plt.title("Run Expectancy by Game State")

for i in range(re_pivot.shape[0]):
    for j in range(re_pivot.shape[1]):
        value = re_pivot.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "run_expectancy_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #4 saved")


# --------------------------------------------------
# 5. Model vs Manager Pull Rates
# --------------------------------------------------
pull_rates = pd.Series(
    {
        "Model": (df["model_recommendation"] == "pull").mean(),
        "Manager": (df["manager_decision"] == "pull").mean(),
    }
)

plt.figure(figsize=(7, 5))

pull_rates.plot(kind="bar")

plt.title("Model vs. Manager Pull Rates")
plt.ylabel("Pull Rate")
plt.xlabel("")

plt.xticks(rotation=0)

for i, value in enumerate(pull_rates):
    plt.text(i, value + 0.002, f"{value:.1%}", ha="center")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_vs_manager_pull_rates.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #5 saved")


# --------------------------------------------------
# 6. Decision Value Distribution
# --------------------------------------------------
plt.figure(figsize=(9, 5))

df["decision_value"].hist(bins=60)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Optimal decision",
)

plt.title("Distribution of Decision Value")

plt.xlim(right=0.125)

plt.xlabel("Runs Lost Relative to Model")
plt.ylabel("Number of Decisions")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "decision_value_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #6 saved")


# --------------------------------------------------
# 7. Manager Pull Rate by Pull Advantage
# --------------------------------------------------
df["pull_advantage_bucket"] = pd.cut(
    df["pull_advantage"],
    bins=20,
)

bucket_df = (
    df.groupby("pull_advantage_bucket", observed=False)
    .agg(
        manager_pull_rate=(
            "manager_decision",
            lambda x: (x == "pull").mean(),
        ),
        count=("manager_decision", "size"),
    )
    .reset_index()
)

bucket_df["bucket_midpoint"] = (
    bucket_df["pull_advantage_bucket"]
    .apply(lambda x: x.mid)
)

plt.figure(figsize=(9, 5))

plt.plot(
    bucket_df["bucket_midpoint"],
    bucket_df["manager_pull_rate"],
    marker="o",
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Break-even",
    color="orange",
)

plt.axvline(
    0.03,
    linestyle="--",
    linewidth=2,
    label="Model pull threshold",
    color="green",
)

plt.title("Manager Pull Rate by Model Pull Advantage")

plt.xlabel(
    "Pull Advantage: Starter Expected Runs - Reliever Expected Runs"
)

plt.ylabel("Manager Pull Rate")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "manager_pull_rate_by_advantage.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #7 saved")


# --------------------------------------------------
# 8. Manager vs Model Confusion Matrix
# --------------------------------------------------
comparison = pd.crosstab(
    df["model_recommendation"],
    df["manager_decision"],
    normalize="index",
)

plt.figure(figsize=(6, 5))

plt.imshow(comparison, aspect="auto")

plt.colorbar(label="Share of Model Recommendation")

plt.xticks(
    range(len(comparison.columns)),
    comparison.columns,
)

plt.yticks(
    range(len(comparison.index)),
    comparison.index,
)

plt.xlabel("Manager Decision")
plt.ylabel("Model Recommendation")

plt.title("Manager Decisions vs. Model Recommendations")

for i in range(comparison.shape[0]):
    for j in range(comparison.shape[1]):
        value = comparison.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:.1%}",
            ha="center",
            va="center",
        )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "manager_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Visualization #8 saved")