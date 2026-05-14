from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_decision_output(file_path: Path) -> pd.DataFrame:
    """
    Load decision framework output.
    """

    return pd.read_csv(file_path, low_memory=False)


# --------------------------------------------------
# Manager/team identification
# --------------------------------------------------
def add_manager_unit(decision_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Identify the manager or team unit for each pitching decision.

    If manager IDs are available, use the pitching manager.
    Otherwise, fall back to the fielding team as a team-level proxy.
    """

    decision_df = decision_df.copy()

    if {"HOME_MGR_ID", "AWAY_MGR_ID", "BAT_HOME_ID"}.issubset(decision_df.columns):
        decision_df["manager_unit"] = np.where(
            decision_df["BAT_HOME_ID"] == 1,
            decision_df["AWAY_MGR_ID"],
            decision_df["HOME_MGR_ID"],
        )
        unit_label = "manager_id"
    else:
        decision_df["manager_unit"] = decision_df["FLD_TEAM_ID"]
        unit_label = "team_id_proxy"

    # Combine Oakland/Athletics codes into one team label
    decision_df["manager_unit"] = decision_df["manager_unit"].replace(
        {"OAK": "ATH"}
    )

    return decision_df, unit_label


# --------------------------------------------------
# Summarize performance
# --------------------------------------------------
def summarize_manager_performance(
    decision_df: pd.DataFrame,
    min_decisions: int = 500,
) -> pd.DataFrame:
    """
    Summarize decision value and alignment metrics by manager/team unit.
    """

    manager_summary = (
        decision_df.groupby("manager_unit")
        .agg(
            total_decisions=("decision_value", "count"),
            total_runs_lost=("decision_value", "sum"),
            avg_decision_value=("decision_value", "mean"),
            alignment_rate=("manager_aligned", "mean"),
            model_pull_rate=(
                "model_recommendation",
                lambda x: (x == "pull").mean(),
            ),
            manager_pull_rate=(
                "manager_decision",
                lambda x: (x == "pull").mean(),
            ),
        )
        .reset_index()
    )

    manager_summary = manager_summary[
        manager_summary["total_decisions"] >= min_decisions
    ].copy()

    # Lower average decision value = better
    manager_summary = manager_summary.sort_values("avg_decision_value")

    return manager_summary


# --------------------------------------------------
# Visualizations
# --------------------------------------------------
def plot_avg_decision_cost(
    manager_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot teams/managers by average decision cost.
    """

    plot_df = manager_summary.copy()

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_df["manager_unit"].astype(str),
        plot_df["avg_decision_value"],
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title("Team by Average Decision Cost")
    plt.xlabel("Average Runs Lost per Decision")
    plt.ylabel("Team")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_alignment_quadrant(
    manager_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot alignment rate against average decision cost.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        manager_summary["alignment_rate"],
        manager_summary["avg_decision_value"],
    )

    for _, row in manager_summary.iterrows():
        plt.text(
            row["alignment_rate"],
            row["avg_decision_value"],
            row["manager_unit"],
            fontsize=8,
        )

    avg_alignment = manager_summary["alignment_rate"].mean()
    avg_cost = manager_summary["avg_decision_value"].mean()

    plt.axvline(avg_alignment, linestyle="--", linewidth=1)
    plt.axhline(avg_cost, linestyle="--", linewidth=1)

    plt.xlabel("Alignment Rate")
    plt.ylabel("Average Decision Cost")
    plt.title("Manager Alignment vs Decision Cost")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":

    decision_output_path = TABLES_DIR / "decision_framework_output.csv"
    summary_output_path = TABLES_DIR / "manager_performance_summary.csv"

    avg_cost_plot_path = FIGURES_DIR / "manager_performance.png"
    quadrant_plot_path = FIGURES_DIR / "manager_performance_quad_chart.png"

    decision_df = load_decision_output(decision_output_path)

    decision_df, unit_label = add_manager_unit(decision_df)

    manager_summary = summarize_manager_performance(
        decision_df,
        min_decisions=500,
    )

    manager_summary.to_csv(summary_output_path, index=False)

    plot_avg_decision_cost(manager_summary, avg_cost_plot_path)
    plot_alignment_quadrant(manager_summary, quadrant_plot_path)

    print(f"\nRanking unit used: {unit_label}")

    print("\nTeams by average decision value:")
    print(manager_summary)

    print(f"\nSaved manager summary to: {summary_output_path}")
    print(f"Saved average decision cost plot to: {avg_cost_plot_path}")
    print(f"Saved quadrant chart to: {quadrant_plot_path}")