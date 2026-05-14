import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load Data

file_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\decision_framework_output.csv"

decision_df = pd.read_csv(file_path, low_memory=False)

# Identify Manager/Team Unit

# If manager IDs exist, assign the pitching manager.
# If not, fall back to FLD_TEAM_ID as a team/manager proxy.
if {"HOME_MGR_ID", "AWAY_MGR_ID", "BAT_HOME_ID"}.issubset(decision_df.columns):
    decision_df["manager_unit"] = np.where(
        decision_df["BAT_HOME_ID"] == 1,
        decision_df["AWAY_MGR_ID"],
        decision_df["HOME_MGR_ID"]
    )
    unit_label = "manager_id"
else:
    decision_df["manager_unit"] = decision_df["FLD_TEAM_ID"]
    unit_label = "team_id_proxy"

# Combine Oakland/Athletics codes into one team label
decision_df["manager_unit"] = decision_df["manager_unit"].replace({
    "OAK": "ATH"
})

# Summarize Performance

manager_summary = (
    decision_df.groupby("manager_unit")
    .agg(
        total_decisions=("decision_value", "count"),
        total_runs_lost=("decision_value", "sum"),
        avg_decision_value=("decision_value", "mean"),
        alignment_rate=("manager_aligned", "mean"),
        model_pull_rate=("model_recommendation", lambda x: (x == "pull").mean()),
        manager_pull_rate=("manager_decision", lambda x: (x == "pull").mean())
    )
    .reset_index()
)

# Filter out very small samples
manager_summary = manager_summary[manager_summary["total_decisions"] >= 500].copy()

# Lower average decision value = better
manager_summary = manager_summary.sort_values("avg_decision_value")

# Print best and worst teams

print(f"\nRanking unit used: {unit_label}")

print("\nTeams by average decision value:")
print(manager_summary)

# Save Results

output_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\manager_performance_summary.csv"

manager_summary.to_csv(output_path, index=False)

# Visualizations

# Team by Average Decision Cost
plot_df = manager_summary.copy()

plt.figure(figsize=(10, 7))
plt.barh(plot_df["manager_unit"].astype(str), plot_df["avg_decision_value"])

plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Team by Average Decision Cost")
plt.xlabel("Average Runs Lost per Decision")
plt.ylabel("Team")

plt.tight_layout()

visual_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\Visualizations\manager_performance.png"
plt.savefig(visual_path, dpi=300)

plt.show()

print(f"Saved visualization to: {visual_path}")

# Alignment Rate vs Average Decision Value Quadrant Chart
plt.figure(figsize=(8,6))

plt.scatter(
    manager_summary["alignment_rate"],
    manager_summary["avg_decision_value"]
)

# Add team labels
for _, row in manager_summary.iterrows():
    plt.text(
        row["alignment_rate"],
        row["avg_decision_value"],
        row["manager_unit"],
        fontsize=8
    )

# Compute averages
avg_alignment = manager_summary["alignment_rate"].mean()
avg_cost = manager_summary["avg_decision_value"].mean()

# Add quadrant lines
plt.axvline(avg_alignment, linestyle="--", linewidth=1)
plt.axhline(avg_cost, linestyle="--", linewidth=1)

plt.xlabel("Alignment Rate")
plt.ylabel("Average Decision Cost")
plt.title("Manager Alignment vs Decision Cost")

plt.tight_layout()

visual_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\Visualizations\manager_performance_quad_chart.png"
plt.savefig(visual_path, dpi=300)

plt.show()

