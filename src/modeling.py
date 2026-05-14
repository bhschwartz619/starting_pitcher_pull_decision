import pandas as pd
from sklearn.linear_model import LinearRegression

from run_expectancy import (
    load_data,
    prepare_plate_appearances,
    add_run_expectancy_features
)

from pitcher_features import add_pitcher_features


# Load and prep data

# Path to combined Retrosheet data
file_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\evcsvs\combined_2022_2025.csv"

# Load raw data
df = load_data(file_path)

# Prepare plate appearance-level dataset and compute run expectancy target
pa_df = prepare_plate_appearances(df)
pa_df = add_run_expectancy_features(pa_df)

# Add pitcher-level features (fatigue, performance, pitch count, etc.)
pa_df = add_pitcher_features(pa_df)

# Add context features

# Score differential from the pitching team's perspective
# Positive values indicate the pitching team is ahead
# Provides game context, as managers behave differently depending on score
pa_df["score_diff"] = (
    pa_df["START_FLD_SCORE_CT"] - pa_df["START_BAT_SCORE_CT"]
)

# Inning number (used as a proxy for game progression)
pa_df["inning"] = pa_df["INN_CT"]

# Bucket score differential to capture nonlinear decision behavior
pa_df["score_diff_bucket"] = pd.cut(
    pa_df["score_diff"],
    bins=[-100, -3, -1, 1, 3, 100],
    labels=["down_big", "down_small", "tie", "up_small", "up_big"],
    include_lowest=True
)

# Build model dataset

# Define modeling dataset
# Target variable: runs_from_state (expected future runs from current state)
# Aligns with the decision problem (minimizing expected runs allowed)
model_df = pa_df[[
    "runs_from_state",
    "outs",
    "base_state",
    "pitch_count_bucket",
    "tto_bucket",
    "starter_flag",
    "inning",
    "score_diff_bucket",
    "runs_allowed_before_pa",
    "outs_recorded_before_pa",
    "hits_allowed_before_pa",
    "walk_hbp_allowed_before_pa",
    "recent_runs_allowed_3pa",
    "recent_hits_allowed_3pa",
    "recent_walk_hbp_allowed_3pa"
]].copy()

# Convert categorical variables into dummy variables
# drop_first=True avoids perfect multicollinearity
model_df = pd.get_dummies(
    model_df,
    columns=["base_state", "pitch_count_bucket", "tto_bucket", "score_diff_bucket"],
    drop_first=True
)

# Fit Model

# Separate predictors and target
X = model_df.drop(columns=["runs_from_state"])
y = model_df["runs_from_state"]

# Use linear regression for interpretability. Allows direct interpretation of how features impact expected runs
model = LinearRegression()
model.fit(X, y)

# Inspect Results

# Extract model coefficients by absolute magnitude
coeffs = pd.DataFrame({
    "feature": X.columns,
    "coef": model.coef_
}).sort_values("coef", ascending=False)

import matplotlib.pyplot as plt

# Plot most important coefficients by absolute magnitude
coef_plot_df = coeffs.copy()
coef_plot_df["abs_coef"] = coef_plot_df["coef"].abs()

# Keep top 15 most influential features for plotting
coef_plot_df = (
    coef_plot_df
    .sort_values("abs_coef", ascending=False)
    .head(15)
    .sort_values("coef")
)

plt.figure(figsize=(9, 6))

plt.barh(coef_plot_df["feature"], coef_plot_df["coef"])

# Vertical line at zero helps interpret positive vs. negative effects
plt.axvline(0, linestyle="--", linewidth=1, color="green")

plt.title("Most Important Model Coefficients")
plt.xlabel("Coefficient Impact on Expected Runs")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\Visualizations\model_coefficients.png",
    dpi=300
)

print(coeffs)