from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from run_expectancy import (
    load_data,
    prepare_plate_appearances,
    add_run_expectancy_features,
)

from pitcher_features import add_pitcher_features


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Build modeling dataset
# --------------------------------------------------
def build_modeling_dataset(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model-ready dataset used to estimate expected future runs.
    """

    pa_df = pa_df.copy()

    # Score differential from the pitching team's perspective
    pa_df["score_diff"] = (
        pa_df["START_FLD_SCORE_CT"] - pa_df["START_BAT_SCORE_CT"]
    )

    # Inning number as game progression context
    pa_df["inning"] = pa_df["INN_CT"]

    # Bucket score differential to capture nonlinear game-state behavior
    pa_df["score_diff_bucket"] = pd.cut(
        pa_df["score_diff"],
        bins=[-100, -3, -1, 1, 3, 100],
        labels=["down_big", "down_small", "tie", "up_small", "up_big"],
        include_lowest=True,
    )

    model_df = pa_df[
        [
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
            "recent_walk_hbp_allowed_3pa",
        ]
    ].copy()

    model_df = pd.get_dummies(
        model_df,
        columns=[
            "base_state",
            "pitch_count_bucket",
            "tto_bucket",
            "score_diff_bucket",
        ],
        drop_first=True,
    )

    return model_df


# --------------------------------------------------
# Train model
# --------------------------------------------------
def train_linear_model(model_df: pd.DataFrame):
    """
    Train an interpretable linear regression model.
    """

    X = model_df.drop(columns=["runs_from_state"])
    y = model_df["runs_from_state"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X, y


# --------------------------------------------------
# Extract coefficients
# --------------------------------------------------
def get_model_coefficients(model: LinearRegression, X: pd.DataFrame) -> pd.DataFrame:
    """
    Return model coefficients in a tidy DataFrame.
    """

    coeffs = pd.DataFrame(
        {
            "feature": X.columns,
            "coef": model.coef_,
        }
    ).sort_values("coef", ascending=False)

    return coeffs


# --------------------------------------------------
# Plot coefficients
# --------------------------------------------------
def plot_top_coefficients(
    coeffs: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Plot the most influential coefficients by absolute magnitude.
    """

    coef_plot_df = coeffs.copy()
    coef_plot_df["abs_coef"] = coef_plot_df["coef"].abs()

    coef_plot_df = (
        coef_plot_df
        .sort_values("abs_coef", ascending=False)
        .head(top_n)
        .sort_values("coef")
    )

    plt.figure(figsize=(9, 6))
    plt.barh(coef_plot_df["feature"], coef_plot_df["coef"])

    # Vertical line at zero helps interpret positive vs. negative effects
    plt.axvline(0, linestyle="--", linewidth=1)

    plt.title("Most Important Model Coefficients")
    plt.xlabel("Coefficient Impact on Expected Runs")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":

    combined_data_path = PROCESSED_DATA_DIR / "combined_2022_2025.csv"

    coefficient_table_path = TABLES_DIR / "model_coefficients.csv"
    coefficient_plot_path = FIGURES_DIR / "model_coefficients.png"

    # Load and prepare data
    df = load_data(combined_data_path)

    pa_df = prepare_plate_appearances(df)
    pa_df = add_run_expectancy_features(pa_df)
    pa_df = add_pitcher_features(pa_df)

    # Build model dataset
    model_df = build_modeling_dataset(pa_df)

    # Fit model
    model, X, y = train_linear_model(model_df)

    # Save and plot coefficients
    coeffs = get_model_coefficients(model, X)
    coeffs.to_csv(coefficient_table_path, index=False)

    plot_top_coefficients(coeffs, coefficient_plot_path)

    print("\nModel training complete.")
    print(f"Coefficient table saved to: {coefficient_table_path}")
    print(f"Coefficient plot saved to: {coefficient_plot_path}")
    print("\nTop coefficients:")
    print(coeffs.head(15))