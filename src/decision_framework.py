from pathlib import Path

import numpy as np
import pandas as pd
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
TABLES_DIR = OUTPUT_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Context features
# --------------------------------------------------
def add_context_features(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game-context features used in the decision framework.
    """

    pa_df = pa_df.copy()

    # Score differential from the pitching team's perspective
    pa_df["score_diff"] = (
        pa_df["START_FLD_SCORE_CT"] - pa_df["START_BAT_SCORE_CT"]
    )

    pa_df["inning"] = pa_df["INN_CT"]

    pa_df["score_diff_bucket"] = pd.cut(
        pa_df["score_diff"],
        bins=[-100, -3, -1, 1, 3, 100],
        labels=["down_big", "down_small", "tie", "up_small", "up_big"],
        include_lowest=True,
    )

    return pa_df


# --------------------------------------------------
# Modeling helpers
# --------------------------------------------------
def build_model_df(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model dataset using the same feature set used for prediction.
    """

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


def train_expected_runs_model(model_df: pd.DataFrame):
    """
    Train an interpretable linear regression model for expected future runs.
    """

    X = model_df.drop(columns=["runs_from_state"])
    y = model_df["runs_from_state"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns


# --------------------------------------------------
# Decision framework assumptions
# --------------------------------------------------
def bullpen_cost(inning: int) -> float:
    """
    Estimate the cost of using the bullpen earlier in the game.

    The cost increases when more innings remain, reflecting the workload burden
    of asking the bullpen to cover more outs.
    """

    remaining = max(0, 9 - inning)
    return 0.03 * remaining + 0.002 * (remaining ** 2)


def reliever_quality_adjustment(inning: int) -> float:
    """
    Approximate reliever quality by inning.

    Negative values reduce expected runs, representing stronger reliever usage
    later in games. This is a modeling assumption rather than a learned estimate.
    """

    if inning <= 4:
        return 0.05
    elif inning <= 6:
        return -0.02
    else:
        return -0.10


# --------------------------------------------------
# Prediction helpers
# --------------------------------------------------
def make_prediction_data(
    pa_df: pd.DataFrame,
    feature_columns,
) -> pd.DataFrame:
    """
    Build prediction data and align columns with the trained model.
    """

    pred_df = pa_df[
        [
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

    pred_df = pd.get_dummies(
        pred_df,
        columns=[
            "base_state",
            "pitch_count_bucket",
            "tto_bucket",
            "score_diff_bucket",
        ],
        drop_first=True,
    )

    pred_df = pred_df.reindex(columns=feature_columns, fill_value=0)

    return pred_df


# --------------------------------------------------
# Decision framework
# --------------------------------------------------
def add_decision_framework(
    pa_df: pd.DataFrame,
    model: LinearRegression,
    feature_columns,
    pull_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Compare model-based starter stay/pull recommendations against
    observed manager decisions.
    """

    # Only evaluate decisions when the current pitcher is the starter
    decision_df = pa_df[pa_df["starter_flag"] == 1].copy()

    # Option 1: starter stays
    stay_X = make_prediction_data(decision_df, feature_columns)
    decision_df["starter_expected_runs"] = model.predict(stay_X)

    # Option 2: starter is replaced by a fresh reliever
    reliever_df = decision_df.copy()

    reliever_df["starter_flag"] = 0
    reliever_df["batters_faced"] = 1
    reliever_df["tto_bucket"] = 1

    reliever_df["runs_allowed_before_pa"] = 0
    reliever_df["outs_recorded_before_pa"] = 0
    reliever_df["hits_allowed_before_pa"] = 0
    reliever_df["walk_hbp_allowed_before_pa"] = 0
    reliever_df["recent_runs_allowed_3pa"] = 0
    reliever_df["recent_hits_allowed_3pa"] = 0
    reliever_df["recent_walk_hbp_allowed_3pa"] = 0

    pull_X = make_prediction_data(reliever_df, feature_columns)
    decision_df["raw_reliever_expected_runs"] = model.predict(pull_X)

    decision_df["bullpen_cost"] = decision_df["inning"].apply(bullpen_cost)

    decision_df["reliever_quality_adjustment"] = (
        decision_df["inning"].apply(reliever_quality_adjustment)
    )

    decision_df["reliever_expected_runs"] = (
        decision_df["raw_reliever_expected_runs"]
        + decision_df["bullpen_cost"]
        + decision_df["reliever_quality_adjustment"]
    )

    # Positive pull advantage means pulling is better
    decision_df["pull_advantage"] = (
        decision_df["starter_expected_runs"]
        - decision_df["reliever_expected_runs"]
    )

    decision_df["model_recommendation"] = np.where(
        decision_df["pull_advantage"] > pull_threshold,
        "pull",
        "stay",
    )

    decision_df["manager_decision"] = np.where(
        decision_df["pitching_change_next"] == 1,
        "pull",
        "stay",
    )

    decision_df["manager_aligned"] = (
        decision_df["model_recommendation"] == decision_df["manager_decision"]
    ).astype(int)

    decision_df["optimal_runs"] = np.minimum(
        decision_df["starter_expected_runs"],
        decision_df["reliever_expected_runs"],
    )

    decision_df["manager_runs"] = np.where(
        decision_df["manager_decision"] == "pull",
        decision_df["reliever_expected_runs"],
        decision_df["starter_expected_runs"],
    )

    decision_df["decision_value"] = (
        decision_df["manager_runs"] - decision_df["optimal_runs"]
    )

    return decision_df


# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":

    combined_data_path = PROCESSED_DATA_DIR / "combined_2022_2025.csv"
    output_path = TABLES_DIR / "decision_framework_output.csv"

    df = load_data(combined_data_path)

    pa_df = prepare_plate_appearances(df)
    pa_df = add_run_expectancy_features(pa_df)
    pa_df = add_pitcher_features(pa_df)
    pa_df = add_context_features(pa_df)

    model_df = build_model_df(pa_df)
    model, feature_columns = train_expected_runs_model(model_df)

    decision_df = add_decision_framework(
        pa_df,
        model,
        feature_columns,
        pull_threshold=0.03,
    )

    decision_df.to_csv(output_path, index=False)

    print("\nModel recommendations:")
    print(decision_df["model_recommendation"].value_counts())

    print("\nManager decisions:")
    print(decision_df["manager_decision"].value_counts())

    print("\nAlignment rate:")
    print(decision_df["manager_aligned"].mean())

    print("\nTotal runs lost by managers:")
    print(decision_df["decision_value"].sum())

    print("\nAverage decision value:")
    print(decision_df["decision_value"].mean())

    print("\nBy inning:")
    print(decision_df.groupby("inning")["decision_value"].mean())

    print("\nBy TTO:")
    print(decision_df.groupby("tto_bucket")["decision_value"].mean())

    print(f"\nSaved decision framework output to: {output_path}")