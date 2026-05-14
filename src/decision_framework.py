import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from run_expectancy import (
    load_data,
    prepare_plate_appearances,
    add_run_expectancy_features
)

from pitcher_features import add_pitcher_features

# Helper Functions

def add_context_features(pa_df):
    # Score differential from the pitching team's perspective
    # Positive = pitching team is winning, negative = losing
    # Captures game context influencing managerial decisions
    pa_df["score_diff"] = (
        pa_df["START_FLD_SCORE_CT"] - pa_df["START_BAT_SCORE_CT"]
    )

    pa_df["inning"] = pa_df["INN_CT"]

    # Score differential bucketing
    pa_df["score_diff_bucket"] = pd.cut(
        pa_df["score_diff"],
        bins=[-100, -3, -1, 1, 3, 100],
        labels=["down_big", "down_small", "tie", "up_small", "up_big"],
        include_lowest=True
    )

    return pa_df


def build_model_df(pa_df):
    # Build modeling dataset using same feature set as modeling.py
    # Ensures consistency between training and prediction
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

    # Encoding categorical features
    model_df = pd.get_dummies(
        model_df,
        columns=["base_state", "pitch_count_bucket", "tto_bucket", "score_diff_bucket"],
        drop_first=True
    )

    return model_df

def train_expected_runs_model(model_df):

    X = model_df.drop(columns=["runs_from_state"])
    y = model_df["runs_from_state"]

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns


def bullpen_cost(inning):
    # Cost of going to the bullpen. Increases with remaining innings to reflect workload burden
    # Nonlinear form prevents unrealistic early-game pulling
    remaining = max(0, 9 - inning)
    return 0.03 * remaining + 0.002 * (remaining ** 2)


def reliever_quality_adjustment(inning):
    # Approximate reliever quality based on inning. Assumes better relievers are used later in games
    # This is a modeling assumption (not learned from data)
    # Change in expected runs returned
    if inning <= 4:
        return 0.05
    elif inning <= 6:
        return -0.02
    else:
        return -0.10


def make_prediction_data(pa_df, feature_columns):
    # Build prediction dataset using same features as training
    pred_df = pa_df[[
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

    # Encode categorical variables
    pred_df = pd.get_dummies(
        pred_df,
        columns=["base_state", "pitch_count_bucket", "tto_bucket", "score_diff_bucket"],
        drop_first=True
    )

    # Align columns with training data
    pred_df = pred_df.reindex(columns=feature_columns, fill_value=0)

    return pred_df


def add_decision_framework(pa_df, model, feature_columns, pull_threshold=0.10):
    # Only evaluate decisions when a starter is pitching to isolate the decision problem (pulling starter for reliever)
    decision_df = pa_df[pa_df["starter_flag"] == 1].copy()

    # Option 1: Starter stays

    stay_X = make_prediction_data(decision_df, feature_columns)
    decision_df["starter_expected_runs"] = model.predict(stay_X)

    # Option 2: Pull starter for reliever

    # Create a scenario where a fresh reliever enters
    reliever_df = decision_df.copy()

    # Reset fatigue and performance features to represent a fresh pitcher
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

    # Predict expected runs for reliever scenario
    pull_X = make_prediction_data(reliever_df, feature_columns)
    decision_df["raw_reliever_expected_runs"] = model.predict(pull_X)

    # Add realism adjustments

    # Add bullpen cost
    decision_df["bullpen_cost"] = decision_df["inning"].apply(bullpen_cost)

    # Add reliever quality adjustment
    decision_df["reliever_quality_adjustment"] = decision_df["inning"].apply(
        reliever_quality_adjustment
    )

    # Final expected runs for reliever option
    decision_df["reliever_expected_runs"] = (
        decision_df["raw_reliever_expected_runs"]
        + decision_df["bullpen_cost"]
        + decision_df["reliever_quality_adjustment"]
    )

    # Decision logic

    # Positive value means pulling is better
    decision_df["pull_advantage"] = (
        decision_df["starter_expected_runs"]
        - decision_df["reliever_expected_runs"]
    )

    # Model recommendation
    # Threshold prevents overreacting to small differences
    decision_df["model_recommendation"] = np.where(
        decision_df["pull_advantage"] > pull_threshold,
        "pull",
        "stay"
    )

    # Actual manager decision (observed from data)
    decision_df["manager_decision"] = np.where(
        decision_df["pitching_change_next"] == 1,
        "pull",
        "stay"
    )

    # Alignment indicator
    decision_df["manager_aligned"] = (
        decision_df["model_recommendation"] == decision_df["manager_decision"]
    ).astype(int)

    # Decision Evaluation

    # Optimal expected runs (stay vs. pull)
    decision_df["optimal_runs"] = np.minimum(
        decision_df["starter_expected_runs"],
        decision_df["reliever_expected_runs"]
    )

    # Runs based on manager decision
    decision_df["manager_runs"] = np.where(
        decision_df["manager_decision"] == "pull",
        decision_df["reliever_expected_runs"],
        decision_df["starter_expected_runs"]
    )

    # Decision value: Cost of manager decision relative to optimal
    decision_df["decision_value"] = (
        decision_df["manager_runs"] - decision_df["optimal_runs"]
    )

    return decision_df


# Main script

if __name__ == "__main__":

    file_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\evcsvs\combined_2022_2025.csv"

    df = load_data(file_path)

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
        pull_threshold=0.03
    )

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

    output_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\decision_framework_output.csv"
    decision_df.to_csv(output_path, index=False)

    print(f"\nSaved decision framework output to: {output_path}")