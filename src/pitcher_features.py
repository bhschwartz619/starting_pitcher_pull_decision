from run_expectancy import (
    load_data,
    prepare_plate_appearances,
    add_run_expectancy_features,
)

import pandas as pd

# Helper function to count pitches from Retrosheet pitch sequence
def count_pitches(seq):
    # Retrosheet stores pitch-by-pitch data as a string
    # Count alphabetical characters as individual pitches
    # Provides a proxy for pitch count at the plate appearance level
    if seq is None:
        return 0

    seq = str(seq)

    if seq == "" or seq.lower() == "nan":
        return 0

    return sum(1 for char in seq if char.isalpha())

# Main feature engineering function
def add_pitcher_features(pa_df):
    # Ensure correct chronological ordering within each game
    # Critical because many features rely on cumulative calculations
    pa_df = pa_df.sort_values(
        ["GAME_ID", "INN_CT", "BAT_HOME_ID", "EVENT_ID"]
    ).reset_index(drop=True)

    # Grouping keys defining a unique pitcher within a game
    pitcher_keys = ["GAME_ID", "FLD_TEAM_ID", "PIT_ID"]

    # Pitcher workload and fatigue features

    # Total batters faced by the pitcher up to the current plate appearance
    # Used to calculate pitcher's times through the order (tto)
    pa_df["batters_faced"] = (
        pa_df.groupby(pitcher_keys).cumcount() + 1
    )

    # Number of times the pitcher has faced this specific batter
    # Captures familiarity effects between pitcher and hitter
    pa_df["times_faced_batter"] = (
        pa_df.groupby(["GAME_ID", "FLD_TEAM_ID", "PIT_ID", "BAT_ID"])
        .cumcount() + 1
    )

    # Times through the order (TTO)
    # Cap at 3 since the third time is most relevant
    pa_df["tto"] = ((pa_df["batters_faced"] - 1) // 9) + 1
    pa_df["tto_bucket"] = pa_df["tto"].clip(upper=3)

    # Indicator for whether the pitcher is the starting pitcher
    pa_df["starter_flag"] = (pa_df["PIT_START_FL"] == "T").astype(int)

    # Pitch count features

    # Number of pitches thrown in this plate appearance
    pa_df["pitches_this_pa"] = pa_df["PITCH_SEQ_TX"].apply(count_pitches)

    # Cumulative pitch count before the current plate appearance
    # Ensures no future information leakage
    pa_df["pitch_count_before_pa"] = (
        pa_df.groupby(pitcher_keys)["pitches_this_pa"].cumsum()
        - pa_df["pitches_this_pa"]
    )

    # Bucket pitch count into real-world thresholds to capture nonlinear fatigue effects
    pa_df["pitch_count_bucket"] = pd.cut(
        pa_df["pitch_count_before_pa"],
        bins=[0, 50, 75, 90, 105, 200],
        labels=["0-50", "50-75", "75-90", "90-105", "105+"],
        include_lowest=True
    )

    # Handle rare edge cases where pitch count exceeds defined bins
    pa_df["pitch_count_bucket"] = pa_df["pitch_count_bucket"].fillna("105+")

    # Pitching change label (target for evaluation)

    # Identify the next pitcher for the pitching team
    pa_df["next_pitcher"] = (
        pa_df.groupby(["GAME_ID", "FLD_TEAM_ID"])["PIT_ID"].shift(-1)
    )

    # Indicator for whether a pitching change occurs after this plate appearance
    # Represents the manager's decision
    pa_df["pitching_change_next"] = (
        (pa_df["PIT_ID"] != pa_df["next_pitcher"])
        & pa_df["next_pitcher"].notna()
    ).astype(int)

    # Reliever usage context

    # Inning when the pitcher entered the game
    pa_df["pitcher_entry_inning"] = (
        pa_df.groupby(pitcher_keys)["INN_CT"].transform("min")
    )

    # Categorize relievers based on entry timing
    # Approximates bullpen role
    pa_df["reliever_entry_bucket"] = "starter"

    pa_df.loc[
        (pa_df["starter_flag"] == 0) & (pa_df["pitcher_entry_inning"] <= 4),
        "reliever_entry_bucket"
    ] = "early"

    pa_df.loc[
        (pa_df["starter_flag"] == 0)
        & (pa_df["pitcher_entry_inning"] >= 5)
        & (pa_df["pitcher_entry_inning"] <= 6),
        "reliever_entry_bucket"
    ] = "middle"

    pa_df.loc[
        (pa_df["starter_flag"] == 0)
        & (pa_df["pitcher_entry_inning"] >= 7)
        & (pa_df["pitcher_entry_inning"] <= 9),
        "reliever_entry_bucket"
    ] = "late"

    pa_df.loc[
        (pa_df["starter_flag"] == 0) & (pa_df["pitcher_entry_inning"] >= 10),
        "reliever_entry_bucket"
    ] = "extras"

    # Number of innings remaining in regulation
    # Used later to model bullpen cost
    pa_df["remaining_innings"] = (9 - pa_df["INN_CT"]).clip(lower=0)

    # In-game pitcher performance features (all features constructed using only past information)

    # Total runs allowed before the current plate appearance
    pa_df["runs_allowed_before_pa"] = (
        pa_df.groupby(pitcher_keys)["runs_scored"].cumsum()
        - pa_df["runs_scored"]
    )

    # Total outs recorded before the current plate appearance
    pa_df["outs_recorded_before_pa"] = (
        pa_df.groupby(pitcher_keys)["EVENT_OUTS_CT"].cumsum()
        - pa_df["EVENT_OUTS_CT"]
    )

    # Indicator for whether a hit was allowed in this plate appearance
    pa_df["hit_allowed"] = (pa_df["H_CD"] > 0).astype(int)

    # Cumulative hits allowed before current plate appearance
    pa_df["hits_allowed_before_pa"] = (
        pa_df.groupby(pitcher_keys)["hit_allowed"].cumsum()
        - pa_df["hit_allowed"]
    )

    # Indicator for walk or hit-by-pitch
    pa_df["walk_hbp_allowed"] = pa_df["EVENT_CD"].isin([14, 15, 16]).astype(int)

    # Cumulative walks/HBP allowed before current plate appearance
    pa_df["walk_hbp_allowed_before_pa"] = (
        pa_df.groupby(pitcher_keys)["walk_hbp_allowed"].cumsum()
        - pa_df["walk_hbp_allowed"]
    )

    # Recent performance over last 3 plat appearances (lagged to avoid leakage)
    pa_df["recent_runs_allowed_3pa"] = (
        pa_df.groupby(pitcher_keys)["runs_scored"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
    )

    pa_df["recent_hits_allowed_3pa"] = (
        pa_df.groupby(pitcher_keys)["hit_allowed"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
    )

    pa_df["recent_walk_hbp_allowed_3pa"] = (
        pa_df.groupby(pitcher_keys)["walk_hbp_allowed"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
    )

    return pa_df


if __name__ == "__main__":
    file_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\evcsvs\combined_2022_2025.csv"

    df = load_data(file_path)

    pa_df = prepare_plate_appearances(df)
    pa_df = add_run_expectancy_features(pa_df)
    pa_df = add_pitcher_features(pa_df)

    print(pa_df[[
        "GAME_ID", "INN_CT", "BAT_HOME_ID", "FLD_TEAM_ID", "PIT_ID",
        "BAT_ID", "batters_faced", "times_faced_batter",
        "tto_bucket", "starter_flag", "pitch_count_before_pa",
        "pitch_count_bucket", "pitching_change_next",
        "runs_allowed_before_pa", "hits_allowed_before_pa",
        "walk_hbp_allowed_before_pa", "recent_runs_allowed_3pa",
        "pitcher_entry_inning", "reliever_entry_bucket",
        "remaining_innings"
    ]].head(20))

    print("\nPitch count summary:")
    print(pa_df["pitch_count_before_pa"].describe())

    print("\nStarter flag counts:")
    print(pa_df["starter_flag"].value_counts())

    print("\nTTO bucket counts:")
    print(pa_df["tto_bucket"].value_counts().sort_index())

    print("\nPitching change next counts:")
    print(pa_df["pitching_change_next"].value_counts())

    print("\nReliever entry bucket counts:")
    print(pa_df["reliever_entry_bucket"].value_counts())