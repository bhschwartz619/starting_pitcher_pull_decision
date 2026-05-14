import pandas as pd

# 1. Load data
def load_data(file_path):
    # Load the combined Retrosheet dataset
    df = pd.read_csv(file_path, low_memory=False)
    return df

# 2. Filter + basic feature prep
def prepare_plate_appearances(df):
    # Filter to plate appearances only
    # BAT_EVENT_FL =="T" ensures each row represents a completed batting event
    # Important because run expectancy is defined at the plate appearance level
    pa_df = df[df["BAT_EVENT_FL"] == "T"].copy()

    # Sort data to ensure chronological order within each half-inning
    # Critical for correctly computing cumulative statistics later
    pa_df = pa_df.sort_values(
        ["GAME_ID", "INN_CT", "BAT_HOME_ID", "EVENT_ID"]
    ).reset_index(drop=True)

    # Construct base state

    # Create binary indicators for whether a runner is on each base
    pa_df["on_1b"] = pa_df["BASE1_RUN_ID"].notna().astype(int)
    pa_df["on_2b"] = pa_df["BASE2_RUN_ID"].notna().astype(int)
    pa_df["on_3b"] = pa_df["BASE3_RUN_ID"].notna().astype(int)

    # Combine into a 3-digit string for standard representation used in run expectancy models
    pa_df["base_state"] = (
        pa_df["on_1b"].astype(str)
        + pa_df["on_2b"].astype(str)
        + pa_df["on_3b"].astype(str)
    )

    # Number of outs at the start of the plate appearance
    pa_df["outs"] = pa_df["OUTS_CT"]

    # Runs scored during this plate appearance
    pa_df["runs_scored"] = pa_df["EVENT_RUNS_CT"]

    return pa_df


# 3. Add run expectancy features
def add_run_expectancy_features(pa_df):
    # Define grouping keys for each half-inning
    # Run expectancy calculated within innings, not across games
    inning_keys = ["GAME_ID", "INN_CT", "BAT_HOME_ID"]

    # Total runs scored in the inning
    # Serves as the final outcome to allocate back to earlier states
    pa_df["inning_total_runs"] = pa_df.groupby(inning_keys)["runs_scored"].transform("sum")

    # Runs scored before the current plate appearance
    # Ensures use of only information available at the time of the decision
    pa_df["runs_scored_before_pa"] = (
        pa_df.groupby(inning_keys)["runs_scored"].cumsum() - pa_df["runs_scored"]
    )

    # Runs scored from this state through the end of the inning
    # Key modeling target: Expected future runs from the current state
    pa_df["runs_from_state"] = (
        pa_df["inning_total_runs"] - pa_df["runs_scored_before_pa"]
    )

    return pa_df


# 4. Build RE table
def build_re_table(pa_df):
    # Aggregate expected runs by out and base state to produce classic run expectancy matrix
    re_table = (
        pa_df.groupby(["outs", "base_state"])["runs_from_state"]
        .mean()
        .reset_index()
        .sort_values(["outs", "base_state"])
    )
    return re_table

#5. Save the matrix to a CSV
def save_csv(df, path):
    df.to_csv(path, index=False)


# -------------------------------
# MAIN SCRIPT
# -------------------------------
if __name__ == "__main__":
    # Path to the combined dataset created in load_combine_files.py
    file_path = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\evcsvs\combined_2022_2025.csv"

    # Load and prepare data
    df = load_data(file_path)
    pa_df = prepare_plate_appearances(df)
    pa_df = add_run_expectancy_features(pa_df)

    # Build run expectancy matrix
    re_table = build_re_table(pa_df)

    print(re_table)

    re_table.to_csv(
        r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\re_table.csv",
        index=False
    )