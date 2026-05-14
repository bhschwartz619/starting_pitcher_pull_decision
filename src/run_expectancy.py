from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------
# Assumes script location:
# project_folder/src/run_expectancy.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local-only data directory (ignored by GitHub)
DATA_DIR = PROJECT_ROOT / "data"

# Processed data location
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"

# Create output directories if they do not exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the combined Retrosheet dataset.
    """
    df = pd.read_csv(file_path, low_memory=False)
    return df


# --------------------------------------------------
# 2. Filter + basic feature preparation
# --------------------------------------------------
def prepare_plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to completed plate appearances and engineer
    base-out state features for run expectancy modeling.
    """

    # Keep only completed batting events
    pa_df = df[df["BAT_EVENT_FL"] == "T"].copy()

    # Ensure chronological ordering
    pa_df = pa_df.sort_values(
        ["GAME_ID", "INN_CT", "BAT_HOME_ID", "EVENT_ID"]
    ).reset_index(drop=True)

    # Binary indicators for occupied bases
    pa_df["on_1b"] = pa_df["BASE1_RUN_ID"].notna().astype(int)
    pa_df["on_2b"] = pa_df["BASE2_RUN_ID"].notna().astype(int)
    pa_df["on_3b"] = pa_df["BASE3_RUN_ID"].notna().astype(int)

    # Standard base-state representation
    pa_df["base_state"] = (
        pa_df["on_1b"].astype(str)
        + pa_df["on_2b"].astype(str)
        + pa_df["on_3b"].astype(str)
    )

    # Outs at start of plate appearance
    pa_df["outs"] = pa_df["OUTS_CT"]

    # Runs scored during the plate appearance
    pa_df["runs_scored"] = pa_df["EVENT_RUNS_CT"]

    return pa_df


# --------------------------------------------------
# 3. Add run expectancy features
# --------------------------------------------------
def add_run_expectancy_features(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute future runs scored from each base-out state.
    """

    inning_keys = ["GAME_ID", "INN_CT", "BAT_HOME_ID"]

    # Total runs scored in the inning
    pa_df["inning_total_runs"] = (
        pa_df.groupby(inning_keys)["runs_scored"]
        .transform("sum")
    )

    # Runs scored before current plate appearance
    pa_df["runs_scored_before_pa"] = (
        pa_df.groupby(inning_keys)["runs_scored"]
        .cumsum()
        - pa_df["runs_scored"]
    )

    # Future runs from current state
    pa_df["runs_from_state"] = (
        pa_df["inning_total_runs"]
        - pa_df["runs_scored_before_pa"]
    )

    return pa_df


# --------------------------------------------------
# 4. Build run expectancy table
# --------------------------------------------------
def build_re_table(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build run expectancy matrix by base-out state.
    """

    re_table = (
        pa_df.groupby(["outs", "base_state"])["runs_from_state"]
        .mean()
        .reset_index()
        .sort_values(["outs", "base_state"])
    )

    return re_table


# --------------------------------------------------
# 5. Save output table
# --------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to CSV.
    """
    df.to_csv(path, index=False)


# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":

    # Combined dataset created in load_combine_files.py
    combined_data_path = (
        PROCESSED_DATA_DIR / "combined_2022_2025.csv"
    )

    # Output file
    re_table_output_path = (
        TABLES_DIR / "run_expectancy_table.csv"
    )

    # Load and prepare data
    df = load_data(combined_data_path)

    pa_df = prepare_plate_appearances(df)

    pa_df = add_run_expectancy_features(pa_df)

    # Build run expectancy table
    re_table = build_re_table(pa_df)

    # Save output
    save_csv(re_table, re_table_output_path)

    print("\nRun expectancy table created successfully.")
    print(f"Saved to: {re_table_output_path}")