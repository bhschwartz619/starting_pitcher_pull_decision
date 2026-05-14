from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------
# This assumes the script is located in:
# project_folder/src/load_combine_files.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local data folder. This folder is not included in the GitHub repo.
DATA_DIR = PROJECT_ROOT / "data"

# Folder containing yearly Retrosheet event CSV files
RAW_DATA_DIR = DATA_DIR / "raw"

# Folder where the combined dataset will be saved
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load and combine yearly event files
# --------------------------------------------------
def combine_event_files(
    input_dir: Path = RAW_DATA_DIR,
    output_file: Path = PROCESSED_DATA_DIR / "combined_2022_2025.csv",
) -> pd.DataFrame:
    """
    Load yearly Retrosheet event CSV files and combine them into one dataset.

    Expected input files:
        data/raw/events_2022.csv
        data/raw/events_2023.csv
        data/raw/events_2024.csv
        data/raw/events_2025.csv

    Parameters
    ----------
    input_dir : Path
        Folder containing yearly event CSV files.
    output_file : Path
        File path where the combined CSV should be saved.

    Returns
    -------
    pd.DataFrame
        Combined event-level dataset.
    """

    files = sorted(input_dir.glob("events_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No event files found in {input_dir}. "
            "Expected files named like events_2022.csv, events_2023.csv, etc."
        )

    print("Files found:")
    for file in files:
        print(f" - {file.name}")

    df = pd.concat(
        (pd.read_csv(file, low_memory=False) for file in files),
        ignore_index=True,
    )

    df.to_csv(output_file, index=False)

    print(f"\nCombined dataset saved to: {output_file}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")

    return df


# --------------------------------------------------
# Run script
# --------------------------------------------------
if __name__ == "__main__":
    combine_event_files()