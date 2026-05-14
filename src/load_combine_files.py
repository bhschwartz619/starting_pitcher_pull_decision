import pandas as pd
import glob
import os


# Folder containing the yearly Retrosheet event CSV files created using Chadwick tools
# Each file represents one MLB season from 2022 through 2025
folder = r"C:\Users\bhsch\OneDrive\Documents\MSBA\Spring 2026\Predictive Modeling in Sports\Project\evcsvs"

# Find all yearly event CSV files in the folder
files = glob.glob(os.path.join(folder, "events_*.csv"))

# Read each yearly CSV and combine them into one master DataFrame.
# low_memory=False prevents pandas from guessing column types in chunks, which is helpful because Retrosheet files have
# many columns with mixed values
# ignore_index=True resets the raw index after stacking the files together
df = pd.concat((pd.read_csv(file, low_memory=False) for file in files), ignore_index=True)

# Save the combined dataset so later scripts can load one file instead of recombining the yearly files every time
output_path = os.path.join(folder, "combined_2022_2025.csv")

# Save without the pandas index to avoid creating an unnecessary extra column
df.to_csv(output_path, index=False)

# Confirm the combined file was created
print(f"Saved to: {output_path}")