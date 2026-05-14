# Starting Pitcher Pull Decision Framework

This project uses Retrosheet event-level MLB data from 2022–2025 to evaluate when managers should remove a starting pitcher from a game.

The analysis builds a run expectancy framework, engineers pitcher fatigue and game-context features, and compares historical managerial decisions against model-based recommendations.

## Project Goals

- Build a run expectancy model using play-by-play MLB data
- Engineer pitcher fatigue and game-context features
- Estimate the expected cost of leaving a starter in the game
- Compare historical managerial decisions to model-based recommendations

## Data

The project uses Retrosheet event-level data processed with Chadwick Tools.

Raw data files are not included in this repository because of file size. Instructions for recreating the dataset will be added.

## Methods

This project includes:

- Run expectancy table construction
- Pitcher feature engineering
- Linear regression modeling
- Decision comparison between actual manager choices and model recommendations

## Repository Structure

```text
src/        Python scripts
data/       Local data folder, raw files not uploaded
outputs/    Charts and tables
notebooks/  Exploratory analysis
report/     Final writeup or presentation