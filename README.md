# Starting Pitcher Pull Decision Framework

This project uses Retrosheet event-level MLB data from 2022–2025 to evaluate when managers should remove a starting pitcher from a game.

The analysis builds a run expectancy framework, engineers pitcher fatigue and game-context features, and compares historical managerial decisions against model-based recommendations.

## Project Goals

- Build a run expectancy model using play-by-play MLB data
- Engineer pitcher fatigue and game-context features
- Estimate the expected cost of leaving a starter in the game
- Compare historical managerial decisions to model-based recommendations

## Data

This project uses event-level Major League Baseball play-by-play data from the 2022–2025 seasons obtained from Retrosheet and processed using Chadwick Tools.

### Data Sources

- Retrosheet event files: https://www.retrosheet.org/
- Chadwick Tools: https://github.com/chadwickbureau/chadwick

Retrosheet event files contain detailed pitch-by-pitch and plate appearance-level game information, including:
- inning and game state
- outs and baserunner configuration
- batter and pitcher identifiers
- scoring events
- substitutions and pitching changes
- play outcomes

The raw event files were converted into CSV format using Chadwick Tools before being combined into a unified event-level dataset for analysis.

### Data Scope

The analysis focuses on MLB regular season games from 2022–2025, corresponding to the modern universal designated hitter era. Using multiple seasons allowed the model to evaluate managerial decision-making across a large sample of pitching appearances and game situations.

### Feature Engineering

Several baseball-specific features were engineered from the raw play-by-play data, including:
- run expectancy by base-out state
- batters faced
- times through the order
- inning context
- score differential buckets
- baserunner state indicators
- starter versus reliever identification

These features were used to estimate the expected impact of leaving a starting pitcher in the game versus making a pitching change.

### Data Availability

Raw Retrosheet event-level datasets are not included in this repository because of file size limitations. The repository instead focuses on the analytical framework, feature engineering pipeline, modeling approach, and project outputs.

Users interested in reproducing the full analysis can obtain the raw data directly from Retrosheet and process the files using Chadwick Tools.

## Project Pipeline

The project is organized as a modular Python pipeline, with each script handling a specific stage of the analysis.

```text
src/
├── load_combine_files.py       # Loads yearly Retrosheet CSV files and combines them into one dataset
├── run_expectancy.py           # Builds run expectancy features using base-out states
├── pitcher_features.py         # Engineers pitcher workload and fatigue-related features
├── modeling.py                 # Trains the model used to estimate expected run outcomes
├── decision_framework.py       # Compares model-based stay/pull recommendations against actual decisions
├── manager_performance.py      # Summarizes manager and team-level alignment with the model
└── visualizations.py           # Generates charts and tables used in the final analysis
```

### Pipeline Flow

```text
Raw Retrosheet event files
        ↓
load_combine_files.py
        ↓
run_expectancy.py
        ↓
pitcher_features.py
        ↓
modeling.py
        ↓
decision_framework.py
        ↓
manager_performance.py
        ↓
visualizations.py
        ↓
Final charts, tables, and presentation outputs
```

This structure separates data loading, feature engineering, modeling, decision evaluation, and visualization into distinct scripts, making the project easier to understand, maintain, and extend.

## Methods

This project includes:

- Run expectancy table construction
- Pitcher feature engineering
- Linear regression modeling
- Decision comparison between actual manager choices and model recommendations

## Repository Structure

```text
src/        Python scripts
outputs/    Charts
report/     Final presentation slides
```

## Final Presentation

[View the final presentation](report/starting_pitcher_final_presentation.pdf)