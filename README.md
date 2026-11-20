# Evaluating Holiday and Weather Inputs for Traffic Forecasting

CV7001 mini-project: a controlled comparison of three one-hour-ahead random-forest
forecasters. This is the updated traffic project, with measured results, all three
trained models and a guided notebook.

At the end of hour `t-1`, forecast the vehicle count during hour `t`. All three
versions use the same target hours, splits, algorithm and settings:

| Version | Inputs |
| --- | --- |
| A | Target hour, weekday and month, and traffic counts at t-1, t-24 and t-168 |
| B | A plus a known-ahead, whole-day holiday indicator |
| C | B plus observed weather from exactly t-1: temperature, rainfall, snowfall, cloud cover and general weather-category flags |

The experiment tests whether additional information improves prediction. A small
improvement is not assumed in advance, and the extra variables are retained in
the comparison even if they do not help.

## Measured results

<!-- FEATURE_RESULTS_START -->
| Version | Inputs | Validation MAE | Test MAE | Test RMSE | Test R-squared |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Calendar + traffic history | 162.6 | 151.4 | 241.1 | 0.985 |
| B | A + holiday flag | 162.3 | 150.9 | 240.5 | 0.985 |
| C | B + previous-hour weather | 161.6 | 151.4 | 240.8 | 0.985 |

Default prediction model: **C**, selected by validation MAE. B changed test MAE relative to A by -0.35% and C changed test MAE relative to A by -0.02% (negative means lower error). Small differences are descriptive and no significance test or causal interpretation is claimed.
<!-- FEATURE_RESULTS_END -->

The table records the supplied reference run. After rerunning training, read
`results/feature_metrics.csv` or `results/feature_metrics.json` for your new scores.

MAE and RMSE are in vehicles/hour. Lower is better. R-squared is unitless and
does not mean classification accuracy. Validation scores use fitting before
2017 and test scores follow refitting through 2017. The original A forecast has
been reproduced exactly, giving a direct comparison with the earlier project.

The holiday-subgroup comparison is also useful: there are 166 eligible holiday
hours in the test period. Their MAE changes from 185.9 (A) to 176.6 (B) and 180.1
(C). The overall score changes much less because this is a small subgroup. On
the 400 hours preceded by positive recorded precipitation, MAE is 133.0 (A),
130.8 (B) and 131.0 (C). Groups overlap, meaning they are not separate data partitions.
All 400 evaluated targets with positive recorded precipitation in the previous hour fall in August-September 2018, and that subgroup does not cover all rainy/snowy conditions.
These are descriptive, single-seed results from this period, not significance
tests or proof of causal weather/holiday effects.

## Run the updated study

Use **Python 3.11** and open a terminal in the extracted project folder
containing `feature_study.py`. The reference environment used Python 3.11.9.
On Windows, use `py -3.11` instead of `python` if appropriate.
On macOS/Linux, the executable may be `python3`.

```bash
python -m pip install -r requirements.txt
python feature_study.py train
python verify.py
python feature_study.py predict --variant selected --inputs examples/context_inputs.json
```

`feature_study.py` runs the A/B/C comparison and uses shared functions
from `traffic_ml.py`.

`baseline_archive/` contains two reference files: `metrics.json` and
`test_predictions.csv`. These are used to check that version A reproduces
the original baseline results.

Training uses two CPU workers. The recorded runtime, including fitting and
compressing three model files, is in `results/feature_metrics.json`. There is no
GPU, web application or API key requirement for training or prediction.
The supplied models can be used immediately after installing the dependencies.

Optional isolated environment, without shell activation:

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python feature_study.py train
.venv\Scripts\python verify.py
.venv\Scripts\python feature_study.py predict --variant C
```

```bash
# macOS/Linux
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python feature_study.py train
.venv/bin/python verify.py
.venv/bin/python feature_study.py predict --variant C
```

For a guided interface, open `Traffic_Forecasting_Walkthrough.ipynb` in Jupyter or
upload the notebook to https://colab.research.google.com/. It includes a cell to
upload and extract this full project ZIP in Colab. Run its cells in order. Each
training or prediction cell executes the same script used in the terminal.

## Understand when information is available

At **09:00**, after the 08:00-09:00 interval has finished, predict the count for
09:00-10:00. You may use the completed 08:00-09:00 traffic and weather observations
and the known calendar for the target date. You cannot use the actual weather
that will occur during 09:00-10:00. Reporting latency is assumed zero for this
experiment. It has not been validated for an operational data feed.

Recorded weather for the target hour is never a model input. This is a rolling
one-hour-ahead evaluation: later test predictions may use measurements from
earlier test hours once those hours have finished. Model parameters remain fixed
throughout testing. This is not a whole-year forecast made at the end of 2017.

## Predict new inputs

Run any of the three supplied models:

```bash
python feature_study.py predict --variant A
python feature_study.py predict --variant B
python feature_study.py predict --variant C
```

All default to `examples/context_inputs.json`. The examples were selected by
predeclared calendar/weather criteria, not by prediction accuracy: New Year's
Day at 08:00, the first test target preceded by precipitation, and the first
non-holiday weekday target at 08:00. They contain real historical inputs and no
target traffic labels. Reference predictions are included for all versions.

For your own input, save a JSON object or list of objects, for example:

```json
{
  "date_time": "2018-07-04 09:00:00",
  "volume_lag_1h": 1000,
  "volume_lag_24h": 4200,
  "volume_lag_168h": 4500,
  "previous_hour_weather": {
    "date_time": "2018-07-04 08:00:00",
    "temperature_c": 24.5,
    "rain_mm": 0.5,
    "snow_mm": 0.0,
    "cloud_cover_percent": 90,
    "conditions": ["Rain", "Mist"]
  }
}
```

The numeric values above are hypothetical to demonstrate the schema. Supply
real measured history for practical prediction. For the 09:00 target, `volume_lag_1h`
is the 08:00-09:00 count, and `volume_lag_24h`/`volume_lag_168h` are the counts
for 09:00-10:00 on the previous day/week. Weather timestamps identify the start
of the completed measurement hour. Date strings are naive local times consistent
with the original dataset, not converted to UTC or Singapore time.

```bash
python feature_study.py predict --variant C --inputs my_input.json --output results/my_prediction.json
```

For A and B the weather object is optional. C requires it. A weather timestamp
equal to the target hour is explicitly rejected. Unknown numeric weather can be
`null` and will use the fitting-period imputer. Use the short `weather_main`
categories such as `Clear`, `Clouds`, `Rain`, `Mist`, `Snow` or `Thunderstorm`,
not the longer descriptions. Several simultaneous categories are allowed.

Holiday status is calculated from the target date. An optional `"is_holiday": 0`
or `1` overrides it, useful for regional events outside the source period.
Federal observed dates and actual fixed-date holidays are generated by calendar
rules. Regional dates such as State Fair are known here only where labelled in
the original dataset. Model A ignores the holiday/weather features.

## Data access and preparation

John Hogue (2019), **Metro Interstate Traffic Volume**, UCI Machine Learning
Repository, https://doi.org/10.24432/C5X60B, CC BY 4.0.
Source: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume.
Location: westbound I-94, Minnesota DOT station 301, USA, October 2012-September 2018.

The unchanged compressed source is bundled under `data/raw/`. It can also be
obtained with `python traffic_ml.py download --force`. Its SHA-256 is:

```text
0b3679ac15173f79c6dc6c5ef8a0798d806fa5c5d7f05c84a5fa711bd1b05f07
```

The source has 48,204 rows and 40,575 unique hourly timestamps. Duplicate traffic
targets must agree before being collapsed. Exact clock-time traffic lags leave
34,731 eligible targets. Weather does not remove any additional target hours.
The original source bytes are not edited.

Preparation details:

- **Holidays:** original names occur only at midnight on 53 dates. The new flag
  covers whole days, using the union of source-labelled dates, pandas US federal
  observed holidays, and actual fixed-date federal holidays. There are 70 calendar
  dates within the recorded date range. This is not a complete regional-event
  calendar or a measurement of whether each employer closes.
- **Weather timing:** all fields are looked up at exactly `target_time - 1 hour`.
  An earlier recorded row is not assumed to be one hour ago when timestamps have gaps.
- **Weather duplicates:** numeric measurements use the median within the hour,
  the union of all simultaneous general weather labels is retained as binary flags.
- **Units/quality:** Kelvin is converted to Celsius. Ten raw rows with zero Kelvin
  and one rainfall value above 1000 mm/hour are marked missing. Negative precipitation
  and cloud cover outside 0-100% are also screened. The 1000-mm limit is a broad
  sanity screen selected before fitting, not an estimated climatological threshold.
- **Imputation:** medians and missing indicators are fitted within each fitting
  partition. Final models use training+validation medians, never test medians.
- **Categories:** general `weather_main` categories are used, the more detailed
  `weather_description` field is omitted to keep the comparison focused.

See `results/feature_data_audit.json`, `data/holiday_calendar.csv` and
`results/feature_inputs.csv` for the exact cleaned inputs, source times and counts.

## Fair evaluation and selection

| Partition | Dates | Eligible target hours |
| --- | --- | ---: |
| Training | October 2012-December 2016 | 19,643 |
| Validation | January-December 2017 | 8,603 |
| Test | January-September 2018 | 6,485 |

All three use 150 trees, maximum depth 18, minimum leaf size 2, all features
available per split, seed 42 and two CPU workers. These settings were retained
from the original baseline and there is no new parameter search on the test period.

First, each version is fitted on training data and evaluated on validation data.
The default version is selected by lowest validation MAE. All fixed versions are
then refitted through 2017 and evaluated on 2018. Each saved model corresponds to
the refitted instance. A test-time input is not used to update weights.

The 2018 period was already evaluated in the earlier baseline. This is a fixed
follow-up comparison held out from fitting, not a newly untouched blind test.
Small differences should not be overinterpreted: this is one station, one fixed
seed, a partial-year test, and limited holiday/rare-weather coverage. The three
versions have not been independently optimized. The results do not establish
that weather is unimportant in general or that predictors have causal effects.

## Verification and illustrations

`python verify.py` checks that A reproduces the original predictions, all models
use the same target hours, weather comes from t-1, holiday flags cover the day,
imputation uses fitting data only, exported metrics agree with predictions, and
JSON inference reproduces saved example predictions. It also checks that target-hour
weather input is rejected.

`results/feature_comparison.png`, `context_errors.png` and `holiday_example.png`
are drawn by Matplotlib from actual scores and predictions. The holiday plot
uses the predeclared date 1 January 2018. Source CSV/JSON files and
`results/figure_sources.json` identify the data behind each figure.

## Download your results

Section 9 of `Traffic_Forecasting_Walkthrough.ipynb` downloads
`My_Traffic_Forecast_Results.zip` with the trained models, result tables, plots,
example predictions and code. Keep the full project ZIP for the original dataset
and baseline reference outputs. You can also access the generated files directly
under `models/` and `results/` after a terminal run.

## Credits and disclosure

The dataset is Hogue/UCI's public source, not newly collected data. The models
use scikit-learn RandomForestRegressor and pandas supplies the federal calendar.
NumPy, SciPy, Matplotlib and joblib support numerical computation, plots and
serialization. No pretrained model or external solution repository was used.

OpenAI ChatGPT/Codex assisted with study design, code development, and documentation.

References: https://doi.org/10.24432/C5X60B;
https://jmlr.org/papers/v12/pedregosa11a.html;
https://pandas.pydata.org/docs/reference/api/pandas.tseries.holiday.USFederalHolidayCalendar.html.
