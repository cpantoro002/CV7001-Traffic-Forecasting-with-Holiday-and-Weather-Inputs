"""Reproducible one-hour-ahead traffic forecasting for the CV7001 mini-project.

At the end of hour t-1, forecast the vehicle count in hour t. Only calendar
information and measured counts at t-1, t-24 and t-168 are used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/Metro_Interstate_Traffic_Volume.csv.gz"
URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
RAW_SHA256 = "0b3679ac15173f79c6dc6c5ef8a0798d806fa5c5d7f05c84a5fa711bd1b05f07"
SEED = 42
LAGS = ["volume_lag_1h", "volume_lag_24h", "volume_lag_168h"]
CALENDAR = ["hour", "day_of_week", "month"]
FEATURES = CALENDAR + LAGS
TRAIN_END = pd.Timestamp("2017-01-01")
VALIDATION_END = pd.Timestamp("2018-01-01")


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def obtain_data(force: bool = False) -> Path:
    """Use the bundled copy, or obtain the same public data without an API key."""
    if not RAW.exists() or force:
        print("Downloading the public UCI dataset...", flush=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "CV7001-traffic-mini-project/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            contents = archive.read("Metro_Interstate_Traffic_Volume.csv.gz")
        if hashlib.sha256(contents).hexdigest() != RAW_SHA256:
            raise ValueError("Downloaded data differ from the documented version; review before using.")
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(contents)
    actual_hash = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if actual_hash != RAW_SHA256:
        raise ValueError("Local dataset checksum mismatch. Restore the supplied file or use download --force.")
    return RAW


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["date_time"], errors="raise")
    result["hour"] = timestamp.dt.hour
    result["day_of_week"] = timestamp.dt.dayofweek  # Monday = 0.
    result["month"] = timestamp.dt.month
    return result


def prepare_data() -> tuple[pd.DataFrame, dict]:
    raw_path = obtain_data()
    raw = pd.read_csv(raw_path, keep_default_na=False, parse_dates=["date_time"])
    if raw["date_time"].isna().any() or (raw["traffic_volume"] < 0).any():
        raise ValueError("Invalid timestamps or negative vehicle counts.")
    # Weather labels can create repeated timestamps. Verify that their targets
    # agree before collapsing them; do not count them as independent hours.
    grouped = raw.groupby("date_time", sort=True)["traffic_volume"]
    conflicting = int(grouped.nunique().gt(1).sum())
    if conflicting:
        raise ValueError(f"Found {conflicting} timestamps with conflicting traffic counts.")
    hourly = grouped.first().to_frame()
    total_grid = len(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))
    for hours in (1, 24, 168):
        # Look up exact clock times, not previous rows: records contain gaps.
        previous_times = hourly.index - pd.Timedelta(hours=hours)
        hourly[f"volume_lag_{hours}h"] = hourly["traffic_volume"].reindex(previous_times).to_numpy()
    eligible = hourly.dropna(subset=LAGS).reset_index()
    eligible = add_calendar(eligible)
    eligible["split"] = np.where(
        eligible["date_time"] < TRAIN_END, "train",
        np.where(eligible["date_time"] < VALIDATION_END, "validation", "test"),
    )
    audit = {
        "raw_rows": len(raw), "unique_hours": len(hourly),
        "repeated_timestamp_rows_removed": len(raw) - len(hourly),
        "conflicting_timestamp_targets": conflicting,
        "start": str(hourly.index.min()), "end": str(hourly.index.max()),
        "hours_on_complete_grid": total_grid,
        "unobserved_hours": total_grid - len(hourly),
        "rows_without_all_three_lags_removed": len(hourly) - len(eligible),
        "eligible_rows": len(eligible), "raw_sha256": RAW_SHA256,
        "split_counts": {k: int(v) for k, v in eligible["split"].value_counts().items()},
        "split_date_ranges": {
            name: {"first": str(part.date_time.min()), "last": str(part.date_time.max())}
            for name, part in eligible.groupby("split")
        },
    }
    return eligible, audit


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "n": len(actual),
    }


def build_model(family: str, parameter: float | int) -> Pipeline:
    if family == "ridge":
        preprocessing = ColumnTransformer([
            ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CALENDAR),
            ("past_counts", StandardScaler(), LAGS),
        ])
        return Pipeline([("features", preprocessing), ("model", Ridge(alpha=float(parameter)))])
    return Pipeline([("model", RandomForestRegressor(
        n_estimators=150, max_depth=18, min_samples_leaf=int(parameter),
        max_features=1.0, random_state=SEED, n_jobs=2,
    ))])


def make_plots(predictions: pd.DataFrame, comparison: dict, selected: str, output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # Fixed before inspection; not selected for visually favourable performance.
    week = predictions.set_index("date_time").reindex(
        pd.date_range("2018-02-05", "2018-02-11 23:00", freq="h")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1.75, 1]})
    axes[0].plot(week.index, week["actual"], label="Observed", color="#23364a", lw=1.3)
    axes[0].plot(week.index, week[selected], label="Forecast", color="#d67625", lw=1.1, alpha=0.9)
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set(xlabel="2018, local recorded time", ylabel="Vehicles per hour", title="(a) Fixed illustrative test week")
    axes[0].legend(frameon=False, ncols=2, fontsize=8)
    errors = (predictions[selected] - predictions["actual"]).abs().groupby(predictions.date_time.dt.hour).mean()
    axes[1].bar(errors.index, errors.values, color="#367b89", width=0.82)
    axes[1].set(xlabel="Hour of day", ylabel="MAE (vehicles/hour)", title="(b) All held-out test hours", xticks=[0, 6, 12, 18, 23])
    fig.tight_layout()
    fig.savefig(output / "forecast_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ordered = sorted(comparison, key=lambda key: comparison[key]["test"]["mae"])
    values = [comparison[key]["test"]["mae"] for key in ordered]
    ax.barh(ordered, values, color=["#367b89" if key == selected else "#93a4b4" for key in ordered])
    ax.invert_yaxis()
    ax.set(xlabel="Test MAE (vehicles/hour)", title="One-hour-ahead forecast comparison")
    for i, value in enumerate(values):
        ax.text(value + 4, i, f"{value:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, max(values) * 1.17)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train() -> None:
    frame, audit = prepare_data()
    training = frame[frame.split == "train"].copy()
    validation = frame[frame.split == "validation"].copy()
    testing = frame[frame.split == "test"].copy()
    for part in (training, validation, testing):
        if part.empty:
            raise ValueError("A chronological split has no eligible observations.")
    assert training.date_time.max() < validation.date_time.min() < testing.date_time.min()
    comparison = {}
    for name, lag in [("Last hour", LAGS[0]), ("Previous day", LAGS[1]), ("Previous week", LAGS[2])]:
        comparison[name] = {"validation": metrics(validation.traffic_volume, validation[lag]), "test": {}}
    # This small candidate grid is fixed in code. Test targets are never used
    # for parameter or model selection. Early stopping and random splits are absent.
    candidates = {"Ridge regression": ("ridge", [1.0, 10.0, 100.0]),
                  "Random forest": ("forest", [2, 8, 20])}
    tuning = []
    best_parameters = {}
    for name, (family, values) in candidates.items():
        scores = []
        for parameter in values:
            estimator = build_model(family, parameter)
            estimator.fit(training[FEATURES], training.traffic_volume)
            score = metrics(validation.traffic_volume, np.maximum(0, estimator.predict(validation[FEATURES])))
            scores.append((score["mae"], parameter, score))
            tuning.append({"model": name, "parameter": parameter, **score})
            print(f"Validation: {name}, parameter={parameter}, MAE={score['mae']:.2f}", flush=True)
        _, parameter, score = min(scores, key=lambda item: item[0])
        best_parameters[name] = {"family": family, "parameter": parameter}
        comparison[name] = {"validation": score, "test": {}}
    selected = min(best_parameters, key=lambda name: comparison[name]["validation"]["mae"])
    baseline = min(["Last hour", "Previous day", "Previous week"], key=lambda name: comparison[name]["validation"]["mae"])
    print(f"Selected by validation MAE: {selected}; strongest validation baseline: {baseline}", flush=True)
    predictions = testing[["date_time"]].copy()
    predictions["actual"] = testing.traffic_volume.to_numpy()
    for name, lag in [("Last hour", LAGS[0]), ("Previous day", LAGS[1]), ("Previous week", LAGS[2])]:
        predictions[name] = testing[lag].to_numpy()
        comparison[name]["test"] = metrics(testing.traffic_volume, predictions[name])
    combined = pd.concat([training, validation], ignore_index=True)
    fitted = {}
    for name, settings in best_parameters.items():
        estimator = build_model(settings["family"], settings["parameter"])
        estimator.fit(combined[FEATURES], combined.traffic_volume)
        predictions[name] = np.maximum(0, estimator.predict(testing[FEATURES]))
        comparison[name]["test"] = metrics(testing.traffic_volume, predictions[name])
        fitted[name] = estimator
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    summary = {
        "forecast_horizon_hours": 1, "random_seed": SEED,
        "selected_model_by_validation": selected,
        "strongest_baseline_by_validation": baseline,
        "best_parameters": best_parameters,
        "models": comparison,
        "selected_model_test_mae_reduction_vs_baseline_percent": 100 * (
            1 - comparison[selected]["test"]["mae"] / comparison[baseline]["test"]["mae"]
        ),
        "test_mean_traffic_volume": float(testing.traffic_volume.mean()),
        "test_median_traffic_volume": float(testing.traffic_volume.median()),
    }
    dump_json(output / "metrics.json", summary)
    dump_json(output / "data_audit.json", audit)
    dump_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__,
    })
    pd.DataFrame(tuning).to_csv(output / "validation_tuning.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame([
        {"model": name, "split": split, **scores}
        for name, parts in comparison.items() for split, scores in parts.items()
    ]).to_csv(output / "metrics.csv", index=False)
    artifact = {"pipeline": fitted[selected], "features": FEATURES,
                "selected_model": selected, "best_parameters": best_parameters[selected],
                "training_end_exclusive": str(VALIDATION_END), "raw_sha256": RAW_SHA256}
    joblib.dump(artifact, output / "model.joblib", compress=("xz", 3))
    # Demonstration inputs are actual held-out records, with labels omitted.
    examples = testing.iloc[:3][["date_time"] + LAGS].copy()
    examples["date_time"] = examples.date_time.astype(str)
    dump_json(ROOT / "examples/inputs.json", examples.to_dict(orient="records"))
    dump_json(ROOT / "examples/reference_predictions.json", [
        {"date_time": row.date_time, "predicted_vehicles_per_hour": float(value)}
        for row, value in zip(examples.itertuples(), predictions[selected].iloc[:3])
    ])
    make_plots(predictions, comparison, selected, output)
    print(json.dumps(summary, indent=2), flush=True)


def predict(input_path: Path, model_path: Path, output_path: Path | None = None) -> list[dict]:
    # Load only a trusted model file; the supplied one is produced by train().
    artifact = joblib.load(model_path)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError("Inputs must be a JSON object or a non-empty list of objects.")
    frame = pd.DataFrame(records)
    required = ["date_time"] + LAGS
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Each input requires {required}")
    for column in LAGS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f"{column} must contain finite, non-negative vehicle counts.")
    parsed = pd.to_datetime(frame["date_time"], errors="raise")
    if parsed.isna().any() or not parsed.eq(parsed.dt.floor("h")).all():
        raise ValueError("date_time must give an exact local hour, for example 2018-01-01 08:00:00.")
    if parsed.dt.tz is not None:
        raise ValueError("Use naive local timestamps consistent with the original dataset.")
    frame = add_calendar(frame)
    values = np.maximum(0, artifact["pipeline"].predict(frame[artifact["features"]]))
    result = [{"date_time": str(timestamp), "predicted_vehicles_per_hour": float(value)}
              for timestamp, value in zip(parsed, values)]
    if output_path is not None:
        dump_json(output_path, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download", help="Obtain/check the public dataset")
    download_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("train", help="Train, validate, refit, evaluate and save results")
    predict_parser = subparsers.add_parser("predict", help="Run the trained model on JSON inputs")
    predict_parser.add_argument("--inputs", type=Path, default=ROOT / "examples/inputs.json")
    predict_parser.add_argument("--model", type=Path, default=ROOT / "results/model.joblib")
    predict_parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "download":
        print(obtain_data(arguments.force))
    elif arguments.command == "train":
        train()
    else:
        predict(arguments.inputs, arguments.model, arguments.output)


if __name__ == "__main__":
    main()
