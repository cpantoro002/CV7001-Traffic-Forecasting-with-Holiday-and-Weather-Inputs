"""Known-ahead holidays and strictly lagged weather for the feature comparison."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from traffic_ml import ROOT, RAW, FEATURES, prepare_data

WEATHER_TYPES = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm", "Other"]
WEATHER_NUMERIC = ["temperature_c_lag_1h", "rain_mm_lag_1h", "snow_mm_lag_1h", "cloud_percent_lag_1h"]
WEATHER_FLAGS = ["weather_" + name.lower() + "_lag_1h" for name in WEATHER_TYPES]
FEATURE_SETS = {"A": list(FEATURES), "B": list(FEATURES) + ["is_holiday"],
                "C": list(FEATURES) + ["is_holiday"] + WEATHER_NUMERIC + WEATHER_FLAGS}
NAMES = {"A": "Calendar + past traffic", "B": "A + whole-day holiday", "C": "B + previous-hour weather"}
FIXED_DATES = [(1, 1, "New Year's Day (actual)"), (7, 4, "Independence Day (actual)"),
               (11, 11, "Veterans Day (actual)"), (12, 25, "Christmas Day (actual)")]


def source_holidays(raw):
    labelled = raw.loc[raw.holiday.ne("None") & raw.holiday.ne(""), ["date_time", "holiday"]].copy()
    labelled["date"] = labelled.date_time.dt.normalize()
    return [{"date": str(date.date()), "name": str(name)}
            for date, name in labelled[["date", "holiday"]].drop_duplicates().itertuples(index=False, name=None)]


def calendar_table(start, end, source_dates):
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    dates, origins = defaultdict(set), defaultdict(set)
    # Rules are calendar covariates; no traffic targets are involved.
    federal = USFederalHolidayCalendar().holidays(start=start, end=end, return_name=True)
    for date, name in federal.items():
        dates[date.normalize()].add(str(name)); origins[date.normalize()].add("federal_calendar")
    for year in range(start.year, end.year + 1):
        fixed = list(FIXED_DATES)
        if year >= 2021: fixed.append((6, 19, "Juneteenth (actual)"))
        for month, day, name in fixed:
            date = pd.Timestamp(year=year, month=month, day=day)
            if start <= date <= end:
                dates[date].add(name); origins[date].add("actual_fixed_date")
    for item in source_dates:
        date = pd.Timestamp(item["date"])
        if start <= date <= end:
            dates[date].add(item["name"]); origins[date].add("source_holiday_label")
    return pd.DataFrame([{"date": date, "names": " | ".join(sorted(names)),
                          "origin": " | ".join(sorted(origins[date]))}
                         for date, names in sorted(dates.items())], columns=["date", "names", "origin"])


def clean_weather(raw):
    weather = raw[["date_time", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main"]].copy()
    checks = {"temp": weather.temp.le(0), "rain_1h": weather.rain_1h.lt(0) | weather.rain_1h.gt(1000),
              "snow_1h": weather.snow_1h.lt(0) | weather.snow_1h.gt(1000),
              "clouds_all": weather.clouds_all.lt(0) | weather.clouds_all.gt(100)}
    invalid = {key: int(mask.sum()) for key, mask in checks.items()}
    for key, mask in checks.items(): weather.loc[mask, key] = np.nan
    weather["temperature_c_lag_1h"] = weather.temp - 273.15
    weather["rain_mm_lag_1h"] = weather.rain_1h
    weather["snow_mm_lag_1h"] = weather.snow_1h
    weather["cloud_percent_lag_1h"] = weather.clouds_all
    known = set(WEATHER_TYPES) - {"Other"}
    for name, flag in zip(WEATHER_TYPES, WEATHER_FLAGS):
        weather[flag] = (weather.weather_main.eq(name) if name != "Other" else ~weather.weather_main.isin(known)).astype(int)
    numeric = weather.groupby("date_time", sort=True)[WEATHER_NUMERIC].median()
    flags = weather.groupby("date_time", sort=True)[WEATHER_FLAGS].max()
    # Repeated timestamps can have several weather labels. Keep their union.
    return numeric.join(flags), invalid


def prepare_comparison():
    frame, base_audit = prepare_data()
    raw = pd.read_csv(RAW, keep_default_na=False, parse_dates=["date_time"])
    source_dates = source_holidays(raw)
    holiday_dates = calendar_table(raw.date_time.min(), raw.date_time.max(), source_dates)
    frame["is_holiday"] = frame.date_time.dt.normalize().isin(holiday_dates.date).astype(int)
    hourly_weather, invalid = clean_weather(raw)
    previous_times = pd.DatetimeIndex(frame.date_time) - pd.Timedelta(hours=1)
    frame["weather_observed_at"] = previous_times
    for column in WEATHER_NUMERIC + WEATHER_FLAGS:
        frame[column] = hourly_weather[column].reindex(previous_times).to_numpy()
    if frame[WEATHER_FLAGS].isna().any().any():
        raise ValueError("A selected hour has no preceding weather row despite its traffic lag")
    if not frame.weather_observed_at.lt(frame.date_time).all():
        raise ValueError("Weather must precede the forecast target hour")
    assert not frame[FEATURE_SETS["A"] + ["is_holiday"]].isna().any().any()
    audit = {**base_audit, "study_versions": FEATURE_SETS, "holiday_definition":
             "Whole-day indicator: US federal calendar observed dates, actual fixed-date federal holidays, plus dates labelled in the source; regional dates outside the source need an override.",
             "source_holiday_label_dates": len(source_dates),
             "source_holiday_label_hours": sorted(int(v) for v in raw.loc[raw.holiday.ne("None"), "date_time"].dt.hour.unique()),
             "calendar_dates_in_record_range": len(holiday_dates),
             "holiday_target_hours_by_split": {split: int(part.is_holiday.sum()) for split, part in frame.groupby("split")},
             "invalid_raw_weather_rows": invalid,
             "weather_rules": {"temperature": "Kelvin <=0 -> missing, then subtract 273.15 for Celsius", "precipitation": "Negative or >1000 mm/hour -> missing (broad sanity screen)",
                               "clouds": "Outside [0,100] percent -> missing", "duplicate_hours": "Median numeric measurements and union of categorical weather labels", "timing": "Exact t-1 timestamp only; no target-hour weather"},
             "weather_missing_feature_values_by_split": {split: int(part[WEATHER_NUMERIC].isna().sum().sum()) for split, part in frame.groupby("split")},
             "imputation": "Training-fold medians plus missing indicators; no weather-based target rows dropped", "same_target_hours_all_versions": True,
             "evaluation_disclosure": "2018 was evaluated in the earlier baseline. This fixed follow-up comparison keeps it out of fitting and model selection, but does not claim a new blind test."}
    return frame, audit, source_dates, holiday_dates


def example_inputs(frame):
    test = frame.loc[frame.split.eq("test")]
    examples = []
    candidates = [test.loc[test.date_time.eq(pd.Timestamp("2018-01-01 08:00"))],
                  test.loc[(test.rain_mm_lag_1h.gt(0) | test.snow_mm_lag_1h.gt(0))],
                  test.loc[test.is_holiday.eq(0) & test.date_time.dt.dayofweek.lt(5) & test.date_time.dt.hour.eq(8)]]
    for candidate in candidates:
        if candidate.empty: continue
        row = candidate.iloc[0]
        item = {"date_time": str(row.date_time), **{name: float(row[name]) for name in FEATURES if name.startswith("volume_lag_")}}
        item["previous_hour_weather"] = {"date_time": str(row.weather_observed_at),
            "temperature_c": None if pd.isna(row.temperature_c_lag_1h) else float(row.temperature_c_lag_1h),
            "rain_mm": None if pd.isna(row.rain_mm_lag_1h) else float(row.rain_mm_lag_1h),
            "snow_mm": None if pd.isna(row.snow_mm_lag_1h) else float(row.snow_mm_lag_1h),
            "cloud_cover_percent": None if pd.isna(row.cloud_percent_lag_1h) else float(row.cloud_percent_lag_1h),
            "conditions": [name for name, flag in zip(WEATHER_TYPES, WEATHER_FLAGS) if row[flag] > 0]}
        if item["date_time"] not in [x["date_time"] for x in examples]: examples.append(item)
    return examples


def inference_frame(records, variant, source_dates):
    if isinstance(records, dict): records = [records]
    if not isinstance(records, list) or not records: raise ValueError("Supply a JSON object or non-empty list")
    rows = []
    for item in records:
        time = pd.Timestamp(item["date_time"])
        if pd.isna(time) or time.tzinfo is not None or time != time.floor("h"):
            raise ValueError("date_time must be an exact naive local hour")
        row = {"date_time": time, "hour": time.hour, "day_of_week": time.dayofweek, "month": time.month}
        for name in ["volume_lag_1h", "volume_lag_24h", "volume_lag_168h"]:
            value = float(item[name])
            if not np.isfinite(value) or value < 0: raise ValueError(f"Invalid count: {name}")
            row[name] = value
        if "is_holiday" in item:
            flag = item["is_holiday"]
            if flag not in (0, 1, False, True): raise ValueError("is_holiday override must be 0 or 1")
            row["is_holiday"] = int(flag)
        else:
            table = calendar_table(time.normalize(), time.normalize(), source_dates)
            row["is_holiday"] = int(not table.empty)
        for name in WEATHER_NUMERIC: row[name] = np.nan
        for flag in WEATHER_FLAGS: row[flag] = 0
        weather = item.get("previous_hour_weather")
        if variant == "C" and weather is None: raise ValueError("Version C requires previous_hour_weather")
        if weather is not None:
            weather_time = pd.Timestamp(weather["date_time"])
            if weather_time != time - pd.Timedelta(hours=1):
                raise ValueError("previous_hour_weather.date_time must be exactly one hour before the target")
            for key, column in zip(["temperature_c", "rain_mm", "snow_mm", "cloud_cover_percent"], WEATHER_NUMERIC):
                value = weather.get(key)
                if value is None: continue
                value = float(value)
                if not np.isfinite(value): raise ValueError(f"Non-finite weather value: {key}")
                if key == "temperature_c" and value <= -273.15: value = np.nan
                if key in ("rain_mm", "snow_mm") and not 0 <= value <= 1000: value = np.nan
                if key == "cloud_cover_percent" and not 0 <= value <= 100: value = np.nan
                row[column] = value
            conditions = weather.get("conditions", [])
            if isinstance(conditions, str): conditions = [conditions]
            if not isinstance(conditions, list): raise ValueError("conditions must be a string or list")
            if not conditions: conditions = ["Other"]
            for name in conditions:
                name = name if name in WEATHER_TYPES else "Other"
                row[WEATHER_FLAGS[WEATHER_TYPES.index(name)]] = 1
        rows.append(row)
    return pd.DataFrame(rows)
