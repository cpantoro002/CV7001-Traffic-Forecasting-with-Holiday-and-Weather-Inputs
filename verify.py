"""Verify forecast timing, identical comparison hours and saved-model inference."""
import json
import joblib
import numpy as np
import pandas as pd
from traffic_ml import ROOT, RAW
from traffic_context import prepare_comparison, FEATURE_SETS, WEATHER_TYPES, WEATHER_FLAGS, inference_frame, calendar_table
from feature_study import predict_records


def main():
    frame,audit,source_dates,_=prepare_comparison()
    raw=pd.read_csv(RAW,keep_default_na=False,parse_dates=["date_time"])
    test=frame.loc[frame.split.eq("test")]
    predictions=pd.read_csv(ROOT/"results/test_predictions_features.csv",parse_dates=["date_time"])
    summary=json.loads((ROOT/"results/feature_metrics.json").read_text())
    original=pd.read_csv(ROOT/"baseline_archive/test_predictions.csv",parse_dates=["date_time"])
    assert frame.weather_observed_at.eq(frame.date_time-pd.Timedelta(hours=1)).all()
    pd.testing.assert_series_equal(predictions.date_time.reset_index(drop=True),original.date_time.reset_index(drop=True))
    np.testing.assert_array_equal(predictions.actual,original.actual)
    np.testing.assert_allclose(predictions.A,original["Random forest"],atol=1e-8)
    assert frame.loc[frame.split.eq("train"),"date_time"].max() < pd.Timestamp("2017-01-01")
    assert frame.loc[frame.split.eq("validation"),"date_time"].between("2017-01-01","2017-12-31 23:00").all()
    assert test.date_time.ge(pd.Timestamp("2018-01-01")).all()
    for row in frame.iloc[::211].itertuples():
        previous=raw.loc[raw.date_time.eq(row.date_time-pd.Timedelta(hours=1))]
        valid=previous.loc[previous.temp.gt(0),"temp"]
        expected=valid.median()-273.15 if not valid.empty else np.nan
        np.testing.assert_allclose(row.temperature_c_lag_1h,expected,atol=1e-10,equal_nan=True)
        for name,flag in zip(WEATHER_TYPES[:-1],WEATHER_FLAGS[:-1]):
            assert getattr(row,flag)==int(previous.weather_main.eq(name).any())
    for date in ["2018-07-04","2016-12-25","2016-12-26"]:
        assert not calendar_table(date,date,source_dates).empty
    noon=pd.Timestamp("2018-07-04 12:00")
    if frame.date_time.eq(noon).any():assert frame.loc[frame.date_time.eq(noon),"is_holiday"].iloc[0]==1
    assert all(hour==0 for hour in audit["source_holiday_label_hours"])
    fit_data=frame.loc[frame.date_time.lt("2018-01-01")]
    for variant,columns in FEATURE_SETS.items():
        artifact=joblib.load(ROOT/"models"/f"model_{variant}.joblib")
        expected=np.nanmedian(fit_data[columns].to_numpy(dtype=float),axis=0)
        imputer=next(step for step in artifact["pipeline"].named_steps.values() if hasattr(step,"statistics_"))
        np.testing.assert_allclose(imputer.statistics_,expected,equal_nan=True)
        error=predictions[variant].to_numpy()-predictions.actual.to_numpy()
        measured=[np.mean(np.abs(error)),np.sqrt(np.mean(error**2)),1-np.sum(error**2)/np.sum((predictions.actual-predictions.actual.mean())**2)]
        stored=[summary["versions"][variant]["test"][key] for key in ["mae","rmse","r2"]]
        np.testing.assert_allclose(measured,stored,atol=1e-9)
        for key,value in {"n_estimators":150,"max_depth":18,"min_samples_leaf":2,"random_state":42,"max_features":1.0}.items():
            assert artifact["pipeline"].named_steps["model"].get_params()[key]==value
    inputs=json.loads((ROOT/"examples/context_inputs.json").read_text())
    reference=json.loads((ROOT/"examples/context_reference_predictions.json").read_text())
    for variant in FEATURE_SETS:
        produced=predict_records(inputs,variant,quiet=True)
        np.testing.assert_allclose([r["predicted_vehicles_per_hour"] for r in produced],
                                   [r["predicted_vehicles_per_hour"] for r in reference[variant]],atol=1e-8)
        features=inference_frame(inputs,variant,source_dates)
        for row in features.itertuples():
            original_row=frame.loc[frame.date_time.eq(row.date_time)].iloc[0]
            for column in FEATURE_SETS[variant]:
                np.testing.assert_allclose(getattr(row,column),original_row[column],atol=1e-9,equal_nan=True)
    bad=json.loads(json.dumps(inputs[0]));bad["previous_hour_weather"]["date_time"]=bad["date_time"]
    try:inference_frame([bad],"C",source_dates)
    except ValueError:pass
    else:raise AssertionError("Target-hour weather should be rejected")
    expected=min(FEATURE_SETS,key=lambda v:summary["versions"][v]["validation"]["mae"])
    assert summary["selected_by_validation"]==expected
    print("PASS: original baseline, identical hours, lagged weather, whole-day holidays, training-only imputation, metrics and inference.")


if __name__=="__main__":main()
