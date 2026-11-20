"""Compare calendar/traffic forecasts with added holidays and lagged weather."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from traffic_ml import ROOT, SEED, metrics, dump_json
from traffic_context import FEATURE_SETS, NAMES, WEATHER_NUMERIC, WEATHER_FLAGS, prepare_comparison, example_inputs, inference_frame

SETTINGS = {"n_estimators":150, "max_depth":18, "min_samples_leaf":2,
            "max_features":1.0, "random_state":SEED, "n_jobs":2}


def estimator():
    return Pipeline([("missing",SimpleImputer(strategy="median",add_indicator=True)),
                     ("model",RandomForestRegressor(**SETTINGS))])


def subgroup_results(test, predictions):
    masks = {"All hours":np.ones(len(test),dtype=bool), "Holiday":test.is_holiday.eq(1).to_numpy(),
             "Previous-hour precipitation":(test.rain_mm_lag_1h.gt(0) | test.snow_mm_lag_1h.gt(0)).to_numpy()}
    rows=[]
    for group, mask in masks.items():
        for variant in FEATURE_SETS:
            if mask.any():rows.append({"group":group,"variant":variant,**metrics(test.traffic_volume.to_numpy()[mask],predictions[variant].to_numpy()[mask])})
    return pd.DataFrame(rows)


def plot_results(summary, prediction_table, subgroups):
    output=ROOT/"results"
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    colours={"A":"#4e6786","B":"#2a8b80","C":"#d48a32"}
    fig,axes=plt.subplots(1,2,figsize=(9,3.1))
    for ax,split,title in zip(axes,["validation","test"],["2017 validation","2018 test"]):
        values=[summary["versions"][v][split]["mae"] for v in FEATURE_SETS]
        ax.bar(list(FEATURE_SETS),values,color=[colours[v] for v in FEATURE_SETS],width=0.6)
        ax.set(title=title,ylabel="MAE (vehicles/hour)",ylim=(0,max(values)*1.2),xlabel="Feature version")
        for x,value in enumerate(values):ax.text(x,value+max(values)*0.025,f"{value:.1f}",ha="center",fontsize=9)
    fig.tight_layout();fig.savefig(output/"feature_comparison.png",dpi=200,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.7,3.2))
    groups=list(subgroups.group.unique()); positions=np.arange(len(groups)); width=0.23
    for j,variant in enumerate(FEATURE_SETS):
        part=subgroups.loc[subgroups.variant.eq(variant)].set_index("group").reindex(groups)
        ax.bar(positions+(j-1)*width,part.mae,width,label=variant+": "+NAMES[variant],color=colours[variant])
    counts=subgroups.groupby("group")["n"].first()
    ax.set_xticks(positions,[name+f"\n(n={int(counts[name]):,})" for name in groups])
    ax.set(ylabel="Test MAE (vehicles/hour)",title="Performance in predefined subgroups")
    ax.legend(frameon=False,fontsize=8,loc="upper left")
    fig.tight_layout();fig.savefig(output/"context_errors.png",dpi=200,bbox_inches="tight");plt.close(fig)
    day=prediction_table.set_index("date_time").reindex(pd.date_range("2018-01-01 00:00","2018-01-01 23:00",freq="h"))
    fig,ax=plt.subplots(figsize=(8.8,2.6))
    ax.plot(day.index,day.actual,label="Observed",color="#1d2c39",lw=1.9)
    for variant in FEATURE_SETS:ax.plot(day.index,day[variant],label=variant,color=colours[variant],lw=1.2)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4));ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set(xlabel="1 January 2018, recorded local time",ylabel="Vehicles/hour",title="A holiday example chosen by date before comparison")
    ax.legend(frameon=False,ncols=4,fontsize=8)
    fig.tight_layout();fig.savefig(output/"holiday_example.png",dpi=200,bbox_inches="tight");plt.close(fig)
    dump_json(output/"figure_sources.json",{
        "feature_comparison.png":"Measured validation/test MAE in feature_metrics.json; no illustrative scores.",
        "context_errors.png":"Measured MAE in subgroup_metrics.csv; groups overlap, and precipitation refers to the previous hour.",
        "holiday_example.png":"Observed source counts and actual saved predictions for the predeclared date 1 January 2018 from test_predictions_features.csv.",
        "renderer":"Matplotlib; all graphs are generated from the executed experiment, with no generated imagery."
    })


def train():
    started=time.perf_counter()
    frame,audit,source_dates,holiday_dates=prepare_comparison()
    training=frame.loc[frame.split.eq("train")].copy()
    validation=frame.loc[frame.split.eq("validation")].copy()
    test=frame.loc[frame.split.eq("test")].copy()
    output=ROOT/"results";output.mkdir(exist_ok=True)
    models_dir=ROOT/"models";models_dir.mkdir(exist_ok=True)
    combined=pd.concat([training,validation],ignore_index=True)
    summary={"question":"Do whole-day holidays and previous-hour weather improve one-hour-ahead volume forecasts?",
             "forecast_horizon_hours":1,"settings":SETTINGS,"fixed_before_comparison":True,
             "training_dates":"before 2017","validation_dates":"2017","test_dates":"January-September 2018",
             "versions":{},"evaluation_disclosure":audit["evaluation_disclosure"]}
    # Stage 1: choose the feature version on validation only.
    for variant,columns in FEATURE_SETS.items():
        model=estimator();model.fit(training[columns],training.traffic_volume)
        score=metrics(validation.traffic_volume,np.maximum(0,model.predict(validation[columns])))
        summary["versions"][variant]={"description":NAMES[variant],"input_columns":columns,"validation":score}
        print(f"{variant} validation MAE = {score['mae']:.3f}",flush=True)
    selected=min(FEATURE_SETS,key=lambda v:summary["versions"][v]["validation"]["mae"])
    summary["selected_by_validation"]=selected
    dump_json(output/"feature_selection_before_test.json",{
        "criterion":"lowest 2017 validation MAE","selected":selected,"settings":SETTINGS,
        "validation_mae":{v:summary['versions'][v]['validation']['mae'] for v in FEATURE_SETS},
        "test_used_for_selection":False,"previous_test_exposure":audit["evaluation_disclosure"]})
    print(f"Selected on validation: {selected}. Refitting fixed versions and evaluating the 2018 period.",flush=True)
    predictions=test[["date_time","is_holiday","rain_mm_lag_1h","snow_mm_lag_1h"]].copy()
    predictions["actual"]=test.traffic_volume.to_numpy()
    for variant,columns in FEATURE_SETS.items():
        model=estimator();model.fit(combined[columns],combined.traffic_volume)
        values=np.maximum(0,model.predict(test[columns]))
        predictions[variant]=values
        summary["versions"][variant]["test"]=metrics(test.traffic_volume,values)
        artifact={"variant":variant,"description":NAMES[variant],"features":columns,"pipeline":model,
                  "source_holidays":source_dates,"training_end_exclusive":"2018-01-01",
                  "settings":SETTINGS,"raw_sha256":audit["raw_sha256"]}
        joblib.dump(artifact,models_dir/f"model_{variant}.joblib",compress=("xz",5))
        print(f"{variant} test MAE = {summary['versions'][variant]['test']['mae']:.3f}; weights saved.",flush=True)
    predictions["Previous week"]=test.volume_lag_168h.to_numpy()
    summary["previous_week_persistence"]={"validation":metrics(validation.traffic_volume,validation.volume_lag_168h),
                                          "test":metrics(test.traffic_volume,test.volume_lag_168h)}
    summary["test_mae_reduction_vs_A_percent"]={v:100*(1-summary['versions'][v]['test']['mae']/summary['versions']['A']['test']['mae']) for v in FEATURE_SETS}
    summary["runtime_seconds"]=time.perf_counter()-started
    original=json.loads((ROOT/"baseline_archive/metrics.json").read_text())
    previous_a=original["models"]["Random forest"]["test"]["mae"]
    audit["original_A_mae_reproduced"]=bool(abs(summary["versions"]["A"]["test"]["mae"]-previous_a)<1e-6)
    if not audit["original_A_mae_reproduced"]:raise ValueError("Version A did not reproduce the original baseline; investigate before reporting")
    dump_json(output/"feature_metrics.json",summary)
    dump_json(output/"feature_data_audit.json",audit)
    dump_json(output/"feature_environment.json",{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                                               "scikit_learn":sklearn.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"joblib":joblib.__version__})
    dump_json(models_dir/"selection.json",{"selected":selected,"criterion":"lowest 2017 validation MAE"})
    dump_json(ROOT/"data/source_holiday_dates.json",source_dates)
    holiday_dates.to_csv(ROOT/"data/holiday_calendar.csv",index=False)
    pd.DataFrame([{"variant":v,"split":split,**summary["versions"][v][split]} for v in FEATURE_SETS for split in ["validation","test"]]).to_csv(output/"feature_metrics.csv",index=False)
    predictions.to_csv(output/"test_predictions_features.csv",index=False)
    frame.to_csv(output/"feature_inputs.csv",index=False)
    subgroups=subgroup_results(test,predictions)
    subgroups.to_csv(output/"subgroup_metrics.csv",index=False)
    inputs=example_inputs(frame)
    dump_json(ROOT/"examples/context_inputs.json",inputs)
    reference={v:predict_records(inputs,v,quiet=True) for v in FEATURE_SETS}
    dump_json(ROOT/"examples/context_reference_predictions.json",reference)
    plot_results(summary,predictions,subgroups)
    print(json.dumps({"selected":selected,"test_mae":{v:summary["versions"][v]["test"]["mae"] for v in FEATURE_SETS},
                      "runtime_minutes":summary['runtime_seconds']/60},indent=2),flush=True)


def predict_records(records,variant="selected",quiet=False):
    if variant=="selected":variant=json.loads((ROOT/"models/selection.json").read_text())["selected"]
    artifact=joblib.load(ROOT/"models"/f"model_{variant}.joblib")
    frame=inference_frame(records,variant,artifact["source_holidays"])
    values=np.maximum(0,artifact["pipeline"].predict(frame[artifact["features"]]))
    result=[{"variant":variant,"date_time":str(row.date_time),"is_holiday":int(row.is_holiday),
             "predicted_vehicles_per_hour":float(value)} for row,value in zip(frame.itertuples(),values)]
    if not quiet:print(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("train")
    sub.add_parser("prepare")
    infer=sub.add_parser("predict")
    infer.add_argument("--inputs",type=Path,default=ROOT/"examples/context_inputs.json")
    infer.add_argument("--variant",choices=["selected","A","B","C"],default="selected")
    infer.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.command=="train":train()
    elif args.command=="prepare":
        frame,audit,_,_=prepare_comparison();print(json.dumps(audit,indent=2))
    else:
        records=json.loads(args.inputs.read_text(encoding="utf-8"))
        results=predict_records(records,args.variant)
        if args.output:dump_json(args.output,results)


if __name__=="__main__":main()
