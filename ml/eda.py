"""Exploratory data analysis. Writes reports/eda.json (used by the About page)."""
import json
import numpy as np
import pandas as pd
from ml.features import data_quality_report, load_data, MODE_COLS

import sys
df = load_data(sys.argv[1] if len(sys.argv) > 1 else "data/raw/ai4i2020.csv")
df["temp_diff"] = df.process_temp - df.air_temp
df["power_w"] = df.torque * df.rpm * 2 * np.pi / 60
df["wear_torque"] = df.tool_wear * df.torque
out = {}

out["data_quality"] = data_quality_report(df)
out["rows"] = len(df)
out["missing_values"] = out["data_quality"]["missing_values"]
out["duplicate_rows"] = out["data_quality"]["duplicate_rows"]
out["failures"] = int(df.failure.sum())
out["failure_rate_pct"] = round(df.failure.mean() * 100, 2)
out["type_stats"] = {t: {"count": int(g.shape[0]), "failures": int(g.failure.sum()),
                         "failure_rate_pct": round(g.failure.mean() * 100, 2)}
                     for t, g in df.groupby("type")}
out["mode_counts"] = {m: int(df[m].sum()) for m in MODE_COLS}
out["failures_without_mode_flag"] = int(((df.failure == 1) & (df[MODE_COLS].sum(axis=1) == 0)).sum())
out["nonfailures_with_mode_flag"] = int(((df.failure == 0) & (df[MODE_COLS].sum(axis=1) > 0)).sum())

feats = ["air_temp", "process_temp", "rpm", "torque", "tool_wear", "temp_diff", "power_w", "wear_torque"]
out["summary"] = {f: {k: round(float(v), 2) for k, v in df[f].describe().items()} for f in feats}
out["summary_by_class"] = {f: {"no_failure_mean": round(float(df[df.failure == 0][f].mean()), 2),
                               "failure_mean": round(float(df[df.failure == 1][f].mean()), 2)} for f in feats}
out["correlation_with_failure"] = {f: round(float(df[f].corr(df.failure)), 3) for f in feats}

# failure rate across bins of the most informative variables
def rate_by_bins(col, bins):
    b = pd.cut(df[col], bins=bins, include_lowest=True)
    g = df.groupby(b, observed=True).failure.agg(["mean", "sum", "count"])
    return [{"bin": str(i), "failure_rate_pct": round(float(r["mean"] * 100), 2),
             "failures": int(r["sum"]), "count": int(r["count"])} for i, r in g.iterrows()]
out["rate_by_tool_wear"] = rate_by_bins("tool_wear", [0, 50, 100, 150, 200, 260])
out["rate_by_temp_diff"] = rate_by_bins("temp_diff", [7, 8.5, 9, 9.5, 10, 10.5, 12])
out["rate_by_rpm"] = rate_by_bins("rpm", [1100, 1300, 1380, 1500, 1700, 2000, 3000])
out["rate_by_torque"] = rate_by_bins("torque", [3, 20, 30, 40, 50, 60, 80])
out["rate_by_power"] = rate_by_bins("power_w", [0, 3500, 5000, 6500, 8000, 9000, 15000])

# structure of the failure-mode columns (verified against the data)
dT_rpm = (df.temp_diff < 8.6) & (df.rpm < 1380)
pw = (df.power_w < 3500) | (df.power_w > 9000)
lim = df.type.map({"L": 11000, "M": 12000, "H": 13000})
ov = df.wear_torque > lim
out["rule_checks"] = {
    "HDF": {"rule": "temp_diff < 8.6 K and rpm < 1380", "rule_true": int(dT_rpm.sum()),
            "flag_true": int(df.HDF.sum()), "mismatches": int((dT_rpm != (df.HDF == 1)).sum())},
    "PWF": {"rule": "power < 3500 W or > 9000 W", "rule_true": int(pw.sum()),
            "flag_true": int(df.PWF.sum()), "mismatches": int((pw != (df.PWF == 1)).sum())},
    "OSF": {"rule": "tool_wear*torque > 11000/12000/13000 (L/M/H)", "rule_true": int(ov.sum()),
            "flag_true": int(df.OSF.sum()), "mismatches": int((ov != (df.OSF == 1)).sum())},
    "TWF": {"wear_min": int(df[df.TWF == 1].tool_wear.min()), "wear_max": int(df[df.TWF == 1].tool_wear.max()),
            "records_with_wear_ge_200": int((df.tool_wear >= 200).sum()),
            "failure_rate_wear_ge_200_pct": round(float(df[df.tool_wear >= 200].failure.mean() * 100), 1)},
    "RNF": {"flagged": int(df.RNF.sum()), "of_which_machine_failure": int(df[df.RNF == 1].failure.sum())},
}
hi, lo = df[df.tool_wear >= 200], df[df.tool_wear < 200]
out["tool_wear_rates"] = {"ge_200_pct": round(float(hi.failure.mean() * 100), 1), "lt_200_pct": round(float(lo.failure.mean() * 100), 1),
                          "ge_200_count": int(len(hi)), "lt_200_count": int(len(lo))}
json.dump(out, open("reports/eda.json", "w"), indent=1)

print("rows", out["rows"], "| failures", out["failures"], f"({out['failure_rate_pct']}%) | missing", out["missing_values"], "| dup rows", out["duplicate_rows"])
print("type:", out["type_stats"])
print("corr with failure:", out["correlation_with_failure"])
for k in ["rate_by_tool_wear", "rate_by_temp_diff", "rate_by_rpm", "rate_by_torque", "rate_by_power"]:
    print(k, [(r["bin"], r["failure_rate_pct"]) for r in out[k]])
print(json.dumps(out["rule_checks"], indent=0))
