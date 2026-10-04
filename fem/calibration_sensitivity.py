"""Descriptive leave-one-LED-out sensitivity; not a measurement uncertainty.

Run: python calibration_sensitivity.py --output-dir results/calibration_sensitivity
Uses the same five pairs as exp_3.py and never changes its global fit.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLORS = ["White", "Blue", "Green", "Yellow", "Red"]
DISTANCE_CM = np.array([5., 6., 6.5, 12., 17.])
ASSIGNED_V = np.array([3.3, 3.1, 2.5, 2.1, 2.])


def fit(distance, voltage):
    slope, intercept = np.polyfit(np.log(distance), np.log(voltage), 1)
    return float(np.exp(intercept)), float(-slope)


def analyze():
    a, gamma = fit(DISTANCE_CM, ASSIGNED_V)
    grid = np.linspace(5., 17., 1201)
    full = a * grid ** (-gamma)
    rows = []
    for i, color in enumerate(COLORS):
        keep = np.arange(5) != i
        ai, gi = fit(DISTANCE_CM[keep], ASSIGNED_V[keep])
        omitted_prediction = ai * DISTANCE_CM[i] ** (-gi)
        rows.append({"omitted_LED": color, "A_V": ai, "gamma": gi,
                     "omitted_distance_cm": float(DISTANCE_CM[i]),
                     "assigned_voltage_V": float(ASSIGNED_V[i]),
                     "omitted_prediction_V": float(omitted_prediction),
                     "omitted_error_V": float(omitted_prediction-ASSIGNED_V[i]),
                     "maximum_relative_change_5_17_cm_percent": float(100*np.max(np.abs(ai*grid**(-gi)/full-1)))})
    log_error = np.log(ASSIGNED_V) - np.log(a*DISTANCE_CM**(-gamma))
    summary = {"method": "Five refits, each omitting one documented LED pair; no new observations or refitting of reported ML models.",
               "interpretation": "Descriptive sensitivity to the five assigned pairs, not a confidence interval, measurement uncertainty, or independent field validation.",
               "full_fit_A_V": a, "full_fit_gamma": gamma,
               "full_fit_R2_log": float(1-np.sum(log_error**2)/np.sum((np.log(ASSIGNED_V)-np.log(ASSIGNED_V).mean())**2)),
               "A_range_V": [min(r["A_V"] for r in rows), max(r["A_V"] for r in rows)],
               "gamma_range": [min(r["gamma"] for r in rows), max(r["gamma"] for r in rows)],
               "maximum_relative_change_5_17_cm_percent": max(r["maximum_relative_change_5_17_cm_percent"] for r in rows),
               "omitted_pair_MAE_V": float(np.mean([abs(r["omitted_error_V"]) for r in rows])),
               "fits": rows}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/calibration_sensitivity"))
    args = parser.parse_args()
    summary = analyze()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (args.output_dir/"leave_one_LED_out.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary["fits"][0])
        writer.writeheader(); writer.writerows(summary["fits"])
    print(json.dumps({k:v for k,v in summary.items() if k != "fits"}, indent=2))


if __name__ == "__main__":
    main()
