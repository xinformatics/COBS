"""
rebuttal_whas500.py — SA-BN vs binary BN on WHAS500 (Worcester Heart Attack
Study; ships with scikit-survival). One non-cancer cohort (acute MI /
cardiology) to test whether the binarization effect is disease-agnostic.
2-year binarization, matching the paper's SVy2. Reuses the GBSG2/TCGA helpers.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

from sksurv.datasets import load_whas500

from rebuttal_gbsg2 import run

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "rebuttal_whas500_results.json"


def load_whas_df():
    X, y = load_whas500()
    df = X.copy().reset_index(drop=True)
    df["event"] = y["fstat"].astype(int)
    df["time_months"] = y["lenfol"].astype(float) / 30.4375
    df = df[df["time_months"] > 0].copy()
    feat_cols = ["age", "bmi", "hr", "sysbp", "diasbp", "gender", "afb",
                 "chf", "cvd", "sho", "av3", "mitype", "miord", "los"]
    return df, feat_cols


def main():
    df, fc = load_whas_df()
    res = run("WHAS500 (acute MI, non-cancer)", df, fc)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
