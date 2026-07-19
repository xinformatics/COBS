"""
rebuttal_gbsg2.py — SA-BN vs binary BN on the GBSG2 breast-cancer cohort
(German Breast Cancer Study Group 2; ships with scikit-survival). Independent
of TCGA and of head/neck: a classic clinical-trial cohort with routine
clinico-pathologic features. Uses 2-year binarization to match the original
RADCURE/HANCOCK SVy2 endpoint. Reuses the TCGA pipeline helpers.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import pandas as pd

from sksurv.datasets import load_gbsg2

from rebuttal_tcga_cancers import (discretize, binary_bn_mb, make_binary_5yr)
from rebuttal_tcga_metrics import metrics
from surv_bn_aware import survival_ensemble_bn, survival_mb_discovery

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "rebuttal_gbsg2_results.json"
SEED = 42
HORIZON_MONTHS = 24  # 2-year binarization, matching the paper's SVy2


def load_gbsg2_df():
    X, y = load_gbsg2()
    df = X.copy().reset_index(drop=True)
    df["event"] = y["cens"].astype(int)
    df["time_months"] = y["time"].astype(float) / 30.4375  # days -> months
    df = df[df["time_months"] > 0].copy()
    feat_cols = ["age", "estrec", "horTh", "menostat", "pnodes",
                 "progrec", "tgrade", "tsize"]
    return df, feat_cols


def run(name, df, feat_cols, horizon_months=HORIZON_MONTHS):
    print(f"\n{'='*64}\n  {name}  (2-year binarization)\n{'='*64}")
    n = len(df)
    print(f"  n={n}, events={int(df['event'].sum())} "
          f"({df['event'].mean():.0%}), median FU "
          f"{df['time_months'].median():.1f} mo")

    rng = np.random.RandomState(SEED)
    idx = np.arange(n); rng.shuffle(idx)
    n_train = int(0.7 * n)
    train_mask = df.index.isin(df.index[idx[:n_train]])

    # 2-year binary label (drops censored-before-2yr patients)
    df = df.copy()
    df["surv2y"] = make_binary_5yr(df, horizon_months=horizon_months)
    n_dropped = int((df["surv2y"] == "dropped").sum())
    print(f"  2yr labels: alive={int((df['surv2y']=='alive').sum())}, "
          f"dead={int((df['surv2y']=='dead').sum())}, "
          f"dropped={n_dropped} ({n_dropped/n:.1%})")

    disc_df = discretize(df, feat_cols, train_mask, n_bins=3)
    disc_df["surv2y"] = df["surv2y"].values

    print("  [1/2] binary BN ...")
    bin_mb, _ = binary_bn_mb(disc_df, feat_cols, train_mask, target_name="surv2y")
    n_bin_used = int(train_mask.sum() -
                     (disc_df.loc[train_mask, "surv2y"] == "dropped").sum())
    print(f"    binary MB ({len(bin_mb)}): {sorted(bin_mb)}  "
          f"(trained on {n_bin_used}/{int(train_mask.sum())})")

    print("  [2/2] SA-BN ...")
    train_disc = disc_df[train_mask].copy()
    t = df.loc[train_mask, "time_months"].values
    e = df.loc[train_mask, "event"].values
    G, _ = survival_ensemble_bn(train_disc, feat_cols, t, e, verbose=False)
    smb = survival_mb_discovery(G, train_disc, feat_cols, t, e, verbose=False)
    sa_mb = set(smb["consensus"])
    print(f"    SA-BN MB ({len(sa_mb)}): {sorted(sa_mb)}  "
          f"(trained on {int(train_mask.sum())}, no exclusion)")

    overlap = bin_mb & sa_mb
    jac = len(overlap) / max(len(bin_mb | sa_mb), 1)
    print(f"    overlap={sorted(overlap)} | SA-only={sorted(sa_mb-bin_mb)} | "
          f"binary-only={sorted(bin_mb-sa_mb)} | Jaccard={jac:.2f}")

    train_df, test_df = df[train_mask].copy(), df[~train_mask].copy()
    m_bin = metrics(train_df, test_df, sorted(bin_mb))
    m_sa = metrics(train_df, test_df, sorted(sa_mb))
    def fmt(m):
        g = lambda k: "NA" if m[k] is None else f"{m[k]:.3f}"
        return f"C={g('harrell_c')} Uno={g('uno_ipcw_c')} tAUC={g('tauc')} IBS={g('ibs')}"
    print(f"  binary-MB: {fmt(m_bin)}")
    print(f"  SA-MB:     {fmt(m_sa)}")

    return {"cohort": name, "n": n, "n_events": int(df["event"].sum()),
            "median_fu_months": float(df["time_months"].median()),
            "horizon_months": horizon_months,
            "n_dropped_by_binary": n_dropped, "pct_dropped": n_dropped / n,
            "binary_mb": sorted(bin_mb), "sabn_mb": sorted(sa_mb),
            "sa_only": sorted(sa_mb - bin_mb),
            "binary_only": sorted(bin_mb - sa_mb), "jaccard": jac,
            "binary": m_bin, "sabn": m_sa}


def main():
    df, fc = load_gbsg2_df()
    res = run("GBSG2 (breast, independent of TCGA)", df, fc)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
