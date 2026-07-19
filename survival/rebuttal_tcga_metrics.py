"""
rebuttal_tcga_metrics.py — full metric suite (Harrell C, Uno IPCW C, tAUC, IBS)
for SA-MB vs binary-MB on TCGA-BRCA and TCGA-COAD, to complete Table 2.
Reuses the deterministic MB feature lists from rebuttal_tcga_cancers_results.json
and the same train/test split as rebuttal_tcga_cancers.py (SEED=42, 70/30).
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            integrated_brier_score, cumulative_dynamic_auc)

from rebuttal_tcga_cancers import load_brca, load_coad, encode_for_cox

HERE = Path(__file__).resolve().parent
JSON_IN = HERE / "notebooks" / "rebuttal_tcga_cancers_results.json"
OUT = HERE / "notebooks" / "rebuttal_tcga_metrics_results.json"
SEED = 42


def split(df):
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(0.7 * len(df))
    tr = df.index[idx[:n_train]]
    mask = df.index.isin(tr)
    return df[mask].copy(), df[~mask].copy()


def y_surv(df):
    return np.array([(bool(e), float(t)) for e, t in
                     zip(df["event"], df["time_months"])],
                    dtype=[("event", bool), ("time", float)])


def metrics(df_tr, df_te, feats):
    feats = [f for f in feats if f in df_tr.columns]
    Xtr = encode_for_cox(df_tr, feats)
    Xte = encode_for_cox(df_te, feats).reindex(columns=Xtr.columns, fill_value=0.0)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr.values), sc.transform(Xte.values)
    ytr, yte = y_surv(df_tr), y_surv(df_te)
    cox = CoxPHSurvivalAnalysis(alpha=1e-2)
    cox.fit(Xtr_s, ytr)
    risk = cox.predict(Xte_s)
    harrell = concordance_index_censored(yte["event"], yte["time"], risk)[0]
    # time grid inside test follow-up and below the max *event* time of both
    # train and test (IPCW estimators require this).
    te = yte["time"]
    t_tr = ytr["time"][ytr["event"]].max() if ytr["event"].any() else ytr["time"].max()
    t_te = yte["time"][yte["event"]].max() if yte["event"].any() else yte["time"].max()
    t_cap = min(float(t_tr), float(t_te))
    lo = max(np.percentile(te[te > 0], 15), te.min() + 1e-3)
    hi = min(np.percentile(te, 80), 0.95 * t_cap)
    if hi <= lo:
        hi = lo + 1e-3
    times = np.linspace(lo, hi, 6)
    out = {"k": len(feats), "harrell_c": float(harrell),
           "uno_ipcw_c": None, "tauc": None, "ibs": None}
    try:
        out["uno_ipcw_c"] = float(
            concordance_index_ipcw(ytr, yte, risk, tau=float(times[-1]))[0])
    except Exception:
        pass
    try:
        _, auc = cumulative_dynamic_auc(ytr, yte, risk, times)
        out["tauc"] = float(auc)
    except Exception:
        pass
    try:
        surv = np.asarray([[fn(t) for t in times]
                           for fn in cox.predict_survival_function(Xte_s)])
        out["ibs"] = float(integrated_brier_score(ytr, yte, surv, times))
    except Exception:
        pass
    return out


def main():
    stored = {r["cohort"]: r for r in json.loads(JSON_IN.read_text())}
    loaders = {"TCGA-BRCA (breast)": load_brca, "TCGA-COAD (colorectal)": load_coad}
    out = {}
    for cohort, loader in loaders.items():
        df, _ = loader()
        df_tr, df_te = split(df)
        s = stored[cohort]
        res = {"binary_mb": s["binary_mb"], "sabn_mb": s["sabn_mb"],
               "binary": metrics(df_tr, df_te, s["binary_mb"]),
               "sabn": metrics(df_tr, df_te, s["sabn_mb"])}
        out[cohort] = res
        print(f"\n{cohort}")
        print(f"  binary-MB {s['binary_mb']}")
        b = res["binary"]; sa = res["sabn"]
        print(f"    binary: C={b['harrell_c']:.3f} Uno={b['uno_ipcw_c']:.3f} "
              f"tAUC={b['tauc']:.3f} IBS={b['ibs']:.3f}")
        print(f"  SA-MB {s['sabn_mb']}")
        print(f"    SA:     C={sa['harrell_c']:.3f} Uno={sa['uno_ipcw_c']:.3f} "
              f"tAUC={sa['tauc']:.3f} IBS={sa['ibs']:.3f}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
