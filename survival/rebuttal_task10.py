"""
rebuttal_task10.py — four-model continuous-time metric suite (fp3P / Task 10).
For each cohort x feature set (SA-MB, binary-MB), fit Cox PH, RSF, GBS, and
penalized (Coxnet/LASSO) Cox, and report:
  - Harrell's C-index
  - Uno's IPCW C-index
  - Integrated Brier score (lower is better)
  - Time-dependent AUC (mean over the grid)
Goal: confirm the SA-MB advantage over the binary-MB holds across models/metrics.
Loads /tmp/surv_cache.pkl.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# sksurv (built for numpy>=2.0) calls np.trapezoid; alias to np.trapz on older numpy
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            integrated_brier_score, cumulative_dynamic_auc)
from sklearn.preprocessing import StandardScaler

from rebuttal_verify import encode_for_survival, RADCURE_MB, HANCOCK_MB
from rebuttal_task45 import RADCURE_BIN_MB, HANCOCK_BIN_MB
from common import RADCURE_CONFIG, HANCOCK_CONFIG, _add_interactions

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/surv_cache.pkl")
OUT = HERE / "notebooks" / "rebuttal_task10_results.json"
SEED = 42


def encode_aligned(tr, te, mb):
    n = len(tr)
    both = pd.concat([tr, te], ignore_index=True)
    X = encode_for_survival(both, mb)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def time_grid(y_train, y_test):
    """Times strictly inside the test follow-up and below max train time."""
    te = y_test["time"]
    tmax_train = y_train["time"].max()
    lo = np.percentile(te[te > 0], 15)
    hi = min(np.percentile(te, 80), tmax_train * 0.95)
    lo = max(lo, te.min() + 1e-3)
    return np.linspace(lo, hi, 6)


def surv_prob_matrix(model, X, times, alpha=None):
    """n_samples x n_times survival probabilities."""
    if alpha is not None:
        fns = model.predict_survival_function(X, alpha=alpha)
    else:
        fns = model.predict_survival_function(X)
    return np.asarray([[fn(t) for t in times] for fn in fns])


def eval_model(name, fit_fn, Xtr, ytr, Xte, yte, times, needs_scale):
    try:
        model, risk, alpha = fit_fn(Xtr, ytr, Xte)
        harrell = concordance_index_censored(yte["event"], yte["time"], risk)[0]
        tau = float(times[-1])
        uno = concordance_index_ipcw(ytr, yte, risk, tau=tau)[0]
        auc_vals, auc_mean = cumulative_dynamic_auc(ytr, yte, risk, times)
        try:
            surv = surv_prob_matrix(model, Xte, times, alpha)
            ibs = integrated_brier_score(ytr, yte, surv, times)
        except Exception as e:
            ibs = None
        return {"model": name, "harrell_c": float(harrell),
                "uno_ipcw_c": float(uno), "auc_mean": float(auc_mean),
                "ibs": (None if ibs is None else float(ibs))}
    except Exception as e:
        return {"model": name, "error": str(e)[:160]}


def make_fitters():
    def cox(Xtr, ytr, Xte):
        m = CoxPHSurvivalAnalysis(alpha=1.0)
        m.fit(Xtr, ytr)
        return m, m.predict(Xte), None

    def rsf(Xtr, ytr, Xte):
        m = RandomSurvivalForest(n_estimators=500, max_depth=7,
                                 min_samples_leaf=4, min_samples_split=8,
                                 random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m, m.predict(Xte), None

    def gbs(Xtr, ytr, Xte):
        m = GradientBoostingSurvivalAnalysis(n_estimators=200, max_depth=3,
                                             learning_rate=0.1,
                                             min_samples_leaf=10,
                                             random_state=SEED)
        m.fit(Xtr, ytr)
        return m, m.predict(Xte), None

    def lasso(Xtr, ytr, Xte):
        m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01,
                                   fit_baseline_model=True, max_iter=100000)
        m.fit(Xtr, ytr)
        alphas = m.alphas_
        a = float(alphas[len(alphas) // 2])  # mid-path penalty
        risk = m.predict(Xte, alpha=a)
        return m, risk, a

    return {"Cox PH": (cox, True), "RSF": (rsf, False),
            "GBS": (gbs, False), "Penalized Cox (LASSO)": (lasso, True)}


def run_cohort(name, cfg, sa_mb, bin_mb, cache):
    ds = cache[name]
    tr, _ = _add_interactions(ds["train_full"].copy(), cfg)
    te, _ = _add_interactions(ds["test_full"].copy(), cfg)
    ytr, yte = ds["y_train_surv"], ds["y_test_surv"]
    out = {}
    for mb_name, mb in [("SA-MB", sa_mb), ("binary-MB", bin_mb)]:
        Xtr_df, Xte_df = encode_aligned(tr, te, mb)
        times = time_grid(ytr, yte)
        fitters = make_fitters()
        rows = []
        print(f"\n  {name} / {mb_name} (k={Xtr_df.shape[1]} cols, "
              f"grid {times[0]:.0f}-{times[-1]:.0f})")
        for mname, (fn, scale) in fitters.items():
            if scale:
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr_df.values)
                Xte = sc.transform(Xte_df.values)
            else:
                Xtr, Xte = Xtr_df.values, Xte_df.values
            r = eval_model(mname, fn, Xtr, ytr, Xte, yte, times, scale)
            rows.append(r)
            if "error" in r:
                print(f"    {mname:22s} ERROR: {r['error']}")
            else:
                print(f"    {mname:22s} C={r['harrell_c']:.3f} "
                      f"Uno={r['uno_ipcw_c']:.3f} AUC={r['auc_mean']:.3f} "
                      f"IBS={r['ibs'] if r['ibs'] is None else round(r['ibs'],3)}")
        out[mb_name] = rows
    return out


def main():
    cache = pickle.load(open(CACHE, "rb"))
    res = {}
    res["RADCURE"] = run_cohort("RADCURE", RADCURE_CONFIG, RADCURE_MB,
                                RADCURE_BIN_MB, cache)
    res["HANCOCK"] = run_cohort("HANCOCK", HANCOCK_CONFIG, HANCOCK_MB,
                                HANCOCK_BIN_MB, cache)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n-> {OUT}")

    # summary: SA-MB vs binary-MB per metric per model
    print("\n=== SA-MB advantage (SA minus binary), by model x metric ===")
    for coh in ["RADCURE", "HANCOCK"]:
        sa = {r["model"]: r for r in res[coh]["SA-MB"] if "error" not in r}
        bn = {r["model"]: r for r in res[coh]["binary-MB"] if "error" not in r}
        print(f"\n{coh}:")
        for m in sa:
            if m in bn:
                dC = sa[m]["harrell_c"] - bn[m]["harrell_c"]
                dU = sa[m]["uno_ipcw_c"] - bn[m]["uno_ipcw_c"]
                dA = sa[m]["auc_mean"] - bn[m]["auc_mean"]
                dI = (None if sa[m]["ibs"] is None or bn[m]["ibs"] is None
                      else bn[m]["ibs"] - sa[m]["ibs"])  # IBS: lower better
                print(f"  {m:22s} dHarrell={dC:+.3f} dUno={dU:+.3f} "
                      f"dAUC={dA:+.3f} dIBS(bin-SA)="
                      f"{'NA' if dI is None else f'{dI:+.3f}'}")


if __name__ == "__main__":
    main()
