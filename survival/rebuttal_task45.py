"""
rebuttal_task45.py — Task 4 (RSF on binary-MB vs SA-MB, C-index + bootstrap CI)
and Task 5 (runtime benchmark). Loads cohorts from /tmp/surv_cache.pkl.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import pickle
import time as _time
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: F401

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from rebuttal_verify import (encode_for_survival, RADCURE_MB, HANCOCK_MB)
from common import RADCURE_CONFIG, HANCOCK_CONFIG, _add_interactions

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/surv_cache.pkl")
OUT = HERE / "notebooks" / "rebuttal_task45_results.json"
SEED = 42

# Paper-Table-2 binary-proxy MBs (the cache's current loader drifts slightly:
# RADCURE drops TxN -> 6, HANCOCK adds Hb/pT -> 5; we use the published lists).
RADCURE_BIN_MB = ["age", "ecog_ps", "gtvp_cm3", "site", "t_stage", "tx_modality", "TxN"]
HANCOCK_BIN_MB = ["infiltration_depth_in_mm", "invasion_burden", "pN_grouped"]


def encode_aligned(train_full, test_full, mb):
    """Encode train+test together so dummy columns align, then split."""
    n_tr = len(train_full)
    both = pd.concat([train_full, test_full], axis=0, ignore_index=True)
    X = encode_for_survival(both, mb)
    return X.iloc[:n_tr].reset_index(drop=True), X.iloc[n_tr:].reset_index(drop=True)


def rsf_cindex(Xtr, ytr, Xte, yte, n_boot=1000):
    rsf = RandomSurvivalForest(
        n_estimators=2000, max_depth=7, min_samples_leaf=4,
        min_samples_split=8, random_state=SEED, n_jobs=-1)
    rsf.fit(Xtr.values, ytr)
    risk = rsf.predict(Xte.values)
    ev = yte["event"]
    tm = yte["time"]
    c = concordance_index_censored(ev, tm, risk)[0]
    # bootstrap CI over test set
    rng = np.random.RandomState(SEED)
    n = len(risk)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if ev[idx].sum() < 2:
            continue
        try:
            boots.append(concordance_index_censored(ev[idx], tm[idx], risk[idx])[0])
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"cindex": float(c), "ci_lower": float(lo), "ci_upper": float(hi),
            "n_features": Xtr.shape[1]}


def task4(cache):
    print("\n=== Task 4: RSF binary-MB vs SA-MB ===")
    out = {}
    specs = [
        ("RADCURE", RADCURE_CONFIG, RADCURE_BIN_MB, RADCURE_MB),
        ("HANCOCK", HANCOCK_CONFIG, HANCOCK_BIN_MB, HANCOCK_MB),
    ]
    for name, cfg, bin_mb, sa_mb in specs:
        ds = cache[name]
        tr, _ = _add_interactions(ds["train_full"].copy(), cfg)
        te, _ = _add_interactions(ds["test_full"].copy(), cfg)
        ytr, yte = ds["y_train_surv"], ds["y_test_surv"]
        print(f"\n  {name}: binary-MB {bin_mb}")
        Xtr_b, Xte_b = encode_aligned(tr, te, bin_mb)
        rb = rsf_cindex(Xtr_b, ytr, Xte_b, yte)
        print(f"    binary-MB RSF C={rb['cindex']:.3f} "
              f"[{rb['ci_lower']:.2f},{rb['ci_upper']:.2f}] k={rb['n_features']}")
        print(f"  {name}: SA-MB {sa_mb}")
        Xtr_s, Xte_s = encode_aligned(tr, te, sa_mb)
        rs = rsf_cindex(Xtr_s, ytr, Xte_s, yte)
        print(f"    SA-MB RSF C={rs['cindex']:.3f} "
              f"[{rs['ci_lower']:.2f},{rs['ci_upper']:.2f}] k={rs['n_features']}")
        out[name] = {"binary_mb": bin_mb, "sa_mb": sa_mb,
                     "binary_rsf": rb, "sa_rsf": rs,
                     "delta_cindex": rs["cindex"] - rb["cindex"]}
    return out


def task5(cache):
    print("\n=== Task 5: runtime benchmark ===")
    from surv_bn_aware import survival_ensemble_bn, load_all_patients
    out = {}
    for name, cfg in [("RADCURE", RADCURE_CONFIG), ("HANCOCK", HANCOCK_CONFIG)]:
        disc_df, feat_cols, tm, ev, train_mask = load_all_patients(cfg, verbose=False)
        tr = disc_df[train_mask]
        ttm = tm[train_mask.values]
        tev = ev[train_mask.values]
        t0 = _time.time()
        survival_ensemble_bn(tr, feat_cols, ttm, tev, verbose=False)
        t_full = _time.time() - t0
        # n~100 subsample single pass
        sub = tr.sample(n=min(100, len(tr)), random_state=SEED)
        si = sub.index
        pos = [tr.index.get_loc(i) for i in si]
        t0 = _time.time()
        survival_ensemble_bn(sub, feat_cols, ttm[pos], tev[pos], verbose=False)
        t_sub = _time.time() - t0
        out[name] = {"n_train": int(len(tr)),
                     "structure_pass_sec": round(t_full, 2),
                     "n100_pass_sec": round(t_sub, 2)}
        print(f"  {name}: 1 ensemble pass {t_full:.1f}s (n={len(tr)}); "
              f"n=100 pass {t_sub:.1f}s")
    return out


def main():
    cache = pickle.load(open(CACHE, "rb"))
    res = {"task4": task4(cache)}
    OUT.write_text(json.dumps(res, indent=2, default=str))  # save before slow task5
    res["task5"] = task5(cache)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n-> {OUT}")

    print("\n=== VERIFY ===")
    t4 = res["task4"]
    targets = {"RADCURE": (0.801, 0.770), "HANCOCK": (0.726, 0.648)}
    for name, (sa_t, bin_t) in targets.items():
        sa = t4[name]["sa_rsf"]["cindex"]
        bn = t4[name]["binary_rsf"]["cindex"]
        print(f"  {name} SA-MB: actual {sa:.3f} vs target {sa_t:.3f} "
              f"({'OK' if abs(sa-sa_t)<0.04 else 'DRIFT'})")
        print(f"  {name} binary-MB: actual {bn:.3f} vs target {bin_t:.3f} "
              f"({'OK' if abs(bn-bin_t)<0.04 else 'DRIFT'})")


if __name__ == "__main__":
    main()
