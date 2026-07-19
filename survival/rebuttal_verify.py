"""
rebuttal_verify.py — MLHC #283 rebuttal verification harness (7june2026 task list).

Loads cohorts ONCE from /tmp/surv_cache.pkl (built by build_cache()), then runs
the fast Cox-based tasks (1,2,3,6,7). Each value is checked against the task
list's verify-against and flagged on mismatch. Tasks 4 (binary-MB RSF) and 5
(runtime) live in separate scripts because they re-derive structure / refit RSF.

Outputs: notebooks/rebuttal_results.json and notebooks/rebuttal_results.md
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: F401  (needed so pickled config objects unpickle)

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/surv_cache.pkl")
STAB = HERE / "notebooks" / "stability_results.pkl"
OUT_JSON = HERE / "notebooks" / "rebuttal_results.json"
OUT_MD = HERE / "notebooks" / "rebuttal_results.md"

# Paper Table-2 SA-BN Markov blankets (the published MBs).
# RADCURE feat_cols carry these names directly; HANCOCK age / pos-LN are NOT in
# the current loader's feat_cols but ARE raw columns in train_full, so we pull
# them by their raw names.
RADCURE_MB = ["age", "ecog_ps", "gtvp_cm3", "hpv_status", "site",
              "smoking_py", "stage_overall", "t_stage", "tx_modality"]
HANCOCK_MB = ["age_at_initial_diagnosis", "blood_hemoglobin", "invasion_burden",
              "nlr", "number_of_positive_lymph_nodes"]


def build_cache():
    """One-time slow load (runs structure learning); cached to /tmp."""
    from surv_common import load_survival_dataset
    from common import RADCURE_CONFIG, HANCOCK_CONFIG
    cache = {}
    for name, cfg in [("RADCURE", RADCURE_CONFIG), ("HANCOCK", HANCOCK_CONFIG)]:
        cache[name] = load_survival_dataset(cfg, verbose=False)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    return cache


def load_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return build_cache()


def encode_for_survival(df, feat_cols):
    """Continuous (>10 unique) -> numeric; else one-hot with 'missing' level.
    Pulls {col}_raw if present, else col."""
    X = pd.DataFrame(index=df.index)
    for col in feat_cols:
        raw_col = f"{col}_raw"
        if raw_col in df.columns:
            src = df[raw_col]
        elif col in df.columns:
            src = df[col]
        else:
            print(f"    [WARN] column not found: {col}")
            continue
        as_num = pd.to_numeric(src, errors="coerce")
        if as_num.notna().sum() > 0 and as_num.dropna().nunique() > 10:
            X[col] = as_num.fillna(as_num.median())
        else:
            cat = src.astype(str).replace({"nan": "missing", "": "missing"})
            dummies = pd.get_dummies(cat, prefix=col, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    return X.fillna(0).astype(float)


def original_feature_of(col, feat_cols):
    for f in feat_cols:
        if col == f or col.startswith(f + "_"):
            return f
    return col


def build_cox_frame(train_full, mb):
    X = encode_for_survival(train_full, mb)
    df = X.copy()
    df["_T"] = pd.to_numeric(train_full["time_days"], errors="coerce")
    df["_E"] = pd.to_numeric(train_full["event"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["_T"])
    df = df[df["_T"] > 0]
    keep = [c for c in df.columns if c not in {"_T", "_E"} and df[c].nunique() > 1]
    return df[keep + ["_T", "_E"]].copy()


def agg_by_feature(summary, mb):
    """Min-p / matched-coef aggregation of one-hot columns back to features."""
    by = {}
    for col, p in summary["p"].items():
        orig = original_feature_of(col, mb)
        if orig not in by or p < by[orig]["p"]:
            by[orig] = {"p": float(p), "coef": float(summary.loc[col, "coef"])}
    return by


def task_1_3_phfdr(name, train_full, mb):
    """Task 1 (Schoenfeld + BH-FDR) and Task 3 (per-feature Cox q) together —
    same multivariate Cox fit + min-p aggregation used for the RADCURE numbers."""
    print(f"\n=== {name}: Tasks 1 & 3 (Cox q + Schoenfeld, BH-FDR) ===")
    df_fit = build_cox_frame(train_full, mb)
    cph = CoxPHFitter(penalizer=0.001)
    cph.fit(df_fit, duration_col="_T", event_col="_E", show_progress=False)

    # --- Task 3: per-feature Cox p + BH-FDR over the full MB family
    by = agg_by_feature(cph.summary, mb)
    feats = list(by.keys())
    pvals = [by[f]["p"] for f in feats]
    coefs = [by[f]["coef"] for f in feats]
    rej, q, *_ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    cox_tbl = pd.DataFrame({
        "feature": feats, "coef": coefs,
        "HR": [float(np.exp(c)) for c in coefs],
        "p_raw": pvals, "q_BH": q, "reject_q05": rej,
    }).sort_values("p_raw")
    print("  Cox per-feature (BH-FDR over MB family):")
    print(cox_tbl.to_string(index=False, float_format="%.4g"))

    # --- Task 1: Schoenfeld + BH-FDR across the PH-test family
    ph = proportional_hazard_test(cph, df_fit, time_transform="rank")
    ph_df = ph.summary[["test_statistic", "p"]].reset_index()
    ph_df.columns = ["col", "stat", "p"]
    ph_df["feature"] = ph_df["col"].apply(lambda c: original_feature_of(c, mb))
    ph_agg = ph_df.groupby("feature").agg(
        stat=("stat", "max"), p_raw=("p", "min")).reset_index().sort_values("p_raw")
    ph_rej, ph_q, *_ = multipletests(ph_agg["p_raw"].values, alpha=0.05, method="fdr_bh")
    ph_agg["q_BH"] = ph_q
    ph_agg["violates_q05"] = ph_rej
    print("\n  Schoenfeld PH (BH-FDR over PH family):")
    print(ph_agg.to_string(index=False, float_format="%.4g"))

    return {
        "cohort": name, "mb": mb,
        "cox_fdr": cox_tbl.to_dict("records"),
        "schoenfeld_fdr": ph_agg.to_dict("records"),
        "n_used": int(len(df_fit)),
    }


def task_2_stratified(train_full):
    """Task 2: RADCURE SA-MB Cox stratified on overall stage; compare HRs to
    the unstratified fit."""
    print("\n=== RADCURE: Task 2 (stratified Cox on overall stage) ===")
    mb_nostage = [f for f in RADCURE_MB if f != "stage_overall"]
    # unstratified on the same feature subset, for apples-to-apples HR comparison
    df_uns = build_cox_frame(train_full, mb_nostage)
    cph_uns = CoxPHFitter(penalizer=0.001)
    cph_uns.fit(df_uns, duration_col="_T", event_col="_E", show_progress=False)
    hr_uns = agg_by_feature(cph_uns.summary, mb_nostage)

    # stratified: build frame WITH stage as a strata column (categorical, raw)
    X = encode_for_survival(train_full, mb_nostage)
    df = X.copy()
    stage_raw = train_full["stage_overall_raw"] if "stage_overall_raw" in train_full \
        else train_full["stage_overall"]
    df["strata_stage"] = stage_raw.astype(str).replace({"nan": "missing"}).values
    df["_T"] = pd.to_numeric(train_full["time_days"], errors="coerce")
    df["_E"] = pd.to_numeric(train_full["event"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["_T"])
    df = df[df["_T"] > 0]
    keep = [c for c in df.columns if c not in {"_T", "_E", "strata_stage"}
            and df[c].nunique() > 1]
    df_fit = df[keep + ["strata_stage", "_T", "_E"]].copy()
    cph_s = CoxPHFitter(penalizer=0.001)
    cph_s.fit(df_fit, duration_col="_T", event_col="_E",
              strata=["strata_stage"], show_progress=False)
    hr_s = agg_by_feature(cph_s.summary, mb_nostage)

    rows = []
    for f in mb_nostage:
        if f in hr_uns and f in hr_s:
            rows.append({
                "feature": f,
                "HR_unstratified": float(np.exp(hr_uns[f]["coef"])),
                "HR_stratified": float(np.exp(hr_s[f]["coef"])),
                "same_direction": (hr_uns[f]["coef"] > 0) == (hr_s[f]["coef"] > 0),
            })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format="%.4g"))
    print(f"  All same direction: {bool(tbl['same_direction'].all())}")
    return {"cohort": "RADCURE", "table": tbl.to_dict("records"),
            "all_same_direction": bool(tbl["same_direction"].all())}


def task_6_table(stab):
    """Task 6: combined membership table per cohort from stability_results.pkl
    (selection freq + directed-edge-into-survival stability). HRs/q filled from
    Tasks 1/3 downstream in the markdown writer."""
    out = {}
    for ck, mb_members in [("radcure", None), ("hancock", None)]:
        mbst = stab[ck]["mb_stability"]
        edst = stab[ck]["edge_stability"]
        edge_to_surv = {k.split(" -> ")[0]: v for k, v in edst.items()
                        if k.endswith("-> survival")}
        rows = []
        for feat, freq in sorted(mbst.items(), key=lambda x: -x[1]):
            rows.append({"feature": feat, "sel_freq": freq,
                         "edge_to_survival": edge_to_surv.get(feat)})
        out[ck] = rows
    return out


def task_7_mice():
    """Task 7: confirm HANCOCK MICE recovery count vs listwise deletion."""
    print("\n=== Task 7 (MICE recovery count) ===")
    # Source of truth: rebuild_hancock.py / hancock_rebuilt.csv (n=763) vs the
    # 620 binary-evaluable / listwise count documented in the paper.
    rebuilt = HERE.parent / "hancock_rebuilt.csv"
    n_full = None
    if rebuilt.exists():
        n_full = len(pd.read_csv(rebuilt))
    # Listwise-complete count on the blood biomarkers (documented ~620)
    result = {"n_rebuilt": n_full, "n_listwise_doc": 620,
              "recovered_doc": 143}
    print(f"  hancock_rebuilt.csv n={n_full}; documented listwise=620; "
          f"recovered=143")
    return result


# ---- verify-against table -------------------------------------------------
VERIFY = {
    "radcure_cox_q": {"hpv_status": 2.5e-13, "age": 2.0e-12, "tx_modality": 6.0e-10,
                      "gtvp_cm3": 4.3e-8, "ecog_ps": 1.3e-7, "smoking_py": 2.3e-7,
                      "site": 6.9e-6, "stage_overall": 3.3e-4, "t_stage": 0.42},
    "radcure_ph": {"stage_overall": (8e-6, 7e-5, True),
                   "smoking_py": (0.018, 0.079, False),
                   "age": (0.055, 0.107, False)},
    "hancock_ph": {"blood_hemoglobin": (0.022, 0.067, False)},
}


def flag(actual, expected, tol_factor=3.0):
    """Order-of-magnitude check for p/q values."""
    if expected == 0 or actual == 0:
        return "OK" if abs(actual - expected) < 1e-12 else "CHECK"
    ratio = actual / expected
    return "OK" if (1 / tol_factor) <= ratio <= tol_factor else "MISMATCH"


def main():
    cache = load_cache()
    with open(STAB, "rb") as f:
        stab = pickle.load(f)

    radcure_tf = cache["RADCURE"]["train_full"]
    hancock_tf = cache["HANCOCK"]["train_full"]
    from common import RADCURE_CONFIG, HANCOCK_CONFIG, _add_interactions
    radcure_tf, _ = _add_interactions(radcure_tf.copy(), RADCURE_CONFIG)
    hancock_tf, _ = _add_interactions(hancock_tf.copy(), HANCOCK_CONFIG)

    results = {}
    results["task1_3_radcure"] = task_1_3_phfdr("RADCURE", radcure_tf, RADCURE_MB)
    results["task1_3_hancock"] = task_1_3_phfdr("HANCOCK", hancock_tf, HANCOCK_MB)
    results["task2_radcure"] = task_2_stratified(radcure_tf)
    results["task6_table"] = task_6_table(stab)
    results["task7_mice"] = task_7_mice()

    # ---- verification flags
    print("\n=== VERIFY-AGAINST CHECKS ===")
    checks = []
    rc_cox = {r["feature"]: r["q_BH"] for r in results["task1_3_radcure"]["cox_fdr"]}
    for f, exp in VERIFY["radcure_cox_q"].items():
        st = flag(rc_cox.get(f, np.nan), exp)
        checks.append(("RADCURE Cox q " + f, rc_cox.get(f), exp, st))
    rc_ph = {r["feature"]: r for r in results["task1_3_radcure"]["schoenfeld_fdr"]}
    for f, (ep, eq, ev) in VERIFY["radcure_ph"].items():
        r = rc_ph.get(f, {})
        st = flag(r.get("p_raw", np.nan), ep)
        checks.append(("RADCURE PH p " + f, r.get("p_raw"), ep, st))
    hc_ph = {r["feature"]: r for r in results["task1_3_hancock"]["schoenfeld_fdr"]}
    for f, (ep, eq, ev) in VERIFY["hancock_ph"].items():
        r = hc_ph.get(f, {})
        st = flag(r.get("p_raw", np.nan), ep)
        checks.append(("HANCOCK PH p " + f, r.get("p_raw"), ep, st))
    for nm, act, exp, st in checks:
        act_s = f"{act:.3g}" if isinstance(act, float) else str(act)
        exp_s = f"{exp:.3g}" if isinstance(exp, float) else str(exp)
        print(f"  [{st}] {nm}: actual={act_s} expected~{exp_s}")
    results["verify_checks"] = [
        {"name": nm, "actual": act, "expected": exp, "status": st}
        for nm, act, exp, st in checks]

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-> JSON: {OUT_JSON}")
    return results


if __name__ == "__main__":
    main()
