"""
rebuttal_paired_bootstrap.py — Paired bootstrap difference test on C-index
between SA-BN and each survival-native feature-selection baseline, with
Benjamini-Hochberg FDR correction across the test family.

Reuses the feature sets cached from rebuttal_baselines.py to avoid the
expensive RSF permutation-importance step.

For each cohort:
  - Fit Cox PH on test set using each method's selected features
  - Generate B=1000 paired bootstrap resamples of test set
  - For each pair of methods, compute distribution of (C_A - C_B)
  - Two-sided p-value = 2 * min(P(diff>=0), P(diff<=0))
  - Apply BH-FDR across all method pairs within each cohort
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from surv_common import load_survival_dataset
from common import RADCURE_CONFIG, HANCOCK_CONFIG, SEED, _add_interactions


HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "notebooks" / "rebuttal_paired_bootstrap.md"


def encode(df, feat_cols):
    X = pd.DataFrame(index=df.index)
    for col in feat_cols:
        raw_col = f"{col}_raw"
        src = df[raw_col] if raw_col in df.columns else df.get(col)
        if src is None:
            continue
        as_num = pd.to_numeric(src, errors="coerce")
        if as_num.notna().sum() > 0 and as_num.dropna().nunique() > 10:
            X[col] = as_num.fillna(as_num.median())
        else:
            cat = src.astype(str).replace({"nan": "missing", "": "missing"})
            dummies = pd.get_dummies(cat, prefix=col, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    X = X.fillna(0).astype(float)
    return X


def get_risk_scores(train_full, test_full, y_train, feats):
    """Fit Cox PH on train, return test-set risk scores."""
    if not feats:
        return None
    Xtr = encode(train_full, feats)
    Xte_full = encode(test_full, feats)
    Xte = Xte_full.reindex(columns=Xtr.columns, fill_value=0.0)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr.values)
    Xte_s = scaler.transform(Xte.values)
    cox = CoxPHSurvivalAnalysis(alpha=1e-3)
    try:
        cox.fit(Xtr_s, y_train)
        return cox.predict(Xte_s)
    except Exception as e:
        print(f"    Cox fit FAILED on {feats[:3]}...: {e}")
        return None


def paired_bootstrap_diff(risk_a, risk_b, y_test, n_boot=1000):
    """Paired bootstrap on C-index difference.

    Returns (mean_diff, ci_lower, ci_upper, p_value_two_sided).
    """
    event = y_test["event"]
    time = y_test["time"]
    n = len(event)
    rng = np.random.RandomState(SEED)

    # Point estimates
    c_a, *_ = concordance_index_censored(event, time, risk_a)
    c_b, *_ = concordance_index_censored(event, time, risk_b)
    point_diff = c_a - c_b

    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if y_test["event"][idx].sum() < 3:
            continue
        try:
            ca_b, *_ = concordance_index_censored(
                event[idx], time[idx], risk_a[idx]
            )
            cb_b, *_ = concordance_index_censored(
                event[idx], time[idx], risk_b[idx]
            )
            diffs.append(ca_b - cb_b)
        except Exception:
            continue
    diffs = np.array(diffs)
    if len(diffs) < 50:
        return point_diff, np.nan, np.nan, np.nan

    ci_l, ci_u = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p-value: 2 * min(P(diff >= 0), P(diff <= 0)) under H0: diff=0
    p_pos = (diffs >= 0).mean()
    p_neg = (diffs <= 0).mean()
    p_two = 2 * min(p_pos, p_neg)
    return point_diff, ci_l, ci_u, max(p_two, 1.0 / n_boot)  # floor


def run_cohort(name, config, mb_sabn, lasso_sm, lasso_cv, rsf_topk):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    ds = load_survival_dataset(config, verbose=False)
    train_full = ds["train_full"]
    test_full = ds["test_full"]
    y_train = ds["y_train_surv"]
    y_test = ds["y_test_surv"]
    feat_cols = ds["feat_cols"]
    train_full, _ = _add_interactions(train_full.copy(), config)
    test_full, _ = _add_interactions(test_full.copy(), config)

    # Filter MBs to features available in feat_cols
    methods = {
        "SA-BN MB": [f for f in mb_sabn if f in feat_cols],
        "LASSO-Cox (size-matched)": [f for f in lasso_sm if f in feat_cols],
        "LASSO-Cox (CV-best)": [f for f in lasso_cv if f in feat_cols],
        "RSF-VI (top-k)": [f for f in rsf_topk if f in feat_cols],
    }
    print(f"  Test n={len(test_full)}, events={int(y_test['event'].sum())}")
    for k, v in methods.items():
        print(f"    {k:30s} k={len(v):2d}  feats={v}")

    # Fit Cox PH on each method's features, get test risk scores
    print("\n  Fitting Cox PH and computing test risk scores...")
    risk_scores = {}
    c_indices = {}
    for k, v in methods.items():
        r = get_risk_scores(train_full, test_full, y_train, v)
        if r is None:
            continue
        risk_scores[k] = r
        c, *_ = concordance_index_censored(y_test["event"], y_test["time"], r)
        c_indices[k] = float(c)
        print(f"    {k:30s} test C={c:.3f}")

    # Pairwise comparisons (SA-BN vs each baseline)
    print("\n  Paired-bootstrap pairwise C-index differences (B=1000):")
    pairs = []
    baseline = "SA-BN MB"
    if baseline not in risk_scores:
        print(f"  ERROR: {baseline} not fittable on this cohort")
        return None
    others = [k for k in risk_scores if k != baseline]
    for o in others:
        pairs.append((baseline, o))
    # Also LASSO_sm vs RSF for completeness
    if "LASSO-Cox (size-matched)" in risk_scores and "RSF-VI (top-k)" in risk_scores:
        pairs.append(("LASSO-Cox (size-matched)", "RSF-VI (top-k)"))

    rows = []
    for a, b in pairs:
        diff, ci_l, ci_u, p = paired_bootstrap_diff(
            risk_scores[a], risk_scores[b], y_test, n_boot=1000
        )
        rows.append({
            "method_A": a,
            "method_B": b,
            "C_A": c_indices[a],
            "C_B": c_indices[b],
            "diff_AmB": diff,
            "diff_CI_lower": ci_l,
            "diff_CI_upper": ci_u,
            "p_raw": p,
        })
        print(f"    {a:25s} vs {b:25s} ΔC={diff:+.3f} "
              f"[{ci_l:+.3f}, {ci_u:+.3f}]  p={p:.3g}")

    # BH-FDR within cohort
    pvals = [r["p_raw"] for r in rows]
    reject, qvals, *_ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    for r, q, rj in zip(rows, qvals, reject):
        r["q_BH"] = float(q)
        r["reject_q05"] = bool(rj)

    print(f"\n  BH-FDR q-values (m={len(rows)} tests):")
    for r in rows:
        print(f"    {r['method_A']:25s} vs {r['method_B']:25s} "
              f"q={r['q_BH']:.3g}  {'REJECT' if r['reject_q05'] else 'fail to reject'}")

    return {"cohort": name, "c_indices": c_indices, "pairs": rows}


def main():
    # Feature sets from prior rebuttal_baselines.py run
    rad_results = run_cohort(
        "RADCURE",
        RADCURE_CONFIG,
        mb_sabn=["age", "ecog_ps", "gtvp_cm3", "site", "t_stage", "tx_modality"],
        lasso_sm=["age", "ecog_ps", "gtvp_cm3", "hpv_status", "smoking_py", "tx_modality"],
        lasso_cv=["TxN", "age", "ecog_ps", "gtvp_cm3", "hpv_status", "n_stage",
                  "sex", "site", "smoking_py", "stage_overall", "t_stage", "tx_modality"],
        rsf_topk=["age", "ecog_ps", "gtvp_cm3", "hpv_status", "smoking_py", "tx_modality"],
    )
    han_results = run_cohort(
        "HANCOCK",
        HANCOCK_CONFIG,
        mb_sabn=["blood_hemoglobin", "infiltration_depth_in_mm",
                 "invasion_burden", "pN_grouped", "pT_grouped"],
        lasso_sm=["blood_hemoglobin", "infiltration_depth_in_mm",
                  "invasion_burden", "nlr"],
        lasso_cv=["TxN", "blood_hemoglobin", "grade_grouped",
                  "hpv_association_p16", "infiltration_depth_in_mm",
                  "invasion_burden", "nlr", "pN_grouped", "pT_grouped",
                  "primary_tumor_site", "sex", "smoking_status"],
        rsf_topk=["blood_hemoglobin", "grade_grouped",
                  "infiltration_depth_in_mm", "invasion_burden", "nlr"],
    )

    # Save markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Paired-bootstrap C-index difference + BH-FDR (rebuttal)",
             "",
             "Methodology: B=1000 paired bootstrap resamples of test set. "
             "Two-sided p-value via fraction of bootstrap diffs on either side "
             "of 0. BH-FDR applied within each cohort's family of tests.",
             ""]
    for r in [rad_results, han_results]:
        if r is None:
            continue
        lines += [f"## {r['cohort']}", ""]
        lines += ["**Test-set C-index (Cox PH on selected features):**", ""]
        for k, v in r["c_indices"].items():
            lines.append(f"- {k}: C = {v:.3f}")
        lines += ["",
                  "**Paired-bootstrap differences (B=1000):**",
                  "",
                  "| A | B | C_A | C_B | ΔC | 95% CI | p_raw | q_BH | reject@0.05 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in r["pairs"]:
            lines.append(
                f"| {p['method_A']} | {p['method_B']} | "
                f"{p['C_A']:.3f} | {p['C_B']:.3f} | "
                f"{p['diff_AmB']:+.3f} | "
                f"[{p['diff_CI_lower']:+.3f}, {p['diff_CI_upper']:+.3f}] | "
                f"{p['p_raw']:.3g} | {p['q_BH']:.3g} | "
                f"{'YES' if p['reject_q05'] else 'no'} |"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"\n→ markdown saved to: {OUT_MD}")


if __name__ == "__main__":
    main()
