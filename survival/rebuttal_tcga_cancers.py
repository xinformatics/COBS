"""
rebuttal_tcga_cancers.py — SA-BN vs binary BN on TCGA-BRCA and TCGA-COAD.

Direct reply to xj2P / fp3P "only two cohorts" critique. Shows the
binarization-drops-features phenomenon generalizes to breast and colorectal
cancer (the most-cited cancer types in the binary-ML literature beyond HNC).

Pipeline per cohort:
  1. Load cBioPortal clinical_patient data, harmonize features
  2. Encode categoricals (one-hot), keep continuous (median imputation)
  3. Discretize continuous to quantile bins for pgmpy
  4. Binary BN: HillClimbSearch on binary 5-year survival target; extract MB
  5. SA-BN: surv_bn_aware.survival_ensemble_bn on Cox-scored target; extract MB
  6. Compare MBs (overlap, Jaccard, binary-only, survival-only)
  7. Report # patients used by binary vs SA-BN
  8. Fit Cox PH on test set with each MB; report C-index

Outputs to notebooks/rebuttal_tcga_cancers_results.{md,json}.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from surv_common import make_y_surv, concordance_bootstrap
from surv_bn_aware import survival_ensemble_bn, survival_mb_discovery

import pgmpy.estimators as pe
from pgmpy.models import DiscreteBayesianNetwork

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "tcga_extra"
OUT_MD = HERE / "notebooks" / "rebuttal_tcga_cancers_results.md"
OUT_JSON = HERE / "notebooks" / "rebuttal_tcga_cancers_results.json"

SEED = 42


# ── BRCA harmonization ──────────────────────────────────────────────────────

def load_brca():
    df = pd.read_csv(DATA_DIR / "brca_tcga_clinical_patient.txt",
                     sep="\t", comment="#", low_memory=False)
    # Continuous
    df["age"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["pos_ln_count"] = pd.to_numeric(
        df["LYMPH_NODES_EXAMINED_HE_COUNT"], errors="coerce")

    # Categoricals
    df["er_status"] = df["ER_STATUS_BY_IHC"].astype(str).str.lower().map(
        {"positive": "positive", "negative": "negative"}).fillna("unknown")
    df["pr_status"] = df["PR_STATUS_BY_IHC"].astype(str).str.lower().map(
        {"positive": "positive", "negative": "negative"}).fillna("unknown")
    df["her2_status"] = df["IHC_HER2"].astype(str).str.lower().map(
        {"positive": "positive", "negative": "negative",
         "equivocal": "equivocal"}).fillna("unknown")

    # T/N stage grouping
    t = df["AJCC_TUMOR_PATHOLOGIC_PT"].astype(str).str.upper()
    df["t_stage"] = t.apply(
        lambda x: "T1-T2" if ("T1" in x or "T2" in x)
        else "T3-T4" if ("T3" in x or "T4" in x)
        else "Tx")
    n = df["AJCC_NODES_PATHOLOGIC_PN"].astype(str).str.upper()
    df["n_stage"] = n.apply(
        lambda x: "N0" if "N0" in x
        else "N+" if any(s in x for s in ["N1", "N2", "N3"])
        else "Nx")

    # Overall stage
    s = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.upper()
    df["stage_overall"] = s.apply(
        lambda x: "IV" if "IV" in x else "III" if "III" in x
        else "II" if "II" in x else "I" if x.startswith("STAGE I")
        else "Unknown")

    # Histology
    h = df["HISTOLOGICAL_DIAGNOSIS"].astype(str).str.lower()
    df["histology"] = h.apply(
        lambda x: "ductal" if "ductal" in x
        else "lobular" if "lobular" in x
        else "other")

    # Menopause
    m = df["MENOPAUSE_STATUS"].astype(str).str.lower()
    df["menopause"] = m.apply(
        lambda x: "post" if "post" in x else "pre" if "pre" in x else "unknown")

    # Treatment
    df["adj_radiation"] = df["RADIATION_TREATMENT_ADJUVANT"].astype(str).str.lower().map(
        {"yes": "yes", "no": "no"}).fillna("unknown")
    df["adj_pharm"] = df["PHARMACEUTICAL_TX_ADJUVANT"].astype(str).str.lower().map(
        {"yes": "yes", "no": "no"}).fillna("unknown")

    # Survival
    df["time_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    s = df["OS_STATUS"].astype(str).str.strip().str.lower()
    df["event"] = s.str.startswith("1").astype(int)
    df = df[df["time_months"].notna() & (df["time_months"] > 0)].copy()

    feat_cols = ["age", "er_status", "pr_status", "her2_status",
                 "t_stage", "n_stage", "stage_overall", "histology",
                 "menopause", "pos_ln_count", "adj_radiation", "adj_pharm"]
    return df, feat_cols


# ── COAD harmonization ─────────────────────────────────────────────────────

def load_coad():
    df = pd.read_csv(DATA_DIR / "coadread_tcga_clinical_patient.txt",
                     sep="\t", comment="#", low_memory=False)
    df["age"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["pos_ln_count"] = pd.to_numeric(
        df["LYMPH_NODES_EXAMINED_HE_COUNT"], errors="coerce")

    df["sex"] = df["SEX"].astype(str).str.lower()

    t = df["AJCC_TUMOR_PATHOLOGIC_PT"].astype(str).str.upper()
    df["t_stage"] = t.apply(
        lambda x: "T1-T2" if ("T1" in x or "T2" in x)
        else "T3-T4" if ("T3" in x or "T4" in x)
        else "Tx")
    n = df["AJCC_NODES_PATHOLOGIC_PN"].astype(str).str.upper()
    df["n_stage"] = n.apply(
        lambda x: "N0" if "N0" in x
        else "N+" if any(s in x for s in ["N1", "N2", "N3"])
        else "Nx")
    m = df["AJCC_METASTASIS_PATHOLOGIC_PM"].astype(str).str.upper()
    df["m_stage"] = m.apply(
        lambda x: "M0" if "M0" in x
        else "M1" if "M1" in x else "Mx")

    s = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.upper()
    df["stage_overall"] = s.apply(
        lambda x: "IV" if "IV" in x else "III" if "III" in x
        else "II" if "II" in x else "I" if x.startswith("STAGE I")
        else "Unknown")

    h = df["HISTOLOGICAL_DIAGNOSIS"].astype(str).str.lower()
    df["histology"] = h.apply(
        lambda x: "mucinous" if "mucinous" in x
        else "adenocarcinoma" if "adenocarcinoma" in x else "other")

    df["lvi"] = df["LYMPHOVASCULAR_INVASION_INDICATOR"].astype(str).str.lower().map(
        {"yes": "yes", "no": "no"}).fillna("unknown")
    df["braf"] = df["BRAF_GENE_ANALYSIS_RESULT"].astype(str).str.lower().map(
        {"abnormal": "mutant", "normal": "wildtype"}).fillna("unknown")

    df["adj_radiation"] = df["RADIATION_TREATMENT_ADJUVANT"].astype(str).str.lower().map(
        {"yes": "yes", "no": "no"}).fillna("unknown")
    df["adj_pharm"] = df["PHARMACEUTICAL_TX_ADJUVANT"].astype(str).str.lower().map(
        {"yes": "yes", "no": "no"}).fillna("unknown")

    df["time_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    so = df["OS_STATUS"].astype(str).str.strip().str.lower()
    df["event"] = so.str.startswith("1").astype(int)
    df = df[df["time_months"].notna() & (df["time_months"] > 0)].copy()

    feat_cols = ["age", "sex", "t_stage", "n_stage", "m_stage",
                 "stage_overall", "histology", "lvi", "braf",
                 "pos_ln_count", "adj_radiation", "adj_pharm"]
    return df, feat_cols


# ── Discretization for pgmpy ──────────────────────────────────────────────

def discretize(df, feat_cols, train_mask, n_bins=3):
    """Bin continuous features into quantiles based on training cohort."""
    out = df.copy()
    for col in feat_cols:
        v = pd.to_numeric(out[col], errors="coerce")
        if v.notna().sum() > 0 and v.dropna().nunique() > 10:
            train_vals = v[train_mask].dropna()
            try:
                edges = np.unique(np.quantile(train_vals, np.linspace(0, 1, n_bins + 1)))
                if len(edges) < 3:
                    out[col] = v.fillna(v.median()).astype(int).astype(str)
                else:
                    edges[0] -= 1e-6
                    edges[-1] += 1e-6
                    binned = pd.cut(v, edges, labels=False, include_lowest=True)
                    out[col] = binned.fillna(binned.median()).astype(int).astype(str)
            except Exception:
                out[col] = v.fillna(v.median()).astype(int).astype(str)
        else:
            out[col] = out[col].astype(str).replace({"nan": "missing"})
    return out


# ── Binary BN on 5-year survival ──────────────────────────────────────────

def binary_bn_mb(disc_df, feat_cols, train_mask, target_name="surv5y"):
    """Run HillClimbSearch with BDeu on binary 5yr survival target.

    Returns Markov blanket (set of feature names).
    """
    train_df = disc_df[train_mask].copy()
    train_df = train_df[~train_df[target_name].isin(["dropped"])]
    train_df[target_name] = train_df[target_name].astype(str)

    cols = feat_cols + [target_name]
    sub = train_df[cols].copy()
    # Ensure all string for pgmpy
    for c in cols:
        sub[c] = sub[c].astype(str)
    try:
        from pgmpy.estimators import HillClimbSearch, BDeu
        hc = HillClimbSearch(sub)
        bdeu = BDeu(sub, equivalent_sample_size=10)
        best = hc.estimate(scoring_method=bdeu, max_indegree=4,
                           max_iter=100, show_progress=False)
    except Exception as e:
        print(f"  Binary BN error: {e}")
        return set(), None

    # Markov blanket = parents + children + spouses
    parents = set(best.predecessors(target_name)) if target_name in best.nodes() else set()
    children = set(best.successors(target_name)) if target_name in best.nodes() else set()
    spouses = set()
    for c in children:
        spouses |= set(best.predecessors(c))
    mb = (parents | children | spouses) - {target_name}
    return mb, best


def make_binary_5yr(df, time_col="time_months", event_col="event",
                    horizon_months=60):
    """Build binary 5-year survival label, marking unevaluable patients as 'dropped'.

    Conventional clinical-ML binarization: patients alive but censored before
    5 years are excluded.
    """
    label = pd.Series("dropped", index=df.index)
    t = df[time_col]
    e = df[event_col]
    label[(e == 0) & (t >= horizon_months)] = "alive"
    label[(e == 1) & (t < horizon_months)] = "dead"
    label[(e == 1) & (t >= horizon_months)] = "alive"  # alive at 5yr even if later died
    return label


# ── Cox PH C-index evaluation ──────────────────────────────────────────────

def encode_for_cox(df, feat_cols):
    X = pd.DataFrame(index=df.index)
    for col in feat_cols:
        src = df[col]
        as_num = pd.to_numeric(src, errors="coerce")
        if as_num.notna().sum() > 0 and as_num.dropna().nunique() > 10:
            X[col] = as_num.fillna(as_num.median())
        else:
            cat = src.astype(str).replace({"nan": "missing", "": "missing"})
            dummies = pd.get_dummies(cat, prefix=col, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    return X.fillna(0).astype(float)


def cox_test_cindex(df_train, df_test, feats, time_col="time_months",
                    event_col="event"):
    if not feats:
        return None
    feats = [f for f in feats if f in df_train.columns]
    if not feats:
        return None
    Xtr = encode_for_cox(df_train, feats)
    Xte_raw = encode_for_cox(df_test, feats)
    Xte = Xte_raw.reindex(columns=Xtr.columns, fill_value=0.0)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr.values)
    Xte_s = sc.transform(Xte.values)
    y_train = np.array(
        [(bool(e), float(t)) for e, t in zip(df_train[event_col],
                                              df_train[time_col])],
        dtype=[("event", bool), ("time", float)])
    y_test = np.array(
        [(bool(e), float(t)) for e, t in zip(df_test[event_col],
                                              df_test[time_col])],
        dtype=[("event", bool), ("time", float)])
    cox = CoxPHSurvivalAnalysis(alpha=1e-2)
    try:
        cox.fit(Xtr_s, y_train)
    except Exception as e:
        return {"cindex": None, "error": str(e), "n_features": len(feats)}
    risk = cox.predict(Xte_s)
    boot = concordance_bootstrap(y_test, risk, n_boot=500)
    boot["n_features"] = len(feats)
    return boot


# ── Per-cohort driver ──────────────────────────────────────────────────────

def run_cohort(name, df, feat_cols):
    print(f"\n{'='*70}\n  COHORT: {name}\n{'='*70}")
    print(f"  Loaded: {len(df)} patients, {len(feat_cols)} candidate features")
    print(f"  Events: {df['event'].sum()} ({df['event'].mean():.1%}), "
          f"median FU: {df['time_months'].median():.1f} months")

    # Train/test split
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(0.7 * len(df))
    train_idx = df.index[idx[:n_train]]
    test_idx = df.index[idx[n_train:]]
    train_mask = df.index.isin(train_idx)
    print(f"  Train n={train_mask.sum()}, Test n={(~train_mask).sum()}")

    # Binary 5-year label
    df["surv5y"] = make_binary_5yr(df)
    print(f"  5-yr labels: alive={int((df['surv5y']=='alive').sum())}, "
          f"dead={int((df['surv5y']=='dead').sum())}, "
          f"dropped={int((df['surv5y']=='dropped').sum())} "
          f"({(df['surv5y']=='dropped').mean():.1%} dropped by binarization)")

    # Discretize for pgmpy
    disc_df = discretize(df, feat_cols, train_mask, n_bins=3)
    disc_df["surv5y"] = df["surv5y"].values

    # Binary BN MB
    print("\n  [1/2] Binary BN structure learning...")
    t0 = time.time()
    bin_mb, _ = binary_bn_mb(disc_df, feat_cols, train_mask,
                             target_name="surv5y")
    t_bin = time.time() - t0
    n_binary_used = int(train_mask.sum() -
                        (disc_df.loc[train_mask, "surv5y"] == "dropped").sum())
    print(f"    binary MB ({len(bin_mb)}): {sorted(bin_mb)}")
    print(f"    binary fit time: {t_bin:.1f}s")
    print(f"    patients used: {n_binary_used} / {train_mask.sum()} train")

    # SA-BN MB
    print("\n  [2/2] SA-BN structure learning (Cox-scored)...")
    t0 = time.time()
    train_disc = disc_df[train_mask].copy()
    train_time = df.loc[train_mask, "time_months"].values * 30.4375
    train_event = df.loc[train_mask, "event"].values
    try:
        G_sa, _graphs = survival_ensemble_bn(
            train_disc, feat_cols, train_time, train_event,
            target_name="survival", verbose=False,
        )
        smb_info = survival_mb_discovery(
            G_sa, train_disc, feat_cols, train_time, train_event,
            target_name="survival", verbose=False,
        )
        sa_mb = set(smb_info["consensus"])
    except Exception as e:
        print(f"    SA-BN error: {e}")
        sa_mb = set()
    t_sa = time.time() - t0
    print(f"    SA-BN MB ({len(sa_mb)}): {sorted(sa_mb)}")
    print(f"    SA-BN fit time: {t_sa:.1f}s")
    n_survival_used = train_mask.sum()
    print(f"    patients used: {n_survival_used} / {n_survival_used} "
          f"(no exclusion for censoring)")

    # Compare MBs
    overlap = bin_mb & sa_mb
    only_bin = bin_mb - sa_mb
    only_sa = sa_mb - bin_mb
    jaccard = (len(overlap) / max(len(bin_mb | sa_mb), 1)) if (bin_mb | sa_mb) else 0
    print(f"\n  MB comparison:")
    print(f"    overlap: {sorted(overlap)} ({len(overlap)})")
    print(f"    binary-only: {sorted(only_bin)} ({len(only_bin)})")
    print(f"    SA-BN-only: {sorted(only_sa)} ({len(only_sa)})")
    print(f"    Jaccard: {jaccard:.3f}")

    # Cox C-index on test set
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()
    print("\n  Test Cox PH C-index:")
    cidx_bin = cox_test_cindex(train_df, test_df, sorted(bin_mb))
    cidx_sa = cox_test_cindex(train_df, test_df, sorted(sa_mb))
    cidx_all = cox_test_cindex(train_df, test_df, feat_cols)
    for label, c in [("binary MB", cidx_bin), ("SA-BN MB", cidx_sa),
                     ("All features", cidx_all)]:
        if c is None or c.get("cindex") is None:
            print(f"    {label}: not fittable")
            continue
        ci_l = c.get("ci_lower", float("nan"))
        ci_u = c.get("ci_upper", float("nan"))
        print(f"    {label}: C={c['cindex']:.3f} "
              f"[{ci_l:.3f}, {ci_u:.3f}] n_feat={c['n_features']}")

    return {
        "cohort": name,
        "n_patients": int(len(df)),
        "n_events": int(df["event"].sum()),
        "median_fu_months": float(df["time_months"].median()),
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "n_dropped_by_binary": int((df["surv5y"] == "dropped").sum()),
        "n_binary_train_used": n_binary_used,
        "n_sabn_train_used": int(n_survival_used),
        "binary_mb": sorted(bin_mb),
        "sabn_mb": sorted(sa_mb),
        "overlap_mb": sorted(overlap),
        "binary_only_mb": sorted(only_bin),
        "sabn_only_mb": sorted(only_sa),
        "jaccard": float(jaccard),
        "cindex_binary_mb": cidx_bin,
        "cindex_sabn_mb": cidx_sa,
        "cindex_all": cidx_all,
        "fit_time_binary_s": float(t_bin),
        "fit_time_sabn_s": float(t_sa),
    }


def main():
    results = []
    print("\n>>> Loading + processing TCGA-BRCA")
    df_brca, fc_brca = load_brca()
    results.append(run_cohort("TCGA-BRCA (breast)", df_brca, fc_brca))

    print("\n>>> Loading + processing TCGA-COAD")
    df_coad, fc_coad = load_coad()
    results.append(run_cohort("TCGA-COAD (colorectal)", df_coad, fc_coad))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))

    lines = [
        "# SA-BN vs Binary BN on TCGA-BRCA and TCGA-COAD (rebuttal)",
        "",
        "Direct reply to xj2P / fp3P 'only two cohorts' critique. The "
        "binarization-vs-time-to-event question is reframed across breast and "
        "colorectal cancer, where binary 5-year survival classification is the "
        "dominant ML approach in the published literature.",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['cohort']}", "",
            f"- n = {r['n_patients']} ({r['n_events']} events, "
            f"{r['n_events']/r['n_patients']:.1%}); median follow-up "
            f"{r['median_fu_months']:.1f} months",
            f"- Train n = {r['n_train']}, test n = {r['n_test']}",
            f"- **Patients dropped by binary 5-year labeling**: "
            f"{r['n_dropped_by_binary']} "
            f"({100*r['n_dropped_by_binary']/r['n_patients']:.1f}%)",
            f"- Binary BN train used: {r['n_binary_train_used']} patients",
            f"- SA-BN train used: {r['n_sabn_train_used']} patients (no censoring exclusion)",
            "",
            f"**Binary MB ({len(r['binary_mb'])})**: {', '.join(r['binary_mb']) or '(none)'}",
            "",
            f"**SA-BN MB ({len(r['sabn_mb'])})**: {', '.join(r['sabn_mb']) or '(none)'}",
            "",
            f"- Overlap ({len(r['overlap_mb'])}): {', '.join(r['overlap_mb']) or '-'}",
            f"- Binary-only ({len(r['binary_only_mb'])}): {', '.join(r['binary_only_mb']) or '-'}",
            f"- **SA-BN-only ({len(r['sabn_only_mb'])}): {', '.join(r['sabn_only_mb']) or '-'}**",
            f"- Jaccard: {r['jaccard']:.3f}",
            "",
            "**Test-set Cox PH C-index:**",
            "",
        ]
        for k, v in [("Binary MB", r["cindex_binary_mb"]),
                     ("SA-BN MB", r["cindex_sabn_mb"]),
                     ("All features", r["cindex_all"])]:
            if v is None or v.get("cindex") is None:
                lines.append(f"- {k}: not fittable")
            else:
                lines.append(
                    f"- {k}: **{v['cindex']:.3f}** "
                    f"[{v.get('ci_lower', float('nan')):.3f}, "
                    f"{v.get('ci_upper', float('nan')):.3f}] "
                    f"(n_features={v['n_features']})"
                )
        lines += ["",
                  f"Fit time: binary BN {r['fit_time_binary_s']:.1f}s, "
                  f"SA-BN {r['fit_time_sabn_s']:.1f}s",
                  ""]
    OUT_MD.write_text("\n".join(lines))
    print(f"\n→ JSON: {OUT_JSON}")
    print(f"→ MD:   {OUT_MD}")


if __name__ == "__main__":
    main()
