"""
rebuttal_tcga_more.py — extend the TCGA generalization to two larger non-HNC
cohorts: TCGA-KIRC (kidney) and TCGA-LUAD (lung adeno). Same 5-year-binarization
pipeline as rebuttal_tcga_cancers.py (for Table 2 consistency) plus the full
metric suite (Uno IPCW C, tAUC, IBS). Reports results honestly.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import pandas as pd

from rebuttal_tcga_cancers import run_cohort
from rebuttal_tcga_metrics import metrics, split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "tcga_extra"
OUT = HERE / "notebooks" / "rebuttal_tcga_more_results.json"


def load_tcga_ajcc(study):
    df = pd.read_csv(DATA / f"{study}_tcga_clinical_patient.txt",
                     sep="\t", comment="#", low_memory=False)
    df["age"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["sex"] = df["SEX"].astype(str).str.lower()
    t = df["AJCC_TUMOR_PATHOLOGIC_PT"].astype(str).str.upper()
    df["t_stage"] = t.apply(lambda x: "T1-T2" if ("T1" in x or "T2" in x)
                            else "T3-T4" if ("T3" in x or "T4" in x) else "Tx")
    n = df["AJCC_NODES_PATHOLOGIC_PN"].astype(str).str.upper()
    df["n_stage"] = n.apply(lambda x: "N0" if "N0" in x
                            else "N+" if any(s in x for s in ["N1", "N2", "N3"])
                            else "Nx")
    m = df["AJCC_METASTASIS_PATHOLOGIC_PM"].astype(str).str.upper()
    df["m_stage"] = m.apply(lambda x: "M0" if "M0" in x
                            else "M1" if "M1" in x else "Mx")
    s = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.upper()
    df["stage_overall"] = s.apply(lambda x: "IV" if "IV" in x else "III" if "III" in x
                                  else "II" if "II" in x
                                  else "I" if x.startswith("STAGE I") else "Unknown")
    feat_cols = ["age", "sex", "t_stage", "n_stage", "m_stage", "stage_overall"]
    if "GRADE" in df.columns:
        g = df["GRADE"].astype(str).str.upper()
        df["grade"] = g.apply(lambda x: "G1-G2" if ("G1" in x or "G2" in x)
                              else "G3-G4" if ("G3" in x or "G4" in x) else "Gx")
        feat_cols.append("grade")
    df["time_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    so = df["OS_STATUS"].astype(str).str.strip().str.lower()
    df["event"] = so.str.startswith("1").astype(int)
    df = df[df["time_months"].notna() & (df["time_months"] > 0)].copy()
    df = df.reset_index(drop=True)
    return df, feat_cols


def add_metrics(name, study):
    df, fc = load_tcga_ajcc(study)
    r = run_cohort(name, df, fc)  # 5-year pipeline: MBs, drop%, Cox C
    # full metric suite on the same split
    df2 = df.copy()
    df_tr, df_te = split(df2)
    r["metrics_binary"] = metrics(df_tr, df_te, r["binary_mb"])
    r["metrics_sabn"] = metrics(df_tr, df_te, r["sabn_mb"])
    b, s = r["metrics_binary"], r["metrics_sabn"]
    print(f"  binary-MB metrics: C={b['harrell_c']:.3f} Uno={b['uno_ipcw_c']} tAUC={b['tauc']} IBS={b['ibs']}")
    print(f"  SA-MB metrics:     C={s['harrell_c']:.3f} Uno={s['uno_ipcw_c']} tAUC={s['tauc']} IBS={s['ibs']}")
    return r


def main():
    results = []
    results.append(add_metrics("TCGA-KIRC (kidney)", "kirc"))
    results.append(add_metrics("TCGA-LUAD (lung adeno)", "luad"))
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
