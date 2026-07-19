"""
rebuttal_tcga_2yr.py — TCGA generalization at the 2-YEAR endpoint (matching the
paper's SVy2), for BRCA, COAD, KIRC. SA-BN vs binary BN + full metric suite.
Reuses the generic run() driver (binary BN MB, SA-BN MB, metrics) at
horizon_months=24.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

from rebuttal_gbsg2 import run
from rebuttal_tcga_cancers import load_brca, load_coad
from rebuttal_tcga_more import load_tcga_ajcc

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "rebuttal_tcga_2yr_results.json"


def main():
    df_b, fc_b = load_brca()
    df_c, fc_c = load_coad()
    df_k, fc_k = load_tcga_ajcc("kirc")
    cohorts = [
        ("TCGA-BRCA (breast)", df_b.reset_index(drop=True), fc_b),
        ("TCGA-COAD (colorectal)", df_c.reset_index(drop=True), fc_c),
        ("TCGA-KIRC (kidney)", df_k.reset_index(drop=True), fc_k),
    ]
    results = []
    for name, df, fc in cohorts:
        r = run(name, df, fc, horizon_months=24)
        results.append(r)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
