# COBS — The Cost of Binarizing Survival

Code & data for **"The Cost of Binarizing Survival Outcomes in Clinical Prognostic
Modeling"** (Yadav, Routman, Foong), *Machine Learning for Healthcare (MLHC) 2026*.

The paper introduces the **Survival-Aware Bayesian Network (SA-BN)**: it replaces the
binary scoring in Bayesian-network feature selection with the Cox partial log-likelihood,
and quantifies how much prognostic signal binarizing a survival endpoint discards.

This repository reproduces the **analyses added in the camera-ready** in response to the
reviews (TCGA cohorts, proportional-hazards testing, multiple-comparison correction,
LASSO-Cox / survival-native comparisons, quantitative selection tables, and runtime).
The core SA-BN method (`surv_bn_aware.py`) is **unchanged** from the submission —
these scripts apply that same code to new cohorts and add evaluation.

Every number below was verified to reproduce from a clean run of this bundle (see
**Verification status**).

---

## Layout

```
common.py                     # dataset configs + feature engineering (repo-root module)
shah_replication.csv          # RADCURE radiation cohort (n=2994)
hancock_rebuilt.csv           # HANCOCK surgical cohort, MICE-imputed (n=763)
tcga_extra/                   # TCGA clinical files (public; cBioPortal)
survival/
  surv_common.py, surv_bn_aware.py   # SA-BN engine (unchanged from original release)
  rebuttal_verify.py          # Tasks 1,2,3,6,7 -> rebuttal_results.json  (RUN FIRST)
  rebuttal_assemble.py        # assembles rebuttal_results.md from the JSONs
  rebuttal_task10.py          # LASSO 4-model x 3-metric benchmark
  rebuttal_task45.py          # parity C-indices (task4) + runtime (task5)
  rebuttal_paired_bootstrap.py# paired-bootstrap C-index diff + BH-FDR
  rebuttal_tcga_2yr.py        # TCGA breast/colorectal/kidney at 2-year endpoint
  rebuttal_tcga_cancers.py, rebuttal_tcga_metrics.py, rebuttal_tcga_more.py,
  rebuttal_gbsg2.py, rebuttal_whas500.py   # TCGA/robustness helpers (see note 3)
  notebooks/
    stability_results.pkl     # 50-bootstrap MB selection frequencies (precomputed)
    rebuttal_*_results.{json,md}   # reference outputs = the exact camera-ready values
```

Paths are resolved relative to this layout (scripts do `sys.path.insert(parent.parent)`
and read `common.py`/data from the bundle root, `tcga_extra/` from the root, and write to
`survival/notebooks/`). Run scripts from `survival/`; do not move files between levels.

## Dependencies

```bash
pip install numpy pandas scikit-learn scikit-survival lifelines statsmodels pgmpy networkx
```

## How to reproduce (run order matters)

`rebuttal_verify.py` must run first: it performs SA-BN structure learning once and caches
the loaded cohorts to `/tmp/surv_cache.pkl`, which `rebuttal_task10.py` and
`rebuttal_task45.py` then reuse.

```bash
cd survival
python rebuttal_verify.py          # ~3-4 min (structure learning); builds /tmp/surv_cache.pkl
python rebuttal_assemble.py        # writes notebooks/rebuttal_results.md
python rebuttal_task10.py          # LASSO/RSF/GBS/Cox benchmark
python rebuttal_paired_bootstrap.py
python rebuttal_task45.py          # parity + runtime (task5 re-times structure learning)
python rebuttal_tcga_2yr.py        # TCGA breast/colorectal/kidney
```

Or `bash reproduce.sh` from the bundle root.

---

## Map: camera-ready element -> script -> reference output -> headline numbers

| Camera-ready | Script | Reference output | Headline values |
|---|---|---|---|
| §4 Missing data (MICE) | `rebuttal_verify.py` (Task 7) | `rebuttal_results.md` | 143 patients recovered (763 vs listwise 620) |
| §5.6 + Table (TCGA) | `rebuttal_tcga_2yr.py` | `rebuttal_tcga_2yr_results.json` | BRCA .711/.652, COAD .702/.649, KIRC .767/.741 (SA/binary C) |
| App G PH tests (Schoenfeld+FDR) | `rebuttal_verify.py` (Tasks 1,2) | `rebuttal_results.json/.md` | only radiation overall stage violates (q=7.2e-5); surgical none (Hb q=0.169) |
| App H membership (freq/HR/q) | `rebuttal_verify.py` (Tasks 3,6) | `rebuttal_results.md`, `rebuttal_membership_*.csv` | radiation 8/9 survive q<0.05 (T stage q=0.42); surgical 5/5 |
| App I LASSO 4-model x 3-metric | `rebuttal_task10.py` | `rebuttal_task10_results.json` | radiation SA-MB RSF .804/.770; surgical RSF .721/.678 |
| App J parity (paired-bootstrap q) | `rebuttal_task45.py` + `rebuttal_paired_bootstrap.py` | `rebuttal_task45_results.json`, `rebuttal_paired_bootstrap.md` | SA vs LASSO/RSF-VI q=0.64/0.57; LASSO-CV q=0.04 |
| App K runtime | `rebuttal_task45.py` (task5) | `rebuttal_task45_results.json` | 1 pass ~137s (rad)/~13s (surg); 50-boot ~114/~11 min |

## Verification status (clean re-run of this bundle)

- **App G (PH):** exact — all 9 radiation + 5 surgical Schoenfeld q-values match to 4 dp.
- **App H (membership) / stratified Cox / MICE:** exact.
- **App I (LASSO):** exact — 0 mismatches across 16 cells x 3 metrics.
- **App J (parity q):** exact — RADCURE q=0.635/0.04, HANCOCK q=0.566.
- **§5.6 TCGA:** SA-MB exact on all three cohorts; COAD/KIRC binary exact. The **TCGA-BRCA
  binary-MB** baseline shows ~±0.01 run-to-run variance (0.652–0.662) from stochasticity in
  the *binary* BN discovery; the survival-aware result and the SA>binary conclusion are stable.
- **App K runtime:** wall-clock is machine-dependent; the recorded values are single-thread
  references, not exact reproductions.

## Notes

1. **Method code is unchanged.** `surv_bn_aware.py` is identical to the original code release;
   the additional results did not modify the algorithm.
2. **`rebuttal_ph_fdr.py` is intentionally excluded.** It is a superseded 3-feature PH draft;
   the camera-ready App G uses the 5-feature surgical family produced by `rebuttal_verify.py`.
3. **Robustness cohorts are included for transparency.** `rebuttal_gbsg2.py` (breast),
   `rebuttal_whas500.py` (non-cancer MI) and the LUAD path in `rebuttal_tcga_more.py` are
   cohorts we ran but did **not** report in the paper, consistent with the boundary conditions
   in §5.7 ("Scope and Limitations"): SA-BN and binary BN converge when little is censored
   (GBSG2, 9% dropped) or the MB is already minimal (TCGA-LUAD), and the effect is specific to
   oncology cohorts with substantial censoring. They are shipped so reviewers can run them.

## Data provenance

- **RADCURE** (radiation): public — https://www.cancerimagingarchive.net/collection/radcure/
- **HANCOCK** (surgical): public — https://hancock.research.fau.eu/
- **TCGA** (BRCA/COAD/KIRC): public clinical files via cBioPortal (included in `tcga_extra/`).
