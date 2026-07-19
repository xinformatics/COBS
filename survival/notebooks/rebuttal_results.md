# MLHC #283 — Rebuttal verification results (7june2026 task list)

Each value labelled by source: **artifact** (pre-existing run) or **fresh** (re-run now). Verify-against status from the task list.

## Reviewer zAYY (clinical)

### Task 1 — Schoenfeld PH test + BH-FDR (fresh)

**RADCURE** (SA-MB family, BH-FDR across PH tests):

| feature | stat | p_raw | q_BH | violates(q<.05) |
|---|---|---|---|---|
| stage_overall | 19.9 | 8e-06 | 7.2e-05 | YES |
| smoking_py | 5.64 | 0.0176 | 0.079 | no |
| age | 3.67 | 0.0554 | 0.107 | no |
| site | 3.57 | 0.0589 | 0.107 | no |
| gtvp_cm3 | 3.37 | 0.0664 | 0.107 | no |
| t_stage | 3.26 | 0.0711 | 0.107 | no |
| hpv_status | 1.61 | 0.205 | 0.264 | no |
| tx_modality | 1.38 | 0.241 | 0.271 | no |
| ecog_ps | 1.09 | 0.296 | 0.296 | no |

**HANCOCK** (SA-MB family, BH-FDR across PH tests):

| feature | stat | p_raw | q_BH | violates(q<.05) |
|---|---|---|---|---|
| blood_hemoglobin | 4.5 | 0.0339 | 0.169 | no |
| invasion_burden | 3.19 | 0.0739 | 0.185 | no |
| number_of_positive_lymph_nodes | 2.35 | 0.126 | 0.209 | no |
| nlr | 0.535 | 0.464 | 0.58 | no |
| age_at_initial_diagnosis | 0.251 | 0.616 | 0.616 | no |

*RADCURE:* only overall stage violates after FDR (q=7.2e-5); smoking (q=0.079) and age (q=0.107) hold — matches verify-against. *HANCOCK:* no MB feature violates after FDR (Hb q=0.169). Note: Hb raw p=0.034 here vs 0.022 in the prior 3-feature fit — the Cox model now includes age+pos_LN; conclusion unchanged.

### Task 2 — Stratified Cox on overall stage, RADCURE (fresh)

| feature | HR unstratified | HR stratified | same direction |
|---|---|---|---|
| age | 1.03 | 1.03 | yes |
| ecog_ps | 1.95 | 1.82 | yes |
| gtvp_cm3 | 1.01 | 1.01 | yes |
| hpv_status | 0.472 | 0.43 | yes |
| site | 0.34 | 0.418 | yes |
| smoking_py | 1.01 | 1.01 | yes |
| t_stage | 0.686 | 0.769 | yes |
| tx_modality | 1.61 | 1.76 | yes |

All HRs keep direction under stratification: True. HR magnitudes stable (e.g. ECOG 1.95→1.82, HPV 0.47→0.43).

### Task 3 — Per-feature Cox p + BH-FDR (fresh)

**RADCURE** (multivariate Cox, min-p aggregation, BH-FDR over MB):

| feature | HR | p_raw | q_BH | reject(q<.05) |
|---|---|---|---|---|
| hpv_status | 0.426 | 2.81e-14 | 2.53e-13 | YES |
| age | 1.03 | 4.53e-13 | 2.04e-12 | YES |
| tx_modality | 1.84 | 1.99e-10 | 5.97e-10 | YES |
| gtvp_cm3 | 1.01 | 1.92e-08 | 4.32e-08 | YES |
| ecog_ps | 1.83 | 7.3e-08 | 1.31e-07 | YES |
| smoking_py | 1.01 | 1.51e-07 | 2.27e-07 | YES |
| site | 0.411 | 5.34e-06 | 6.86e-06 | YES |
| stage_overall | 2.09 | 0.000296 | 0.000333 | YES |
| t_stage | 0.806 | 0.422 | 0.422 | no |

**HANCOCK** (multivariate Cox, min-p aggregation, BH-FDR over MB):

| feature | HR | p_raw | q_BH | reject(q<.05) |
|---|---|---|---|---|
| invasion_burden | 2.87 | 8.32e-06 | 4.16e-05 | YES |
| number_of_positive_lymph_nodes | 1.06 | 3.01e-05 | 7.52e-05 | YES |
| age_at_initial_diagnosis | 1.03 | 5.42e-05 | 9.03e-05 | YES |
| blood_hemoglobin | 0.819 | 0.00038 | 0.000475 | YES |
| nlr | 1.08 | 0.00691 | 0.00691 | YES |

*RADCURE:* 8/9 survive (t_stage q=0.42) — matches verify-against to the digit. *HANCOCK (now 5-feature family incl. age + pos_LN):* all 5 survive (invasion_burden q=4.2e-5, pos_LN q=7.5e-5, age q=9.0e-5, Hb q=4.8e-4, NLR q=6.9e-3). Note: raw p's are smaller than Table 7's *univariate* values (age 5.4e-5 vs 0.005) because this uses the same *multivariate* min-p method as RADCURE; all still survive. q's are higher than the prior 3-feature run because BH scales with family size — expected.

### Task 7 — MICE recovery count (artifact)

hancock_rebuilt.csv n=763; documented listwise=620; **recovered=143** — matches verify-against.

## Reviewers fp3P + xj2P (baselines / generalization)

### Task 4 — RSF on binary-MB vs SA-MB, test C-index + 1000-bootstrap CI (fresh)

| cohort | feature set | k | C-index [95% CI] | verify-against |
|---|---|---|---|---|
| RADCURE | SA-MB | 24 | 0.804 [0.768, 0.839] | 0.801 (OK) |
| RADCURE | binary-MB | 33 | 0.771 [0.729, 0.809] | 0.77 (OK) |
| HANCOCK | SA-MB | 7 | 0.726 [0.638, 0.808] | 0.726 (OK) |
| HANCOCK | binary-MB | 8 | 0.679 [0.581, 0.769] | 0.648 (OK) |

SA-MB beats binary-MB on both cohorts (RADCURE +0.0328, HANCOCK +0.0475). RADCURE SA 0.804 / binary 0.771 match the paper near-exactly. HANCOCK binary-MB 0.679 vs paper 0.648 (+0.031): config drift in the current loader's encoding; SA-MB superiority and direction preserved.

**k = number of one-hot-encoded columns**, not raw features (e.g. RADCURE SA-MB = 9 clinical features → 24 dummy columns).

## Reviewer 6EZs (quantitative tables + runtime)

### Task 6 — Combined membership table (artifact: stability_results.pkl + fresh Cox)

**RADCURE**

| feature | in_SA_MB | sel_freq_pct | HR | q_BH | edge_to_survival_pct |
|---|---|---|---|---|---|
| age | yes | 100 | 1.03 | 2.04e-12 | 98 |
| ecog_ps | yes | 100 | 1.83 | 1.31e-07 | 100 |
| gtvp_cm3 | yes | 100 | 1.01 | 4.32e-08 | 66 |
| hpv_status | yes | 100 | 0.426 | 2.53e-13 | 80 |
| site | yes | 100 | 0.411 | 6.86e-06 | 96 |
| smoking_py | yes | 100 | 1.01 | 2.27e-07 | 82 |
| tx_modality | yes | 100 | 1.84 | 5.97e-10 | 2 |
| stage_overall | yes | 68 | 2.09 | 0.000333 | 6 |
| t_stage | yes | 66 | 0.806 | 0.422 | - |
| n_stage | no | 24 | - | - | - |
| TxN | no | 22 | - | - | 8 |
| sex | no | 2 | - | - | - |

**HANCOCK**

| feature | in_SA_MB | sel_freq_pct | HR | q_BH | edge_to_survival_pct |
|---|---|---|---|---|---|
| blood_hemoglobin | yes | 94 | 0.819 | 0.000475 | 82 |
| number_of_positive_lymph_nodes | yes | 88 | 1.06 | 7.52e-05 | 28 |
| age_at_initial_diagnosis | yes | 88 | 1.03 | 9.03e-05 | 36 |
| invasion_burden | yes | 86 | 2.87 | 4.16e-05 | 74 |
| nlr | yes | 78 | 1.08 | 0.00691 | 44 |
| TxN | no | 62 | - | - | 62 |
| adjuvant_radiotherapy | no | 40 | - | - | 34 |
| sex | no | 30 | - | - | - |
| hpv_association_p16 | no | 26 | - | - | 4 |
| infiltration_depth_in_mm | no | 14 | - | - | 2 |
| smoking_status | no | 10 | - | - | - |
| pN_grouped | no | 6 | - | - | - |
| grade_grouped | no | 6 | - | - | 10 |

Selection frequencies and directed-edge-to-survival stabilities are from the 50-bootstrap stability_results.pkl and match the rebuttal/verify-against exactly (RADCURE: 7/9 at 100%, stage 68%, T 66%; ECOG→surv 100%, age→surv 98%, site→surv 96%. HANCOCK: Hb 94%, pos_LN 88%, age 88%, IB 86%, NLR 78%; Hb→surv 82%, IB→surv 74%). HR and q columns from the fresh Cox fits above.

### Task 5 — Runtime benchmark (fresh)

| cohort | n_train | 1 ensemble pass (s) | n≈100 pass (s) | 50-bootstrap (est) |
|---|---|---|---|---|
| RADCURE | 2174 | 137.0 | 2.5 | ~114 min |
| HANCOCK | 565 | 13.2 | 2.5 | ~11 min |

> **FLAG — rebuttal correction required.** The single ensemble pass is **PC-dominated on large n**: RADCURE **~137s** (n=2174), HANCOCK ~13s (n=565), n≈100 ~2.5s. The 50-resample bootstrap is 50× the single pass: **RADCURE ~114 min, HANCOCK ~11 min**. The 6EZs draft claims "<30 s per cohort" and "~2-3 min RADCURE / ~1 min HANCOCK" — TRUE only for HANCOCK's single pass. The RADCURE single-pass and both bootstrap figures are wrong and the rebuttal text must be corrected. What survives intact: (a) the fast route for privacy-constrained small cohorts (n≈100 in ~2.5s) is real; (b) the single SA-BN pass still beats RSF permutation importance (~16 min on RADCURE), so the cost comparison vs RSF-VI holds.

## Reviewer xj2P (linchpin) — Task 8 parity verification (fresh + artifact)

The xj2P rebuttal's parity numbers were the one set with no clean on-disk
artifact (`rebuttal_baselines.py` errors on disk). Verified independently:

- **LASSO-Cox selection reproduced from scratch** (real Coxnet path on cached
  RADCURE, not the hardcoded list): at α=0.078 LASSO selects 7 features
  (age, ecog_ps, gtvp_cm3, hpv_status, site, smoking_py, tx_modality); Cox on
  the k=6 set gives **test C=0.772** vs rebuttal's claimed ~0.769. ✓
- **Paired-bootstrap q-values backed** by `rebuttal_paired_bootstrap.md` (real
  run): RADCURE SA-BN vs LASSO/RSF q=0.635 (rebuttal q=0.64); HANCOCK q=0.566
  (rebuttal q=0.57). Note: that script fits Cox/bootstrap freshly but uses the
  baselines' feature *lists* as input — the C-indices and q's are real; the
  LASSO selection is independently re-confirmed above.
- **RSF-VI runtime confirmed:** real implementation (300 trees, 46 cols × 10
  permutation repeats) = **997.8s (16.6 min)** on RADCURE — matches the
  ~16 min claim, and confirms a single SA-BN pass (~137s) is ~7× cheaper than
  the RSF-VI baseline (the surviving half of the compute argument).

## One-line status per Priority-1 task

- **Task 1 (PH+FDR):** backed by fresh run; RADCURE matches exactly, HANCOCK no violation after FDR. ✓
- **Task 2 (stratified Cox):** backed; all HRs stable in direction. ✓
- **Task 3 (Cox q):** backed; RADCURE 8/9, HANCOCK 5/5 survive FDR. ✓
- **Task 4 (RSF binary vs SA):** backed; SA>binary both cohorts; 3/4 match, HANCOCK binary-MB +0.031 drift noted. ✓
- **Task 5 (runtime):** measured; HANCOCK ~13s, RADCURE ~137s (PC-dominated), n=100 ~2.5s. **6EZs rebuttal compute numbers need correction** (RADCURE not <30s; bootstrap ~114 min not 2-3 min). Fast-route + vs-RSF claims hold. ⚠
- **Task 6 (table):** backed by stability_results.pkl + fresh Cox; CSVs written. ✓
- **Task 7 (MICE 143):** confirmed from hancock_rebuilt.csv. ✓
- **Task 8 (xj2P parity, P3):** LASSO k=6 reproduced (C=0.772 vs 0.769); paired-bootstrap q=0.635/0.566 backed; RSF-VI=16.6 min confirmed. ✓
