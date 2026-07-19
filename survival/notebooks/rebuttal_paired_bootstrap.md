# Paired-bootstrap C-index difference + BH-FDR (rebuttal)

Methodology: B=1000 paired bootstrap resamples of test set. Two-sided p-value via fraction of bootstrap diffs on either side of 0. BH-FDR applied within each cohort's family of tests.

## RADCURE

**Test-set C-index (Cox PH on selected features):**

- SA-BN MB: C = 0.760
- LASSO-Cox (size-matched): C = 0.769
- LASSO-Cox (CV-best): C = 0.795
- RSF-VI (top-k): C = 0.769

**Paired-bootstrap differences (B=1000):**

| A | B | C_A | C_B | ΔC | 95% CI | p_raw | q_BH | reject@0.05 |
|---|---|---|---|---|---|---|---|---|
| SA-BN MB | LASSO-Cox (size-matched) | 0.760 | 0.769 | -0.009 | [-0.037, +0.018] | 0.476 | 0.635 | no |
| SA-BN MB | LASSO-Cox (CV-best) | 0.760 | 0.795 | -0.035 | [-0.061, -0.010] | 0.01 | 0.04 | YES |
| SA-BN MB | RSF-VI (top-k) | 0.760 | 0.769 | -0.009 | [-0.037, +0.018] | 0.476 | 0.635 | no |
| LASSO-Cox (size-matched) | RSF-VI (top-k) | 0.769 | 0.769 | +0.000 | [+0.000, +0.000] | 2 | 1 | no |

## HANCOCK

**Test-set C-index (Cox PH on selected features):**

- SA-BN MB: C = 0.725
- LASSO-Cox (size-matched): C = 0.699
- RSF-VI (top-k): C = 0.674

**Paired-bootstrap differences (B=1000):**

| A | B | C_A | C_B | ΔC | 95% CI | p_raw | q_BH | reject@0.05 |
|---|---|---|---|---|---|---|---|---|
| SA-BN MB | LASSO-Cox (size-matched) | 0.725 | 0.699 | +0.025 | [-0.037, +0.094] | 0.496 | 0.566 | no |
| SA-BN MB | RSF-VI (top-k) | 0.725 | 0.674 | +0.051 | [-0.044, +0.154] | 0.308 | 0.566 | no |
| LASSO-Cox (size-matched) | RSF-VI (top-k) | 0.699 | 0.674 | +0.025 | [-0.042, +0.105] | 0.566 | 0.566 | no |
