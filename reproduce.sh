#!/usr/bin/env bash
# Reproduce the camera-ready rebuttal analyses (MLHC 2026 #283).
# Run order matters: rebuttal_verify.py builds /tmp/surv_cache.pkl that later scripts reuse.
set -e
cd "$(dirname "$0")/survival"

echo "[1/6] rebuttal_verify.py  (PH + FDR + membership + MICE; builds cache)"
python rebuttal_verify.py
echo "[2/6] rebuttal_assemble.py  (assemble rebuttal_results.md)"
python rebuttal_assemble.py
echo "[3/6] rebuttal_task10.py  (LASSO/RSF/GBS/Cox x 3 metrics)"
python rebuttal_task10.py
echo "[4/6] rebuttal_paired_bootstrap.py  (parity q-values)"
python rebuttal_paired_bootstrap.py
echo "[5/6] rebuttal_task45.py  (parity C-indices + runtime)"
python rebuttal_task45.py
echo "[6/6] rebuttal_tcga_2yr.py  (TCGA breast/colorectal/kidney)"
python rebuttal_tcga_2yr.py

echo "Done. Outputs in survival/notebooks/ ; compare against the shipped *_results.{json,md}."
