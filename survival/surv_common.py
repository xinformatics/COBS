"""
surv_common.py — Survival-specific utilities
=============================================
Structured arrays, C-index bootstrap, time-dependent AUC, IBS,
D-calibration, and dataset loading for the survival pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory so we can import common
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from common import (
    SEED, SURVIVAL_THRESHOLDS_YEARS, setup_plotting,
    CB_BLUE, CB_ORANGE, CB_GREEN, CB_SKY, CB_VERMIL, CB_PINK, CB_BLACK,
    DatasetConfig, RADCURE_CONFIG, HANCOCK_CONFIG, OPC_CONFIG,
    load_dataset, get_bn_graph_and_mb, define_subgroups,
    harmonize_pair, harmonize_all,
    _simplify_t, _simplify_n, _simplify_site,
)

FIG_BASE = PARENT_DIR / "figures"
N_BOOT_CI = 1000


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED SURVIVAL ARRAY
# ══════════════════════════════════════════════════════════════════════════════

def make_y_surv(df, time_col="time_days", event_col="event"):
    """Convert DataFrame columns to sksurv structured array.

    Args:
        df: DataFrame with time and event columns.
        time_col: column name for follow-up time (days).
        event_col: column name for event indicator (1=event, 0=censored).

    Returns:
        Structured numpy array with (event, time) dtype.
    """
    event = df[event_col].astype(bool).values
    time = pd.to_numeric(df[time_col], errors="coerce").values
    return np.array(
        list(zip(event, time)),
        dtype=[("event", bool), ("time", float)],
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENCODING FOR SURVIVAL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def encode_for_survival(df, feat_cols):
    """One-hot encode BN features for survival models, returning numeric matrix.

    Prefers _raw columns (continuous) when available, dummies for categoricals.
    """
    X = pd.DataFrame(index=df.index)

    for col in feat_cols:
        raw_col = f"{col}_raw"
        if raw_col in df.columns:
            vals = pd.to_numeric(df[raw_col], errors="coerce")
            if vals.notna().sum() > 0:
                X[col] = vals.fillna(vals.median())
                continue
        # Categorical: one-hot
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    X = X.fillna(0).astype(float)
    return X


# ══════════════════════════════════════════════════════════════════════════════
# C-INDEX BOOTSTRAP CI
# ══════════════════════════════════════════════════════════════════════════════

def concordance_bootstrap(y_surv, risk_scores, n_boot=N_BOOT_CI, alpha=0.05):
    """Bootstrap 95% CI for Harrell's concordance index.

    Args:
        y_surv: structured array (event, time).
        risk_scores: predicted risk scores (higher = higher risk).
        n_boot: number of bootstrap resamples.
        alpha: significance level.

    Returns:
        dict with 'cindex', 'ci_lower', 'ci_upper'.
    """
    from sksurv.metrics import concordance_index_censored

    event = y_surv["event"]
    time = y_surv["time"]

    # Point estimate
    c_stat, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        event, time, risk_scores
    )

    rng = np.random.RandomState(SEED)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(event), len(event), replace=True)
        e_b, t_b, r_b = event[idx], time[idx], risk_scores[idx]
        if e_b.sum() < 2:
            continue
        try:
            c_b, *_ = concordance_index_censored(e_b, t_b, r_b)
            boot_scores.append(c_b)
        except Exception:
            continue

    if len(boot_scores) < 50:
        return {"cindex": c_stat, "ci_lower": np.nan, "ci_upper": np.nan}

    return {
        "cindex": c_stat,
        "ci_lower": np.percentile(boot_scores, 100 * alpha / 2),
        "ci_upper": np.percentile(boot_scores, 100 * (1 - alpha / 2)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TIME-DEPENDENT AUC (CUMULATIVE/DYNAMIC)
# ══════════════════════════════════════════════════════════════════════════════

def time_dependent_auc(y_train_surv, y_test_surv, risk_scores,
                       times_years=None):
    """Compute cumulative/dynamic AUC at multiple time horizons.

    Args:
        y_train_surv: training structured array (for KM estimator).
        y_test_surv: test structured array.
        risk_scores: predicted risk scores on test set.
        times_years: list of years at which to evaluate. Defaults to [1,2,3,5].

    Returns:
        DataFrame with columns ['years', 'days', 'auc'].
    """
    from sksurv.metrics import cumulative_dynamic_auc

    if times_years is None:
        times_years = [1, 2, 3, 5]

    test_times = y_test_surv["time"]
    # Filter to times within observed range
    t_min = test_times.min()
    t_max = test_times.max()

    results = []
    for yr in times_years:
        t_days = yr * 365.25
        if t_days < t_min or t_days > t_max * 0.95:
            continue
        try:
            auc_vals, mean_auc = cumulative_dynamic_auc(
                y_train_surv, y_test_surv, risk_scores, times=[t_days]
            )
            results.append({"years": yr, "days": t_days, "auc": auc_vals[0]})
        except Exception:
            continue

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATED BRIER SCORE
# ══════════════════════════════════════════════════════════════════════════════

def integrated_brier(y_train_surv, y_test_surv, surv_funcs, times_days=None):
    """Compute integrated Brier score over a time grid.

    Args:
        y_train_surv: training structured array (for IPCW).
        y_test_surv: test structured array.
        surv_funcs: array-like of survival function estimates at times_days.
                    Shape: (n_test, n_times).
        times_days: 1-D array of evaluation times. If None, auto-generated.

    Returns:
        dict with 'ibs' and 'brier_at_times' DataFrame.
    """
    from sksurv.metrics import integrated_brier_score, brier_score

    test_times = y_test_surv["time"]
    if times_days is None:
        times_days = np.linspace(
            max(test_times.min(), 30),
            test_times.max() * 0.9,
            50,
        )

    # Filter valid times
    train_times = y_train_surv["time"]
    t_lo = max(train_times.min(), test_times.min()) + 1
    t_hi = min(train_times[y_train_surv["event"]].max(),
               test_times.max()) * 0.95
    valid_mask = (times_days >= t_lo) & (times_days <= t_hi)
    times_days = times_days[valid_mask]

    if len(times_days) < 3:
        return {"ibs": np.nan, "brier_at_times": pd.DataFrame()}

    try:
        ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_funcs[:, valid_mask],
                                     times_days)
    except Exception:
        ibs = np.nan

    # Brier at specific year marks
    brier_results = []
    for yr in [1, 2, 3, 5]:
        t_d = yr * 365.25
        if t_d < t_lo or t_d > t_hi:
            continue
        idx = np.argmin(np.abs(times_days - t_d))
        try:
            _, bs = brier_score(y_train_surv, y_test_surv,
                                surv_funcs[:, valid_mask][:, idx], times_days[idx])
            brier_results.append({"years": yr, "brier": bs[-1] if hasattr(bs, '__len__') else bs})
        except Exception:
            continue

    return {"ibs": ibs, "brier_at_times": pd.DataFrame(brier_results)}


# ══════════════════════════════════════════════════════════════════════════════
# D-CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def d_calibration_test(y_test_surv, surv_probs_at_t, n_bins=10):
    """D-calibration test: are predicted survival probabilities well-calibrated?

    Uses the Hosmer-Lemeshow-style binning on predicted survival probabilities
    at a specific time point.

    Args:
        y_test_surv: structured array.
        surv_probs_at_t: predicted P(T > t) at a specific time t.
        n_bins: number of bins.

    Returns:
        dict with 'chi2_stat', 'p_value', 'bins_df'.
    """
    from scipy.stats import chi2

    event = y_test_surv["event"]
    time = y_test_surv["time"]
    n = len(event)

    # Bin by predicted survival probability
    bins = pd.qcut(surv_probs_at_t, q=n_bins, duplicates="drop")
    bin_labels = bins.cat.categories

    rows = []
    for b in bin_labels:
        mask = bins == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        pred_surv = surv_probs_at_t[mask].mean()
        obs_surv = 1 - event[mask].mean()  # approximate for simplicity
        rows.append({
            "bin": str(b), "n": n_b,
            "pred_surv": pred_surv, "obs_surv": obs_surv,
            "diff": obs_surv - pred_surv,
        })

    bins_df = pd.DataFrame(rows)
    if bins_df.empty:
        return {"chi2_stat": np.nan, "p_value": np.nan, "bins_df": bins_df}

    # Chi-squared statistic
    stat = sum(
        r["n"] * (r["obs_surv"] - r["pred_surv"])**2 / max(r["pred_surv"] * (1 - r["pred_surv"]), 1e-6)
        for _, r in bins_df.iterrows()
    )
    df = len(bins_df) - 1
    p_val = chi2.sf(stat, df) if df > 0 else np.nan

    return {"chi2_stat": stat, "p_value": p_val, "bins_df": bins_df}


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING FOR SURVIVAL
# ══════════════════════════════════════════════════════════════════════════════

def load_survival_dataset(config, verbose=True):
    """Load dataset via parent pipeline, then attach survival arrays.

    Returns dict with:
        train_df, test_df: DataFrames with BN features (from binary pipeline).
        feat_cols, target: from binary pipeline.
        train_df_full, test_df_full: DataFrames with ALL patients (incl. censored).
        y_train_surv, y_test_surv: structured arrays for sksurv.
        G_consensus, mb_info: from BN discovery.
    """
    # Run the standard binary pipeline (BN/MB on SVy{k})
    train_df, test_df, feat_cols, target = load_dataset(config, verbose=verbose)

    if verbose:
        print(f"\n  Survival endpoint: using ALL patients (no censoring exclusion)")

    # We need the full pre-BN DataFrame with time_days and event for ALL patients
    # Reload raw data, preprocess, but don't filter by SVy evaluability
    raw_df = pd.read_csv(config.csv_path)
    if verbose:
        print(f"  Raw CSV: {len(raw_df)} patients")

    # Import preprocessing functions
    from common import (
        build_survival_endpoints, _preprocess_hancock, _preprocess_opc,
        _preprocess_mda, _preprocess_radiation, _preprocess_radcure,
    )

    if config.name == "hancock":
        raw_df = _preprocess_hancock(raw_df)
    elif config.name == "opc":
        raw_df = _preprocess_opc(raw_df)
    elif config.name == "mda":
        raw_df = _preprocess_mda(raw_df)
    elif config.name == "radiation":
        raw_df = _preprocess_radiation(raw_df)
    elif config.name == "radcure":
        raw_df = _preprocess_radcure(raw_df)

    raw_df = build_survival_endpoints(raw_df, config)

    # Filter patients with valid time and event
    raw_df["time_days"] = pd.to_numeric(raw_df["time_days"], errors="coerce")
    raw_df["event"] = pd.to_numeric(raw_df["event"], errors="coerce").fillna(0).astype(int)
    full_df = raw_df.dropna(subset=["time_days"]).copy()
    full_df = full_df[full_df["time_days"] > 0].copy()

    if verbose:
        print(f"  Patients with valid survival data: {len(full_df)}")
        print(f"  Events: {full_df['event'].sum()} ({full_df['event'].mean():.1%})")
        median_fu = full_df["time_days"].median() / 365.25
        print(f"  Median follow-up: {median_fu:.1f} years")

    # Split full data the same way as the binary pipeline
    if config.split_strategy == "temporal" and config.split_year:
        year_col = config.columns.year
        if year_col and year_col in full_df.columns:
            yr = pd.to_numeric(full_df[year_col], errors="coerce")
        elif "_rt_year" in full_df.columns:
            yr = pd.to_numeric(full_df["_rt_year"], errors="coerce")
        elif "_diag_year" in full_df.columns:
            yr = pd.to_numeric(full_df["_diag_year"], errors="coerce")
        else:
            yr = pd.Series(np.nan, index=full_df.index)

        train_full = full_df[yr <= config.split_year].copy()
        test_full = full_df[yr > config.split_year].copy()
    else:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            full_df.index, test_size=1-config.split_ratio, random_state=SEED
        )
        train_full = full_df.loc[train_idx].copy()
        test_full = full_df.loc[test_idx].copy()

    if verbose:
        print(f"  Full split: train={len(train_full)}, test={len(test_full)}")
        print(f"  Train events: {train_full['event'].sum()}, "
              f"Test events: {test_full['event'].sum()}")

    # Structured arrays
    y_train_surv = make_y_surv(train_full)
    y_test_surv = make_y_surv(test_full)

    # BN/MB discovery (on binary-evaluable subset)
    G_consensus, all_graphs, mb_info = get_bn_graph_and_mb(
        train_df, feat_cols, target, verbose=verbose
    )

    return {
        "train_df": train_df,           # binary-evaluable (for BN)
        "test_df": test_df,             # binary-evaluable
        "feat_cols": feat_cols,
        "target": target,
        "train_full": train_full,       # ALL patients
        "test_full": test_full,         # ALL patients
        "y_train_surv": y_train_surv,
        "y_test_surv": y_test_surv,
        "G_consensus": G_consensus,
        "all_graphs": all_graphs,
        "mb_info": mb_info,
        "config": config,
    }
