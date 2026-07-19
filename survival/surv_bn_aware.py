"""
surv_bn_aware.py — Survival-Aware Bayesian Network Structure Learning
=====================================================================

Methodological contribution
---------------------------
Traditional BN-based feature selection uses binary classification targets
(e.g. "survived >= 2 years?"), which:
  (a) excludes 12-19% of censored patients, and
  (b) discards the temporal structure of survival data.

We propose **Survival-Aware BN (SA-BN)** structure learning that replaces the
binary target node's scoring function with the Cox partial log-likelihood,
allowing ALL patients (including censored) to contribute to structure learning.

Three consensus algorithms:
  1. SA-HC/Cox  — Hill-Climbing with BDeu (feature edges) + Cox BIC (target edges)
  2. SA-HC/LR   — Hill-Climbing with BDeu (feature edges) + log-rank (target edges)
  3. SA-PC/LR   — PC algorithm with stratified log-rank CI tests

Three MB discovery methods:
  1. DAG-derived — parents/children/co-parents from SA-BN
  2. Cox stepwise — forward/backward selection with likelihood-ratio tests
  3. C-index ranking — top-k features by univariate concordance index

Usage:
    from surv_bn_aware import main
    results = main(config)           # full pipeline + comparison
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from common import (
    SEED, MAX_INDEGREE, N_BOOTSTRAP,
    DatasetConfig, RADCURE_CONFIG, HANCOCK_CONFIG,
    build_survival_endpoints, _discretize_yj_gmm, _add_interactions,
    _preprocess_hancock, _preprocess_radcure, _preprocess_opc,
    load_dataset, get_bn_graph_and_mb,
)

from surv_common import (
    make_y_surv, encode_for_survival, concordance_bootstrap,
    time_dependent_auc, load_survival_dataset, FIG_BASE,
    CB_BLUE, CB_ORANGE, CB_GREEN, CB_VERMIL,
)

RNG = np.random.RandomState(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: LOAD ALL PATIENTS (NO CENSORING EXCLUSION)
# ══════════════════════════════════════════════════════════════════════════════

_PREPROC = {
    "hancock": _preprocess_hancock,
    "radcure": _preprocess_radcure,
    "opc": _preprocess_opc,
}


def load_all_patients(config, verbose=True):
    """Load and discretize features for ALL patients with valid survival data.

    Unlike ``load_dataset()``, does NOT exclude censored patients.  Feature
    discretisation (Yeo-Johnson + GMM) is fit on training data and applied
    to both splits, identical to the binary pipeline.

    Returns
    -------
    disc_df : DataFrame   – discretized features for ALL patients
    feat_cols : list[str]  – feature column names (after filtering)
    time : ndarray         – survival time in days
    event : ndarray        – event indicator (1=death, 0=censored)
    train_mask : Series    – boolean mask for training set
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING ALL PATIENTS — {config.name.upper()} (no censoring exclusion)")
        print(f"{'='*70}")

    # ── 1. Raw load + dataset-specific preprocessing ──────────────────────
    df = pd.read_csv(config.csv_path)
    n_raw = len(df)
    preproc = _PREPROC.get(config.name)
    if preproc is None:
        raise ValueError(f"No preprocessor for '{config.name}'")
    df = preproc(df)
    df = build_survival_endpoints(df, config)

    # ── 2. Keep all patients with valid survival data ─────────────────────
    df["time_days"] = pd.to_numeric(df["time_days"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
    df = df[df["time_days"].notna() & (df["time_days"] > 0)].copy()
    df = df.reset_index(drop=True)

    time = df["time_days"].values.astype(float)
    event = df["event"].values.astype(int)

    if verbose:
        print(f"  Raw CSV: {n_raw} patients")
        print(f"  Valid survival data: {len(df)} patients "
              f"({len(df) - event.sum()} censored, {event.sum()} events)")

    # ── 3. Temporal train/test split (same as binary pipeline) ────────────
    if config.split_strategy == "temporal" and config.split_year:
        year_col = config.columns.year
        if year_col and year_col in df.columns:
            yr = pd.to_numeric(df[year_col], errors="coerce")
        elif "_rt_year" in df.columns:
            yr = pd.to_numeric(df["_rt_year"], errors="coerce")
        elif "_diag_year" in df.columns:
            yr = pd.to_numeric(df["_diag_year"], errors="coerce")
        else:
            yr = pd.Series(np.nan, index=df.index)
        train_mask = pd.Series(yr <= config.split_year, index=df.index)
    else:
        from sklearn.model_selection import train_test_split
        train_idx, _ = train_test_split(
            df.index, test_size=1 - config.split_ratio,
            random_state=SEED, stratify=event,
        )
        train_mask = pd.Series(df.index.isin(train_idx), index=df.index)

    n_train = train_mask.sum()
    n_test = (~train_mask).sum()
    if verbose:
        print(f"  Split: train={n_train}, test={n_test}")

    # ── 4. Discretize continuous features (fit on training) ───────────────
    disc_df = df.copy()
    cont_cols = [c for c in config.cont_candidates if c in disc_df.columns]
    if cont_cols:
        disc_df = _discretize_yj_gmm(disc_df, cont_cols, train_mask)

    # ── 5. Normalize categoricals ─────────────────────────────────────────
    for col in config.cat_candidates:
        if col in disc_df.columns:
            disc_df[col] = (disc_df[col].astype(str).str.strip().str.lower()
                            .replace({"nan": np.nan, "none": np.nan, "": np.nan}))

    # ── 6. Add interaction features ───────────────────────────────────────
    disc_df, interaction_cols = _add_interactions(disc_df, config)

    # ── 7. Assemble feature list (same filtering as binary, minus MI) ─────
    all_cands = []
    for c in config.cat_candidates + config.cont_candidates + interaction_cols:
        if c in disc_df.columns and c not in all_cands:
            all_cands.append(c)

    feat_cols = []
    for c in all_cands:
        vals = disc_df[c].dropna()
        if len(vals) < 10:
            continue
        nuniq = vals.nunique()
        if nuniq < 2 or nuniq > 20:
            continue
        miss_pct = disc_df[c].isna().mean()
        if miss_pct > 0.40:
            continue
        feat_cols.append(c)

    if verbose:
        print(f"  Features ({len(feat_cols)}): {feat_cols}")

    # ── 8. Mode imputation (fit on training) ──────────────────────────────
    for c in feat_cols:
        if disc_df[c].isna().any():
            mode_val = disc_df.loc[train_mask, c].mode()
            if len(mode_val) > 0:
                disc_df[c] = disc_df[c].fillna(mode_val.iloc[0])

    # Drop rows still missing features (rare)
    disc_df = disc_df.dropna(subset=feat_cols)
    keep_idx = disc_df.index
    train_mask = train_mask.loc[keep_idx]
    time = time[keep_idx]
    event = event[keep_idx]
    disc_df = disc_df.reset_index(drop=True)
    train_mask = train_mask.reset_index(drop=True)

    if verbose:
        print(f"  Final: {len(disc_df)} patients (train={train_mask.sum()}, "
              f"test={(~train_mask).sum()})")
        evaluable = disc_df[config.columns.time_days if hasattr(config.columns, 'time_days') else "time_days"]
        # How many would be evaluable for binary SVy2?
        target_col = f"SVy{config.primary_threshold_years}"
        if target_col in disc_df.columns:
            n_eval = disc_df[target_col].notna().sum()
            gained = len(disc_df) - n_eval
            print(f"  Binary-evaluable: {n_eval} | GAINED: {gained} patients "
                  f"({gained/len(disc_df):.0%})")

    return disc_df, feat_cols, time, event, train_mask


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: SURVIVAL SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _encode_parents(disc_df, parents):
    """One-hot encode discretized parent features for Cox model."""
    if not parents:
        return pd.DataFrame(index=disc_df.index)
    X = pd.DataFrame(index=disc_df.index)
    for col in parents:
        raw_col = f"{col}_raw"
        if raw_col in disc_df.columns:
            vals = pd.to_numeric(disc_df[raw_col], errors="coerce")
            if vals.notna().sum() > 0:
                X[col] = vals.fillna(vals.median())
                continue
        dummies = pd.get_dummies(disc_df[col], prefix=col, drop_first=True,
                                 dtype=float)
        X = pd.concat([X, dummies], axis=1)
    return X.fillna(0).astype(float)


def cox_bic_score(disc_df, parents, time, event):
    """Cox partial log-likelihood with BIC penalty for a parent set.

    Higher is better.  Returns −∞ if the model cannot be fit.
    """
    from lifelines import CoxPHFitter

    if not parents:
        return 0.0

    X = _encode_parents(disc_df, list(parents))
    fit_df = X.copy()
    fit_df["T"] = time
    fit_df["E"] = event
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(fit_df) < 30 or fit_df["E"].sum() < 10:
        return -np.inf

    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(fit_df, duration_col="T", event_col="E")
        ll = cph.log_likelihood_
        k = len(cph.params_)
        n = len(fit_df)
        return ll - (k / 2.0) * np.log(n)          # BIC-penalised
    except Exception:
        return -np.inf


def logrank_score(disc_df, feat, time, event):
    """Log-rank chi-squared for a single feature's survival discrimination."""
    from lifelines.statistics import multivariate_logrank_test

    groups = disc_df[feat].astype(str)
    valid = groups.notna() & (time > 0)
    groups = groups[valid]
    t = time[valid]
    e = event[valid]

    if groups.nunique() < 2 or len(groups) < 20:
        return 0.0

    try:
        res = multivariate_logrank_test(t, groups, e)
        return res.test_statistic
    except Exception:
        return 0.0


def concordance_score(disc_df, feat, time, event):
    """Univariate concordance index for a single feature."""
    from sksurv.metrics import concordance_index_censored

    X = _encode_parents(disc_df, [feat])
    if X.shape[1] == 0:
        return 0.5

    # Simple risk score: sum of encoded columns
    risk = X.sum(axis=1).values
    ev = event.astype(bool)
    t = time.astype(float)

    valid = np.isfinite(risk) & np.isfinite(t) & (t > 0)
    if valid.sum() < 20:
        return 0.5

    try:
        c, *_ = concordance_index_censored(ev[valid], t[valid], risk[valid])
        return max(c, 1 - c)  # Unsigned concordance
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: SURVIVAL-AWARE BN ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def _feature_graph_pgmpy(disc_df, feat_cols, scoring="bdeu"):
    """Learn feature-feature edges using standard pgmpy HC."""
    from pgmpy.estimators import HillClimbSearch, BDeu as BDeuScore, K2 as K2Score

    data = disc_df[feat_cols].copy()
    # pgmpy >=1.0 requires pd.Categorical dtype, not str
    for col in data.columns:
        data[col] = pd.Categorical(data[col].astype(str))

    try:
        hc = HillClimbSearch(data)
        score_fn = (BDeuScore(data, equivalent_sample_size=10) if scoring == "bdeu"
                    else K2Score(data))
        model = hc.estimate(scoring_method=score_fn, max_indegree=MAX_INDEGREE)
        G = nx.DiGraph()
        G.add_nodes_from(feat_cols)
        G.add_edges_from(model.edges())
        return G
    except Exception as e:
        print(f"    pgmpy HC ({scoring}) failed: {e}")
        G = nx.DiGraph()
        G.add_nodes_from(feat_cols)
        return G


def _cox_null_ll(time, event):
    """Compute Cox partial log-likelihood for the null (no-covariate) model.

    For the Breslow approximation: ll_null = -sum_{i: event_i=1} log(|R_i|)
    where R_i is the risk set at time t_i.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    order = np.argsort(-t)  # descending time
    ll = 0.0
    n_at_risk = 0
    for idx in order:
        n_at_risk += 1
        if e[idx] == 1:
            ll -= np.log(n_at_risk)
    return ll


def _add_target_edges_cox(G, disc_df, feat_cols, time, event, target_name,
                          max_parents=MAX_INDEGREE, verbose=True):
    """Forward selection of target parents using Cox LRT.

    Uses likelihood-ratio tests between nested models.  The first feature
    is tested against the proper null model (no covariates).  All models
    are fit on the same complete-case rows to ensure valid LRT comparisons.
    """
    from lifelines import CoxPHFitter
    from scipy.stats import chi2 as chi2_dist

    G = G.copy()
    G.add_node(target_name)

    # ── Compute complete-case mask across ALL candidate features ──────────
    # This ensures every LRT comparison uses identical rows.
    full_X = _encode_parents(disc_df, list(feat_cols))
    full_X["T"] = time
    full_X["E"] = event
    complete_mask = full_X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    cc_time = np.asarray(time)[complete_mask.values]
    cc_event = np.asarray(event)[complete_mask.values]
    cc_df = disc_df[complete_mask].reset_index(drop=True)

    if len(cc_df) < 30 or cc_event.sum() < 10:
        if verbose:
            print(f"    Cox target parents: [] (insufficient complete cases)")
        return G

    # ── Null model log-likelihood ─────────────────────────────────────────
    prev_ll = _cox_null_ll(cc_time, cc_event)
    prev_n_params = 0

    selected = []
    remaining = list(feat_cols)

    while remaining and len(selected) < max_parents:
        best_feat = None
        best_p = 1.0
        best_ll = None
        best_n_params = 0

        for feat in remaining:
            candidates = selected + [feat]
            X = _encode_parents(cc_df, candidates)
            fit_df = X.copy()
            fit_df["T"] = cc_time
            fit_df["E"] = cc_event

            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(fit_df, duration_col="T", event_col="E")
                ll = cph.log_likelihood_
                n_params = len(cph.params_)

                lr_stat = 2 * (ll - prev_ll)
                dof = max(n_params - prev_n_params, 1)
                p_val = chi2_dist.sf(lr_stat, dof)

                if p_val < best_p:
                    best_p = p_val
                    best_feat = feat
                    best_ll = ll
                    best_n_params = n_params
            except Exception:
                continue

        if best_feat is None or best_p > 0.05:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        prev_ll = best_ll
        prev_n_params = best_n_params
        G.add_edge(best_feat, target_name)

    if verbose:
        print(f"    Cox target parents: {selected}")

    return G


def _add_target_edges_logrank(G, disc_df, feat_cols, time, event,
                              target_name, p_threshold=0.01, verbose=True):
    """Stepwise forward selection of target parents using conditional log-rank.

    At each step, selects the feature with the smallest conditional log-rank
    p-value (conditioned on already-selected features).  Stops when no
    remaining feature is significant.
    """
    G = G.copy()
    G.add_node(target_name)

    selected = []
    remaining = list(feat_cols)

    while remaining and len(selected) < MAX_INDEGREE:
        best_feat = None
        best_p = 1.0

        for feat in remaining:
            # Conditional log-rank: test feat ⊥ survival | selected
            p_val = _logrank_ci_pvalue(disc_df, feat, time, event, selected)
            if p_val < best_p:
                best_p = p_val
                best_feat = feat

        if best_feat is None or best_p > p_threshold:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        G.add_edge(best_feat, target_name)

    if verbose:
        print(f"    Log-rank target parents: {selected}")

    return G


def surv_hc_cox(disc_df, feat_cols, time, event, target_name="survival",
                verbose=True):
    """SA-HC/Cox: BDeu feature edges + Cox BIC target edges."""
    if verbose:
        print("  [1/3] SA-HC/Cox")
    G = _feature_graph_pgmpy(disc_df, feat_cols, scoring="bdeu")
    G = _add_target_edges_cox(G, disc_df, feat_cols, time, event,
                              target_name, verbose=verbose)
    return G


def surv_hc_logrank(disc_df, feat_cols, time, event, target_name="survival",
                    verbose=True):
    """SA-HC/LR: K2 feature edges + log-rank target edges."""
    if verbose:
        print("  [2/3] SA-HC/LR")
    G = _feature_graph_pgmpy(disc_df, feat_cols, scoring="k2")
    G = _add_target_edges_logrank(G, disc_df, feat_cols, time, event,
                                  target_name, verbose=verbose)
    return G


def surv_pc_logrank(disc_df, feat_cols, time, event, target_name="survival",
                    alpha=0.05, max_cond=3, verbose=True):
    """SA-PC/LR: PC algorithm with stratified log-rank CI tests.

    For feature-feature edges: chi-squared CI test.
    For feature-target edges: stratified log-rank CI test.
    """
    from scipy.stats import chi2_contingency, chi2 as chi2_dist
    from lifelines.statistics import multivariate_logrank_test

    if verbose:
        print("  [3/3] SA-PC/LR")

    all_nodes = feat_cols + [target_name]
    G = nx.Graph()  # Start undirected
    G.add_nodes_from(all_nodes)

    # Start fully connected
    for i, u in enumerate(all_nodes):
        for v in all_nodes[i + 1:]:
            G.add_edge(u, v)

    sep_sets = {}

    # ── Skeleton discovery (removing edges) ───────────────────────────────
    for cond_size in range(max_cond + 1):
        edges_to_test = list(G.edges())
        for u, v in edges_to_test:
            if not G.has_edge(u, v):
                continue

            neighbors = set(G.neighbors(u)) - {v}
            if len(neighbors) < cond_size:
                continue

            from itertools import combinations
            for Z in combinations(neighbors, cond_size):
                Z = set(Z)

                # CI test depends on whether target is involved
                if target_name in (u, v):
                    feat = u if v == target_name else v
                    p_val = _logrank_ci_pvalue(disc_df, feat, time, event,
                                              list(Z - {target_name}))
                else:
                    # target_name is a virtual node (time+event), not
                    # an observable column in disc_df.  We approximate
                    # u ⊥ v | Z by u ⊥ v | Z\{target}, which is
                    # conservative: it may fail to remove edges that
                    # would be removed if we could condition on
                    # survival directly.
                    p_val = _chi2_ci_pvalue(disc_df, u, v,
                                           list(Z - {target_name}))

                if p_val > alpha:
                    G.remove_edge(u, v)
                    sep_sets[(u, v)] = Z
                    sep_sets[(v, u)] = Z
                    break

    # ── Orient edges ──────────────────────────────────────────────────────
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(all_nodes)

    # V-structures
    for node in all_nodes:
        nbrs = list(G.neighbors(node))
        from itertools import combinations
        for u, v in combinations(nbrs, 2):
            if G.has_edge(u, v):
                continue
            sep = sep_sets.get((u, v), set())
            if node not in sep:
                G_dir.add_edge(u, node)
                G_dir.add_edge(v, node)

    # Add remaining edges (prefer feature → target direction)
    for u, v in G.edges():
        if G_dir.has_edge(u, v) or G_dir.has_edge(v, u):
            continue
        if v == target_name:
            G_dir.add_edge(u, v)
        elif u == target_name:
            G_dir.add_edge(v, u)
        else:
            G_dir.add_edge(u, v)  # arbitrary direction

    # Break cycles
    while not nx.is_directed_acyclic_graph(G_dir):
        try:
            cycle = nx.find_cycle(G_dir)
            G_dir.remove_edge(*cycle[0])
        except nx.NetworkXNoCycle:
            break

    if verbose:
        target_parents = list(G_dir.predecessors(target_name))
        print(f"    PC target parents: {target_parents}")

    return G_dir


def _logrank_ci_pvalue(disc_df, feat, time, event, cond_vars):
    """Stratified log-rank p-value: feat ⊥ survival | cond_vars.

    NOTE: Uses summed chi-squared statistics across strata (CMH-type
    heuristic).  This is an approximation; a formal Mantel-Haenszel
    stratified log-rank test would use weighted statistics.  Strata
    with < 10 observations are skipped to mitigate small-sample
    chi-squared approximation issues.
    """
    from lifelines.statistics import multivariate_logrank_test
    from scipy.stats import chi2 as chi2_dist

    groups = disc_df[feat].astype(str)
    valid = groups.notna() & (time > 0) & np.isfinite(time)

    if not cond_vars:
        # Unconditional
        g, t, e = groups[valid], time[valid], event[valid]
        if g.nunique() < 2 or len(g) < 20:
            return 1.0
        try:
            res = multivariate_logrank_test(t, g, e)
            return res.p_value
        except Exception:
            return 1.0

    # Stratified: combine log-rank statistics across strata
    strata = disc_df[cond_vars].astype(str).apply(tuple, axis=1)
    total_chi2, total_dof = 0.0, 0

    for stratum_val in strata[valid].unique():
        mask = valid & (strata == stratum_val)
        if mask.sum() < 10:
            continue
        g_s = groups[mask]
        if g_s.nunique() < 2:
            continue
        try:
            res = multivariate_logrank_test(time[mask], g_s, event[mask])
            total_chi2 += res.test_statistic
            total_dof += max(g_s.nunique() - 1, 1)
        except Exception:
            continue

    if total_dof == 0:
        return 1.0
    return chi2_dist.sf(total_chi2, total_dof)


def _chi2_ci_pvalue(disc_df, u, v, cond_vars):
    """Chi-squared CI test p-value: u ⊥ v | cond_vars."""
    from scipy.stats import chi2_contingency

    if not cond_vars:
        try:
            ct = pd.crosstab(disc_df[u].astype(str), disc_df[v].astype(str))
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                return 1.0
            _, p, _, _ = chi2_contingency(ct)
            return p
        except Exception:
            return 1.0

    # Stratified chi-squared
    strata = disc_df[cond_vars].astype(str).apply(tuple, axis=1)
    from scipy.stats import chi2 as chi2_dist
    total_chi2, total_dof = 0.0, 0
    for sv in strata.unique():
        mask = strata == sv
        if mask.sum() < 10:
            continue
        try:
            ct = pd.crosstab(disc_df.loc[mask, u].astype(str),
                             disc_df.loc[mask, v].astype(str))
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            stat, _, dof, _ = chi2_contingency(ct)
            total_chi2 += stat
            total_dof += dof
        except Exception:
            continue

    if total_dof == 0:
        return 1.0
    return chi2_dist.sf(total_chi2, total_dof)


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: CONSENSUS
# ══════════════════════════════════════════════════════════════════════════════

def survival_ensemble_bn(disc_df, feat_cols, time, event,
                         target_name="survival", verbose=True):
    """Consensus of 3 survival-aware BN algorithms.

    An edge is included if it appears in >= 2/3 algorithms.
    """
    if verbose:
        print(f"\n  SURVIVAL-AWARE ENSEMBLE BN ({len(disc_df)} patients, "
              f"{len(feat_cols)} features)")

    graphs = {
        "SA-HC/Cox": surv_hc_cox(disc_df, feat_cols, time, event,
                                 target_name, verbose),
        "SA-HC/LR": surv_hc_logrank(disc_df, feat_cols, time, event,
                                    target_name, verbose),
        "SA-PC/LR": surv_pc_logrank(disc_df, feat_cols, time, event,
                                    target_name, verbose=verbose),
    }

    # ── Consensus (direction-aware) ──────────────────────────────────────
    # Count directed edges separately; an undirected "presence" requires
    # >= 2/3 algorithms, and direction is determined by majority vote.
    from collections import Counter
    directed_counts = Counter()   # (u, v) directed
    for G in graphs.values():
        for u, v in G.edges():
            directed_counts[(u, v)] += 1

    # Group by undirected edge to check presence threshold
    undirected_support = Counter()
    for (u, v), cnt in directed_counts.items():
        key = (min(u, v), max(u, v))
        undirected_support[key] += cnt  # total votes for this edge pair

    threshold = 2  # >= 2/3
    G_consensus = nx.DiGraph()
    G_consensus.add_nodes_from(feat_cols + [target_name])

    for (a, b), total in undirected_support.items():
        if total < threshold:
            continue
        # Direction by majority vote
        fwd = directed_counts.get((a, b), 0)
        bwd = directed_counts.get((b, a), 0)
        if fwd >= bwd:
            G_consensus.add_edge(a, b)
        else:
            G_consensus.add_edge(b, a)

    # Break cycles by removing the least-supported edge
    while not nx.is_directed_acyclic_graph(G_consensus):
        try:
            cycle = nx.find_cycle(G_consensus)
            min_edge = min(cycle, key=lambda e: directed_counts.get(
                (e[0], e[1]), 0))
            G_consensus.remove_edge(*min_edge[:2])
        except nx.NetworkXNoCycle:
            break

    if verbose:
        target_parents = sorted(G_consensus.predecessors(target_name))
        print(f"\n  Consensus: {G_consensus.number_of_nodes()} nodes, "
              f"{G_consensus.number_of_edges()} edges")
        print(f"  Target parents: {target_parents}")

    return G_consensus, graphs


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: SURVIVAL MARKOV BLANKET DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _mb_dag_derived(G, target_name, feat_cols):
    """MB Method 1: parents + children + co-parents from DAG."""
    parents = set(G.predecessors(target_name))
    children = set(G.successors(target_name))
    co_parents = set()
    for ch in children:
        co_parents |= set(G.predecessors(ch)) - {target_name}
    mb = (parents | children | co_parents) & set(feat_cols)
    return sorted(mb)


def _mb_cox_stepwise(disc_df, feat_cols, time, event, verbose=True):
    """MB Method 2: Forward selection with Cox LRT.

    Greedily adds features that significantly improve the Cox partial
    log-likelihood (likelihood-ratio test, p < 0.05).  All models are
    fit on the same complete-case rows for valid LRT comparisons.
    """
    from lifelines import CoxPHFitter
    from scipy.stats import chi2 as chi2_dist

    # ── Complete-case mask across all features ────────────────────────────
    full_X = _encode_parents(disc_df, list(feat_cols))
    full_X["T"] = time
    full_X["E"] = event
    cc_mask = full_X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    cc_df = disc_df[cc_mask].reset_index(drop=True)
    cc_time = np.asarray(time)[cc_mask.values]
    cc_event = np.asarray(event)[cc_mask.values]

    if len(cc_df) < 30 or cc_event.sum() < 10:
        if verbose:
            print(f"    Cox stepwise MB: [] (insufficient complete cases)")
        return []

    # ── Null model baseline ───────────────────────────────────────────────
    prev_ll = _cox_null_ll(cc_time, cc_event)
    prev_n_params = 0

    selected = []
    remaining = list(feat_cols)

    while remaining:
        best_feat = None
        best_p = 1.0
        best_ll = None
        best_n_params = 0

        for feat in remaining:
            candidates = selected + [feat]
            X = _encode_parents(cc_df, candidates)
            fit_df = X.copy()
            fit_df["T"] = cc_time
            fit_df["E"] = cc_event

            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(fit_df, "T", "E")
                ll = cph.log_likelihood_
                n_params = len(cph.params_)

                lr_stat = 2 * (ll - prev_ll)
                dof = max(n_params - prev_n_params, 1)
                p_val = chi2_dist.sf(lr_stat, dof)

                if p_val < best_p:
                    best_p = p_val
                    best_feat = feat
                    best_ll = ll
                    best_n_params = n_params
            except Exception:
                continue

        if best_feat is None or best_p > 0.05:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        prev_ll = best_ll
        prev_n_params = best_n_params

    # ── Backward pruning on same complete-case data ───────────────────────
    if len(selected) > 1:
        pruned = list(selected)
        for feat in selected:
            others = [f for f in pruned if f != feat]
            if not others:
                break

            try:
                X_full = _encode_parents(cc_df, pruned)
                X_full["T"] = cc_time
                X_full["E"] = cc_event
                cph_full = CoxPHFitter(penalizer=0.01)
                cph_full.fit(X_full, "T", "E")

                X_red = _encode_parents(cc_df, others)
                X_red["T"] = cc_time
                X_red["E"] = cc_event
                cph_red = CoxPHFitter(penalizer=0.01)
                cph_red.fit(X_red, "T", "E")

                lr = 2 * (cph_full.log_likelihood_ - cph_red.log_likelihood_)
                n_lost = len(cph_full.params_) - len(cph_red.params_)
                p = chi2_dist.sf(lr, max(n_lost, 1))
                if p > 0.10:
                    pruned.remove(feat)
            except Exception:
                continue
        selected = pruned

    if verbose:
        print(f"    Cox stepwise MB: {selected}")
    return selected


def _mb_cindex_topk(disc_df, feat_cols, time, event, k=None, verbose=True):
    """MB Method 3: Top-k features by univariate concordance index."""
    scores = {}
    for feat in feat_cols:
        scores[feat] = concordance_score(disc_df, feat, time, event)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if k is None:
        k = max(3, len(feat_cols) // 3)

    topk = [feat for feat, _ in ranked[:k]]
    if verbose:
        for feat, c in ranked[:k]:
            print(f"      {feat}: C={c:.3f}")
        print(f"    C-index top-{k} MB: {topk}")
    return topk


def survival_mb_discovery(G_consensus, disc_df, feat_cols, time, event,
                          target_name="survival", verbose=True):
    """Three-way survival Markov blanket discovery.

    Consensus: feature in MB if selected by >= 2/3 methods.
    """
    if verbose:
        print(f"\n  SURVIVAL MB DISCOVERY")

    # Method 1: DAG-derived
    mb_dag = _mb_dag_derived(G_consensus, target_name, feat_cols)
    if verbose:
        print(f"    DAG-derived MB: {mb_dag}")

    # Method 2: Cox stepwise
    mb_cox = _mb_cox_stepwise(disc_df, feat_cols, time, event, verbose)

    # Method 3: C-index top-k (use max of DAG and Cox stepwise sizes)
    k = max(len(mb_dag), len(mb_cox), 5)
    mb_cindex = _mb_cindex_topk(disc_df, feat_cols, time, event, k, verbose)

    # ── Consensus ─────────────────────────────────────────────────────────
    all_vars = set(mb_dag) | set(mb_cox) | set(mb_cindex)
    consensus = []
    for var in all_vars:
        count = sum([var in mb_dag, var in mb_cox, var in mb_cindex])
        if count >= 2:
            consensus.append(var)
    consensus = sorted(consensus)

    if verbose:
        print(f"\n  Survival MB consensus ({len(consensus)}): {consensus}")
        for var in consensus:
            methods = []
            if var in mb_dag:
                methods.append("DAG")
            if var in mb_cox:
                methods.append("Cox")
            if var in mb_cindex:
                methods.append("C-idx")
            print(f"    {var}: {'+'.join(methods)}")

    return {
        "dag_derived": mb_dag,
        "cox_stepwise": mb_cox,
        "cindex_topk": mb_cindex,
        "consensus": consensus,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: BOOTSTRAP STABILITY
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_mb_stability(disc_df, feat_cols, time, event, train_mask,
                           n_boot=50, verbose=True):
    """Bootstrap stability of survival MB features."""
    if verbose:
        print(f"\n  BOOTSTRAP STABILITY ({n_boot} resamples)")

    train_df = disc_df[train_mask]
    train_time = time[train_mask.values]
    train_event = event[train_mask.values]

    feat_counts = {f: 0 for f in feat_cols}
    rng = np.random.RandomState(SEED)

    for b in range(n_boot):
        idx = rng.choice(len(train_df), len(train_df), replace=True)
        boot_df = train_df.iloc[idx].reset_index(drop=True)
        boot_time = train_time[idx]
        boot_event = train_event[idx]

        try:
            G_b, _ = survival_ensemble_bn(boot_df, feat_cols, boot_time,
                                          boot_event, verbose=False)
            mb_b = survival_mb_discovery(G_b, boot_df, feat_cols, boot_time,
                                         boot_event, verbose=False)
            for f in mb_b["consensus"]:
                if f in feat_counts:
                    feat_counts[f] += 1
        except Exception:
            continue

    stability = {f: count / n_boot * 100
                 for f, count in feat_counts.items() if count > 0}
    stability = dict(sorted(stability.items(), key=lambda x: -x[1]))

    if verbose:
        for f, pct in stability.items():
            tag = "[ROBUST]" if pct >= 80 else "[moderate]" if pct >= 50 else "[weak]"
            print(f"    {f}: {pct:.0f}% {tag}")

    return stability


# ══════════════════════════════════════════════════════════════════════════════
# PART 7: COMPARE BINARY-PROXY vs SURVIVAL-AWARE
# ══════════════════════════════════════════════════════════════════════════════

def compare_binary_vs_survival(config, verbose=True):
    """Head-to-head comparison of binary-proxy MB vs survival-aware MB.

    Trains survival models (Cox PH, RSF, GBS) on both MBs and compares
    C-index on the test set.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import (RandomSurvivalForest,
                                 GradientBoostingSurvivalAnalysis)
    from sksurv.metrics import concordance_index_censored

    if verbose:
        print(f"\n{'='*70}")
        print(f"BINARY-PROXY vs SURVIVAL-AWARE MB — {config.name.upper()}")
        print(f"{'='*70}")

    # ── 1. Binary-proxy MB (existing pipeline) ───────────────────────────
    if verbose:
        print("\n--- Binary-Proxy MB ---")
    data_binary = load_survival_dataset(config, verbose=verbose)
    binary_mb = data_binary["mb_info"]["consensus"]
    n_binary = len(data_binary["train_df"]) + len(data_binary["test_df"])

    # ── 2. Survival-aware MB (new method) ─────────────────────────────────
    if verbose:
        print("\n--- Survival-Aware MB ---")
    disc_df, feat_cols, time, event, train_mask = load_all_patients(
        config, verbose=verbose
    )
    n_survival = len(disc_df)

    # Train-only for BN learning
    train_df_sa = disc_df[train_mask]
    train_time = time[train_mask.values]
    train_event = event[train_mask.values]

    G_sa, graphs_sa = survival_ensemble_bn(
        train_df_sa, feat_cols, train_time, train_event, verbose=verbose
    )
    smb_info = survival_mb_discovery(
        G_sa, train_df_sa, feat_cols, train_time, train_event, verbose=verbose
    )
    survival_mb = smb_info["consensus"]

    # ── 3. Print comparison ───────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*70}")
        print("MB COMPARISON")
        print(f"{'='*70}")
        print(f"  Binary-proxy MB ({len(binary_mb)}):   {sorted(binary_mb)}")
        print(f"  Survival-aware MB ({len(survival_mb)}): {sorted(survival_mb)}")
        overlap = set(binary_mb) & set(survival_mb)
        only_binary = set(binary_mb) - set(survival_mb)
        only_survival = set(survival_mb) - set(binary_mb)
        print(f"\n  Overlap:        {sorted(overlap)} ({len(overlap)})")
        print(f"  Binary-only:    {sorted(only_binary)} ({len(only_binary)})")
        print(f"  Survival-only:  {sorted(only_survival)} ({len(only_survival)})")
        jaccard = (len(overlap) / len(set(binary_mb) | set(survival_mb))
                   if (set(binary_mb) | set(survival_mb)) else 0)
        print(f"  Jaccard:        {jaccard:.3f}")
        print(f"\n  Patients used for BN learning:")
        print(f"    Binary-proxy:   {n_binary} (evaluable only)")
        print(f"    Survival-aware: {n_survival} (ALL patients)")
        print(f"    Gained:         {n_survival - n_binary} "
              f"({(n_survival - n_binary) / n_survival:.0%})")

    # ── 4. Evaluate both MBs with survival models ─────────────────────────
    # Use the full survival data (ALL patients) for evaluation
    y_train = data_binary["y_train_surv"]
    y_test = data_binary["y_test_surv"]
    train_df_eval = data_binary["train_df"]
    test_df_eval = data_binary["test_df"]

    model_factories = {
        "Cox PH": lambda: CoxPHSurvivalAnalysis(alpha=0.1),
        "RSF": lambda: RandomSurvivalForest(
            n_estimators=2000, max_depth=7, min_samples_leaf=4,
            min_samples_split=8, random_state=SEED, n_jobs=-1,
        ),
        "GBS": lambda: GradientBoostingSurvivalAnalysis(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=10, random_state=SEED,
        ),
    }

    results = {}
    for mb_name, mb_feats in [("Binary MB", binary_mb),
                               ("Survival MB", survival_mb)]:
        cols = [c for c in mb_feats if c in train_df_eval.columns]
        if not cols:
            if verbose:
                print(f"\n  {mb_name}: no features available in evaluable data")
            continue

        X_train = encode_for_survival(train_df_eval, cols)
        X_test = encode_for_survival(test_df_eval, cols)
        all_cols = sorted(set(X_train.columns) | set(X_test.columns))
        X_train = X_train.reindex(columns=all_cols, fill_value=0)
        X_test = X_test.reindex(columns=all_cols, fill_value=0)

        y_tr = make_y_surv(train_df_eval)
        y_te = make_y_surv(test_df_eval)

        mb_results = {}
        for m_name, factory in model_factories.items():
            try:
                model = factory()
                model.fit(X_train.values, y_tr)
                risk = model.predict(X_test.values)
                ci = concordance_bootstrap(y_te, risk, n_boot=500)
                mb_results[m_name] = ci
            except Exception as e:
                if verbose:
                    print(f"    {mb_name} / {m_name}: FAILED ({e})")
                mb_results[m_name] = {"cindex": np.nan,
                                       "ci_lower": np.nan, "ci_upper": np.nan}

        results[mb_name] = mb_results

    # ── 5. Evaluate Survival MB on ALL patients (full-data advantage) ────
    #    This arm trains on ALL training patients and tests on ALL test
    #    patients, showing the combined benefit of SA-BN features + more data.
    sa_cols = [c for c in survival_mb if c in disc_df.columns]
    if sa_cols:
        from sksurv.metrics import concordance_index_censored as _ci_c
        sa_train = disc_df[train_mask]
        sa_test = disc_df[~train_mask]
        X_tr_all = encode_for_survival(sa_train, sa_cols)
        X_te_all = encode_for_survival(sa_test, sa_cols)
        all_c = sorted(set(X_tr_all.columns) | set(X_te_all.columns))
        X_tr_all = X_tr_all.reindex(columns=all_c, fill_value=0)
        X_te_all = X_te_all.reindex(columns=all_c, fill_value=0)
        y_tr_all = np.array(
            [(bool(e), float(t)) for e, t in
             zip(event[train_mask.values], time[train_mask.values])],
            dtype=[("event", bool), ("time", float)],
        )
        y_te_all = np.array(
            [(bool(e), float(t)) for e, t in
             zip(event[~train_mask.values], time[~train_mask.values])],
            dtype=[("event", bool), ("time", float)],
        )

        mb_results_all = {}
        for m_name, factory in model_factories.items():
            try:
                model = factory()
                model.fit(X_tr_all.values, y_tr_all)
                risk = model.predict(X_te_all.values)
                ci = concordance_bootstrap(y_te_all, risk, n_boot=500)
                mb_results_all[m_name] = ci
            except Exception as e:
                if verbose:
                    print(f"    SA-MB(All) / {m_name}: FAILED ({e})")
                mb_results_all[m_name] = {"cindex": np.nan,
                                           "ci_lower": np.nan, "ci_upper": np.nan}
        results["SA-MB (All)"] = mb_results_all

    # ── 6. Print evaluation table ─────────────────────────────────────────
    if verbose:
        print(f"\n{'='*70}")
        print("SURVIVAL MODEL EVALUATION")
        print(f"{'='*70}")
        print(f"{'MB Type':<18} {'Model':<10} {'C-index':>8} {'95% CI':>18} {'N_test':>8}")
        print("-" * 66)
        for mb_name, mb_res in results.items():
            n_te = (len(test_df_eval) if mb_name != "SA-MB (All)"
                    else (~train_mask).sum())
            for m_name, ci in mb_res.items():
                c = ci["cindex"]
                lo, hi = ci["ci_lower"], ci["ci_upper"]
                ci_str = f"[{lo:.3f}-{hi:.3f}]" if not np.isnan(lo) else ""
                print(f"{mb_name:<18} {m_name:<10} {c:>8.3f} {ci_str:>18} {n_te:>8}")

    return {
        "binary_mb": binary_mb,
        "survival_mb": survival_mb,
        "smb_info": smb_info,
        "G_consensus_sa": G_sa,
        "graphs_sa": graphs_sa,
        "n_binary": n_binary,
        "n_survival": n_survival,
        "model_results": results,
        "disc_df": disc_df,
        "train_mask": train_mask,
        "time": time,
        "event": event,
        "feat_cols": feat_cols,
        "config": config,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 8: MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(config=None, configs=None):
    """Run full survival-aware BN pipeline with comparison.

    Args:
        config: single DatasetConfig (runs one cohort)
        configs: dict of {name: DatasetConfig} (runs multiple cohorts)

    Returns:
        dict of {cohort_name: comparison_results}
    """
    if configs is None:
        if config is None:
            configs = {"radcure": RADCURE_CONFIG, "hancock": HANCOCK_CONFIG}
        else:
            configs = {config.name: config}

    all_results = {}
    for name, cfg in configs.items():
        print(f"\n{'#'*70}")
        print(f"# SURVIVAL-AWARE BN — {name.upper()}")
        print(f"{'#'*70}")
        all_results[name] = compare_binary_vs_survival(cfg, verbose=True)

    # ── Cross-cohort summary ──────────────────────────────────────────────
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-COHORT SUMMARY")
        print(f"{'='*70}")
        print(f"{'Cohort':<12} {'Binary MB':>12} {'Survival MB':>14} "
              f"{'Overlap':>10} {'Patients gained':>18}")
        print("-" * 68)
        for name, res in all_results.items():
            b_mb = res["binary_mb"]
            s_mb = res["survival_mb"]
            overlap = set(b_mb) & set(s_mb)
            gained = res["n_survival"] - res["n_binary"]
            print(f"{name.upper():<12} {len(b_mb):>12} {len(s_mb):>14} "
                  f"{len(overlap):>10} {gained:>14} "
                  f"({gained/res['n_survival']:.0%})")

    return all_results


if __name__ == "__main__":
    results = main()
