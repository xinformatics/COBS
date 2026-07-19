"""
common.py — Generalized data pipeline for multi-cohort BN analysis
===================================================================
Supports HANCOCK (surgical), OPC (radiation-treated OPC), and MDA
(radiation-treated HNSCC) datasets with unified preprocessing,
feature engineering, and harmonization.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
SURVIVAL_THRESHOLDS_YEARS = [1, 2, 3, 5]
MAX_INDEGREE = 5
N_BOOTSTRAP = 100
EDGE_STABILITY_THRESHOLD = 0.60

BASE_DIR = Path(__file__).parent
FIG_BASE = BASE_DIR / "figures"

# Okabe-Ito palette
CB_BLUE   = "#0072B2"
CB_ORANGE = "#D55E00"
CB_GREEN  = "#009E73"
CB_YELLOW = "#F0E442"
CB_SKY    = "#56B4E9"
CB_VERMIL = "#E69F00"
CB_PINK   = "#CC79A7"
CB_BLACK  = "#000000"


def setup_plotting():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11, "figure.titlesize": 16,
    })


# ── Dataset Configuration ────────────────────────────────────────────────────

@dataclass
class ColumnMap:
    """Maps dataset-specific column names to canonical names."""
    patient_id: Optional[str] = None
    age: Optional[str] = None
    sex: Optional[str] = None
    site: Optional[str] = None
    t_stage: Optional[str] = None
    n_stage: Optional[str] = None
    hpv: Optional[str] = None
    smoking: Optional[str] = None
    grade: Optional[str] = None
    time_days: Optional[str] = None
    event: Optional[str] = None
    year: Optional[str] = None


@dataclass
class DatasetConfig:
    name: str
    csv_path: Path
    columns: ColumnMap
    split_strategy: str = "temporal"  # "temporal" or "random"
    split_year: Optional[int] = None
    split_ratio: float = 0.7
    primary_threshold_years: int = 2
    treatment_context: str = "surgical"  # "surgical" or "radiation"
    has_invasion: bool = False
    has_hpv: bool = True
    # Dataset-specific categorical and continuous feature candidates
    cat_candidates: List[str] = field(default_factory=list)
    cont_candidates: List[str] = field(default_factory=list)
    # Interaction features to construct
    interaction_specs: List[str] = field(default_factory=list)
    # Features considered essential (not removed by MI filter)
    essential_features: List[str] = field(default_factory=list)

    @property
    def fig_dir(self):
        return FIG_BASE / self.name

    def ensure_dirs(self):
        for sub in ["bn", "causal", "ood"]:
            (self.fig_dir / sub).mkdir(parents=True, exist_ok=True)


# ── Predefined configs ───────────────────────────────────────────────────────

HANCOCK_CONFIG_ORIGINAL = DatasetConfig(
    name="hancock_original",
    csv_path=BASE_DIR / "hancock.csv",
    columns=ColumnMap(
        patient_id="patient_id",
        age="age_at_initial_diagnosis",
        sex="sex",
        site="primary_tumor_site",
        t_stage="pT_grouped",
        n_stage="pN_grouped",
        hpv="hpv_association_p16",
        smoking="smoking_status",
        grade="grade_grouped",
        time_days="days_to_last_information",
        event=None,  # derived from survival_status
        year="year_of_initial_diagnosis",
    ),
    split_strategy="temporal",
    split_year=2014,
    primary_threshold_years=2,
    treatment_context="surgical",
    has_invasion=True,
    has_hpv=True,
    cat_candidates=[
        "primary_tumor_site", "pT_grouped", "pN_grouped", "grade_grouped",
        "hpv_association_p16", "sex", "smoking_status", "histologic_type_grp",
        "lymphovascular_invasion_L", "vascular_invasion_V",
        "perineural_invasion_Pn", "adjuvant_radiotherapy",
    ],
    cont_candidates=[
        "age_at_initial_diagnosis", "infiltration_depth_in_mm",
        "number_of_positive_lymph_nodes",
    ],
    interaction_specs=["HPVxSite", "TxN", "invasion_burden"],
    essential_features=[
        "primary_tumor_site", "pT_grouped", "pN_grouped",
        "hpv_association_p16", "grade_grouped",
        "lymphovascular_invasion_L", "perineural_invasion_Pn",
    ],
)

# HANCOCK config — clinically comprehensive feature set (n=763, imputed).
# Standard clinical/pathological features + 2 blood biomarkers (Hb, NLR).
# Removed redundant blood features: PLR (r=0.60 with NLR), leukocytes,
#   platelets (r=0.64 with PLR); resected LN count (surgical quality, not
#   biology); resection_status (90% R0); closest margin (22% missing).
# 15 features → comparable to RADCURE's 13.
HANCOCK_CONFIG = DatasetConfig(
    name="hancock",
    csv_path=BASE_DIR / "hancock_rebuilt.csv",
    columns=ColumnMap(
        patient_id="patient_id",
        age="age_at_initial_diagnosis",
        sex="sex",
        site="primary_tumor_site",
        t_stage="pT_grouped",
        n_stage="pN_grouped",
        hpv="hpv_association_p16",
        smoking="smoking_status",
        grade="grade_grouped",
        time_days="days_to_last_information",
        event=None,  # derived from survival_status
        year="year_of_initial_diagnosis",
    ),
    split_strategy="temporal",
    split_year=2016,
    primary_threshold_years=2,
    treatment_context="surgical",
    has_invasion=True,
    has_hpv=True,
    cat_candidates=[
        "primary_tumor_site", "pT_grouped", "pN_grouped", "grade_grouped",
        "hpv_association_p16", "sex", "smoking_status",
        "adjuvant_radiotherapy",
    ],
    cont_candidates=[
        "age_at_initial_diagnosis", "infiltration_depth_in_mm",
        "number_of_positive_lymph_nodes",
        "blood_hemoglobin", "nlr",
    ],
    interaction_specs=["invasion_burden", "TxN"],
    essential_features=[
        "primary_tumor_site", "pT_grouped", "pN_grouped",
        "hpv_association_p16",
    ],
)

OPC_CONFIG = DatasetConfig(
    name="opc",
    csv_path=BASE_DIR / "Radiomics_Outcome_Prediction_in_OPC_ASRM_corrected.csv",
    columns=ColumnMap(
        patient_id="TCIA Radiomics dummy ID of To_Submit_Final",
        age="Age at Diag",
        sex="Gender",
        site="Cancer subsite of origin",
        t_stage="T-category",
        n_stage="N-category",
        hpv="HPV Status",
        smoking="Smoking status",
        grade=None,
        time_days="Overall survival_duration of Merged updated ASRM V2",
        event="Vital status",
        year=None,
    ),
    split_strategy="random",
    split_ratio=0.7,
    primary_threshold_years=5,
    treatment_context="radiation",
    has_invasion=False,
    has_hpv=True,
    cat_candidates=[
        "sex", "smoking_status", "hpv_status", "t_stage", "n_stage",
        "ajcc_stage", "treatment_type", "neck_dissection",
    ],
    cont_candidates=["age", "smoking_py", "rt_dose", "rt_duration"],
    interaction_specs=["TxN"],
    essential_features=["t_stage", "n_stage", "hpv_status"],
)

MDA_CONFIG = DatasetConfig(
    name="mda",
    csv_path=BASE_DIR / "HNSCC-MDA.csv",
    columns=ColumnMap(
        patient_id="TCIA PatientID",
        age="Age",
        sex="Sex",
        site="Site",
        t_stage="T",
        n_stage="N",
        hpv=None,
        smoking="Smoking History",
        grade="Grade",
        time_days="Follow up duration (day)",
        event="Alive or Dead",
        year=None,  # will parse from Offset Date of Diagnosis
    ),
    split_strategy="random",
    split_ratio=0.7,
    primary_threshold_years=5,
    treatment_context="radiation",
    has_invasion=False,
    has_hpv=False,
    cat_candidates=[
        "sex", "site", "grade", "t_stage", "n_stage",
        "overall_stage", "smoking", "surgery_binary",
        "sarcopenia_status",
    ],
    cont_candidates=["age", "rt_dose"],
    interaction_specs=["TxN"],
    essential_features=["t_stage", "n_stage", "site"],
)

RADIATION_CONFIG = DatasetConfig(
    name="radiation",
    csv_path=BASE_DIR / "radiation_merged.csv",
    columns=ColumnMap(
        patient_id="patient_id",
        age="age",
        sex="sex",
        site="site",
        t_stage="t_stage",
        n_stage="n_stage",
        hpv=None,
        smoking="smoking_status",
        grade=None,
        time_days="time_days",
        event="event",
        year=None,
    ),
    split_strategy="random",
    split_ratio=0.7,
    primary_threshold_years=5,
    treatment_context="radiation",
    has_invasion=False,
    has_hpv=False,
    cat_candidates=[
        "sex", "site", "t_stage", "n_stage", "smoking_status",
    ],
    cont_candidates=["age", "rt_dose"],
    interaction_specs=["TxN"],
    essential_features=["t_stage", "n_stage", "site"],
)

RADCURE_CONFIG = DatasetConfig(
    name="radcure",
    csv_path=BASE_DIR / "shah_replication.csv",
    columns=ColumnMap(
        patient_id="patient_id",
        age="age",
        sex="sex",
        site="site",
        t_stage="t_stage",
        n_stage="n_stage",
        hpv="hpv_status",
        smoking="smoking_py",
        grade=None,
        time_days="time_days",
        event="event",
        year="_rt_year",
    ),
    split_strategy="temporal",
    split_year=2007,
    primary_threshold_years=2,
    treatment_context="radiation",
    has_invasion=False,
    has_hpv=True,
    cat_candidates=[
        "sex", "site", "t_stage", "n_stage", "hpv_status",
        "ecog_ps", "stage_overall", "pathology", "tx_modality",
    ],
    cont_candidates=["age", "smoking_py", "gtvp_cm3"],
    interaction_specs=["TxN"],
    essential_features=["site", "t_stage", "n_stage", "hpv_status"],
)


# ── Survival endpoint construction ───────────────────────────────────────────

def build_survival_endpoints(df, config):
    """Build binary SVy{1,2,3,5} endpoints with proper censoring."""
    df = df.copy()

    # Extract event indicator
    if "event" in df.columns and df["event"].dtype in [int, float, np.int64, np.float64]:
        df["_event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
    elif config.columns.event:
        ecol = config.columns.event
        raw = df[ecol].astype(str).str.strip().str.lower()
        df["_event"] = raw.isin(["dead", "deceased", "1", "0"]).astype(int)
        # Handle MDA where "Dead" → event=1, "Alive" → event=0
        if config.name == "mda":
            df["_event"] = raw.isin(["dead", "deceased"]).astype(int)
        elif config.name == "opc":
            df["_event"] = raw.isin(["dead"]).astype(int)
        else:
            df["_event"] = raw.isin(["dead", "deceased", "1"]).astype(int)
    else:
        # HANCOCK: derive from survival_status
        df["_event"] = df["survival_status"].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ["deceased", "dead", "1"] else 0
        )

    # Extract time in days
    tcol = config.columns.time_days
    df["_time_days"] = pd.to_numeric(df[tcol], errors="coerce")

    # For MDA: time may be in days already, verify
    if config.name == "mda":
        # Follow up duration (day) is already in days
        pass

    # Build multi-threshold endpoints
    for yrs in SURVIVAL_THRESHOLDS_YEARS:
        days_thresh = yrs * 365.25
        col = f"SVy{yrs}"

        def _classify(row, dt=days_thresh):
            t, e = row["_time_days"], row["_event"]
            if pd.isna(t):
                return np.nan
            if e == 1 and t < dt:
                return 0
            if t >= dt:
                return 1
            return np.nan  # alive but insufficient FU

        df[col] = df.apply(_classify, axis=1)

    # Conservative censoring (Shah's approach) for comparison
    primary_days = config.primary_threshold_years * 365.25
    def _classify_shah(row):
        t, e = row["_time_days"], row["_event"]
        if pd.isna(t):
            return np.nan
        if e == 1 and t < primary_days:
            return 0
        if t >= primary_days:
            return 1
        return 0  # conservative: insufficient FU → non-survivor
    df["SVy_shah"] = df.apply(_classify_shah, axis=1)

    # Store cleaned time/event
    df["time_days"] = df["_time_days"]
    df["event"] = df["_event"]
    df["time_months"] = df["_time_days"] / 30.44

    df.drop(columns=["_event", "_time_days"], inplace=True, errors="ignore")
    return df


# ── Dataset-specific preprocessing ───────────────────────────────────────────

def _preprocess_hancock(df):
    """HANCOCK-specific feature engineering."""
    # Simplify histologic type
    if "histologic_type" in df.columns:
        def _simplify(x):
            if pd.isna(x): return np.nan
            s = str(x).strip().lower()
            if "keratinizing" in s: return "SCC_Kerat"
            elif "basaloid" in s: return "SCC_Basaloid"
            else: return "SCC_Other"
        df["histologic_type_grp"] = df["histologic_type"].apply(_simplify)
    return df


def _preprocess_opc(df):
    """OPC-specific feature engineering."""
    # Standardize column names
    df["sex"] = df["Gender"].astype(str).str.lower()
    df["age"] = pd.to_numeric(df["Age at Diag"], errors="coerce")

    # Smoking
    df["smoking_status"] = df["Smoking status"].astype(str).str.lower()
    df["smoking_py"] = pd.to_numeric(df["Smoking status (Packs-Years)"], errors="coerce").fillna(0)

    # HPV
    hpv_raw = df["HPV Status"].astype(str).str.strip().str.upper()
    df["hpv_status"] = hpv_raw.map({"P": "positive", "N": "negative"}).fillna("unknown")

    # T stage
    df["t_stage"] = pd.to_numeric(df["T-category"], errors="coerce").apply(
        lambda x: f"t{int(x)}" if pd.notna(x) and 1 <= x <= 4 else np.nan
    )

    # N stage
    df["n_stage"] = pd.to_numeric(df["N-category"], errors="coerce").apply(
        lambda x: "n0" if x == 0 else ("n1" if x == 1 else ("n2+" if x >= 2 else np.nan))
        if pd.notna(x) else np.nan
    )

    # AJCC stage
    df["ajcc_stage"] = df["AJCC Stage (7th edition)"].astype(str).str.strip()

    # Treatment type
    tc = df["Therapeutic Combination"].astype(str).str.lower()
    df["treatment_type"] = tc.apply(lambda x:
        "ccrt" if "concurrent chemo" in x else
        ("induction_ccrt" if "induction" in x and "concurrent" in x else
         ("rt_alone" if "radiation alone" in x else
          ("induction_rt" if "induction" in x else "other")))
    )

    # Neck dissection
    df["neck_dissection"] = df["Neck Dissection after IMRT"].astype(str).str.lower().map(
        {"yes": "yes", "no": "no"}).fillna("unknown")

    # RT parameters
    df["rt_dose"] = pd.to_numeric(df["Total prescribed Radiation treatment dose"], errors="coerce")
    df["rt_duration"] = pd.to_numeric(df["Radiation Treatment_duration"], errors="coerce")

    # Site: all OPC, simplify subsites
    site_raw = df["Cancer subsite of origin"].astype(str).str.lower()
    df["site"] = site_raw.apply(lambda x:
        "base_of_tongue" if "base of tongue" in x else
        ("tonsil" if "tonsil" in x else
         ("soft_palate" if "soft palate" in x else "oropharynx_nos"))
    )

    return df


def _preprocess_mda(df):
    """MDA-specific feature engineering."""
    # Standardize column names
    df["sex"] = df["Sex"].astype(str).str.lower()
    df["age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Site
    site_raw = df["Site"].astype(str).str.lower()
    df["site"] = site_raw.apply(lambda x:
        "oropharynx" if "oropharynx" in x else
        ("larynx" if any(k in x for k in ["larynx", "glottis", "supraglottic", "subglottic"]) else
         ("hypopharynx" if any(k in x for k in ["hypopharynx", "pyriform"]) else
          ("oral_cavity" if any(k in x for k in ["oral", "tongue", "floor of mouth", "buccal", "alveolar"]) else
           "other")))
    )

    # Grade simplification
    grade_raw = df["Grade"].astype(str).str.lower()
    df["grade"] = grade_raw.apply(lambda x:
        "well" if "well" in x and "moderate" not in x else
        ("moderate" if "moderate" in x else
         ("poor" if "poor" in x else
          ("undiff" if "undiff" in x else np.nan)))
    )

    # T stage simplification
    t_raw = df["T"].astype(str).str.lower().str.strip()
    df["t_stage"] = t_raw.apply(lambda x:
        "t1" if x in ["1", "t1"] else
        ("t2" if x in ["2", "t2"] else
         ("t3" if x in ["3", "t3"] else
          ("t4" if any(x.startswith(k) for k in ["4", "t4"]) else np.nan)))
    )

    # N stage simplification
    n_raw = df["N"].astype(str).str.lower().str.strip()
    df["n_stage"] = n_raw.apply(lambda x:
        "n0" if x in ["0", "n0"] else
        ("n1" if x in ["1", "n1"] else
         ("n2+" if any(x.startswith(k) for k in ["2", "n2", "3", "n3"]) else np.nan))
    )

    # Overall stage
    df["overall_stage"] = df["Stage"].astype(str).str.strip()

    # Smoking: 0=never, 1=former, 2=current
    smoke_raw = df["Smoking History"].astype(str).str.strip()
    df["smoking"] = smoke_raw.map({"0": "never", "1": "former", "2": "current"}).fillna("unknown")

    # Surgery binary
    surg_raw = df["Surgery Summary"].astype(str).str.lower()
    df["surgery_binary"] = surg_raw.apply(lambda x: "yes" if x != "no" and x != "nan" else "no")

    # Sarcopenia status
    if "PreRT Skeletal Muscle status" in df.columns:
        df["sarcopenia_status"] = df["PreRT Skeletal Muscle status"].astype(str).str.lower().apply(
            lambda x: "depleted" if "deplet" in x else ("normal" if "normal" in x or "not" in x else np.nan)
        )

    # RT dose
    df["rt_dose"] = pd.to_numeric(df["RT Total Dose (Gy)"], errors="coerce")

    # Try to parse year from diagnosis date for temporal split
    if "Offset Date of Diagnosis" in df.columns:
        dates = pd.to_datetime(df["Offset Date of Diagnosis"], errors="coerce")
        df["_diag_year"] = dates.dt.year

    return df


def _preprocess_radiation(df):
    """Merged radiation cohort — columns already in unified schema from merge_opc_mda.py."""
    # Columns are already: patient_id, source, age, sex, site, t_stage, n_stage,
    # smoking_status, rt_dose, time_days, event
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["rt_dose"] = pd.to_numeric(df["rt_dose"], errors="coerce")
    df["time_days"] = pd.to_numeric(df["time_days"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
    # Ensure string types for categoricals
    for col in ["sex", "site", "t_stage", "n_stage", "smoking_status"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def _preprocess_radcure(df):
    """RADCURE preprocessing — ported from replicate_shah.py."""
    # Age
    df["age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Sex
    df["sex"] = df["Sex"].astype(str).str.lower().str.strip()

    # ECOG PS
    def _clean_ecog(val):
        if pd.isna(val): return np.nan
        val = str(val).strip()
        if val == "Unknown": return np.nan
        for i in range(5):
            if str(i) in val: return i
        return np.nan
    df["ecog_ps"] = df["ECOG PS"].apply(_clean_ecog)
    ecog_median = df["ecog_ps"].median()
    df["ecog_ps"] = df["ecog_ps"].fillna(ecog_median).astype(int).astype(str)

    # Smoking PY (continuous)
    def _clean_smoking(val):
        if pd.isna(val): return np.nan
        val = str(val).strip().lower()
        if val in ["na", "unknown", ""]: return np.nan
        val = val.replace(">", "").replace("<", "")
        try: return float(val)
        except ValueError: return np.nan
    df["smoking_py"] = df["Smoking PY"].apply(_clean_smoking)
    df["smoking_py"] = df["smoking_py"].fillna(df["smoking_py"].median())

    # T stage
    def _clean_t(val):
        if pd.isna(val): return np.nan
        val = str(val).strip()
        if val in ["TX", "Tis"]: return np.nan
        for i in range(1, 5):
            if f"T{i}" in val: return f"t{i}"
        if "T0" in val or "rT0" in val: return np.nan
        return np.nan
    df["t_stage"] = df["T"].apply(_clean_t)

    # N stage
    def _clean_n(val):
        if pd.isna(val): return np.nan
        val = str(val).strip()
        if val == "NX": return np.nan
        if "N0" in val: return "n0"
        if "N1" in val: return "n1"
        if "N2" in val or "N3" in val: return "n2+"
        return np.nan
    df["n_stage"] = df["N"].apply(_clean_n)

    # Disease site
    def _clean_site(val):
        s = str(val).strip().lower()
        if "oropharynx" in s: return "oropharynx"
        if "larynx" in s: return "larynx"
        if "nasopharynx" in s: return "nasopharynx"
        if "hypopharynx" in s: return "hypopharynx"
        if any(k in s for k in ["nasal", "paranasal", "salivary"]): return "other"
        if any(k in s for k in ["oral", "lip"]): return "oral cavity"
        return "other"
    df["site"] = df["Ds Site"].apply(_clean_site)

    # HPV status
    def _clean_hpv(val):
        if pd.isna(val): return "unknown"
        val = str(val).strip().lower()
        if "positive" in val: return "positive"
        if "negative" in val: return "negative"
        return "unknown"
    df["hpv_status"] = df["HPV"].apply(_clean_hpv)

    # Overall stage
    def _clean_stage(val):
        if pd.isna(val): return np.nan
        val = str(val).strip().upper()
        if val in ["X", "UNKNOWN"]: return np.nan
        if val.startswith("IV"): return "IV"
        if val.startswith("III"): return "III"
        if val.startswith("II"): return "II"
        if val.startswith("I") or val == "0": return "I"
        return np.nan
    df["stage_overall"] = df["Stage"].apply(_clean_stage)

    # Pathology
    def _clean_path(val):
        s = str(val).strip().lower()
        if "squamous" in s or "verrucous" in s: return "squamous"
        if "npc" in s: return np.nan
        if "adeno" in s or "basal" in s: return "adeno_other"
        return np.nan
    df["pathology"] = df["Path"].apply(_clean_path)

    # Treatment modality
    def _clean_tx(val):
        val = str(val).strip()
        if "ChemoRT" in val: return "chemort"
        if "RT alone" in val: return "rt_alone"
        if "EGFRI" in val: return "rt_egfri"
        if "Postop" in val: return "postop_rt"
        return "other"
    df["tx_modality"] = df["Tx Modality"].apply(_clean_tx)

    # GTV volume (mm³ → cm³)
    df["gtvp_cm3"] = pd.to_numeric(df["original_shape_VoxelVolume"], errors="coerce") / 1000.0

    # ── Time and event (CRITICAL: Length FU is from diagnosis, not RT start) ──
    df["_rt_start_dt"] = pd.to_datetime(df["RT Start"], errors="coerce")
    df["_rt_year"] = df["_rt_start_dt"].dt.year

    # Compute diagnosis-to-RT gap from dead patients with known dates
    dead_mask = (df["Status"] == "Dead") & df["Date of Death"].notna()
    df.loc[dead_mask, "_death_dt"] = pd.to_datetime(df.loc[dead_mask, "Date of Death"], errors="coerce")
    df.loc[dead_mask, "_surv_from_rt_days"] = (
        df.loc[dead_mask, "_death_dt"] - df.loc[dead_mask, "_rt_start_dt"]
    ).dt.days
    df.loc[dead_mask, "_fu_from_diag_days"] = pd.to_numeric(df.loc[dead_mask, "Length FU"], errors="coerce") * 365.0
    df.loc[dead_mask, "_gap_days"] = df.loc[dead_mask, "_fu_from_diag_days"] - df.loc[dead_mask, "_surv_from_rt_days"]
    median_gap_days = df.loc[dead_mask, "_gap_days"].median()
    if pd.isna(median_gap_days):
        median_gap_days = 43.0  # fallback from Shah replication

    # Compute time_days (from RT start) and event
    def _compute_time(row):
        if row["Status"] == "Dead" and pd.notna(row.get("Date of Death")):
            rt = pd.to_datetime(row["RT Start"], errors="coerce")
            death = pd.to_datetime(row["Date of Death"], errors="coerce")
            if pd.notna(rt) and pd.notna(death):
                return (death - rt).days
        # Alive or missing death date: adjust Length FU
        fu_years = pd.to_numeric(row.get("Length FU"), errors="coerce")
        if pd.notna(fu_years):
            return fu_years * 365.0 - median_gap_days
        return np.nan
    df["time_days"] = df.apply(_compute_time, axis=1)
    df["event"] = (df["Status"] == "Dead").astype(int)

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_") and c != "_rt_year"]
    df.drop(columns=temp_cols, inplace=True, errors="ignore")

    return df


# ── Generic preprocessing pipeline ───────────────────────────────────────────

def _discretize_yj_gmm(df, cont_cols, train_mask):
    """Yeo-Johnson + GMM discretization of continuous variables."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import PowerTransformer

    df = df.copy()
    for col in cont_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_raw"] = vals
        train_vals = vals[train_mask].dropna().values.reshape(-1, 1)
        if len(train_vals) < 20:
            edges = np.quantile(train_vals, [0, 0.33, 0.66, 1.0])
            labels = [f"{col}_low", f"{col}_mid", f"{col}_high"]
            df[col] = pd.cut(vals, bins=np.unique(edges),
                             labels=labels[:len(np.unique(edges))-1],
                             include_lowest=True).astype(str)
            df[col] = df[col].replace("nan", np.nan)
            continue

        pt = PowerTransformer(method="yeo-johnson")
        try:
            transformed = pt.fit_transform(train_vals)
        except Exception:
            transformed = train_vals

        best_aic, best_n = np.inf, 2
        for n in range(2, min(6, len(train_vals) // 15 + 1)):
            gmm = GaussianMixture(n_components=n, random_state=SEED)
            gmm.fit(transformed)
            aic = gmm.aic(transformed)
            if aic < best_aic:
                best_aic, best_n = aic, n

        gmm = GaussianMixture(n_components=best_n, random_state=SEED)
        gmm.fit(transformed)
        means = sorted(gmm.means_.flatten())
        edges_t = [-np.inf] + [(means[i]+means[i+1])/2 for i in range(len(means)-1)] + [np.inf]
        edges_orig = []
        for e in edges_t:
            if np.isinf(e):
                edges_orig.append(e)
            else:
                edges_orig.append(float(pt.inverse_transform([[e]])[0][0]))
        edges_orig = sorted(edges_orig)
        labels = [f"{col}_bin{i}" for i in range(len(edges_orig)-1)]
        df[col] = pd.cut(vals, bins=edges_orig, labels=labels, include_lowest=True).astype(str)
        df[col] = df[col].replace("nan", np.nan)
    return df


def _add_interactions(df, config):
    """Add interaction features based on config specs."""
    interactions = []

    for spec in config.interaction_specs:
        if spec == "HPVxSite":
            hpv_col = "hpv_association_p16" if "hpv_association_p16" in df.columns else "hpv_status"
            site_col = next((c for c in ["primary_tumor_site", "site"] if c in df.columns), None)
            if hpv_col in df.columns and site_col in df.columns:
                df["HPVxSite"] = df[hpv_col].astype(str) + "_" + df[site_col].astype(str)
                interactions.append("HPVxSite")

        elif spec == "TxN":
            t_col = next((c for c in ["pT_grouped", "t_stage"] if c in df.columns), None)
            n_col = next((c for c in ["pN_grouped", "n_stage"] if c in df.columns), None)
            if t_col and n_col:
                df["TxN"] = df[t_col].astype(str) + "_" + df[n_col].astype(str)
                interactions.append("TxN")

        elif spec == "invasion_burden":
            inv_cols = ["lymphovascular_invasion_L", "vascular_invasion_V", "perineural_invasion_Pn"]
            avail = [c for c in inv_cols if c in df.columns]
            if len(avail) >= 2:
                df["invasion_burden"] = 0
                for c in avail:
                    df["invasion_burden"] += df[c].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip().lower()
                                  in ["yes", "l1", "v1", "pn1", "1", "true"] else 0
                    )
                df["invasion_burden"] = df["invasion_burden"].astype(str)
                interactions.append("invasion_burden")

    return df, interactions


def preprocess(df, config, train_mask):
    """Full preprocessing pipeline: feature selection, discretization, MI filter, impute.

    Returns (bn_df, feat_cols).
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    bn_df = df.copy()

    # Normalize categoricals
    cat_cols = [c for c in config.cat_candidates if c in bn_df.columns and bn_df[c].dropna().nunique() >= 2]
    cont_cols = [c for c in config.cont_candidates if c in bn_df.columns]

    for col in cat_cols:
        bn_df[col] = bn_df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else np.nan)
        bn_df[col] = bn_df[col].replace({"nan": np.nan, "none": np.nan, "": np.nan})

    # Discretize continuous
    bn_df = _discretize_yj_gmm(bn_df, cont_cols, train_mask)

    # Add interactions
    bn_df, interaction_cols = _add_interactions(bn_df, config)

    # Target
    target = f"SVy{config.primary_threshold_years}"
    bn_df[target] = bn_df[target].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)

    # Assemble feature list
    all_feat_cols = cat_cols + cont_cols + interaction_cols
    to_drop = []
    for col in all_feat_cols:
        if col not in bn_df.columns:
            to_drop.append(col); continue
        miss = bn_df[col].isna().mean()
        nuniq = bn_df[col].dropna().nunique()
        if miss > 0.40 or nuniq < 2 or nuniq > 20:
            to_drop.append(col)
    feat_cols = [c for c in all_feat_cols if c not in to_drop and c in bn_df.columns]

    # Keep meta columns
    meta_cols = ["time_days", "time_months", "event", "SVy_shah"]
    raw_cols = [f"{c}_raw" for c in cont_cols if f"{c}_raw" in bn_df.columns]
    for yrs in SURVIVAL_THRESHOLDS_YEARS:
        scol = f"SVy{yrs}"
        if scol in bn_df.columns:
            meta_cols.append(scol)
    if config.columns.year and config.columns.year in bn_df.columns:
        meta_cols.append(config.columns.year)
    if config.columns.patient_id and config.columns.patient_id in bn_df.columns:
        meta_cols.append(config.columns.patient_id)
    if "_diag_year" in bn_df.columns:
        meta_cols.append("_diag_year")

    keep = list(dict.fromkeys([c for c in meta_cols if c in bn_df.columns] + feat_cols + raw_cols + [target]))
    bn_df = bn_df[[c for c in keep if c in bn_df.columns]]
    bn_df = bn_df.dropna(subset=[target])

    # MI filter: remove bottom 25% unless essential
    essential = set(config.essential_features)
    complete_mask = bn_df[feat_cols].notna().all(axis=1)
    if complete_mask.sum() > 50 and len(feat_cols) > 4:
        X_mi = pd.DataFrame()
        for c in feat_cols:
            le = LabelEncoder()
            X_mi[c] = le.fit_transform(bn_df.loc[complete_mask, c].astype(str))
        y_mi = bn_df.loc[complete_mask, target].astype(int).values
        mi = mutual_info_classif(X_mi, y_mi, random_state=SEED)
        mi_s = pd.Series(mi, index=feat_cols).sort_values()
        threshold = mi_s.quantile(0.25)
        low_mi_removable = [c for c in mi_s[mi_s < threshold].index if c not in essential]
        if low_mi_removable:
            feat_cols = [c for c in feat_cols if c not in low_mi_removable]

    # Impute missing with training-set mode
    # Align train_mask to bn_df's (possibly filtered) index
    bn_train_mask = train_mask.reindex(bn_df.index, fill_value=False)
    for col in feat_cols:
        miss = bn_df[col].isna().sum()
        if miss > 0:
            train_mode = bn_df.loc[bn_train_mask, col].mode()
            if len(train_mode) > 0:
                bn_df[col] = bn_df[col].fillna(train_mode.iloc[0])

    bn_df = bn_df.dropna(subset=feat_cols + [target])
    return bn_df, feat_cols


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(config, verbose=True):
    """Load, preprocess, and split a dataset.

    Returns (train_df, test_df, feat_cols, target).
    """
    import random
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(config.csv_path)
    if verbose:
        print(f"Loaded {config.name.upper()}: {len(df)} x {df.shape[1]}")

    # Dataset-specific preprocessing
    if config.name == "hancock":
        df = _preprocess_hancock(df)
    elif config.name == "opc":
        df = _preprocess_opc(df)
    elif config.name == "mda":
        df = _preprocess_mda(df)
    elif config.name == "radiation":
        df = _preprocess_radiation(df)
    elif config.name == "radcure":
        df = _preprocess_radcure(df)

    # Build survival endpoints
    df = build_survival_endpoints(df, config)

    target = f"SVy{config.primary_threshold_years}"

    # Determine train mask for preprocessing
    if config.split_strategy == "temporal" and config.split_year:
        year_col = config.columns.year
        if year_col and year_col in df.columns:
            df["_split_year"] = pd.to_numeric(df[year_col], errors="coerce")
        elif "_diag_year" in df.columns:
            df["_split_year"] = df["_diag_year"]
        else:
            df["_split_year"] = np.nan
        train_mask = df["_split_year"] <= config.split_year
        if train_mask.sum() < 30 or (~train_mask).sum() < 30:
            if verbose:
                print(f"  Temporal split too imbalanced, falling back to random 70/30")
            config.split_strategy = "random"

    if config.split_strategy == "random":
        from sklearn.model_selection import train_test_split
        evaluable = df[target].notna()
        idx_eval = df.index[evaluable]
        y_eval = df.loc[idx_eval, target]
        try:
            train_idx, test_idx = train_test_split(
                idx_eval, test_size=1-config.split_ratio,
                random_state=SEED, stratify=y_eval
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                idx_eval, test_size=1-config.split_ratio, random_state=SEED
            )
        train_mask = pd.Series(df.index.isin(train_idx), index=df.index)
    else:
        train_mask = pd.Series(df["_split_year"] <= config.split_year, index=df.index)

    # Preprocess
    bn_df, feat_cols = preprocess(df, config, train_mask)

    # Split
    if config.split_strategy == "temporal":
        if "_split_year" in bn_df.columns:
            split_year_col = "_split_year"
        elif config.columns.year and config.columns.year in bn_df.columns:
            split_year_col = config.columns.year
        elif "_diag_year" in bn_df.columns:
            split_year_col = "_diag_year"
        else:
            split_year_col = None

        if split_year_col:
            yr = pd.to_numeric(bn_df[split_year_col], errors="coerce")
            train_df = bn_df[yr <= config.split_year].copy()
            test_df = bn_df[yr > config.split_year].copy()
        else:
            train_df = bn_df.iloc[:int(len(bn_df)*config.split_ratio)].copy()
            test_df = bn_df.iloc[int(len(bn_df)*config.split_ratio):].copy()
    else:
        train_df = bn_df[bn_df.index.isin(train_idx)].copy()
        test_df = bn_df[bn_df.index.isin(test_idx)].copy()

    if verbose:
        n_train = len(train_df)
        n_test = len(test_df)
        primary_target = target
        pos_rate = train_df[target].astype(int).mean() if len(train_df) > 0 else 0
        print(f"  Split: train={n_train}, test={n_test}, features={len(feat_cols)}")
        print(f"  Target: {target} (pos_rate={pos_rate:.1%}), features: {feat_cols}")
        print(f"  Split strategy: {config.split_strategy}" +
              (f" @ {config.split_year}" if config.split_year and config.split_strategy == "temporal" else ""))

    return train_df, test_df, feat_cols, target


# ── BN + MB discovery ────────────────────────────────────────────────────────

def get_bn_graph_and_mb(train_df, feat_cols, target, verbose=True):
    """Run ensemble BN learning + 3-way MB discovery.
    Returns (G_consensus, all_graphs, mb_info).
    """
    from pgmpy.estimators import HillClimbSearch, BDeu, K2, PC

    bn_cols = feat_cols + [target]
    data = train_df[bn_cols].copy()
    graphs = {}

    # HC + BDeu
    try:
        hc = HillClimbSearch(data)
        hc_model = hc.estimate(scoring_method=BDeu(data, equivalent_sample_size=10),
                               max_indegree=MAX_INDEGREE)
        G_hc = nx.DiGraph(); G_hc.add_nodes_from(bn_cols); G_hc.add_edges_from(hc_model.edges())
        graphs["HC"] = G_hc
        if verbose: print(f"  HC/BDeu: {G_hc.number_of_edges()} edges")
    except Exception as e:
        if verbose: print(f"  HC/BDeu failed: {e}")

    # HC + K2
    try:
        hc2 = HillClimbSearch(data)
        hc2_model = hc2.estimate(scoring_method=K2(data), max_indegree=MAX_INDEGREE)
        G_k2 = nx.DiGraph(); G_k2.add_nodes_from(bn_cols); G_k2.add_edges_from(hc2_model.edges())
        graphs["K2"] = G_k2
        if verbose: print(f"  HC/K2: {G_k2.number_of_edges()} edges")
    except Exception as e:
        if verbose: print(f"  HC/K2 failed: {e}")

    # PC
    try:
        pc = PC(data)
        pc_model = pc.estimate(variant="stable", max_cond_vars=4, significance_level=0.05)
        G_pc = nx.DiGraph(); G_pc.add_nodes_from(bn_cols); G_pc.add_edges_from(pc_model.edges())
        graphs["PC"] = G_pc
        if verbose: print(f"  PC: {G_pc.number_of_edges()} edges")
    except Exception as e:
        if verbose: print(f"  PC failed: {e}")

    # Consensus (≥2/3)
    edge_counts = Counter()
    for G in graphs.values():
        for u, v in G.edges():
            edge_counts[(min(u, v), max(u, v))] += 1

    G_consensus = nx.DiGraph(); G_consensus.add_nodes_from(bn_cols)
    threshold = max(2, len(graphs) // 2 + 1)
    for (u, v), count in edge_counts.items():
        if count >= threshold:
            if "HC" in graphs and graphs["HC"].has_edge(u, v):
                G_consensus.add_edge(u, v)
            elif "HC" in graphs and graphs["HC"].has_edge(v, u):
                G_consensus.add_edge(v, u)
            else:
                G_consensus.add_edge(u, v)

    while not nx.is_directed_acyclic_graph(G_consensus):
        cycle = nx.find_cycle(G_consensus)
        G_consensus.remove_edge(*cycle[0])

    if verbose:
        print(f"  Consensus: {G_consensus.number_of_edges()} edges")

    # MB discovery
    G_ref = graphs.get("HC", G_consensus)

    # DAG-derived MB
    parents = set(G_ref.predecessors(target)) if target in G_ref else set()
    children = set(G_ref.successors(target)) if target in G_ref else set()
    co_parents = set()
    for ch in children:
        for p in G_ref.predecessors(ch):
            if p != target:
                co_parents.add(p)
    mb_dag = parents | children | co_parents

    # CI-test MB
    from scipy.stats import chi2_contingency
    mb_ci = set()
    remaining = set(feat_cols)
    for _ in range(len(feat_cols)):
        best_var, best_stat = None, 0
        for var in remaining:
            try:
                ct = pd.crosstab(data[var], data[target])
                stat, p, _, _ = chi2_contingency(ct)
                if p < 0.05 and stat > best_stat:
                    best_var, best_stat = var, stat
            except Exception:
                continue
        if best_var is None:
            break
        mb_ci.add(best_var)
        remaining.discard(best_var)
    for var in list(mb_ci):
        others = mb_ci - {var}
        if not others: continue
        try:
            combined = data[target].astype(str)
            for o in list(others)[:3]:
                combined = combined + "_" + data[o].astype(str)
            ct = pd.crosstab(data[var], combined)
            _, p, _, _ = chi2_contingency(ct)
            if p > 0.10:
                mb_ci.discard(var)
        except Exception:
            continue

    # MI top-k
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    X = pd.DataFrame()
    for c in feat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(data[c].astype(str))
    y = data[target].astype(int).values
    mi = mutual_info_classif(X, y, random_state=SEED)
    mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
    k = max(len(mb_dag), 3)
    mb_mi = set(mi_s.head(k).index)

    # Consensus MB (≥2/3)
    all_vars = mb_dag | mb_ci | mb_mi
    consensus_mb = set()
    for var in all_vars:
        count = sum([var in mb_dag, var in mb_ci, var in mb_mi])
        if count >= 2:
            consensus_mb.add(var)

    mb_info = {
        "dag_derived": sorted(mb_dag),
        "ci_test": sorted(mb_ci),
        "mi_topk": sorted(mb_mi),
        "consensus": sorted(consensus_mb),
        "parents": sorted(parents),
        "children": sorted(children),
        "co_parents": sorted(co_parents),
    }
    if verbose:
        print(f"  Consensus MB: {mb_info['consensus']}")

    return G_consensus, graphs, mb_info


# ── Subgroup definitions ─────────────────────────────────────────────────────

def define_subgroups(df, config):
    """Return dict of {name: boolean Series} for clinically meaningful subgroups."""
    subs = {}

    # HPV
    if config.has_hpv:
        hpv_col = next((c for c in ["hpv_association_p16", "hpv_status"]
                        if c in df.columns), None)
        if hpv_col:
            hpv = df[hpv_col].astype(str).str.lower()
            subs["HPV+"] = hpv.isin(["positive", "yes, positive", "p16 positive", "p"])
            subs["HPV-"] = hpv.isin(["negative", "yes, negative", "p16 negative", "n"])

    # Site
    site_col = next((c for c in ["primary_tumor_site", "site"] if c in df.columns), None)
    if site_col:
        site = df[site_col].astype(str).str.lower()
        for s in ["oropharynx", "larynx", "oral_cavity", "oral cavity", "hypopharynx"]:
            mask = site.str.contains(s.replace("_", "."), na=False, regex=True)
            if mask.sum() > 0:
                subs[s.replace("_", " ").title()] = mask

    # T stage
    t_col = next((c for c in ["pT_grouped", "t_stage"] if c in df.columns), None)
    if t_col:
        t = df[t_col].astype(str).str.lower()
        for ts in ["t1", "t2", "t3", "t4"]:
            mask = t.str.contains(ts, na=False)
            if mask.sum() > 0:
                subs[ts.upper()] = mask

    # N stage
    n_col = next((c for c in ["pN_grouped", "n_stage"] if c in df.columns), None)
    if n_col:
        n = df[n_col].astype(str).str.lower()
        subs["N0"] = n.str.contains("n0|pn0", na=False)
        subs["N1"] = n.str.contains("n1|pn1", na=False)
        subs["N2+"] = n.str.contains("n2|n3|pn2|pn3", na=False)

    # Sex
    sex_col = next((c for c in ["sex"] if c in df.columns), None)
    if sex_col:
        sex = df[sex_col].astype(str).str.lower()
        subs["Male"] = sex.isin(["male", "m"])
        subs["Female"] = sex.isin(["female", "f"])

    # Age
    age_col = next((c for c in ["age_at_initial_diagnosis_raw", "age_at_initial_diagnosis", "age_raw", "age"]
                    if c in df.columns), None)
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce")
        subs["Age<60"] = age < 60
        subs["Age>=60"] = age >= 60

    # Invasion burden (HANCOCK only)
    if "invasion_burden" in df.columns:
        ib = df["invasion_burden"].astype(str)
        for level in ["0", "1", "2", "3"]:
            if (ib == level).sum() > 0:
                subs[f"IB={level}"] = ib == level

    # Smoking
    smoke_col = next((c for c in ["smoking_status", "smoking"] if c in df.columns), None)
    if smoke_col:
        smoke = df[smoke_col].astype(str).str.lower()
        for s in ["never", "former", "current"]:
            mask = smoke.str.contains(s, na=False)
            if mask.sum() > 0:
                subs[f"Smoke_{s}"] = mask

    return subs


# ── Encoding for sklearn ─────────────────────────────────────────────────────

def encode_features(df, feat_cols, target=None):
    """Label-encode categorical features for sklearn models."""
    from sklearn.preprocessing import LabelEncoder
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(df[c].astype(str))
    if target and target in df.columns:
        y = df[target].astype(int).values
        return X, y
    return X


# ── Harmonization for cross-cohort analysis ──────────────────────────────────

def _simplify_t(x):
    s = str(x).lower()
    for t in ["t1", "t2", "t3", "t4"]:
        if t in s: return t
    try:
        v = int(float(s))
        if 1 <= v <= 4: return f"t{v}"
    except (ValueError, TypeError):
        pass
    return np.nan


def _simplify_n(x):
    s = str(x).lower()
    if "n0" in s or s == "0": return "n0"
    if "n1" in s or s == "1": return "n1"
    if any(k in s for k in ["n2", "n3", "2", "3"]): return "n2+"
    return np.nan


def _simplify_site(x):
    s = str(x).lower()
    if any(k in s for k in ["oropharynx", "tonsil", "base of tongue", "bot", "soft palate",
                            "glossopharyngeal"]): return "oropharynx"
    if any(k in s for k in ["larynx", "glottis", "supraglottic"]): return "larynx"
    if any(k in s for k in ["oral", "tongue", "floor of mouth", "buccal"]): return "oral cavity"
    if any(k in s for k in ["hypopharynx", "pyriform"]): return "hypopharynx"
    return "other"


def harmonize_pair(df_a, df_b, config_a, config_b, target_years=5, verbose=True):
    """Harmonize two datasets to shared features for cross-cohort comparison.

    Returns (ha, hb, shared_cols) where ha/hb have columns: shared_cols + SVy{target_years}.
    """
    a = df_a.copy()
    b = df_b.copy()

    # Age
    for d, cfg in [(a, config_a), (b, config_b)]:
        age_col = cfg.columns.age
        if age_col and age_col in d.columns:
            age = pd.to_numeric(d[age_col], errors="coerce")
        elif f"{cfg.columns.age}_raw" in d.columns:
            age = pd.to_numeric(d[f"{cfg.columns.age}_raw"], errors="coerce")
        else:
            age = pd.Series(np.nan, index=d.index)
        d["age_h"] = pd.cut(age, bins=[0, 50, 60, 70, 200],
                            labels=["<50", "50-60", "60-70", "70+"],
                            include_lowest=True).astype(str).replace("nan", np.nan)

    # Sex
    for d, cfg in [(a, config_a), (b, config_b)]:
        sex_col = next((c for c in ["sex", cfg.columns.sex] if c in d.columns), None)
        if sex_col:
            d["sex_h"] = d[sex_col].astype(str).str.lower().str.strip()
        else:
            d["sex_h"] = np.nan

    # Site
    for d, cfg in [(a, config_a), (b, config_b)]:
        site_col = next((c for c in ["primary_tumor_site", "site", cfg.columns.site]
                        if c is not None and c in d.columns), None)
        if site_col:
            d["site_h"] = d[site_col].astype(str).apply(_simplify_site)
        else:
            d["site_h"] = np.nan

    # T stage
    for d, cfg in [(a, config_a), (b, config_b)]:
        t_col = next((c for c in ["pT_grouped", "t_stage", cfg.columns.t_stage]
                      if c is not None and c in d.columns), None)
        if t_col:
            d["t_h"] = d[t_col].astype(str).apply(_simplify_t)
        else:
            d["t_h"] = np.nan

    # N stage
    for d, cfg in [(a, config_a), (b, config_b)]:
        n_col = next((c for c in ["pN_grouped", "n_stage", cfg.columns.n_stage]
                      if c is not None and c in d.columns), None)
        if n_col:
            d["n_h"] = d[n_col].astype(str).apply(_simplify_n)
        else:
            d["n_h"] = np.nan

    shared_cols = ["age_h", "sex_h", "site_h", "t_h", "n_h"]

    # HPV (if both have it)
    if config_a.has_hpv and config_b.has_hpv:
        for d, cfg in [(a, config_a), (b, config_b)]:
            hpv_col = next((c for c in ["hpv_association_p16", "hpv_status", cfg.columns.hpv]
                           if c is not None and c in d.columns), None)
            if hpv_col:
                hpv = d[hpv_col].astype(str).str.lower()
                d["hpv_h"] = hpv.apply(lambda x:
                    "positive" if any(k in x for k in ["pos", "p16 pos", "yes"]) else
                    ("negative" if any(k in x for k in ["neg", "p16 neg"]) else "unknown"))
            else:
                d["hpv_h"] = "unknown"
        shared_cols.append("hpv_h")

    # Survival target
    svy_col = f"SVy{target_years}"
    for d in [a, b]:
        if svy_col not in d.columns:
            # Build on the fly from time_days and event
            if "time_days" in d.columns and "event" in d.columns:
                days_thresh = target_years * 365.25
                d[svy_col] = d.apply(lambda r:
                    0 if r["event"] == 1 and r["time_days"] < days_thresh else
                    (1 if r["time_days"] >= days_thresh else np.nan), axis=1)

    # Filter to complete cases
    keep = shared_cols + [svy_col]
    a_clean = a[keep].dropna()
    b_clean = b[keep].dropna()

    if verbose:
        print(f"  Harmonized {config_a.name}-{config_b.name}: "
              f"{len(a_clean)} / {len(b_clean)} patients, "
              f"{len(shared_cols)} features" + (" (+HPV)" if "hpv_h" in shared_cols else ""))

    return a_clean, b_clean, shared_cols


def harmonize_all(datasets, configs, target_years=5, verbose=True):
    """Harmonize all datasets to the minimal shared feature set.

    datasets: dict {name: df}
    configs: dict {name: config}
    Returns: dict {name: harmonized_df}, shared_cols
    """
    shared_cols = ["age_h", "sex_h", "site_h", "t_h", "n_h"]
    svy_col = f"SVy{target_years}"
    result = {}

    # Check if all cohorts have HPV
    all_have_hpv = all(configs[name].has_hpv for name in datasets)
    if all_have_hpv:
        shared_cols.append("hpv_h")

    for name, df in datasets.items():
        cfg = configs[name]
        d = df.copy()

        # Age
        age_col = cfg.columns.age
        if age_col and age_col in d.columns:
            age = pd.to_numeric(d[age_col], errors="coerce")
        elif f"{cfg.columns.age}_raw" in d.columns:
            age = pd.to_numeric(d[f"{cfg.columns.age}_raw"], errors="coerce")
        else:
            age = pd.Series(np.nan, index=d.index)
        d["age_h"] = pd.cut(age, bins=[0, 50, 60, 70, 200],
                            labels=["<50", "50-60", "60-70", "70+"],
                            include_lowest=True).astype(str).replace("nan", np.nan)

        # Sex
        sex_col = next((c for c in ["sex", cfg.columns.sex] if c in d.columns), None)
        d["sex_h"] = d[sex_col].astype(str).str.lower().str.strip() if sex_col else np.nan

        # Site
        site_col = next((c for c in ["primary_tumor_site", "site", cfg.columns.site]
                        if c is not None and c in d.columns), None)
        d["site_h"] = d[site_col].astype(str).apply(_simplify_site) if site_col else np.nan

        # T stage
        t_col = next((c for c in ["pT_grouped", "t_stage", cfg.columns.t_stage]
                      if c is not None and c in d.columns), None)
        d["t_h"] = d[t_col].astype(str).apply(_simplify_t) if t_col else np.nan

        # N stage
        n_col = next((c for c in ["pN_grouped", "n_stage", cfg.columns.n_stage]
                      if c is not None and c in d.columns), None)
        d["n_h"] = d[n_col].astype(str).apply(_simplify_n) if n_col else np.nan

        # HPV (if all cohorts have it)
        if all_have_hpv:
            hpv_col = next((c for c in ["hpv_association_p16", "hpv_status", cfg.columns.hpv]
                           if c is not None and c in d.columns), None)
            if hpv_col:
                hpv = d[hpv_col].astype(str).str.lower()
                d["hpv_h"] = hpv.apply(lambda x:
                    "positive" if any(k in x for k in ["pos", "p16 pos", "yes"]) else
                    ("negative" if any(k in x for k in ["neg", "p16 neg"]) else "unknown"))
            else:
                d["hpv_h"] = "unknown"

        # Survival
        if svy_col not in d.columns and "time_days" in d.columns and "event" in d.columns:
            days_thresh = target_years * 365.25
            d[svy_col] = d.apply(lambda r:
                0 if r["event"] == 1 and r["time_days"] < days_thresh else
                (1 if r["time_days"] >= days_thresh else np.nan), axis=1)

        keep = shared_cols + [svy_col]
        result[name] = d[keep].dropna()

    n_feat = len(shared_cols)
    hpv_note = " (+HPV)" if all_have_hpv else ""
    if verbose:
        print(f"Harmonized all cohorts ({n_feat}-feature set{hpv_note}):")
        for name, hdf in result.items():
            print(f"  {name}: n={len(hdf)}")

    return result, shared_cols
