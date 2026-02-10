# app.py
# Steel Chemistry + HAZ Toughness Predictor (Engineering Screening Tool)
# Features: Multi-dataset upload (SDI + USS + additional files), robust cleaning, XGBoost, Pcm, screening CVN/CTOD, Monte Carlo
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm, weibull_min

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="Steel Chemistry + HAZ Toughness Predictor", layout="wide")

# =========================
# Help Manual (HTML) - Enhanced
# =========================
HELP_MANUAL_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: sans-serif; color: #333; line-height: 1.45; }
  h2 { border-bottom: 2px solid #444; padding-bottom: 5px; margin-top: 18px; }
  h3 { color: #005aa5; margin-top: 14px; }
  .box { background: #f5f7fb; padding: 14px; border-radius: 10px; border-left: 6px solid #005aa5; margin: 14px 0; }
  ul { margin-top: 6px; }
  code { background: #eee; padding: 2px 4px; border-radius: 4px; }
</style>
</head>
<body>
<h1>Methodology & Reference Manual</h1>
<div class="box">
<h3>Purpose (Screening)</h3>
<p>
This application is an <b>engineering screening tool</b>. It trains a multi-output ML model on uploaded product-analysis chemistry
to predict <b>C, Nb, Si</b> from key drivers (Mn, N, Cr/Ni/Cu + process route). It then computes weldability/hardenability indicators
(<b>Pcm</b>) and provides optional <b>screening-only</b> toughness outputs (CVN, CTOD) using a physically-motivated mapping and
conservatism factors based on user selections.
</p>
<p>
<b>Do not use screening CVN/CTOD for acceptance.</b> Use measured CVN/CTOD for qualification. The tool is intended to rank risk and
support process-control discussion.
</p>
</div>
<h2>1) Data & Cleaning</h2>
<ul>
  <li>Reads all sheets from SDI/USS/Additional Excel files; detects the chemistry table header row automatically.</li>
  <li>Standardizes element names (C, Si, Mn, Nb, N, Cr, Ni, Cu, Mo, V, B).</li>
  <li>Drops non-numeric/empty rows; fills missing trace elements as 0 (screening convention).</li>
  <li>Tags route: EAF = 0, BOF = 1 (numeric, not categorical).</li>
</ul>
<h2>2) ML Chemistry Model</h2>
<ul>
  <li><b>Model:</b> Multi-output XGBoost regression: inputs = (Mn, N, Cr, Ni, Cu, route) → outputs = (C, Nb, Si).</li>
  <li><b>Why residuals:</b> Cr/Ni/Cu often behave as “scrap fingerprints” (especially EAF) and correlate with achievable low-C and microalloy control in practice (process-control proxy, not causality).</li>
  <li><b>Validation:</b> Holdout split RMSE is reported. For small datasets, do not over-interpret RMSE; expand training data and validate against unseen heats.</li>
</ul>
<h2>3) Weldability / Hardenability: Pcm</h2>
<p>
<b>Ito–Bessyo Pcm</b> is used for low-carbon steels:
</p>
<p><i>Pcm = C + Si/30 + (Mn+Cu+Cr)/20 + Ni/60 + Mo/15 + V/10 + 5B</i></p>
<p>
Higher Pcm generally increases hardenability and risk of harder/brittle HAZ microstructures under fast cooling.
</p>
<h2>4) Screening Toughness (CVN & CTOD) — How the dropdowns affect math</h2>
<ul>
  <li>Start from a conservative upper-shelf CVN baseline by route + grade family.</li>
  <li>Apply a Pcm-based penalty that increases with HAZ severity: CGHAZ/fast-cooling/high heat input → stronger penalty.</li>
  <li>Apply additional conservatism if the selected temperature regime is Transition/Lower shelf/Unknown.</li>
  <li>Convert CVN→K-like metric (screening mapping) and then estimate CTOD using δ ~ K²(1-ν²)/(E σy).</li>
  <li>Constraint/thickness conservatism is applied as a multiplier on the K-like metric (CTOD scales with the square).</li>
</ul>
<h2>5) Uncertainty (Monte Carlo)</h2>
<ul>
  <li>Samples (C, Nb, Si) from Normal distributions using route-specific training scatter.</li>
  <li>Propagates sampled chemistry through Pcm → CVN → CTOD.</li>
  <li>Reports P5/P50/P95 for C, Pcm, CVN, CTOD and probability of CTOD falling below a selected screening threshold.</li>
</ul>
<h2>6) References (Primary / Standards)</h2>
<ul>
  <li>BS 7910 (latest edition) — guidance on fracture assessment and use of Charpy correlations for screening.</li>
  <li>API 5L — line pipe chemistry and mechanical testing framework.</li>
  <li>IIW / carbon equivalent concepts; Ito–Bessyo Pcm as widely used weldability indicator for low-carbon steels.</li>
  <li>Constraint effects and specimen geometry influence on toughness (general fracture mechanics principle).</li>
  <li>Monte Carlo in Fracture: "Probabilistic fracture mechanics using Weibull distribution" (ScienceDirect, 2021).</li>
  <li>ML for Steel Composition: "Machine learning for tramp element forecasting in steel" (Nature Scientific Reports, 2023).</li>
</ul>
</body>
</html>
"""

# =========================
# Units helpers
# =========================
def joule_to_ftlb(j): return float(j) * 0.737562149
def mm_to_in(mm): return float(mm) * 0.0393700787

# =========================
# Metallurgy / Fracture screening functions
# =========================
def calculate_pcm(C, Si, Mn, Cu, Cr, Ni, Mo, V, B):
    return C + (Si/30) + ((Mn + Cu + Cr)/20) + (Ni/60) + (Mo/15) + (V/10) + (5*B)

def estimate_cvn_screening_j(pcm_val: float, route_type: int, cfg: dict):
    base_j = 320.0 if route_type == 0 else 280.0
    pcm0 = cfg["pcm_threshold"]
    alpha = cfg["pcm_sensitivity"]
    haz_mult = 1.0
    if "CGHAZ" in cfg["toughness_location"]: haz_mult *= 1.20
    if cfg["t85"] == "Short (fast cooling)": haz_mult *= 1.25
    penalty = np.exp(-alpha * max(0.0, pcm_val - pcm0) * haz_mult)
    cvn = base_j * penalty
    if cfg["service_env"] in ["Sour (H2S)", "Hydrogen (H2)", "NH3 / ammonia", "Mixed / not sure"]:
        cvn *= 0.90
    if cfg["temperature_regime"] == "Transition / mixed":
        cvn *= 0.85
    elif cfg["temperature_regime"] == "Lower shelf (cleavage risk)":
        cvn *= 0.70
    return float(np.clip(cvn, 15.0, 400.0))

def estimate_ctod_mm_from_cvn(cvn_j: float, yield_strength_mpa: float, cfg: dict):
    if cfg["disable_toughness"]:
        return np.nan
    k = 0.54 * cvn_j + 55.0  # Screening mapping
    k_eff = k * cfg["constraint_factor"]
    E = 207000.0
    nu = 0.30
    ctod_mm = (k_eff**2 * (1.0 - nu**2)) / (E * yield_strength_mpa) * 1000.0
    return float(np.clip(ctod_mm, 0.001, 2.0))

# =========================
# Monte Carlo
# =========================
def monte_carlo_full(pred_mean_vec, pred_std_vec, inputs_fixed, cfg, n=10000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    safe_std = np.maximum(np.asarray(pred_std_vec, dtype=float), 1e-6)
    chem = rng.normal(loc=np.asarray(pred_mean_vec, dtype=float), scale=safe_std, size=(n, 3))
    chem[:, 0] = np.clip(chem[:, 0], 0.0, 0.20) # C
    chem[:, 1] = np.clip(chem[:, 1], 0.0, 0.20) # Nb
    chem[:, 2] = np.clip(chem[:, 2], 0.0, 1.50) # Si
    C_sim = chem[:, 0]
    Nb_sim = chem[:, 1]
    Si_sim = chem[:, 2]
    Mn_in, N_in, Cr_in, Ni_in, Cu_in, Mo_in, V_in, B_in, ys_mpa, route_type = inputs_fixed
    pcm = np.array([calculate_pcm(C_sim[i], Si_sim[i], Mn_in, Cu_in, Cr_in, Ni_in, Mo_in, V_in, B_in) for i in range(n)])
    cvn = np.array([estimate_cvn_screening_j(pcm[i], route_type, cfg) for i in range(n)])
    ctod = np.array([estimate_ctod_mm_from_cvn(cvn[i], ys_mpa, cfg) for i in range(n)])
    def summarize(x):
        x = x[~np.isnan(x)]
        if len(x) == 0: return None
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
        }
    return {
        "C": summarize(C_sim),
        "Nb": summarize(Nb_sim),
        "Si": summarize(Si_sim),
        "Pcm": summarize(pcm),
        "CVN_J": summarize(cvn),
        "CTOD_mm": summarize(ctod),
        "raw": {"C": C_sim, "Nb": Nb_sim, "Si": Si_sim, "Pcm": pcm, "CVN_J": cvn, "CTOD_mm": ctod}
    }

# =========================
# Excel parsing (robust)
# =========================
def _read_excel_all_sheets(file_or_path):
    xls = pd.ExcelFile(file_or_path)
    dfs = []
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, header=None)
        df["__sheet__"] = sh
        dfs.append(df)
    return dfs

def _find_header_row(df_raw, must_have=("C", "Si", "Mn"), search_rows=60):
    n = min(search_rows, len(df_raw))
    best_i = None
    best_score = -1
    for i in range(n):
        row = df_raw.iloc[i].astype(str).fillna("")
        joined = ",".join(row.values).lower()
        score = sum(1 for tok in must_have if tok.lower() in joined)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i if best_score >= 2 else None

def _standardize_colname(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _extract_element_columns(df):
    cols = [_standardize_colname(c) for c in df.columns]
    df.columns = cols
    def find_col(patterns):
        for c in df.columns:
            cl = c.lower()
            for p in patterns:
                if re.search(p, cl):
                    return c
        return None
    mapping = {}
    mapping["C"] = find_col([r"^c\b", r"c\("])
    mapping["Si"] = find_col([r"^si\b", r"si\("])
    mapping["Mn"] = find_col([r"^mn\b", r"mn\("])
    mapping["Nb"] = find_col([r"^nb\b", r"nb\("])
    mapping["N"] = find_col([r"^n\b", r"n\("])
    mapping["Cr"] = find_col([r"^cr\b", r"cr\("])
    mapping["Ni"] = find_col([r"^ni\b", r"ni\("])
    mapping["Cu"] = find_col([r"^cu\b", r"cu\("])
    mapping["Mo"] = find_col([r"^mo\b", r"mo\("])
    mapping["V"] = find_col([r"^v\b", r"v\("])
    mapping["B"] = find_col([r"^b\b", r"b\("])
    mapping = {k: v for k, v in mapping.items() if v is not None}
    return mapping

def parse_product_analysis(file_or_path, route_label: str, route_type: int):
    dfs_raw = _read_excel_all_sheets(file_or_path)
    if not dfs_raw:
        return pd.DataFrame(), {"raw_rows": 0, "clean_rows": 0, "note": "No readable sheets"}
    collected = []
    raw_rows_total = 0
    for raw in dfs_raw:
        raw_rows_total += len(raw)
        hdr = _find_header_row(raw)
        if hdr is None:
            continue
        header = raw.iloc[hdr].astype(str).tolist()
        data = raw.iloc[hdr+1:].copy()
        data.columns = header
        data = data.reset_index(drop=True)
        data = data.dropna(axis=1, how="all")
        mapping = _extract_element_columns(data)
        if "C" not in mapping or "Mn" not in mapping or "Si" not in mapping:
            continue
        out = pd.DataFrame()
        for std_name, col in mapping.items():
            out[std_name] = data[col]
        out = _coerce_numeric(out, ["C","Si","Mn","Nb","N","Cr","Ni","Cu","Mo","V","B"])
        out["Route"] = route_label
        out["process_type"] = int(route_type)
        for el in ["Nb","N","Cr","Ni","Cu","Mo","V","B"]:
            if el not in out.columns:
                out[el] = 0.0
            out[el] = out[el].fillna(0.0)
        out_clean = out.dropna(subset=["C","Si","Mn"]).copy()
        out_clean = out_clean[(out_clean["C"] >= 0) & (out_clean["C"] <= 0.25)]
        out_clean = out_clean[(out_clean["Mn"] >= 0) & (out_clean["Mn"] <= 3.0)]
        out_clean = out_clean[(out_clean["Si"] >= 0) & (out_clean["Si"] <= 1.5)]
        if len(out_clean) > 0:
            collected.append(out_clean)
    if not collected:
        return pd.DataFrame(), {"raw_rows": raw_rows_total, "clean_rows": 0, "note": "No chemistry table detected"}
    df = pd.concat(collected, ignore_index=True)
    df["process_type"] = pd.to_numeric(df["process_type"], errors="coerce").fillna(route_type).astype(int)
    info = {"raw_rows": raw_rows_total, "clean_rows": len(df), "note": "OK"}
    return df, info

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_all_data(sdi_source, uss_source, additional_eaf, additional_bof):
    sdi_df, sdi_info = (pd.DataFrame(), {"raw_rows": 0, "clean_rows": 0, "note": "None"})
    uss_df, uss_info = (pd.DataFrame(), {"raw_rows": 0, "clean_rows": 0, "note": "None"})
    if sdi_source is not None:
        sdi_df, sdi_info = parse_product_analysis(sdi_source, route_label="EAF", route_type=0)
    if uss_source is not None:
        uss_df, uss_info = parse_product_analysis(uss_source, route_label="BOF", route_type=1)
    # Additional files
    if additional_eaf:
        for f in additional_eaf:
            df_add, _ = parse_product_analysis(f, route_label="EAF", route_type=0)
            sdi_df = pd.concat([sdi_df, df_add], ignore_index=True)
    if additional_bof:
        for f in additional_bof:
            df_add, _ = parse_product_analysis(f, route_label="BOF", route_type=1)
            uss_df = pd.concat([uss_df, df_add], ignore_index=True)
    common_cols = ["C","Si","Mn","Nb","N","Cr","Ni","Cu","Mo","V","B","process_type","Route"]
    frames = []
    if not sdi_df.empty:
        frames.append(sdi_df[common_cols])
    if not uss_df.empty:
        frames.append(uss_df[common_cols])
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=common_cols)
    # Ensure numeric
    for c in ["C","Si","Mn","Nb","N","Cr","Ni","Cu","Mo","V","B","process_type"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["process_type"] = df["process_type"].fillna(0).astype(int)
    return df, {"sdi": sdi_info, "uss": uss_info}

# =========================
# Train Model
# =========================
@st.cache_resource(show_spinner=False)
def train_model(df):
    if df is None or df.empty:
        return None, None, None, "No data loaded."
    df_clean = df.dropna(subset=["Mn","N","Cr","Ni","Cu","process_type","C","Nb","Si"]).copy()
    if len(df_clean) < 12:
        return None, None, None, "Model not trained (insufficient clean rows)."
    X = df_clean[["Mn","N","Cr","Ni","Cu","process_type"]].astype(float)
    y = df_clean[["C","Nb","Si"]].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    stats = df_clean.groupby("process_type")[["C","Nb","Si"]].agg(["mean","std"])
    return model, rmse, stats, "OK"

# =========================
# Sidebar: data sources
# =========================
st.sidebar.header("Data Sources")
sdi_upload = st.sidebar.file_uploader("Upload SDI Excel (.xlsx)", type=["xlsx"], key="sdi")
uss_upload = st.sidebar.file_uploader("Upload USS Excel (.xlsx)", type=["xlsx"], key="uss")
st.sidebar.subheader("Additional Datasets (optional)")
additional_eaf = st.sidebar.file_uploader("Additional EAF files", type=["xlsx"], accept_multiple_files=True, key="add_eaf")
additional_bof = st.sidebar.file_uploader("Additional BOF files", type=["xlsx"], accept_multiple_files=True, key="add_bof")
st.sidebar.caption("Multiple files allowed. All will be concatenated into training data.")

# =========================
# Load data + train model
# =========================
df, info = load_all_data(sdi_upload, uss_upload, additional_eaf, additional_bof)
model, rmse, stats, train_msg = train_model(df)

# =========================
# Header
# =========================
st.title("Steel Chemistry + HAZ Toughness Predictor (Screening Tool)")
st.caption("CVN shown in Joules (optionally ft-lb). CTOD shown in mm (optionally inches).")

# Status / counts
c1, c2, c3, c4 = st.columns(4)
c1.metric("SDI rows (raw / clean)", f"{info['sdi']['raw_rows']} / {info['sdi']['clean_rows']}")
c2.metric("USS rows (raw / clean)", f"{info['uss']['raw_rows']} / {info['uss']['clean_rows']}")
c3.metric("Total clean rows", f"{len(df)}")
if model is not None:
    c4.metric("Holdout RMSE (screening)", f"{rmse:.6f}")
else:
    c4.metric("Model status", train_msg)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Prediction Tool", "Visual Analysis", "Data Inspection", "Help Manual"])

# =========================
# TAB 1 — Prediction tool
# =========================
with tab1:
    st.subheader("Prediction + Screening Outputs")
    if model is None:
        st.warning(train_msg)
        st.stop()
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Process Inputs")
        process = st.selectbox("Manufacturing Route", ["EAF", "BOF"], index=1)
        process_type_input = 0 if process == "EAF" else 1
        Mn_in = st.number_input("Mn (%)", value=1.25, step=0.01, format="%.3f")
        N_in = st.number_input("N (%)", value=0.009 if process_type_input == 0 else 0.002, step=0.001, format="%.4f")
        Cr_in = st.number_input("Cr (%)", value=0.06, step=0.01, format="%.3f")
        Ni_in = st.number_input("Ni (%)", value=0.05, step=0.01, format="%.3f")
        Cu_in = st.number_input("Cu (%)", value=0.12, step=0.01, format="%.3f")
        st.markdown("---")
        st.caption("Trace elements for Pcm (used as-entered)")
        Mo_in = st.number_input("Mo (%)", value=0.02, step=0.01, format="%.3f")
        V_in = st.number_input("V (%)", value=0.005, step=0.001, format="%.3f")
        B_in = st.number_input("B (%)", value=0.0005, step=0.0001, format="%.4f")
        st.markdown("---")
        yield_strength_mpa = st.number_input("Assumed yield strength (MPa) for CTOD screening", value=550.0, step=10.0, format="%.1f")
        st.markdown("---")
        st.markdown("### Drop-down assumptions (used in calculations)")
        grade_family = st.selectbox("Grade family", ["X65", "X70", "X80", "Other / Mixed"], index=1)
        product_form = st.selectbox("Product form / thickness constraint", ["Plate (thin)", "Plate (thick)", "Pipe (thin)", "Pipe (thick)"], index=3)
        service_env = st.selectbox("Service environment", ["Sweet / benign", "Sour (H2S)", "Hydrogen (H2)", "NH3 / ammonia", "Mixed / not sure"], index=0)
        toughness_location = st.selectbox("Toughness location screened", ["Base Metal", "FGHAZ", "ICHAZ", "CGHAZ"], index=3)
        temperature_regime = st.selectbox("Temperature regime for intended service/testing", ["Upper shelf (ductile)", "Transition / mixed", "Lower shelf (cleavage risk)", "Unknown"], index=0)
        heat_input = st.selectbox("Heat input level", ["Low", "Medium", "High"], index=1)
        t85 = st.selectbox("Cooling time t8/5", ["Short (fast cooling)", "Medium", "Long (slow cooling)"], index=0)
        kmat_method = st.selectbox("Screening CVN→K mapping", ["BS7910_linear", "FITNET_linear"], index=0)
        if product_form in ["Pipe (thick)", "Plate (thick)"]:
            constraint_factor = 0.85
        else:
            constraint_factor = 0.92
        guard_warn = temperature_regime in ["Transition / mixed", "Lower shelf (cleavage risk)", "Unknown"]
        guard_force = temperature_regime == "Lower shelf (cleavage risk)"
        disable_toughness = st.checkbox("Disable CVN/CTOD screening (chemistry + Pcm only)", value=False)
        predict_btn = st.button("Run Prediction", type="primary")
    with right:
        if predict_btn:
            cfg = {
                "grade_family": grade_family,
                "product_form": product_form,
                "service_env": service_env,
                "toughness_location": toughness_location,
                "temperature_regime": temperature_regime,
                "heat_input": heat_input,
                "t85": t85,
                "kmat_method": kmat_method,
                "disable_toughness": bool(disable_toughness),
                "constraint_factor": float(constraint_factor),
                "guard_warn": bool(guard_warn),
                "guard_force": bool(guard_force),
                "pcm_threshold": 0.14,
                "pcm_sensitivity": 18.0,
            }
            input_data = np.array([[Mn_in, N_in, Cr_in, Ni_in, Cu_in, float(process_type_input)]], dtype=float)
            pred = model.predict(input_data)[0]
            pred_C, pred_Nb, pred_Si = float(pred[0]), float(pred[1]), float(pred[2])
            st.markdown("### 1) Predicted Chemistry")
            a, b, c = st.columns(3)
            a.metric("C (%)", f"{pred_C:.4f}")
            b.metric("Nb (%)", f"{pred_Nb:.4f}")
            c.metric("Si (%)", f"{pred_Si:.4f}")
            pcm_value = float(calculate_pcm(pred_C, pred_Si, Mn_in, Cu_in, Cr_in, Ni_in, Mo_in, V_in, B_in))
            st.markdown("### 2) Weldability / Hardenability (Pcm)")
            st.metric("Pcm", f"{pcm_value:.4f}")
            st.markdown("### 3) Screening Toughness Outputs (Optional)")
            if cfg["guard_warn"]:
                st.warning("Selected temperature regime is not 'Upper shelf'. Treat CVN→CTOD as conservative screening only.")
            cvn_j = estimate_cvn_screening_j(pcm_value, process_type_input, cfg)
            cvn_ftlb = joule_to_ftlb(cvn_j)
            if cfg["disable_toughness"]:
                st.info("Toughness conversion disabled. Showing chemistry + Pcm only.")
            else:
                ctod_mm = estimate_ctod_mm_from_cvn(cvn_j, yield_strength_mpa, cfg)
                ctod_in = mm_to_in(ctod_mm)
                k1, k2, k3 = st.columns(3)
                k1.metric("Screening CVN", f"{cvn_j:.0f} J")
                k2.metric("Screening CVN", f"{cvn_ftlb:.0f} ft-lb")
                k3.metric("Screening CTOD", f"{ctod_mm:.3f} mm")
                st.caption(f"CTOD ≈ {ctod_in:.4f} in")
                st.caption("Screening only. Replace with measured CVN/CTOD for qualification/acceptance.")
            st.markdown("### 4) Monte Carlo Uncertainty (Chemistry + Derived Metrics)")
            if stats is None or process_type_input not in stats.index:
                st.warning("Monte Carlo unavailable: route stats not available.")
            else:
                route_stats = stats.loc[process_type_input]
                std_C = float(route_stats[("C","std")]) if ("C","std") in route_stats.index else np.nan
                std_Nb = float(route_stats[("Nb","std")]) if ("Nb","std") in route_stats.index else np.nan
                std_Si = float(route_stats[("Si","std")]) if ("Si","std") in route_stats.index else np.nan
                std_vec = np.array([std_C, std_Nb, std_Si], dtype=float)
                if np.any(~np.isfinite(std_vec)) or np.any(std_vec <= 0):
                    st.warning("Monte Carlo unavailable: route standard deviations could not be computed reliably.")
                else:
                    inputs_fixed = (
                        float(Mn_in), float(N_in), float(Cr_in), float(Ni_in), float(Cu_in),
                        float(Mo_in), float(V_in), float(B_in),
                        float(yield_strength_mpa),
                        int(process_type_input),
                    )
                    mc = monte_carlo_full(pred_mean_vec=pred, pred_std_vec=std_vec, inputs_fixed=inputs_fixed, cfg=cfg, n=10000)
                    sC = mc["C"]; sP = mc["Pcm"]; sV = mc["CVN_J"]; sD = mc["CTOD_mm"]
                    st.write(f"**C (P5/P50/P95):** {sC['p05']:.4f} / {sC['p50']:.4f} / {sC['p95']:.4f} %")
                    st.write(f"**Pcm (P5/P50/P95):** {sP['p05']:.3f} / {sP['p50']:.3f} / {sP['p95']:.3f}")
                    st.write(f"**CVN (P5/P50/P95):** {sV['p05']:.0f} / {sV['p50']:.0f} / {sV['p95']:.0f} J")
                    if cfg["disable_toughness"]:
                        st.info("CTOD not computed (disabled).")
                    else:
                        st.write(f"**CTOD (P5/P50/P95):** {sD['p05']:.3f} / {sD['p50']:.3f} / {sD['p95']:.3f} mm")
                        ctod_threshold = 0.10
                        prob_low = float(np.mean(mc["raw"]["CTOD_mm"] < ctod_threshold) * 100.0)
                        st.write(f"**P(CTOD < {ctod_threshold:.2f} mm):** {prob_low:.1f}%")
                    if not cfg["disable_toughness"]:
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.hist(mc["raw"]["CTOD_mm"], bins=50, edgecolor="black", alpha=0.85)
                        ax.set_xlabel("CTOD (mm)")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Monte Carlo CTOD distribution (screening)")
                        st.pyplot(fig)

# =========================
# TAB 2 — Visual analysis
# =========================
with tab2:
    st.subheader("Exploratory Data Analysis (Chemistry)")
    if df.empty:
        st.info("Upload Excel files to see data.")
    else:
        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots()
            for rt, sub in df.groupby("Route"):
                ax.hist(sub["C"].dropna(), bins=30, alpha=0.7, label=rt, edgecolor="black")
            ax.set_title("Carbon (C) distribution by route")
            ax.set_xlabel("C (%)")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
        with colB:
            fig, ax = plt.subplots()
            for rt, sub in df.groupby("Route"):
                ax.hist(sub["N"].dropna(), bins=30, alpha=0.7, label=rt, edgecolor="black")
            ax.set_title("Nitrogen (N) distribution by route")
            ax.set_xlabel("N (%)")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)

# =========================
# TAB 3 — Data inspection
# =========================
with tab3:
    st.subheader("Cleaned Training Dataset")
    st.dataframe(df)

# =========================
# TAB 4 — Help manual
# =========================
with tab4:
    st.components.v1.html(HELP_MANUAL_HTML, height=900, scrolling=True)
