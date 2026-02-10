# app.py
# Steel Chemistry + HAZ Toughness Predictor (ECA Solutions Engineering)
# Protected with login + password
# Features: Multi-file upload, robust cleaning, XGBoost, Pcm, screening CVN/CTOD, Monte Carlo

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
st.set_page_config(page_title="ECA Solutions | Steel Predictor", layout="wide")

# =========================
# Login (ECA Solutions Engineering)
# =========================
USERNAME = "User_ECA1"
PASSWORD = "eca2026"  # ← CHANGE THIS! (use st.secrets in production)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown("<h1 style='text-align: center; color: #003366;'>ECA SOLUTIONS ENGINEERING</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Steel Chemistry + HAZ Toughness Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Internal Engineering Screening Tool</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", placeholder="*******")
        password = st.text_input("Password", type="password", placeholder="******")
        if st.button("Login", type="primary", use_container_width=True):
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect username or password")
    st.caption("Contact: aquiles.perez@ecasolutionseng.com")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# Logout
col_logout = st.columns([9, 1])
with col_logout[1]:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

st.sidebar.success("✅ Logged in as Aquiles (ECA Solutions)")

# =========================
# Help Manual (HTML)
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
<p>This is an <b>engineering screening tool</b> for ECA Solutions Engineering. It trains a multi-output ML model on uploaded product-analysis chemistry to predict C, Nb, Si and provides weldability (Pcm) and screening toughness (CVN/CTOD) estimates.</p>
<p><b>Do not use for acceptance or qualification.</b> Use measured data for final decisions.</p>
</div>
<h2>1) Data & Cleaning</h2>
<ul>
  <li>Supports SDI, USS, and additional EAF/BOF files</li>
  <li>Auto-detects header rows and standardizes element names</li>
  <li>Route tagging: EAF = 0, BOF = 1</li>
</ul>
<h2>2) ML Chemistry Model</h2>
<ul>
  <li>XGBoost MultiOutputRegressor (C, Nb, Si ← Mn, N, Cr, Ni, Cu + route)</li>
  <li>Holdout RMSE reported</li>
</ul>
<h2>3) Weldability (Pcm)</h2>
<p>Ito–Bessyo Pcm = C + Si/30 + (Mn+Cu+Cr)/20 + Ni/60 + Mo/15 + V/10 + 5B</p>
<h2>4) Screening Toughness (CVN & CTOD)</h2>
<ul>
  <li>CVN upper-shelf from Pcm + user-selected conservatism (HAZ, cooling, environment)</li>
  <li>CTOD from fracture mechanics approximation (BS 7910 style)</li>
</ul>
<h2>5) Monte Carlo Uncertainty</h2>
<ul>
  <li>Chemicals: Normal distribution (route-specific std)</li>
  <li>Toughness: Weibull distribution (shape=2)</li>
  <li>10,000 samples, P5/P50/P95 + exceedance probability</li>
</ul>
<h2>6) References</h2>
<ul>
  <li>BS 7910 (Annex J) – fracture toughness from Charpy</li>
  <li>API RP 2Z – HAZ toughness testing</li>
  <li>Nature Scientific Reports (2023) – ML for steel composition</li>
  <li>ScienceDirect (2021) – Weibull in fracture mechanics</li>
</ul>
</body>
</html>
"""

# =========================
# Units & Formulas
# =========================
def joule_to_ftlb(j): return float(j) * 0.737562149
def mm_to_in(mm): return float(mm) * 0.0393700787

def calculate_pcm(C, Si, Mn, Cu, Cr, Ni, Mo, V, B):
    return C + (Si/30) + ((Mn + Cu + Cr)/20) + (Ni/60) + (Mo/15) + (V/10) + (5*B)

def estimate_cvn_screening_j(pcm_val: float, route_type: int):
    base_j = 320.0 if route_type == 0 else 280.0
    penalty = np.exp(-15.0 * max(0.0, pcm_val - 0.14))
    return max(float(base_j * penalty), 30.0)

def estimate_ctod_mm_from_cvn(cvn_j: float, yield_strength_mpa: float):
    k_mat = 0.54 * cvn_j + 55.0
    E = 207000.0
    nu = 0.30
    delta_m = (k_mat**2 * (1.0 - nu**2)) / (E * yield_strength_mpa)
    return float(delta_m * 1000.0)

# =========================
# Monte Carlo
# =========================
def monte_carlo_chemistry(pred_mean_vec, pred_std_vec, n_samples=10000, seed=42):
    rng = np.random.default_rng(seed)
    safe_std = np.maximum(np.asarray(pred_std_vec, dtype=float), 1e-8)
    sims = rng.normal(loc=np.asarray(pred_mean_vec, dtype=float), scale=safe_std, size=(n_samples, 3))
    return sims

def monte_carlo_toughness(ctod_mean, n_samples=10000, seed=42, shape=2.0):
    rng = np.random.default_rng(seed)
    scale = ctod_mean / (np.gamma(1 + 1/shape))
    sims = weibull_min.rvs(c=shape, loc=0, scale=scale, size=n_samples)
    return sims

# =========================
# Excel Parsing
# =========================
def _read_excel_all_sheets(file_or_path):
    xls = pd.ExcelFile(file_or_path)
    dfs = []
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, header=None)
        df["__sheet__"] = sh
        dfs.append(df)
    return dfs

def _find_header_row(df_raw, must_have=("C", "Si", "Mn")):
    n = min(60, len(df_raw))
    for i in range(n):
        row = df_raw.iloc[i].astype(str).fillna("").str.lower().str.join(" ")
        score = sum(1 for tok in must_have if tok.lower() in row)
        if score >= 2:
            return i
    return None

def _standardize_colname(x):
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
    return {k: v for k, v in mapping.items() if v is not None}

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
        out = out.apply(pd.to_numeric, errors="coerce")
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

@st.cache_data(show_spinner=False)
def load_all_data(sdi_source, uss_source, additional_eaf, additional_bof):
    sdi_df, sdi_info = (pd.DataFrame(), {"raw_rows": 0, "clean_rows": 0, "note": "None"})
    uss_df, uss_info = (pd.DataFrame(), {"raw_rows": 0, "clean_rows": 0, "note": "None"})
    if sdi_source is not None:
        sdi_df, sdi_info = parse_product_analysis(sdi_source, "EAF", 0)
    if uss_source is not None:
        uss_df, uss_info = parse_product_analysis(uss_source, "BOF", 1)
    if additional_eaf:
        for f in additional_eaf:
            df_add, _ = parse_product_analysis(f, "EAF", 0)
            sdi_df = pd.concat([sdi_df, df_add], ignore_index=True)
    if additional_bof:
        for f in additional_bof:
            df_add, _ = parse_product_analysis(f, "BOF", 1)
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
    for c in ["C","Si","Mn","Nb","N","Cr","Ni","Cu","Mo","V","B","process_type"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["process_type"] = df["process_type"].fillna(0).astype(int)
    return df, {"sdi": sdi_info, "uss": uss_info}

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
    base = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, objective="reg:squarederror", random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    stats = df_clean.groupby("process_type")[["C","Nb","Si"]].agg(["mean","std"])
    return model, rmse, stats, "OK"

# =========================
# Sidebar Data Sources
# =========================
st.sidebar.header("Data Sources")
sdi_upload = st.sidebar.file_uploader("SDI Excel (.xlsx)", type=["xlsx"], key="sdi")
uss_upload = st.sidebar.file_uploader("USS Excel (.xlsx)", type=["xlsx"], key="uss")
st.sidebar.subheader("Additional Files (optional)")
additional_eaf = st.sidebar.file_uploader("Additional EAF files", type=["xlsx"], accept_multiple_files=True, key="add_eaf")
additional_bof = st.sidebar.file_uploader("Additional BOF files", type=["xlsx"], accept_multiple_files=True, key="add_bof")

# =========================
# Load + Train
# =========================
df, info = load_all_data(sdi_upload, uss_upload, additional_eaf, additional_bof)
model, rmse, stats, train_msg = train_model(df)

# =========================
# Header
# =========================
st.title("Steel Chemistry + HAZ Toughness Predictor")
st.caption("ECA Solutions Engineering — Internal Screening Tool")

c1, c2, c3, c4 = st.columns(4)
c1.metric("SDI (raw/clean)", f"{info['sdi']['raw_rows']} / {info['sdi']['clean_rows']}")
c2.metric("USS (raw/clean)", f"{info['uss']['raw_rows']} / {info['uss']['clean_rows']}")
c3.metric("Total clean rows", f"{len(df)}")
if model is not None:
    c4.metric("Holdout RMSE", f"{rmse:.6f}")
else:
    c4.metric("Model status", train_msg)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Prediction Tool", "Visual Analysis", "Data Inspection", "Help Manual"])

# =========================
# TAB 1: Prediction
# =========================
with tab1:
    st.subheader("Prediction + Screening Outputs")
    if model is None:
        st.warning(train_msg)
        st.stop()
    left, right = st.columns([1, 2])
    with left:
        process = st.selectbox("Route", ["EAF", "BOF"])
        process_type_input = 0 if process == "EAF" else 1
        Mn_in = st.number_input("Mn (%)", value=1.25, step=0.01, format="%.3f")
        N_in = st.number_input("N (%)", value=0.009 if process_type_input == 0 else 0.002, step=0.001, format="%.4f")
        Cr_in = st.number_input("Cr (%)", value=0.06, step=0.01, format="%.3f")
        Ni_in = st.number_input("Ni (%)", value=0.05, step=0.01, format="%.3f")
        Cu_in = st.number_input("Cu (%)", value=0.12, step=0.01, format="%.3f")
        Mo_in = st.number_input("Mo (%)", value=0.02, step=0.01, format="%.3f")
        V_in = st.number_input("V (%)", value=0.005, step=0.001, format="%.3f")
        B_in = st.number_input("B (%)", value=0.0005, step=0.0001, format="%.4f")
        ys_mpa = st.number_input("Yield Strength (MPa)", value=550.0, step=10.0, format="%.1f")
        predict_btn = st.button("Run Prediction", type="primary")
    with right:
        if predict_btn:
            X_in = np.array([[Mn_in, N_in, Cr_in, Ni_in, Cu_in, float(process_type_input)]], dtype=float)
            pred = model.predict(X_in)[0]
            pred_C, pred_Nb, pred_Si = float(pred[0]), float(pred[1]), float(pred[2])
            a, b, c = st.columns(3)
            a.metric("C (%)", f"{pred_C:.4f}")
            b.metric("Nb (%)", f"{pred_Nb:.4f}")
            c.metric("Si (%)", f"{pred_Si:.4f}")
            pcm_value = calculate_pcm(pred_C, pred_Si, Mn_in, Cu_in, Cr_in, Ni_in, Mo_in, V_in, B_in)
            st.metric("Pcm", f"{pcm_value:.4f}")
            cvn_j = estimate_cvn_screening_j(pcm_value, process_type_input)
            ctod_mm = estimate_ctod_mm_from_cvn(cvn_j, ys_mpa)
            st.write(f"**Screening CVN:** {cvn_j:.0f} J")
            st.write(f"**Screening CTOD:** {ctod_mm:.3f} mm")
            # Monte Carlo
            if stats is not None and process_type_input in stats.index:
                route_stats = stats.loc[process_type_input]
                std_vec = np.array([float(route_stats[("C","std")]), float(route_stats[("Nb","std")]), float(route_stats[("Si","std")])])
                chem_sims = monte_carlo_chemistry(np.array([pred_C, pred_Nb, pred_Si]), std_vec)
                toughness_sims = monte_carlo_toughness(ctod_mm)
                st.write(f"**C (P5/P50/P95):** {np.quantile(chem_sims[:,0], 0.05):.4f} / {np.mean(chem_sims[:,0]):.4f} / {np.quantile(chem_sims[:,0], 0.95):.4f}")
                st.write(f"**CTOD (P5/P50/P95):** {np.quantile(toughness_sims, 0.05):.3f} / {np.mean(toughness_sims):.3f} / {np.quantile(toughness_sims, 0.95):.3f} mm")

# =========================
# TAB 2: Visual Analysis
# =========================
with tab2:
    st.subheader("Visual Analysis")
    if df.empty:
        st.info("Upload files to see analysis")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            for rt, sub in df.groupby("Route"):
                ax.hist(sub["C"].dropna(), bins=30, alpha=0.7, label=rt)
            ax.set_title("Carbon Distribution")
            ax.legend()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            for rt, sub in df.groupby("Route"):
                ax.hist(sub["N"].dropna(), bins=30, alpha=0.7, label=rt)
            ax.set_title("Nitrogen Distribution")
            ax.legend()
            st.pyplot(fig)

# =========================
# TAB 3: Data Inspection
# =========================
with tab3:
    st.subheader("Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

# =========================
# TAB 4: Help Manual
# =========================
with tab4:
    st.components.v1.html(HELP_MANUAL_HTML, height=900, scrolling=True)
