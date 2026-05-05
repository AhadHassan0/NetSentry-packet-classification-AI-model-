"""
NetSentry – app.py
Streamlit dashboard: upload a CICIDS2017 CSV or a .pcap file,
run XGBoost inference, and explore results + high-risk IPs.

Usage: streamlit run app.py
"""

import io, pickle, subprocess, tempfile, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_FILE = "netsentry_model.pkl"

# Columns that identify a flow but must NOT be fed to the model
IDENTITY_COLS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Timestamp",
    " Flow ID", " Source IP", " Source Port",
    " Destination IP", " Destination Port", " Timestamp",
    "Label",
]
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NetSentry",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; }
h1,h2,h3,h4                        { color: #e6edf3; }
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 6px;
}
.metric-label { font-size:.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:.07em; font-weight:600; }
.metric-value { font-size:2rem; font-weight:700; margin-top:4px; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_model(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    # Force CPU inference regardless of what device the model was trained on.
    # Prevents the cuda/cpu device mismatch crash in XGBoost.
    model.set_params(device="cpu")
    return model, payload["feature_names"]


def metric_card(label: str, value, color: str):
    st.markdown(f"""
    <div class="metric-card" style="border-left:4px solid {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
    </div>""", unsafe_allow_html=True)


def extract_ips(df: pd.DataFrame) -> pd.Series | None:
    """Return the Source IP column if it exists (strip spaces from header)."""
    for col in df.columns:
        if col.strip() in ("Source IP", "Src IP"):
            return df[col].astype(str)
    return None


def pcap_to_df(pcap_bytes: bytes) -> pd.DataFrame:
    """
    Convert raw .pcap bytes → DataFrame of flow features via tshark.
    Returns a DataFrame with CICIDS2017-compatible numeric columns,
    plus 'Source IP' and 'Destination IP' preserved for display.

    Requires tshark (Wireshark CLI) to be installed.
    Falls back gracefully with an error message.
    """
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp.write(pcap_bytes)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                "tshark", "-r", tmp_path,
                "-T", "fields",
                "-e", "ip.src",
                "-e", "ip.dst",
                "-e", "tcp.srcport",
                "-e", "tcp.dstport",
                "-e", "frame.len",
                "-e", "frame.time_delta",
                "-e", "tcp.flags",
                "-e", "ip.proto",
                "-E", "header=y",
                "-E", "separator=,",
                "-E", "quote=d",
                "-E", "occurrence=f",
            ],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "tshark is not installed. Install Wireshark/tshark and retry, "
            "or upload a CICIDS2017 CSV instead."
        )
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(f"tshark error: {result.stderr[:400]}")

    df = pd.read_csv(io.StringIO(result.stdout))
    df.rename(columns={
        "ip.src":          "Source IP",
        "ip.dst":          "Destination IP",
        "tcp.srcport":     "Source Port",
        "tcp.dstport":     "Destination Port",
        "frame.len":       "Total Length of Fwd Packets",
        "frame.time_delta":"Flow IAT Mean",
        "tcp.flags":       "FIN Flag Count",
        "ip.proto":        "Protocol",
    }, inplace=True)
    df.replace("", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def preprocess_for_inference(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Strip identity cols and align to the model's feature set."""
    X = df.copy()
    X.columns = X.columns.str.strip()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with column median (same distribution as training)
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col].fillna(X[col].median(), inplace=True)

    drops = [c for c in IDENTITY_COLS if c in X.columns]
    X.drop(columns=drops, inplace=True)
    X.drop(columns=X.select_dtypes(exclude=[np.number]).columns, inplace=True)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ NetSentry")
    st.markdown("*AI-powered network threat detection*")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload traffic file",
        type=["csv", "pcap"],
        help="CICIDS2017 CSV **or** a raw .pcap capture file.",
    )

    st.markdown("---")
    st.markdown("**Model:** XGBoost · CICIDS2017")
    st.markdown("**Input:** CSV or .pcap")
    st.info(
        "NetSentry classifies flow-level behaviour without inspecting payloads. "
        "Use alongside your existing firewall / IDS for best coverage.",
        icon="ℹ️",
    )

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:0'>🛡️ NetSentry</h1>
<p style='color:#8b949e;margin-top:4px'>Network Intrusion Detection · XGBoost · CICIDS2017</p>
<hr style='border:1px solid #30363d'>
""", unsafe_allow_html=True)

# ── LANDING ───────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:70px 20px;color:#8b949e'>
        <div style='font-size:4rem'>📡</div>
        <h3 style='color:#e6edf3'>Awaiting Traffic File</h3>
        <p>Upload a <b>CICIDS2017 CSV</b> or a <b>.pcap</b> capture using the sidebar.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
try:
    model, feature_names = load_model(MODEL_FILE)
except FileNotFoundError:
    st.error(f"Model file `{MODEL_FILE}` not found. Run `train_model.py` first.")
    st.stop()

# ── PARSE UPLOAD ─────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
is_pcap    = uploaded.name.lower().endswith(".pcap")

with st.spinner("Parsing file …"):
    try:
        if is_pcap:
            raw_df = pcap_to_df(file_bytes)
        else:
            raw_df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        st.stop()

st.success(f"✅  Loaded **{len(raw_df):,} records** from `{uploaded.name}`")

# Preserve IPs before they get stripped
src_ips = extract_ips(raw_df)

# ── PREPROCESS + PREDICT ─────────────────────────────────────────────────────
with st.spinner("Running inference …"):
    try:
        X = preprocess_for_inference(raw_df, feature_names)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    preds = model.predict(X)
    try:
        scores = model.predict_proba(X)[:, 1]
    except Exception:
        scores = preds.astype(float)

total       = len(preds)
n_benign    = int((preds == 0).sum())
n_malicious = int((preds == 1).sum())
pct_mal     = n_malicious / total * 100 if total else 0

# ── METRICS ───────────────────────────────────────────────────────────────────
st.markdown("### 📊 Traffic Summary")
c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("Total Records",     f"{total:,}",        "#4493f8")
with c2: metric_card("Benign",            f"{n_benign:,}",     "#3fb950")
with c3: metric_card("Malicious",         f"{n_malicious:,}",  "#f85149")
with c4: metric_card("Threat Rate",       f"{pct_mal:.1f}%",
                     "#f85149" if pct_mal > 10 else "#d29922")

st.markdown("<br>", unsafe_allow_html=True)

# ── CHARTS ────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Traffic Breakdown")
    fig_pie = px.pie(
        names=["Benign", "Malicious"],
        values=[n_benign, n_malicious],
        color=["Benign", "Malicious"],
        color_discrete_map={"Benign": "#3fb950", "Malicious": "#f85149"},
        hole=0.45,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e6edf3")),
        margin=dict(t=20, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_pie, width="stretch")

with col_b:
    st.markdown("#### Confidence Score Distribution")
    fig_hist = px.histogram(
        x=scores * 100, nbins=40,
        color_discrete_sequence=["#4493f8"],
        labels={"x": "Malicious Confidence (%)"},
    )
    fig_hist.add_vline(x=50, line_dash="dash", line_color="#f85149",
                       annotation_text="Threshold", annotation_font_color="#f85149")
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3",
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        margin=dict(t=20, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_hist, width="stretch")

# ── HIGH-RISK IPs ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 High-Risk Source IPs")

if src_ips is not None:
    mal_mask   = preds == 1
    ip_series  = src_ips.reset_index(drop=True)

    ip_stats = (
        pd.DataFrame({
            "Source IP":      ip_series,
            "Malicious Hits": mal_mask.astype(int),
            "Total Flows":    1,
            "Avg Score":      scores,
        })
        .groupby("Source IP", as_index=False)
        .agg(
            Malicious_Hits=("Malicious Hits", "sum"),
            Total_Flows   =("Total Flows",    "sum"),
            Avg_Score     =("Avg Score",       "mean"),
        )
        .rename(columns={
            "Malicious_Hits": "Malicious Hits",
            "Total_Flows":    "Total Flows",
            "Avg_Score":      "Avg Score (%)",
        })
    )
    ip_stats["Avg Score (%)"] = (ip_stats["Avg Score (%)"] * 100).round(1)
    ip_stats["Threat %"] = (
        ip_stats["Malicious Hits"] / ip_stats["Total Flows"] * 100
    ).round(1)
    ip_stats = ip_stats[ip_stats["Malicious Hits"] > 0].sort_values(
        "Malicious Hits", ascending=False
    )

    if ip_stats.empty:
        st.success("No malicious flows detected — no high-risk IPs to report.")
    else:
        st.warning(f"⚠️  {len(ip_stats):,} source IPs generated malicious traffic.")
        st.dataframe(ip_stats.head(100), width="stretch", height=320)

        csv_ip = ip_stats.to_csv(index=False).encode()
        st.download_button("⬇️ Download High-Risk IP Report", csv_ip,
                           "high_risk_ips.csv", "text/csv")
else:
    st.info(
        "Source IP column not found in this file. "
        "High-risk IP analysis requires a CSV with a **Source IP** column "
        "or a .pcap file parsed via tshark."
    )

# ── FLAGGED RECORDS ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚨 Flagged Malicious Records")

result_df           = raw_df.copy()
result_df["__pred"] = preds
result_df["Threat Score (%)"] = np.round(scores * 100, 2)

mal_df = (
    result_df[result_df["__pred"] == 1]
    .drop(columns=["__pred"])
    .sort_values("Threat Score (%)", ascending=False)
)

if mal_df.empty:
    st.success("🎉 No malicious traffic detected.")
else:
    st.warning(
        f"⚠️  {n_malicious:,} records flagged ({pct_mal:.1f}% of traffic). "
        "Cross-reference with your SIEM."
    )
    MAX_ROWS = 2_000
    st.dataframe(mal_df.head(MAX_ROWS), width="stretch", height=400)
    if len(mal_df) > MAX_ROWS:
        st.caption(f"Showing first {MAX_ROWS:,} of {len(mal_df):,} flagged records.")

    st.download_button(
        "⬇️ Download Malicious Records CSV",
        mal_df.to_csv(index=False).encode(),
        "malicious_records.csv", "text/csv",
    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("NetSentry · XGBoost · CICIDS2017 · For research and demonstration purposes only.")
