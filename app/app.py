"""
app.py  –  Warp-Level Reduction Benchmark Dashboard
Streamlit + Plotly interactive UI.
Run: streamlit run app/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import pathlib
import sys

# Add project root to path so we can import cuda_runner from app/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cuda_runner

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Warp Reduction Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease;
}
.metric-card:hover { border-color: rgba(99,179,237,0.5); }
.metric-card .label { color: #94a3b8; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; }
.metric-card .value { color: #e2e8f0; font-size: 1.7rem; font-weight: 700; }
.metric-card .value.green  { color: #4ade80; }
.metric-card .value.red    { color: #f87171; }
.metric-card .value.yellow { color: #fbbf24; }
.metric-card .value.blue   { color: #60a5fa; }

/* Status badges */
.badge-pass { background:#064e3b; color:#6ee7b7; border:1px solid #059669; border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.85rem; }
.badge-fail { background:#7f1d1d; color:#fca5a5; border:1px solid #dc2626; border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.85rem; }
.badge-saved { background:#1e3a5f; color:#93c5fd; border:1px solid #3b82f6; border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.85rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #60a5fa;
    border-left: 3px solid #3b82f6; padding-left: 10px;
    margin: 1.5rem 0 0.8rem 0; letter-spacing: 0.05em;
}

/* Alert boxes */
.alert-info    { background:#1e3a5f22; border:1px solid #3b82f6; border-radius:8px; padding:0.9rem 1.2rem; color:#93c5fd; margin:0.5rem 0; }
.alert-warn    { background:#78350f22; border:1px solid #f59e0b; border-radius:8px; padding:0.9rem 1.2rem; color:#fcd34d; margin:0.5rem 0; }
.alert-error   { background:#7f1d1d22; border:1px solid #ef4444; border-radius:8px; padding:0.9rem 1.2rem; color:#fca5a5; margin:0.5rem 0; }
.alert-success { background:#064e3b22; border:1px solid #10b981; border-radius:8px; padding:0.9rem 1.2rem; color:#6ee7b7; margin:0.5rem 0; }

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Plotly theme helper
# ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(13,17,23,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(family="Inter", color="#cbd5e1"),
    xaxis=dict(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)"),
    legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="rgba(99,179,237,0.2)", borderwidth=1),
    margin=dict(l=60, r=30, t=60, b=50),
)
COLORS = {
    "shared":  "#60a5fa",   # blue
    "shuffle": "#4ade80",   # green
    "ref":     "#f97316",   # orange
}

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
  <h1 style="font-size:2.4rem; font-weight:800; background:linear-gradient(135deg,#60a5fa,#a78bfa,#4ade80);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
    ⚡ Warp-Level Reduction Benchmarks
  </h1>
  <p style="color:#94a3b8; font-size:1.05rem; margin-top:0.5rem; max-width:700px; margin-left:auto; margin-right:auto;">
    Interactive comparison of <strong style="color:#60a5fa;">Shared-Memory</strong> vs
    <strong style="color:#4ade80;">Warp-Shuffle</strong> CUDA reduction algorithms for SUM, MAX, and ARGMAX.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────
# CUDA availability check
# ──────────────────────────────────────────────────────────────
cuda_available = cuda_runner.is_cuda_available()
gpu_info       = cuda_runner.get_gpu_info() if cuda_available else ""

if cuda_available:
    gpu_label = gpu_info if gpu_info else "GPU detected"
    st.markdown(f'<div class="alert-success">✅ <strong>CUDA mode active</strong> — {gpu_label}. Live benchmarks enabled.</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-warn">⚠️ <strong>CUDA executable not found.</strong>
    Running in <em>saved-results mode</em>. Historical benchmark data will be displayed.
    <br>To enable live benchmarks, compile on a CUDA-capable machine:<br>
    <code>nvcc -O2 -arch=sm_75 src/reduction_bench.cu -o src/reduction_bench</code>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SIDEBAR – Run Configuration
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Run Configuration")
    st.divider()

    operation = st.selectbox(
        "Operation", ["sum", "max", "argmax"],
        format_func=lambda x: {"sum": "∑ SUM", "max": "⬆ MAX", "argmax": "📍 ARGMAX"}[x],
        key="op",
    )
    algorithm = st.selectbox(
        "Algorithm", ["both", "shared", "shuffle"],
        format_func=lambda x: {"both": "↔ Compare Both", "shared": "🟦 Shared Memory", "shuffle": "🟩 Warp Shuffle"}[x],
        key="alg",
    )
    st.divider()
    st.markdown("**Input Size**")
    size_preset = st.selectbox(
        "Preset sizes", [32, 256, 1_000, 65_536, 1_000_000, "Custom"],
        format_func=lambda x: f"{x:,}" if isinstance(x, int) else x,
    )
    if size_preset == "Custom":
        input_size = st.number_input("Custom size", min_value=1, max_value=50_000_000, value=100_000, step=1000)
    else:
        input_size = int(size_preset)

    pattern = st.selectbox(
        "Input pattern",
        ["random_signed", "positive", "all_equal", "dup_max", "max_at_end"],
        format_func=lambda x: {
            "random_signed": "🎲 Random signed",
            "positive": "➕ Positive only",
            "all_equal": "═ All equal",
            "dup_max": "👥 Duplicate maximum",
            "max_at_end": "🔚 Max at last position",
        }[x],
    )
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)

    st.divider()
    reps   = st.slider("Repetitions", 1, 50, 5)
    warmup = st.slider("Warm-up iterations", 0, 200, 50)

    st.divider()
    run_btn = st.button(
        "▶ Run Benchmark",
        use_container_width=True,
        disabled=not cuda_available,
        type="primary",
    )

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab_live, tab_history, tab_charts, tab_about = st.tabs(
    ["🚀 Live Run", "📋 History", "📊 Charts", "ℹ️ About"]
)

# ══════════════════════════════════════════════════════════════
# TAB: LIVE RUN
# ══════════════════════════════════════════════════════════════
with tab_live:
    if not cuda_available:
        st.markdown('<div class="alert-warn">⚠️ Live benchmarks require the compiled CUDA executable. See the saved-results mode in the Charts tab.</div>', unsafe_allow_html=True)
    elif run_btn:
        with st.spinner("Running CUDA benchmark…"):
            try:
                data = cuda_runner.run_benchmark(
                    operation=operation,
                    algorithm=algorithm,
                    size=input_size,
                    seed=seed,
                    repetitions=reps,
                    warmup=warmup,
                    pattern=pattern,
                )
                st.session_state["last_result"] = data
            except Exception as exc:
                st.markdown(f'<div class="alert-error">❌ <strong>Error:</strong> {exc}</div>', unsafe_allow_html=True)
                st.stop()

    if "last_result" in st.session_state:
        data    = st.session_state["last_result"]
        results = data.get("results", {})
        meta    = data.get("metadata", {})
        comp    = data.get("comparison") or {}
        isAm    = data.get("operation") == "argmax"

        st.markdown(f'<div class="section-header">Results – {data["operation"].upper()} | size {data["input_size"]:,} | seed {data["seed"]} | pattern: {data["pattern"]}</div>', unsafe_allow_html=True)

        # ── Overall correctness banner
        if data.get("correct"):
            st.markdown('<div class="alert-success">✅ All algorithms: <strong>PASS</strong> (matches CPU reference)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-error">❌ One or more algorithms failed the CPU-reference correctness check.</div>', unsafe_allow_html=True)

        # ── Per-algorithm result cards
        cols = st.columns(len(results))
        for col, (alg, res) in zip(cols, results.items()):
            badge = '<span class="badge-pass">PASS</span>' if res.get("correct") else '<span class="badge-fail">FAIL</span>'
            val_fmt = f"{res.get('value', 0):,.2f}"
            lat_fmt = f"{res.get('latency_us', 0):.2f} µs"
            am_txt  = f"<br><span style='color:#94a3b8;font-size:0.85rem'>Index: {res.get('argmax_index','—')}</span>" if isAm else ""
            color_cls = "blue" if alg == "shared" else "green"
            col.markdown(f"""
            <div class="metric-card">
                <div class="label">{alg.upper()} MEMORY</div>
                <div class="value {color_cls}">{val_fmt}</div>
                {am_txt}
                <div style="margin-top:0.6rem; color:#94a3b8; font-size:0.85rem">{lat_fmt}</div>
                <div style="margin-top:0.4rem">{badge}</div>
            </div>""", unsafe_allow_html=True)

        # ── Comparison metrics
        if comp:
            st.markdown("")
            c1, c2 = st.columns(2)
            spd = comp.get("speedup_shared_over_shuffle", 1.0)
            lr  = comp.get("latency_reduction_percent", 0.0)
            spd_cls = "green" if spd > 1 else ("yellow" if spd == 1 else "red")
            lr_cls  = "green" if lr  > 0 else "red"
            c1.markdown(f"""<div class="metric-card">
                <div class="label">Speedup (shared / shuffle)</div>
                <div class="value {spd_cls}">{spd:.3f}×</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-card">
                <div class="label">Latency Reduction</div>
                <div class="value {lr_cls}">{lr:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # ── Sample distribution chart
        if len(results) >= 1:
            st.markdown('<div class="section-header">Per-Run Latency Distribution</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for alg, res in results.items():
                samples = res.get("samples_us", [])
                if samples:
                    fig.add_trace(go.Box(
                        y=samples, name=alg.capitalize(),
                        marker_color=COLORS.get(alg, "#aaa"),
                        boxpoints="all", jitter=0.4, pointpos=0,
                        line_width=2,
                    ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Latency (µs) per repetition",
                              yaxis_title="Latency (µs)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Raw JSON expander
        with st.expander("📄 Raw JSON output"):
            st.json(data)
    else:
        if cuda_available:
            st.markdown('<div class="alert-info">👈 Configure your benchmark in the sidebar and click <strong>▶ Run Benchmark</strong>.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB: HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-header">Benchmark History</div>', unsafe_allow_html=True)

    rows = cuda_runner.load_raw_csv()
    legacy = cuda_runner.load_legacy_sum_csv()

    if not rows and not legacy:
        st.markdown('<div class="alert-info">No benchmark history found. Run a live benchmark or place CSV files in results/.</div>', unsafe_allow_html=True)
    else:
        if rows:
            df = pd.DataFrame(rows)
            # Filters
            fc1, fc2, fc3, fc4 = st.columns(4)
            ops_avail = sorted(df["operation"].unique()) if "operation" in df.columns else []
            alg_avail = sorted(df["algorithm"].unique()) if "algorithm" in df.columns else []
            siz_avail = sorted(df["input_size"].unique()) if "input_size" in df.columns else []

            sel_op  = fc1.multiselect("Operation",  ops_avail, default=ops_avail)
            sel_alg = fc2.multiselect("Algorithm",  alg_avail, default=alg_avail)
            sel_siz = fc3.multiselect("Input size",  siz_avail, default=siz_avail)
            sel_cor = fc4.selectbox("Correctness", ["All", "PASS only", "FAIL only"])

            mask = df["operation"].isin(sel_op) & df["algorithm"].isin(sel_alg) & df["input_size"].isin(sel_siz)
            if sel_cor == "PASS only":
                mask = mask & (df["correct"].astype(str).str.lower() == "true")
            elif sel_cor == "FAIL only":
                mask = mask & (df["correct"].astype(str).str.lower() != "true")
            filtered = df[mask]

            st.dataframe(filtered, use_container_width=True, hide_index=True)
            csv_bytes = filtered.to_csv(index=False).encode()
            st.download_button("⬇ Download filtered CSV", csv_bytes, "filtered_results.csv", "text/csv")

        if legacy:
            with st.expander("📜 Legacy SUM benchmark data (benchmark_raw.csv)"):
                st.markdown('<span class="badge-saved">SAVED / HISTORICAL</span>', unsafe_allow_html=True)
                leg_df = pd.DataFrame(legacy)
                st.dataframe(leg_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB: CHARTS
# ══════════════════════════════════════════════════════════════
with tab_charts:
    rows = cuda_runner.load_raw_csv()
    legacy = cuda_runner.load_legacy_sum_csv()

    # Try to build a combined df from history (new CSV) or legacy SUM CSV
    has_new = len(rows) > 0
    has_legacy = len(legacy) > 0

    if has_new:
        df = pd.DataFrame(rows)
        df["latency_us"]  = pd.to_numeric(df["latency_us"],  errors="coerce")
        df["input_size"]  = pd.to_numeric(df["input_size"],  errors="coerce")
        df["speedup_shared_over_shuffle"] = pd.to_numeric(df["speedup_shared_over_shuffle"], errors="coerce")
        df["latency_reduction_percent"]   = pd.to_numeric(df["latency_reduction_percent"],   errors="coerce")
        source_label = "Live benchmark history"
    elif has_legacy:
        # Reshape legacy CSV to match our schema
        rows_reshaped = []
        for r in legacy:
            for alg in ("shared", "shuffle"):
                rows_reshaped.append({
                    "operation": "sum", "algorithm": alg,
                    "input_size": int(r.get("input_size", 0)),
                    "latency_us": float(r.get(f"{alg}_us", 0)),
                    "correct": r.get("correctness", ""),
                    "speedup_shared_over_shuffle": float(r.get("speedup_shared_over_shuffle", 0)),
                    "latency_reduction_percent": "",
                })
        df = pd.DataFrame(rows_reshaped)
        source_label = "Saved SUM-only historical data"
    else:
        st.markdown('<div class="alert-info">No benchmark data found. Run benchmarks from the Live Run tab first.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown(f'<span class="badge-saved">{source_label}</span>', unsafe_allow_html=True)
    st.markdown("")

    # Chart 1 – Latency grouped bar chart
    st.markdown('<div class="section-header">Latency: Shared Memory vs Warp Shuffle</div>', unsafe_allow_html=True)

    ops = sorted(df["operation"].unique()) if "operation" in df.columns else ["sum"]
    sel_op_chart = st.selectbox("Filter by operation", ["All"] + list(ops), key="chart_op")
    chart_df = df if sel_op_chart == "All" else df[df["operation"] == sel_op_chart]

    grouped = chart_df.groupby(["input_size", "algorithm"])["latency_us"].median().reset_index()
    if not grouped.empty:
        fig1 = go.Figure()
        for alg in ["shared", "shuffle"]:
            sub = grouped[grouped["algorithm"] == alg]
            if not sub.empty:
                fig1.add_trace(go.Bar(
                    x=sub["input_size"].astype(str), y=sub["latency_us"],
                    name=alg.capitalize(), marker_color=COLORS.get(alg),
                    text=[f"{v:.2f}µs" for v in sub["latency_us"]],
                    textposition="outside",
                ))
        fig1.update_layout(**PLOTLY_LAYOUT,
                           barmode="group",
                           title="Median Latency by Input Size",
                           xaxis_title="Input Size", yaxis_title="Latency (µs)")
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 – Speedup by input size
    st.markdown('<div class="section-header">Speedup (Shared / Shuffle) by Input Size</div>', unsafe_allow_html=True)
    spd_df = chart_df[chart_df["algorithm"] == "shuffle"].groupby("input_size")["speedup_shared_over_shuffle"].median().reset_index()
    if not spd_df.empty and spd_df["speedup_shared_over_shuffle"].notna().any():
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=spd_df["input_size"].astype(str), y=spd_df["speedup_shared_over_shuffle"],
            mode="lines+markers+text", name="Speedup",
            marker=dict(size=10, color="#a78bfa"),
            line=dict(color="#a78bfa", width=2),
            text=[f"{v:.2f}×" for v in spd_df["speedup_shared_over_shuffle"]],
            textposition="top center",
        ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color=COLORS["ref"], annotation_text="1× baseline")
        fig2.update_layout(**PLOTLY_LAYOUT, title="Speedup (>1 = shuffle is faster)", xaxis_title="Input Size", yaxis_title="Speedup (×)")
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 – Latency reduction %
    st.markdown('<div class="section-header">Latency Reduction %</div>', unsafe_allow_html=True)
    if "latency_reduction_percent" in chart_df.columns:
        lr_df = chart_df[chart_df["algorithm"] == "shuffle"].copy()
        lr_df["latency_reduction_percent"] = pd.to_numeric(lr_df["latency_reduction_percent"], errors="coerce")
        lr_grp = lr_df.groupby("input_size")["latency_reduction_percent"].median().reset_index().dropna()
        if not lr_grp.empty:
            fig3 = go.Figure(go.Bar(
                x=lr_grp["input_size"].astype(str), y=lr_grp["latency_reduction_percent"],
                marker_color=[COLORS["shuffle"] if v > 0 else COLORS["ref"] for v in lr_grp["latency_reduction_percent"]],
                text=[f"{v:.1f}%" for v in lr_grp["latency_reduction_percent"]],
                textposition="outside",
            ))
            fig3.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.2)")
            fig3.update_layout(**PLOTLY_LAYOUT, title="Latency Reduction (%) – positive means shuffle is faster", xaxis_title="Input Size", yaxis_title="Reduction (%)")
            st.plotly_chart(fig3, use_container_width=True)

    # Chart 4 – Log-scale latency
    st.markdown('<div class="section-header">Log-Scale Latency Comparison</div>', unsafe_allow_html=True)
    if not grouped.empty:
        fig4 = go.Figure()
        for alg in ["shared", "shuffle"]:
            sub = grouped[grouped["algorithm"] == alg]
            if not sub.empty:
                fig4.add_trace(go.Scatter(
                    x=sub["input_size"], y=sub["latency_us"],
                    mode="lines+markers", name=alg.capitalize(),
                    marker=dict(size=8, color=COLORS.get(alg)),
                    line=dict(color=COLORS.get(alg), width=2),
                ))
        fig4.update_layout(**PLOTLY_LAYOUT,
                           xaxis_type="log", yaxis_type="log",
                           title="Latency vs Input Size (log-log scale)",
                           xaxis_title="Input Size (log)", yaxis_title="Latency µs (log)")
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB: ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## Warp-Level Primitives for Fast Reduction Operations

### Project Goal
Compare two GPU reduction strategies – **shared-memory tree reduction** and
**warp-shuffle (`__shfl_down_sync`) reduction** – for three operations:
**SUM**, **MAX**, and **ARGMAX**.

---
### Algorithms

| Approach | Description |
|---|---|
| **Shared Memory** | Each thread computes a local partial result, stores it in `__shared__` memory, then participates in a tree reduction with `__syncthreads()` barriers. |
| **Warp Shuffle** | Threads reduce within each 32-thread warp using `__shfl_down_sync`; lane-0 of each warp writes to shared memory; the first warp then reduces those partial sums. |

### ARGMAX Correctness
ARGMAX carries a **(value, index)** pair throughout the entire reduction.
The tie-breaking rule is: **first occurrence wins** (lowest original index for equal maximum values).
The warp-shuffle version shuffles both `val` and `idx` simultaneously to keep the pair consistent.

---
### Benchmark Methodology
- CUDA events wrap the timed section (both kernel stages included).
- Both algorithms warm up on the same input before timing.
- Median latency over N repetitions is reported.
- Speedup = median_shared / median_shuffle.
- Latency reduction (%) = (shared − shuffle) / shared × 100.
- A speedup > 1 means shuffle was faster; < 1 means shared was faster.

---
### Setup & Compile
```bash
# On a CUDA-capable Linux machine / Google Colab:
nvcc -O2 -arch=sm_75 src/reduction_bench.cu -o src/reduction_bench
pip install -r requirements.txt
streamlit run app/app.py
```

### Command-Line Interface
```
./reduction_bench
  --operation   sum|max|argmax
  --algorithm   shared|shuffle|both
  --size        1000000
  --seed        42
  --repetitions 5
  --warmup      50
  --pattern     random_signed|positive|all_equal|dup_max|max_at_end
  --output-format json
```
""")
