"""
app.py
------
Graph-Based Network Risk Simulator
Streamlit web application entry point.

Run with:
    streamlit run app.py
"""

import io
import csv
import math
import streamlit as st
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from graph_utils import (
    build_predefined_graph,
    build_random_graph,
    find_lowest_risk_path,
    find_shortest_time_path,
    node_risk_scores,
    get_edge_data_df,
)
from simulation import simulate_path, hops_to_records, compare_paths

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Risk Simulator",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}
h1, h2, h3, .stMarkdown h1 {
    font-family: 'Share Tech Mono', monospace !important;
}
.stApp { background: #0a0e1a; color: #c9d1e0; }
section[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2a45;
}
.metric-card {
    background: linear-gradient(135deg, #0f1a2e 0%, #162035 100%);
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 6px 0;
    box-shadow: 0 0 12px rgba(0,200,255,0.05);
}
.metric-card .label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #5a7a9a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #00e5ff;
}
.metric-card .value.risk { color: #ff4d6d; }
.metric-card .value.time { color: #ffd166; }
.metric-card .value.detect { color: #06d6a0; }
.path-display {
    font-family: 'Share Tech Mono', monospace;
    background: #060d1a;
    border: 1px solid #1a3050;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #00e5ff;
    letter-spacing: 1px;
    line-height: 1.8;
    overflow-x: auto;
    white-space: nowrap;
}
.section-header {
    font-family: 'Share Tech Mono', monospace;
    color: #00e5ff;
    border-bottom: 1px solid #1a3050;
    padding-bottom: 6px;
    margin: 28px 0 16px 0;
    font-size: 15px;
    letter-spacing: 2px;
}
.stDataFrame { background: #0a0e1a; }
div[data-testid="stExpander"] {
    background: #0d1120;
    border: 1px solid #1e2a45;
    border-radius: 8px;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #5a7a9a !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
}
.stButton > button {
    background: linear-gradient(135deg, #003d6e, #005fa3);
    color: #00e5ff;
    border: 1px solid #007acc;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px;
    font-size: 12px;
    padding: 10px 20px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #005fa3, #0084cc);
    box-shadow: 0 0 16px rgba(0,200,255,0.3);
}
.warning-box {
    background: #1a0a00;
    border-left: 3px solid #ff9800;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #ffa040;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────────────────────────
def init_state():
    defaults = {
        "graph": None,
        "graph_type": None,
        "source": None,
        "target": None,
        "risk_path": None,
        "time_path": None,
        "risk_sim": None,
        "time_sim": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 NETWORK RISK SIMULATOR")
    st.markdown("---")

    graph_mode = st.radio(
        "GRAPH MODE",
        ["Predefined Enterprise Network", "Random Graph Generator"],
        index=0,
    )

    if graph_mode == "Random Graph Generator":
        st.markdown("**GRAPH PARAMETERS**")
        n_nodes = st.slider("Number of Nodes", 5, 30, 10)
        risk_min = st.slider("Min Risk Weight", 0.01, 0.40, 0.05, 0.01)
        risk_max = st.slider("Max Risk Weight", 0.40, 1.00, 0.90, 0.01)
        time_min = st.slider("Min Time Cost", 1, 10, 2)
        time_max = st.slider("Max Time Cost", 10, 60, 20)
        rand_seed = st.number_input("Random Seed", 0, 9999, 42)
    else:
        n_nodes = None
        risk_min = risk_max = time_min = time_max = rand_seed = None

    st.markdown("---")
    st.markdown("**MONITORING SETTINGS**")
    monitoring = st.slider(
        "Monitoring Strength",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Higher = security tools are more sensitive to movement.",
    )

    st.markdown("---")
    if st.button("⚙  GENERATE GRAPH"):
        with st.spinner("Building graph..."):
            if graph_mode == "Predefined Enterprise Network":
                G = build_predefined_graph()
                st.session_state.graph_type = "predefined"
            else:
                G = build_random_graph(
                    n_nodes=n_nodes,
                    risk_min=risk_min,
                    risk_max=risk_max,
                    time_min=time_min,
                    time_max=time_max,
                    seed=int(rand_seed),
                )
                st.session_state.graph_type = "random"

            st.session_state.graph = G
            # Reset paths when graph changes
            for k in ("risk_path", "time_path", "risk_sim", "time_sim", "source", "target"):
                st.session_state[k] = None
        st.success("Graph generated!")

    # Source / Target selection (shown only when graph exists)
    if st.session_state.graph is not None:
        G = st.session_state.graph
        nodes = list(G.nodes())

        st.markdown("---")
        st.markdown("**PATH SELECTION**")
        source = st.selectbox("SOURCE NODE", nodes, index=0)
        target = st.selectbox("TARGET NODE", nodes, index=len(nodes) - 1)

        if st.button("▶  RUN SIMULATION"):
            if source == target:
                st.error("Source and target must be different nodes.")
            else:
                with st.spinner("Computing paths..."):
                    rp, rr, rt = find_lowest_risk_path(G, source, target)
                    tp, tr, tt = find_shortest_time_path(G, source, target)
                    st.session_state.risk_path = rp
                    st.session_state.time_path = tp
                    st.session_state.source = source
                    st.session_state.target = target

                    if rp:
                        st.session_state.risk_sim = simulate_path(G, rp, monitoring, "Lowest-Risk Path")
                    if tp:
                        st.session_state.time_sim = simulate_path(G, tp, monitoring, "Shortest-Time Path")
                st.success("Simulation complete!")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:10px;color:#2a4060;font-family:Share Tech Mono'>"
        "⚠ ACADEMIC RESEARCH TOOL ONLY<br>No real exploit code included.</div>",
        unsafe_allow_html=True,
    )


# ── Main panel ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#00e5ff;font-size:28px;letter-spacing:3px;'>GRAPH-BASED NETWORK RISK SIMULATOR</h1>"
    "<p style='color:#3a5a7a;font-size:13px;font-family:Share Tech Mono;'>Dwell Time & Lateral Movement Analysis · Dijkstra Risk Pathfinding</p>",
    unsafe_allow_html=True,
)

if st.session_state.graph is None:
    st.markdown("""
    <div style="background:#060d1a;border:1px solid #1a3050;border-radius:10px;padding:40px;text-align:center;margin-top:40px;">
        <div style="font-family:Share Tech Mono;color:#00e5ff;font-size:22px;letter-spacing:3px;">← START HERE</div>
        <div style="color:#3a5a7a;margin-top:12px;font-size:14px;">
            Select a graph mode in the sidebar and click <b>GENERATE GRAPH</b> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

G = st.session_state.graph
tab1, tab2, tab3, tab4 = st.tabs(["📡  Graph View", "🔍  Path Analysis", "📊  Simulation Detail", "🗂  Data"])


# ── TAB 1: Graph visualisation ──────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">NETWORK TOPOLOGY</div>', unsafe_allow_html=True)

    risk_path = st.session_state.risk_path
    time_path = st.session_state.time_path

    # Determine layout
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")

    # Node positions
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Node colors by role / sensitivity
    role_colors = {
        "endpoint":  "#1a3a6e",
        "server":    "#1a5a3a",
        "database":  "#5a1a6e",
        "critical":  "#6e1a1a",
        "dmz":       "#5a5a1a",
        "security":  "#1a5a5a",
    }
    node_colors = [
        role_colors.get(G.nodes[n].get("role", "server"), "#1a3060")
        for n in G.nodes()
    ]

    # Edge colors: default, then overlay paths
    edge_colors = ["#1a3050"] * len(G.edges())
    edge_widths = [0.8] * len(G.edges())
    edge_list = list(G.edges())

    def path_edges(path):
        if path:
            return list(zip(path[:-1], path[1:]))
        return []

    risk_edges = set(path_edges(risk_path))
    time_edges = set(path_edges(time_path))

    for i, (u, v) in enumerate(edge_list):
        if (u, v) in risk_edges and (u, v) in time_edges:
            edge_colors[i] = "#ffd166"   # overlap → yellow
            edge_widths[i] = 3.0
        elif (u, v) in risk_edges:
            edge_colors[i] = "#06d6a0"   # green for lowest-risk
            edge_widths[i] = 2.5
        elif (u, v) in time_edges:
            edge_colors[i] = "#ff4d6d"   # red for shortest-time
            edge_widths[i] = 2.5

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowstyle="->",
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        alpha=0.85,
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=900,
        edgecolors="#00e5ff",
        linewidths=0.8,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_color="#c9d1e0",
        font_size=7.5,
        font_family="monospace",
    )

    # Edge risk weight labels (only on highlighted paths for readability)
    path_edge_labels = {}
    for u, v in (risk_edges | time_edges):
        if G.has_edge(u, v):
            r = G[u][v].get("risk_weight", "")
            t = G[u][v].get("time_cost", "")
            path_edge_labels[(u, v)] = f"r={r}\nt={t}"

    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=path_edge_labels,
        font_size=6.5,
        font_color="#aaaaaa",
        font_family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", fc="#0a0e1a", ec="none", alpha=0.7),
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#06d6a0", label="Lowest-Risk Path"),
        mpatches.Patch(color="#ff4d6d", label="Shortest-Time Path"),
        mpatches.Patch(color="#ffd166", label="Shared Edge"),
        mpatches.Patch(color="#1a3050", label="Network Edge"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        facecolor="#0d1120",
        edgecolor="#1e3050",
        labelcolor="#c9d1e0",
        fontsize=8,
        framealpha=0.9,
    )

    ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Heatmap of node risk
    st.markdown('<div class="section-header">NODE RISK HEATMAP</div>', unsafe_allow_html=True)
    scores = node_risk_scores(G)
    nodes_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fig2, ax2 = plt.subplots(figsize=(12, 2.5), facecolor="#0a0e1a")
    ax2.set_facecolor("#0a0e1a")

    names = [n for n, _ in nodes_sorted]
    values = [v for _, v in nodes_sorted]
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=max(values) if values else 1)
    bar_colors = [cmap(norm(v)) for v in values]

    bars = ax2.barh(names, values, color=bar_colors, edgecolor="#0a0e1a", height=0.6)
    ax2.set_xlabel("Composite Risk Score", color="#5a7a9a", fontsize=8)
    ax2.tick_params(colors="#c9d1e0", labelsize=7.5)
    ax2.spines[:].set_visible(False)
    ax2.set_facecolor("#0a0e1a")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1a3050")

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", color="#c9d1e0", fontsize=7,
            fontfamily="monospace",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2, orientation="vertical", fraction=0.02, pad=0.01)
    cbar.ax.tick_params(colors="#5a7a9a", labelsize=7)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ── TAB 2: Path analysis ────────────────────────────────────────────────────────
with tab2:
    risk_sim = st.session_state.risk_sim
    time_sim = st.session_state.time_sim

    if risk_sim is None and time_sim is None:
        st.info("Select source/target nodes in the sidebar and click **RUN SIMULATION** to compute paths.")
        st.stop()

    src = st.session_state.source
    tgt = st.session_state.target
    st.markdown(
        f'<div class="section-header">PATH ANALYSIS · {src} → {tgt}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    # ── Lowest-Risk Path ────────────────────────────────────────────────────────
    with col_a:
        st.markdown("#### 🟢 Lowest-Risk Path")
        if risk_sim:
            path_str = " → ".join(risk_sim.path)
            st.markdown(f'<div class="path-display">{path_str}</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">TOTAL RISK</div>
                    <div class="value risk">{risk_sim.total_risk:.4f}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">TOTAL TIME</div>
                    <div class="value time">{risk_sim.total_time} u</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                det_pct = f"{risk_sim.final_detection_prob:.2%}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">DETECTION PROB</div>
                    <div class="value detect">{det_pct}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No path found between selected nodes.")

    # ── Shortest-Time Path ──────────────────────────────────────────────────────
    with col_b:
        st.markdown("#### 🔴 Shortest-Time Path")
        if time_sim:
            path_str = " → ".join(time_sim.path)
            st.markdown(f'<div class="path-display">{path_str}</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">TOTAL RISK</div>
                    <div class="value risk">{time_sim.total_risk:.4f}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">TOTAL TIME</div>
                    <div class="value time">{time_sim.total_time} u</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                det_pct = f"{time_sim.final_detection_prob:.2%}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">DETECTION PROB</div>
                    <div class="value detect">{det_pct}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No path found between selected nodes.")

    # ── Comparison table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">COMPARISON TABLE</div>', unsafe_allow_html=True)
    comparison_rows = []
    if risk_sim:
        comparison_rows.append({
            "Path Type": "🟢 Lowest-Risk",
            "Hops": len(risk_sim.hops),
            "Total Risk": risk_sim.total_risk,
            "Total Time (units)": risk_sim.total_time,
            "Detection Probability": f"{risk_sim.final_detection_prob:.2%}",
            "Monitoring Strength": monitoring,
        })
    if time_sim:
        comparison_rows.append({
            "Path Type": "🔴 Shortest-Time",
            "Hops": len(time_sim.hops),
            "Total Risk": time_sim.total_risk,
            "Total Time (units)": time_sim.total_time,
            "Detection Probability": f"{time_sim.final_detection_prob:.2%}",
            "Monitoring Strength": monitoring,
        })

    if comparison_rows:
        df_compare = pd.DataFrame(comparison_rows)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

    # ── Detection probability chart over hops ──────────────────────────────────
    st.markdown('<div class="section-header">CUMULATIVE DETECTION PROBABILITY OVER HOPS</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(11, 4), facecolor="#0a0e1a")
    ax3.set_facecolor("#0a0e1a")

    if risk_sim and risk_sim.hops:
        steps_r = [0] + [h.step for h in risk_sim.hops]
        probs_r = [0.0] + [h.cumulative_detection_prob for h in risk_sim.hops]
        ax3.plot(steps_r, probs_r, color="#06d6a0", linewidth=2.2, marker="o", markersize=5, label="Lowest-Risk Path")

    if time_sim and time_sim.hops:
        steps_t = [0] + [h.step for h in time_sim.hops]
        probs_t = [0.0] + [h.cumulative_detection_prob for h in time_sim.hops]
        ax3.plot(steps_t, probs_t, color="#ff4d6d", linewidth=2.2, marker="s", markersize=5, label="Shortest-Time Path")

    ax3.set_xlabel("Hop Number", color="#5a7a9a", fontsize=9)
    ax3.set_ylabel("Cumulative Detection Probability", color="#5a7a9a", fontsize=9)
    ax3.tick_params(colors="#c9d1e0", labelsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.grid(axis="y", color="#1a3050", linestyle="--", alpha=0.5)
    ax3.spines[:].set_edgecolor("#1a3050")
    ax3.legend(facecolor="#0d1120", edgecolor="#1e3050", labelcolor="#c9d1e0", fontsize=9)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ── TAB 3: Simulation detail ────────────────────────────────────────────────────
with tab3:
    if st.session_state.risk_sim is None and st.session_state.time_sim is None:
        st.info("Run a simulation first.")
        st.stop()

    risk_sim = st.session_state.risk_sim
    time_sim = st.session_state.time_sim

    st.markdown('<div class="section-header">HOP-BY-HOP SIMULATION DETAIL</div>', unsafe_allow_html=True)

    if risk_sim:
        with st.expander("🟢  Lowest-Risk Path — Hop Detail", expanded=True):
            df_r = pd.DataFrame(hops_to_records(risk_sim))
            st.dataframe(df_r, use_container_width=True, hide_index=True)

    if time_sim:
        with st.expander("🔴  Shortest-Time Path — Hop Detail", expanded=True):
            df_t = pd.DataFrame(hops_to_records(time_sim))
            st.dataframe(df_t, use_container_width=True, hide_index=True)

    # Monitoring strength sensitivity chart
    st.markdown('<div class="section-header">MONITORING STRENGTH SENSITIVITY</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#5a7a9a;font-size:12px;font-family:Share Tech Mono'>"
        "How detection probability changes as monitoring strength varies (current path fixed).</p>",
        unsafe_allow_html=True,
    )

    m_range = np.linspace(0.1, 3.0, 60)
    fig4, ax4 = plt.subplots(figsize=(11, 4), facecolor="#0a0e1a")
    ax4.set_facecolor("#0a0e1a")

    def detect_for_monitoring(sim, m_vals):
        """Recompute final detection prob for various monitoring strengths."""
        from simulation import simulate_path as _sim
        return [
            _sim(G, sim.path, float(m), "").final_detection_prob
            for m in m_vals
        ]

    if risk_sim:
        ax4.plot(m_range, detect_for_monitoring(risk_sim, m_range),
                 color="#06d6a0", linewidth=2, label="Lowest-Risk Path")
    if time_sim:
        ax4.plot(m_range, detect_for_monitoring(time_sim, m_range),
                 color="#ff4d6d", linewidth=2, label="Shortest-Time Path")

    ax4.axvline(x=monitoring, color="#ffd166", linewidth=1.5, linestyle="--",
                label=f"Current ({monitoring:.1f})")
    ax4.set_xlabel("Monitoring Strength", color="#5a7a9a", fontsize=9)
    ax4.set_ylabel("Final Detection Probability", color="#5a7a9a", fontsize=9)
    ax4.set_ylim(0, 1.05)
    ax4.tick_params(colors="#c9d1e0", labelsize=8)
    ax4.grid(axis="y", color="#1a3050", linestyle="--", alpha=0.5)
    ax4.spines[:].set_edgecolor("#1a3050")
    ax4.legend(facecolor="#0d1120", edgecolor="#1e3050", labelcolor="#c9d1e0", fontsize=9)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)


# ── TAB 4: Data / Export ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">EDGE DATA</div>', unsafe_allow_html=True)
    edge_rows = get_edge_data_df(G)
    df_edges = pd.DataFrame(edge_rows)
    st.dataframe(df_edges, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">NODE DATA</div>', unsafe_allow_html=True)
    node_rows = [
        {
            "Node": n,
            "Role": G.nodes[n].get("role", "-"),
            "Sensitivity": G.nodes[n].get("sensitivity", "-"),
            "Risk Score": node_risk_scores(G).get(n, 0),
        }
        for n in G.nodes()
    ]
    df_nodes = pd.DataFrame(node_rows)
    st.dataframe(df_nodes, use_container_width=True, hide_index=True)

    # CSV export
    st.markdown('<div class="section-header">EXPORT RESULTS</div>', unsafe_allow_html=True)

    def df_to_csv_bytes(df):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode()

    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.download_button(
            "⬇  Download Edge Data CSV",
            data=df_to_csv_bytes(df_edges),
            file_name="network_edges.csv",
            mime="text/csv",
        )
    with col_y:
        st.download_button(
            "⬇  Download Node Data CSV",
            data=df_to_csv_bytes(df_nodes),
            file_name="network_nodes.csv",
            mime="text/csv",
        )
    with col_z:
        if st.session_state.risk_sim and st.session_state.time_sim:
            records_r = hops_to_records(st.session_state.risk_sim)
            records_t = hops_to_records(st.session_state.time_sim)
            for r in records_r:
                r["Path Type"] = "Lowest-Risk"
            for r in records_t:
                r["Path Type"] = "Shortest-Time"
            df_results = pd.DataFrame(records_r + records_t)
            st.download_button(
                "⬇  Download Simulation Results CSV",
                data=df_to_csv_bytes(df_results),
                file_name="simulation_results.csv",
                mime="text/csv",
            )
        else:
            st.button("⬇  Download Simulation Results CSV", disabled=True)