"""
================================================================================
                                CYBERGUARD
       Advanced AI Threat Intelligence, XAI Forensics & Defense Console
================================================================================

Features:
1. Front-and-Center XAI Forensics & Incident Response Hub
2. Native SHAP (TreeExplainer) & LIME (Local Surrogate) Feature Attributions
3. Deep Packet Packaging Diagnostics (Wireshark-grade Header & Payload Forensic Breakdown)
4. Firewall Breaking & Perimeter Penetration Analysis
5. Cyber Officer Action Station with Executable Firewall Rules (iptables, nftables, Snort/Suricata, AWS WAF)
6. Interactive Threat Neutralization & Mitigation Simulator
7. Live Network Flow Triage Table with Persistent Deterministic Flow IDs
8. Dual-Path Routing Sankey Flow & Autoencoder Manifold Scatter Plot
9. One-Click CISO Executive Presentation (.pptx) Generator

Author: CyberGuard AI Engineering Team
Version: 3.0.0
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cyberguard_core import CyberGuardEngine, SyntheticTrafficGenerator, ThreatVerdict
from src.generate_ppt import generate_cyberguard_presentation
from src.cyberguard_xai import (
    CyberGuardXAI,
    AttackAnatomyAnalyzer,
    CyberDefensePlaybook,
    XAIExplanation,
    AttackAnatomy,
    DefensePlaybook,
)
from src.attack_simulator import inject_attack


# ============================================================================
# Streamlit Page Configuration & Modern Cybersecurity Theme
# ============================================================================

st.set_page_config(
    page_title="CyberGuard SOC | Advanced Threat Intelligence & XAI Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Cyber Dark CSS with Neon Accents & Glassmorphism
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp {
        background-color: #060913;
        color: #E2E8F0;
        font-family: 'Outfit', sans-serif;
    }

    /* Top Command Header */
    .cyber-header {
        background: linear-gradient(135deg, rgba(8, 14, 34, 0.95) 0%, rgba(17, 27, 54, 0.9) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 229, 255, 0.12), inset 0 0 16px rgba(0, 229, 255, 0.05);
        backdrop-filter: blur(12px);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .cyber-title {
        font-size: 26px;
        font-weight: 800;
        color: #00E5FF;
        letter-spacing: 0.8px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        text-shadow: 0 0 12px rgba(0, 229, 255, 0.4);
    }
    .cyber-subtitle {
        font-size: 13.5px;
        color: #94A3B8;
        margin-top: 6px;
    }

    /* Metric KPI Cards */
    .metric-card {
        background: rgba(13, 22, 44, 0.8);
        border: 1px solid rgba(30, 48, 80, 0.9);
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: #00E5FF;
        transform: translateY(-3px);
        box-shadow: 0 6px 24px rgba(0, 229, 255, 0.2);
    }
    .metric-val {
        font-size: 27px;
        font-weight: 700;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-lbl {
        font-size: 11px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    /* Analysis & Anatomy Containers */
    .cyber-box {
        background: #091024;
        border: 1px solid #1E2D4A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .cyber-box-title {
        font-size: 16px;
        font-weight: 700;
        color: #00E5FF;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Packet Dissector Terminal Box */
    .packet-dissector {
        background: #030712;
        border: 1px solid #00E5FF50;
        border-radius: 8px;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: #E2E8F0;
        line-height: 1.7;
        box-shadow: inset 0 0 16px rgba(0, 229, 255, 0.08);
        margin: 12px 0;
    }
    .packet-header-tag {
        color: #00E5FF;
        font-weight: 700;
    }
    .packet-val-warn {
        color: #EF4444;
        font-weight: 700;
    }
    .packet-val-ok {
        color: #10B981;
        font-weight: 700;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-benign {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .badge-known {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
    }
    .badge-zeroday {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid #A855F7;
    }
    .badge-lowconf {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }
    .badge-mitigated {
        background: rgba(14, 165, 233, 0.2);
        color: #38BDF8;
        border: 1px solid #38BDF8;
    }

    /* Playbook Step styling */
    .playbook-phase {
        background: #0B142C;
        border-left: 3px solid #00E5FF;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        line-height: 1.5;
    }

    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Cached Model & Data Loaders
# ============================================================================

@st.cache_resource
def load_cyberguard_engine(models_dir: str = "models") -> CyberGuardEngine:
    """Load trained CyberGuard engine artifacts once into memory."""
    engine = CyberGuardEngine(models_dir)
    engine.load()
    return engine


@st.cache_data
def load_dataset(filepath: str) -> pd.DataFrame:
    """Load telemetry dataset from disk."""
    return pd.read_csv(filepath)


@st.cache_resource
def load_xai_explainer(_engine: CyberGuardEngine, _bg_df: pd.DataFrame) -> CyberGuardXAI:
    """Initialize cached SHAP & LIME XAI Orchestrator."""
    return CyberGuardXAI(_engine, background_df=_bg_df)


# ============================================================================
# Main Dashboard Application
# ============================================================================

def main():
    models_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data" / "synthetic"

    # --- Session State Initialization (Guarantees batch size & selection stability) ---
    if "stream_batch_size" not in st.session_state:
        st.session_state.stream_batch_size = 500
    if "selected_flow_id" not in st.session_state:
        st.session_state.selected_flow_id = "FLOW-0002"  # Default to a prominent attack flow
    if "mitigated_flows" not in st.session_state:
        st.session_state.mitigated_flows = set()

    # --- Header Banner ---
    st.markdown(
        """
        <div class="cyber-header">
            <div>
                <div class="cyber-title">
                    <span>🛡️ CYBERGUARD SOC</span>
                    <span style="font-size: 14px; color: #475569; font-weight: 500;">//</span>
                    <span style="font-size: 17px; color: #CBD5E1; font-weight: 600;">Advanced Threat Forensics & XAI Defense Console</span>
                </div>
                <div class="cyber-subtitle">
                    Dual-Path Neural Protection: <b>Path A (XGBoost Signature Classifier)</b> & 
                    <b>Path B (Deep Autoencoder Zero-Day Detector)</b> with <b>SHAP & LIME XAI Forensics</b>
                </div>
            </div>
            <div>
                <span class="badge badge-benign" style="font-size: 12px; padding: 6px 14px;">
                    ● LIVE TELEMETRY INTERCEPTOR: ACTIVE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Verify Models Exist ---
    if not (models_dir / "preprocessor.joblib").exists():
        st.error("⚠️ Trained models not found in `models/`. Please run `python src/cyberguard_core.py --mode demo` first.")
        st.stop()

    engine = load_cyberguard_engine(str(models_dir))

    # ========================================================================
    # Sidebar Controls & Telemetry Stream Ingestion
    # ========================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=56)
        st.subheader("⚙️ Telemetry Controls")

        available_datasets = []
        if data_dir.exists():
            available_datasets = [f.name for f in sorted(data_dir.glob("*.csv"))]

        data_source_mode = st.radio(
            "Telemetry Ingestion Source:",
            ["Synthetic Benchmark Feed", "Upload Custom CSV"],
            help="Select live telemetry stream or upload network flow capture",
        )

        selected_df = None
        if data_source_mode == "Synthetic Benchmark Feed":
            if available_datasets:
                default_idx = 0
                if st.session_state.get("active_dataset_file") in available_datasets:
                    default_idx = available_datasets.index(st.session_state["active_dataset_file"])
                elif "test_known_traffic.csv" in available_datasets:
                    default_idx = available_datasets.index("test_known_traffic.csv")
                chosen_file = st.selectbox("Select Traffic Dataset:", available_datasets, index=default_idx)
                selected_df = load_dataset(str(data_dir / chosen_file))
            else:
                st.warning("No pre-generated synthetic datasets found. Generating live test feed...")
                selected_df = SyntheticTrafficGenerator.generate_dataset(n_samples=2500)
        else:
            uploaded_file = st.file_uploader("Upload Traffic CSV:", type=["csv"])
            if uploaded_file:
                selected_df = pd.read_csv(uploaded_file)
            else:
                st.info("Please upload a CSV with network flow features.")
                st.stop()

        # --- Red Team Hacker Attack Simulator Panel ---
        st.divider()
        st.markdown("### 🔴 Red Team Attack Injector")
        st.caption("Simulate attacking your system as a hacker to watch CyberGuard intercept and resist it in real-time.")

        attack_type_selection = st.selectbox(
            "Attack Vector to Launch:",
            [
                "🌊 TCP SYN Flood (hping3 style)",
                "🕵️ Stealth Port Scan (Nmap style)",
                "💥 Layer-7 DDoS HTTP Flood (LOIC)",
                "🔑 SSH Password Brute Force (Hydra)",
                "☣️ Polymorphic Zero-Day Exploit",
            ],
            key="sim_attack_vector",
        )

        vector_key_map = {
            "🌊 TCP SYN Flood (hping3 style)": "syn_flood",
            "🕵️ Stealth Port Scan (Nmap style)": "port_scan",
            "💥 Layer-7 DDoS HTTP Flood (LOIC)": "http_flood",
            "🔑 SSH Password Brute Force (Hydra)": "brute_force",
            "☣️ Polymorphic Zero-Day Exploit": "zero_day",
        }

        if st.button("🚀 Launch Attack on System", type="secondary", width="stretch"):
            v_key = vector_key_map[attack_type_selection]
            inject_attack(v_key, n_samples=120, mix_benign=True)
            st.session_state.active_dataset_file = "live_attack_stream.csv"
            st.session_state.selected_flow_id = "FLOW-0026"
            st.toast(f"🔥 {attack_type_selection} launched! Ingested into CyberGuard defense engine.", icon="🚨")
            st.rerun()

        # Telemetry Stream Window Controls (Guaranteed Stable via Session State)
        total_available = len(selected_df)
        st.markdown("#### 📡 Stream Ingestion Window")
        
        # Batch size preset pills for instant stable control
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("250 Flows", width="stretch"):
                st.session_state.stream_batch_size = 250
                st.rerun()
        with col_p2:
            if st.button("500 Flows", width="stretch"):
                st.session_state.stream_batch_size = 500
                st.rerun()
        with col_p3:
            if st.button("1000 Flows", width="stretch"):
                st.session_state.stream_batch_size = min(total_available, 1000)
                st.rerun()

        sample_size = st.slider(
            "Telemetry Stream Batch Size:",
            min_value=50,
            max_value=min(total_available, 3000),
            value=min(st.session_state.stream_batch_size, total_available),
            step=50,
            key="slider_batch_size",
        )
        st.session_state.stream_batch_size = sample_size

        # Fixed deterministic head slice with Unique Persistent Flow IDs
        traffic_stream_df = selected_df.head(sample_size).copy().reset_index(drop=True)
        traffic_stream_df["Flow_ID"] = [f"FLOW-{i+1:04d}" for i in range(len(traffic_stream_df))]

        st.divider()
        st.markdown("### 🔍 Model Engine Status")
        st.markdown(f"- **Path A Classes:** `{len(engine.preprocessor.classes_)} classes`")
        st.markdown(f"- **Path B Features:** `{engine.path_b.input_dim} dimensions`")
        st.markdown(f"- **Autoencoder Cutoff ($T$):** `{engine.path_b.threshold:.5f}`")

        st.divider()
        st.markdown("### 📑 CISO Presentation")
        st.caption("Generate a publication-grade 16:9 PowerPoint deck with embedded confusion heatmaps and threat metrics.")

        if st.button("🚀 Generate CISO Deck", type="primary", width="stretch"):
            with st.spinner("Compiling high-resolution charts & generating .pptx..."):
                try:
                    ppt_bytes = generate_cyberguard_presentation(
                        models_dir=str(models_dir),
                        data_dir=str(data_dir),
                        output_path=None,
                    )
                    st.download_button(
                        label="📥 Download Executive Deck (.pptx)",
                        data=ppt_bytes,
                        file_name="CyberGuard_Executive_Briefing.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        width="stretch",
                    )
                    st.success("PowerPoint ready for download!")
                except Exception as e:
                    st.error(f"Error generating PowerPoint: {e}")

    # ========================================================================
    # Inference Engine Execution
    # ========================================================================
    start_inf = time.time()
    verdicts = engine.analyze_traffic(traffic_stream_df)
    inf_duration = time.time() - start_inf

    # Build Analysis DataFrame with Persistent Flow IDs
    analysis_records = []
    for i, v in enumerate(verdicts):
        rec = v.to_dict()
        rec["Flow_ID"] = traffic_stream_df.iloc[i]["Flow_ID"]
        rec["is_mitigated"] = rec["Flow_ID"] in st.session_state.mitigated_flows
        analysis_records.append(rec)
    res_df = pd.DataFrame(analysis_records)

    # Attach Ground Truth and Raw Features
    if "label" in traffic_stream_df.columns:
        res_df["ground_truth"] = traffic_stream_df["label"].values

    # Merge Raw Telemetry Features for XAI & Anatomy
    res_df = pd.concat([res_df, traffic_stream_df[[c for c in traffic_stream_df.columns if c not in res_df.columns]]], axis=1)

    # Initialize XAI Engine with background data
    xai_engine = load_xai_explainer(engine, selected_df.head(150))

    # ========================================================================
    # Top Metrics Banner
    # ========================================================================
    total_inspected = len(res_df)
    benign_count = (res_df["status"] == "BENIGN").sum()
    known_attack_count = (res_df["status"] == "KNOWN_ATTACK").sum()
    zero_day_count = (res_df["status"] == "SUSPECTED_ZERO_DAY").sum()
    low_conf_count = (res_df["status"] == "LOW_CONF_ALERT").sum()
    mitigated_count = len(st.session_state.mitigated_flows)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Flows Inspected</div>
                <div class="metric-val" style="color: #00E5FF;">{total_inspected:,}</div>
                <div style="font-size: 11px; color: #64748B;">{inf_duration*1000:.1f} ms latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Known Attacks (Path A)</div>
                <div class="metric-val" style="color: #EF4444;">{known_attack_count:,}</div>
                <div style="font-size: 11px; color: #64748B;">XGBoost Interceptions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Zero-Day Threats (Path B)</div>
                <div class="metric-val" style="color: #C084FC;">{zero_day_count:,}</div>
                <div style="font-size: 11px; color: #64748B;">Autoencoder Outliers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Benign Baseline</div>
                <div class="metric-val" style="color: #10B981;">{benign_count:,}</div>
                <div style="font-size: 11px; color: #64748B;">{(benign_count/total_inspected)*100:.1f}% normal traffic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Officer Mitigations</div>
                <div class="metric-val" style="color: #38BDF8;">{mitigated_count}</div>
                <div style="font-size: 11px; color: #64748B;">Firewall Rules Enacted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================================
    # Multi-Tab Platform: TAB 1 IS NOW THE PRIMARY XAI & FORENSICS CENTER!
    # ========================================================================
    tab_xai, tab_triage, tab_overview, tab_export = st.tabs([
        "🔬 1. Threat Forensics & XAI Center (SHAP, LIME, Packaging & Defense)",
        "🚨 2. Live Threat Triage Feed Table",
        "📊 3. Executive Defense Overview (Sankey & Loss Manifold)",
        "📑 4. CISO Presentation Export",
    ])

    # ------------------------------------------------------------------------
    # TAB 1: THREAT FORENSICS & XAI CENTER (FRONT AND CENTER!)
    # ------------------------------------------------------------------------
    with tab_xai:
        st.subheader("🔬 Deep Threat Forensics, Explainable AI & Officer Action Station")
        st.caption("Direct investigation of network packet packaging, firewall breaking vectors, SHAP/LIME feature attributions, and actionable mitigation commands.")

        # --- Incident Selection Bar ---
        threat_flows = res_df[res_df["status"] != "BENIGN"]
        threat_options = []
        
        # Build curated label for dropdown
        for _, row in res_df.iterrows():
            f_id = row["Flow_ID"]
            a_type = row["attack_type"]
            sev = row["threat_level"]
            stat = row["status"]
            mit = " [🛡️ MITIGATED]" if row.get("is_mitigated", False) else ""
            threat_options.append(f"{f_id} | [{sev}] {a_type} ({stat}){mit}")

        # Ensure default selected flow is recognized
        current_index = 0
        for idx, opt in enumerate(threat_options):
            if st.session_state.selected_flow_id in opt:
                current_index = idx
                break

        col_select_threat, col_quick_btn = st.columns([3, 1])
        with col_select_threat:
            chosen_threat_label = st.selectbox(
                "🎯 Select Network Flow for Deep Forensic Dissection:",
                threat_options,
                index=current_index,
                help="Choose any active attack or zero-day flow to inspect packaging, firewall breaking, and XAI explanations.",
            )
            # Extract Flow ID
            active_flow_id = chosen_threat_label.split(" | ")[0].strip()
            st.session_state.selected_flow_id = active_flow_id

        with col_quick_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            # Find first zero-day or prominent attack
            zd_rows = res_df[res_df["status"] == "SUSPECTED_ZERO_DAY"]
            if not zd_rows.empty:
                zd_first_id = zd_rows.iloc[0]["Flow_ID"]
                if st.button("⚡ Inspect Zero-Day Anomaly", width="stretch"):
                    st.session_state.selected_flow_id = zd_first_id
                    st.rerun()

        # Retrieve chosen flow record
        flow_record = res_df[res_df["Flow_ID"] == active_flow_id].iloc[0]
        flow_df = pd.DataFrame([flow_record])
        verdict_dict = flow_record.to_dict()

        # Generate Forensics, XAI, and Defense Playbook
        anatomy = AttackAnatomyAnalyzer.analyze(flow_record, verdict_dict)
        playbook = CyberDefensePlaybook.generate(flow_record, verdict_dict)
        is_already_mitigated = active_flow_id in st.session_state.mitigated_flows

        # --- High-Tech Flow Profile HUD Card ---
        status_badge_class = (
            "badge-benign" if flow_record["status"] == "BENIGN"
            else ("badge-zeroday" if flow_record["status"] == "SUSPECTED_ZERO_DAY"
            else "badge-known")
        )
        mitigated_badge = (
            '<span class="badge badge-mitigated">🛡️ DEFENSE ACTIVE (CONTAINED)</span>'
            if is_already_mitigated
            else '<span class="badge badge-known">⚠️ ACTIVE INTRUSION</span>'
        )

        st.markdown(
            f"""
            <div class="cyber-box" style="border-left: 5px solid #00E5FF;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span style="font-size: 22px; font-weight: 800; color: #00E5FF; font-family: 'JetBrains Mono';">
                            {active_flow_id}
                        </span>
                        <span style="margin-left: 12px;" class="badge {status_badge_class}">
                            {flow_record['status']}
                        </span>
                        <span style="margin-left: 8px;" class="badge badge-zeroday">
                            {flow_record['attack_type']}
                        </span>
                        <span style="margin-left: 8px;">{mitigated_badge}</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 14px; color: #94A3B8;">
                        Severity: <b style="color: {'#EF4444' if flow_record['threat_level'] in ['HIGH', 'CRITICAL'] else '#10B981'}; font-size: 16px;">{flow_record['threat_level']}</b>
                    </div>
                </div>
                <div style="display: flex; gap: 20px; font-size: 13.5px; color: #94A3B8; flex-wrap: wrap;">
                    <div>Protocol: <b style="color: #E2E8F0;">{flow_record.get('protocol_type', 'TCP')}</b></div>
                    <div>Target Service: <b style="color: #E2E8F0;">{flow_record.get('service', 'http')}</b></div>
                    <div>TCP Flag: <b style="color: {'#EF4444' if flow_record.get('flag') in ['S0', 'RSTO', 'REJ'] else '#10B981'}; font-weight: 700;">{flow_record.get('flag', 'SF')}</b></div>
                    <div>Payload Size: <b style="color: #E2E8F0;">{flow_record.get('src_bytes', 0)} B in / {flow_record.get('dst_bytes', 0)} B out</b></div>
                    <div>Velocity: <b style="color: #E2E8F0;">{flow_record.get('packet_rate', 0):,.1f} pkts/s</b></div>
                    <div>Path A Conf: <b style="color: #00E5FF;">{flow_record.get('path_a_confidence', 0)*100:.1f}%</b></div>
                    <div>Path B MSE: <b style="color: #C084FC;">{flow_record.get('path_b_recon_error', 0):.5f}</b> (Cutoff: {flow_record.get('path_b_threshold', 0):.5f})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================================
        # SECTION A: PACKET PACKAGING & FIREWALL BREAKING FORENSICS
        # ====================================================================
        st.markdown("### 📦 1. Packet Packaging Dynamics & Firewall Breaking Vectors")
        st.caption("How the attack packets were structured, how they attack or bypass network firewalls, and why the alarm fired.")

        col_pack, col_fw = st.columns([1, 1])

        with col_pack:
            # Simulated realistic packet header inspection box
            protocol_name = flow_record.get('protocol_type', 'TCP')
            service_name = flow_record.get('service', 'http')
            tcp_flag = flow_record.get('flag', 'SF')
            s_bytes = flow_record.get('src_bytes', 0)
            d_bytes = flow_record.get('dst_bytes', 0)
            p_rate = flow_record.get('packet_rate', 0)
            c_count = flow_record.get('count', 0)
            s_err = flow_record.get('serror_rate', 0)

            flag_status_class = "packet-val-warn" if tcp_flag in ["S0", "RSTO", "REJ"] else "packet-val-ok"

            st.markdown(
                f"""
                <div class="cyber-box">
                    <div class="cyber-box-title">
                        <span>📦 Crafted Packet Packaging Dissector</span>
                    </div>
                    <div class="packet-dissector">
                        <div><span class="packet-header-tag">[FRAME]</span> Protocol: <b>{protocol_name}</b> | Service: <b>{service_name}</b></div>
                        <div><span class="packet-header-tag">[LAYER 3]</span> IP Header: Src=192.168.1.105:49152 &rarr; Dst=10.0.0.5:{'80' if service_name=='http' else ('443' if service_name=='ssl' else '22')}</div>
                        <div><span class="packet-header-tag">[LAYER 4]</span> TCP Flags: <span class="{flag_status_class}">[{tcp_flag}]</span> ({AttackAnatomyAnalyzer._flag_description(tcp_flag)})</div>
                        <div><span class="packet-header-tag">[PAYLOAD]</span> Inbound: <b>{s_bytes} Bytes</b> | Outbound: <b>{d_bytes} Bytes</b> (Ratio: {s_bytes/max(d_bytes,1):.2f})</div>
                        <div><span class="packet-header-tag">[VELOCITY]</span> Burst Rate: <b>{p_rate:,.1f} pkts/sec</b> | Concurrency: <b>{c_count} flows</b></div>
                        <div><span class="packet-header-tag">[ANOMALY]</span> SYN Error Rate: <span class="{flag_status_class}">{s_err*100:.1f}%</span></div>
                    </div>
                    <p style="font-size: 13.5px; line-height: 1.6; color: #CBD5E1; margin-top: 10px;">
                        <b>Forensic Packaging Diagnosis:</b><br>{anatomy.packaging_summary}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_fw:
            st.markdown(
                f"""
                <div class="cyber-box">
                    <div class="cyber-box-title" style="color: #EF4444;">
                        <span>🔥 Firewall Breaking & Evasion Mechanics</span>
                    </div>
                    <div style="margin-bottom: 12px;">
                        Perimeter Impact Severity: 
                        <span class="badge {'badge-known' if anatomy.firewall_impact_level == 'CRITICAL' else 'badge-lowconf'}">
                            {anatomy.firewall_impact_level} FIREWALL IMPACT
                        </span>
                    </div>
                    <p style="font-size: 13.5px; line-height: 1.6; color: #CBD5E1;">
                        <b>Perimeter Defense Attack Vector:</b><br>
                        {anatomy.firewall_breaking_vector}
                    </p>
                    <hr style="border-color: #1E2D4A; margin: 14px 0;">
                    <div class="cyber-box-title" style="font-size: 14px; color: #F59E0B;">
                        <span>🚨 Alarm Causality Trigger</span>
                    </div>
                    <p style="font-size: 13px; line-height: 1.5; color: #94A3B8;">
                        {anatomy.alarm_causality}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ====================================================================
        # SECTION B: EXPLAINABLE AI (SHAP & LIME)
        # ====================================================================
        st.markdown("### 🧠 2. Explainable AI (SHAP & LIME) Feature Attributions")
        st.caption("Inspect mathematically rigorous local feature attributions explaining exactly why the models classified this threat.")

        xai_mode = st.radio(
            "Select XAI Mathematical Explainer:",
            [
                "⚡ SHAP (TreeExplainer) - Shapley Feature Attributions (φ)",
                "🔍 LIME (Local Surrogate) - Decision Boundary Rules",
                "🧬 Deep Autoencoder - Latent Manifold Reconstruction Decomposition",
            ],
            horizontal=True,
        )

        # Compute XAI Explanation
        if "SHAP" in xai_mode:
            explanation = xai_engine.explain_flow_shap(flow_df)
        elif "LIME" in xai_mode:
            explanation = xai_engine.explain_flow_lime(flow_df)
        else:
            explanation = xai_engine.explain_autoencoder_anomaly(flow_df)

        feat_df = pd.DataFrame([f.to_dict() for f in explanation.top_features])

        if not feat_df.empty:
            col_chart, col_table = st.columns([3, 2])

            with col_chart:
                st.markdown(f"#### 📊 {explanation.method} Attribution Plot")
                st.caption(explanation.summary_text)

                # Color mapping: Red for pushing towards attack, Green for pushing benign
                colors = ["#EF4444" if imp == "PUSHES_ATTACK" else "#10B981" for imp in feat_df["impact"]]

                fig_xai = go.Figure(
                    go.Bar(
                        x=feat_df["attribution"],
                        y=feat_df["feature"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.4f}" for v in feat_df["attribution"]],
                        textposition="auto",
                    )
                )
                fig_xai.update_layout(
                    paper_bgcolor="#091024",
                    plot_bgcolor="#091024",
                    font=dict(color="#F8FAFC", size=11, family="Outfit"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=320,
                    xaxis=dict(gridcolor="#1E2D4A", title="Attribution Impact (φ)"),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_xai, width="stretch")

            with col_table:
                st.markdown("#### 📋 Feature Decision Rules & Observed Values")
                st.caption("Condition thresholds governing the model's localized decision boundary.")
                st.dataframe(
                    feat_df[["feature", "feature_value", "attribution", "condition", "impact"]].rename(
                        columns={
                            "feature": "Feature",
                            "feature_value": "Telemetry Value",
                            "attribution": "Attribution",
                            "condition": "Rule Condition",
                            "impact": "Direction",
                        }
                    ),
                    width="stretch",
                    height=320,
                )

        # ====================================================================
        # SECTION C: CYBER OFFICER ACTION STATION & FIREWALL PLAYBOOK
        # ====================================================================
        st.markdown("### 🛡️ 3. Cyber Officer Incident Defense & Mitigation Station")
        st.caption("Equipping the SOC Analyst to resist, isolate, and neutralize this attack using executable firewall rules and guided playbooks.")

        # Interactive Containment Action Bar
        col_banner, col_action_btn = st.columns([3, 1])
        with col_banner:
            if is_already_mitigated:
                st.success(f"🛡️ Threat **{active_flow_id}** is **NEUTRALIZED & MITIGATED**. Firewall drop rules active on ingress gateway.")
            elif flow_record["status"] != "BENIGN":
                st.error(f"🚨 **ACTIVE INCIDENT**: Signature **{flow_record['attack_type']}** requires immediate Cyber Officer intervention.")
            else:
                st.info(f"🟢 Flow **{active_flow_id}** is verified Benign. No perimeter countermeasures required.")

        with col_action_btn:
            if not is_already_mitigated and flow_record["status"] != "BENIGN":
                if st.button("🛡️ Neutralize & Deploy Firewall Rule", type="primary", width="stretch"):
                    st.session_state.mitigated_flows.add(active_flow_id)
                    st.rerun()
            elif is_already_mitigated:
                if st.button("🔄 Rollback Mitigation Rule", width="stretch"):
                    st.session_state.mitigated_flows.remove(active_flow_id)
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Executable Firewall Rules
        st.markdown("#### ⚡ Ready-to-Run Firewall Defense Rules")
        rule_tab1, rule_tab2, rule_tab3, rule_tab4 = st.tabs([
            "Linux iptables Script",
            "Modern nftables",
            "Snort / Suricata IDS Signature",
            "Cloud WAF / AWS ACL",
        ])
        with rule_tab1:
            st.code(playbook.iptables_rule, language="bash")
        with rule_tab2:
            st.code(playbook.nftables_rule, language="bash")
        with rule_tab3:
            st.code(playbook.snort_suricata_rule, language="bash")
        with rule_tab4:
            st.code(playbook.aws_waf_rule, language="bash")

        # 3-Phase SOC Playbook Cards
        st.markdown("#### 📋 3-Phase Incident Response Playbook")
        col_pb1, col_pb2, col_pb3 = st.columns(3)
        with col_pb1:
            st.markdown(
                """
                <div class="cyber-box">
                    <div class="cyber-box-title" style="color: #EF4444;">🚨 Phase 1: Containment</div>
                """,
                unsafe_allow_html=True,
            )
            for step in playbook.phase1_containment:
                st.markdown(f'<div class="playbook-phase" style="border-left-color: #EF4444;">{step}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_pb2:
            st.markdown(
                """
                <div class="cyber-box">
                    <div class="cyber-box-title" style="color: #F59E0B;">🧹 Phase 2: Eradication</div>
                """,
                unsafe_allow_html=True,
            )
            for step in playbook.phase2_eradication:
                st.markdown(f'<div class="playbook-phase" style="border-left-color: #F59E0B;">{step}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_pb3:
            st.markdown(
                """
                <div class="cyber-box">
                    <div class="cyber-box-title" style="color: #10B981;">🔒 Phase 3: Hardening</div>
                """,
                unsafe_allow_html=True,
            )
            for step in playbook.phase3_hardening:
                st.markdown(f'<div class="playbook-phase" style="border-left-color: #10B981;">{step}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # TAB 2: LIVE THREAT TRIAGE FEED TABLE
    # ------------------------------------------------------------------------
    with tab_triage:
        st.subheader("🚨 Live Network Telemetry Triage Table")
        st.caption("Filter and search through inspected telemetry events. Click any row to jump into deep XAI forensics.")

        f_col1, f_col2, f_col3 = st.columns([2, 2, 3])
        with f_col1:
            status_filter = st.selectbox(
                "Filter by Hybrid Verdict:",
                ["ALL", "KNOWN_ATTACK", "SUSPECTED_ZERO_DAY", "LOW_CONF_ALERT", "BENIGN"],
                index=0,
            )
        with f_col2:
            severity_filter = st.selectbox(
                "Filter by Severity:",
                ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                index=0,
            )
        with f_col3:
            search_query = st.text_input("Search Attack Type / Keyword / Flow ID:", "")

        # Apply Filters
        filtered_df = res_df.copy()
        if status_filter != "ALL":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]
        if severity_filter != "ALL":
            filtered_df = filtered_df[filtered_df["threat_level"] == severity_filter]
        if search_query:
            filtered_df = filtered_df[
                filtered_df["Flow_ID"].str.contains(search_query, case=False, na=False)
                | filtered_df["attack_type"].str.contains(search_query, case=False, na=False)
                | filtered_df["explanation"].str.contains(search_query, case=False, na=False)
            ]

        # Display Table
        display_cols = [
            "Flow_ID",
            "status",
            "attack_type",
            "threat_level",
            "path_a_prediction",
            "path_a_confidence",
            "path_b_recon_error",
            "path_b_is_anomaly",
            "is_mitigated",
            "explanation",
        ]
        if "ground_truth" in filtered_df.columns:
            display_cols.insert(1, "ground_truth")

        table_render = filtered_df[display_cols].rename(
            columns={
                "Flow_ID": "Flow ID",
                "ground_truth": "Ground Truth",
                "status": "Hybrid Verdict",
                "attack_type": "Attack Type",
                "threat_level": "Severity",
                "path_a_prediction": "Path A Match",
                "path_a_confidence": "Path A Conf",
                "path_b_recon_error": "Path B MSE",
                "path_b_is_anomaly": "Path B Anomaly",
                "is_mitigated": "Mitigated?",
                "explanation": "Triage Reasoning",
            }
        )
        st.dataframe(table_render, width="stretch", height=420)

    # ------------------------------------------------------------------------
    # TAB 3: EXECUTIVE DEFENSE OVERVIEW (SANKEY & MANIFOLD SCATTER)
    # ------------------------------------------------------------------------
    with tab_overview:
        st.subheader("📊 Executive Threat Routing & Autoencoder Manifold")
        st.caption("Visualizing flow triage from Raw Telemetry through Path A (XGBoost) and Path B (Autoencoder).")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🔀 Dual-Path Threat Routing Sankey Flow")
            sankey_labels = [
                f"Raw Telemetry ({total_inspected})",
                "Path A: XGBoost",
                "Path B: Autoencoder",
                f"BENIGN ({benign_count})",
                f"KNOWN ATTACK ({known_attack_count})",
                f"ZERO-DAY ANOMALY ({zero_day_count})",
                f"LOW CONF ALERT ({low_conf_count})",
            ]
            sankey_colors = ["#00E5FF", "#3B82F6", "#8B5CF6", "#10B981", "#EF4444", "#C084FC", "#F59E0B"]

            sources = [0, 0, 1, 1, 2, 2]
            targets = [1, 2, 4, 6, 3, 5]
            values = [
                known_attack_count + low_conf_count,
                benign_count + zero_day_count,
                max(known_attack_count, 1),
                max(low_conf_count, 1),
                max(benign_count, 1),
                max(zero_day_count, 1),
            ]

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=18,
                            thickness=20,
                            line=dict(color="#1E293B", width=1.0),
                            label=sankey_labels,
                            color=sankey_colors,
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                            color="rgba(0, 229, 255, 0.22)",
                        ),
                    )
                ]
            )
            fig_sankey.update_layout(
                paper_bgcolor="#091024",
                plot_bgcolor="#091024",
                font=dict(color="#F8FAFC", size=11, family="Outfit"),
                margin=dict(l=10, r=10, t=20, b=20),
                height=340,
            )
            st.plotly_chart(fig_sankey, width="stretch")

        with col_right:
            st.markdown("#### 📈 Anomaly Reconstruction Loss vs. Threshold")
            plot_df = res_df.copy()
            thresh_val = float(engine.path_b.threshold)

            color_map = {
                "BENIGN": "#10B981",
                "KNOWN_ATTACK": "#EF4444",
                "SUSPECTED_ZERO_DAY": "#C084FC",
                "LOW_CONF_ALERT": "#F59E0B",
            }

            fig_scatter = px.scatter(
                plot_df,
                x="sample_id",
                y="path_b_recon_error",
                color="status",
                color_discrete_map=color_map,
                hover_data=["Flow_ID", "attack_type", "path_a_confidence", "path_b_recon_error", "threat_level"],
                labels={"sample_id": "Flow Sequence #", "path_b_recon_error": "Reconstruction MSE (log)"},
                log_y=True,
            )
            fig_scatter.add_hline(
                y=thresh_val,
                line_dash="dash",
                line_color="#00E5FF",
                line_width=2.0,
                annotation_text=f"Threshold T = {thresh_val:.4f}",
                annotation_position="top left",
                annotation_font_color="#00E5FF",
            )
            fig_scatter.update_layout(
                paper_bgcolor="#091024",
                plot_bgcolor="#091024",
                font=dict(color="#F8FAFC", size=11, family="Outfit"),
                margin=dict(l=10, r=10, t=20, b=20),
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(gridcolor="#1E293B"),
            )
            st.plotly_chart(fig_scatter, width="stretch")

        # Distribution charts
        col_bar, col_pie = st.columns([3, 2])
        with col_bar:
            st.markdown("#### 📊 Top Detected Threat Signatures")
            attack_counts = res_df[res_df["status"] != "BENIGN"]["attack_type"].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Detected Count"]

            if not attack_counts.empty:
                fig_bar = px.bar(
                    attack_counts,
                    x="Detected Count",
                    y="Attack Type",
                    orientation="h",
                    color="Detected Count",
                    color_continuous_scale="Viridis",
                    text="Detected Count",
                )
                fig_bar.update_layout(
                    paper_bgcolor="#091024",
                    plot_bgcolor="#091024",
                    font=dict(color="#F8FAFC", size=11, family="Outfit"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=280,
                    xaxis=dict(gridcolor="#1E293B"),
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("No active attacks detected in current telemetry window.")

        with col_pie:
            st.markdown("#### 🎯 Threat Severity Breakdown")
            sev_counts = res_df["threat_level"].value_counts().reset_index()
            sev_counts.columns = ["Threat Level", "Count"]
            sev_colors = {"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444", "CRITICAL": "#7C3AED"}

            fig_pie = px.pie(
                sev_counts,
                names="Threat Level",
                values="Count",
                color="Threat Level",
                color_discrete_map=sev_colors,
                hole=0.55,
            )
            fig_pie.update_layout(
                paper_bgcolor="#091024",
                plot_bgcolor="#091024",
                font=dict(color="#F8FAFC", size=11, family="Outfit"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1),
            )
            st.plotly_chart(fig_pie, width="stretch")

    # ------------------------------------------------------------------------
    # TAB 4: CISO PRESENTATION EXPORT
    # ------------------------------------------------------------------------
    with tab_export:
        st.subheader("📑 CISO Executive Presentation Deck Generator")
        st.caption("Generate a publication-grade, widescreen 16:9 PowerPoint briefing with high-resolution confusion matrices, dual-path architecture slides, and threat triage breakdowns.")

        col_ppt1, col_ppt2 = st.columns([2, 1])
        with col_ppt1:
            st.markdown(
                """
                <div class="cyber-box">
                    <div class="cyber-box-title">📊 Executive Briefing Deck Highlights</div>
                    <ul style="color: #CBD5E1; font-size: 14px; line-height: 1.8;">
                        <li><b>Slide 1:</b> Executive Title & Cybersecurity Intelligence Summary</li>
                        <li><b>Slide 2:</b> Dual-Path Defense Architecture (Path A XGBoost + Path B Autoencoder)</li>
                        <li><b>Slide 3:</b> Supervised Signature Classification (Confusion Matrix & Class Metrics)</li>
                        <li><b>Slide 4:</b> Zero-Day Deep Autoencoder Reconstruction Analysis (AUC-ROC & Anomaly Loss)</li>
                        <li><b>Slide 5:</b> Live Telemetry Flow Triage & Incident Investigation Table</li>
                        <li><b>Slide 6:</b> Strategic CISO Recommendations & SOC Deployment Roadmap</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_ppt2:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Compile Briefing (.pptx)", type="primary", width="stretch"):
                with st.spinner("Compiling high-resolution charts & generating .pptx..."):
                    try:
                        ppt_bytes = generate_cyberguard_presentation(
                            models_dir=str(models_dir),
                            data_dir=str(data_dir),
                            output_path=None,
                        )
                        st.download_button(
                            label="📥 Download CyberGuard_Executive_Briefing.pptx",
                            data=ppt_bytes,
                            file_name="CyberGuard_Executive_Briefing.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            width="stretch",
                        )
                        st.success("PowerPoint generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PowerPoint: {e}")


if __name__ == "__main__":
    main()
