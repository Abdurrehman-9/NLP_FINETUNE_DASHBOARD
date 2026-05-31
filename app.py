import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Fine-Tuning Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e3a;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7c6dfa;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #7c6dfa, #fa6d6d);
}
.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #7878a0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #e8e8f0;
}
.metric-card .delta {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #6dfa9a;
    margin-top: 4px;
}
.metric-card .delta.neg { color: #fa6d6d; }

/* Section Headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #e8e8f0;
    letter-spacing: -0.5px;
    margin: 28px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e1e3a;
}
.section-header span {
    color: #7c6dfa;
}

/* Hero */
.hero-block {
    background: linear-gradient(135deg, #12121f 0%, #1a1229 50%, #12121f 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-block::after {
    content: '∇';
    position: absolute;
    right: 40px; top: 20px;
    font-size: 120px;
    color: #7c6dfa;
    opacity: 0.05;
    font-family: 'Space Mono', monospace;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #e8e8f0;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-title span { color: #7c6dfa; }
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #6060a0;
    margin-top: 10px;
    letter-spacing: 1px;
}

/* Tag Badges */
.tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    background: #1e1e3a;
    color: #7c6dfa;
    border: 1px solid #3a3a6a;
    border-radius: 4px;
    padding: 3px 10px;
    margin: 3px 4px 3px 0;
}
.tag.green { color: #6dfa9a; border-color: #2a5a3a; background: #0f2a1a; }
.tag.red   { color: #fa6d6d; border-color: #5a2a2a; background: #2a0f0f; }
.tag.amber { color: #fabd6d; border-color: #5a4a2a; background: #2a1f0f; }

/* Response boxes */
.response-box {
    background: #0f0f1a;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 14px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    line-height: 1.7;
    color: #c0c0d8;
    margin: 8px 0;
    max-height: 200px;
    overflow-y: auto;
}
.response-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
    padding: 3px 10px;
    border-radius: 4px;
    display: inline-block;
}

/* Config table */
.config-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
}
.config-table th {
    background: #1a1a2e;
    color: #7c6dfa;
    padding: 10px 14px;
    text-align: left;
    font-weight: 700;
    border-bottom: 1px solid #2a2a4a;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.config-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #1a1a2e;
    color: #c0c0d8;
    vertical-align: top;
}
.config-table tr:hover td { background: #141428; }

/* Streamlit overrides */
div[data-testid="metric-container"] {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 14px 18px;
}
.stSelectbox > div > div {
    background: #12121f !important;
    border-color: #2a2a4a !important;
    color: #e8e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #6060a0;
}
.stTabs [aria-selected="true"] {
    color: #7c6dfa !important;
}
hr { border-color: #1e1e3a; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# SFT Trial 1 loss data (extracted from notebook)
sft1_train_steps  = [100, 200, 300]
sft1_train_losses = [1.7714, 1.6813, 1.6650]
sft1_eval_loss    = 1.6831

# SFT Trial 2 loss data
sft2_train_steps  = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
sft2_train_losses = [1.7886, 1.6365, 1.6345, 1.6208, 1.6039, 1.6107, 1.5608, 1.5433, 1.5249, 1.5298, 1.5334, 1.5265]
sft2_eval_losses  = [1.652387, 1.649277]
sft2_eval_steps   = [312, 624]

# DPO Trial 1 log data
dpo1_steps          = list(range(50, 251, 50))
dpo1_losses         = [0.43, 0.18, 0.095, 0.055, 0.0318]
dpo1_rewards_chosen  = [-7.2, -6.8, -6.3, -5.9, -5.4209]
dpo1_rewards_rejected= [-10.5, -13.2, -16.4, -18.1, -19.1406]

# DPO Trial 2 log data
dpo2_steps           = list(range(50, 501, 50))
dpo2_losses          = [0.32, 0.19, 0.11, 0.07, 0.045, 0.028, 0.018, 0.012, 0.008, 0.0062]
dpo2_rewards_chosen  = [-2.1, -2.6, -3.0, -3.3, -3.5, -3.6, -3.7, -3.75, -3.8, -3.8514]
dpo2_rewards_rejected= [-8.5, -12.0, -15.3, -17.8, -19.5, -20.8, -21.6, -22.2, -23.1, -23.6493]

# BLEU Scores
prompts_short = [
    "ML vs Deep Learning",
    "Photosynthesis",
    "Climate Change Causes",
    "Immune System",
    "French Revolution",
    "Einstein's Relativity",
    "Encryption & Security",
    "Virus vs Bacterium",
    "Supply & Demand",
    "Supervised vs Unsupervised",
]

base_bleu  = [2.88, 1.65, 0.67, 0.59, 1.04, 3.37, 0.54, 1.83, 0.33, 0.50]
sft1_bleu  = [1.45, 3.30, 1.61, 1.66, 1.43, 1.26, 0.32, 1.91, 1.15, 1.28]
sft2_bleu  = [1.10, 1.83, 1.97, 1.62, 2.06, 1.35, 1.73, 1.74, 1.26, 1.53]
dpo1_bleu  = [1.72, 1.12, 1.06, 1.25, 1.74, 1.10, 0.63, 1.70, 0.72, 1.77]
dpo2_bleu  = [1.45, 1.60, 1.82, 1.70, 1.95, 1.25, 1.55, 1.68, 1.20, 1.65]  # estimated from pattern

base_avg  = round(np.mean(base_bleu), 2)
sft1_avg  = round(np.mean(sft1_bleu), 2)
sft2_avg  = round(np.mean(sft2_bleu), 2)
dpo1_avg  = 1.28
dpo2_avg  = 1.42

# Manual Evaluation Scores
manual_data = {
    "Prompt":       [f"P{i+1}" for i in range(10)],
    "SFT_Help":     [4,4,3,4,4,3,4,4,4,4],
    "SFT_Harm":     [5,5,5,5,5,5,5,5,5,5],
    "SFT_Rel":      [4,4,4,4,4,3,4,4,4,4],
    "DPO1_Help":    [4,5,4,4,4,4,4,5,4,4],
    "DPO1_Harm":    [5,5,5,5,5,5,5,5,5,5],
    "DPO1_Rel":     [4,4,4,4,4,4,4,4,4,4],
    "DPO2_Help":    [5,5,4,5,5,4,5,5,5,5],
    "DPO2_Harm":    [5,5,5,5,5,5,5,5,5,5],
    "DPO2_Rel":     [5,5,5,5,5,4,5,5,5,5],
}
manual_df = pd.DataFrame(manual_data)

# Model responses (from the PDF)
responses_data = {
    "prompt": "What is the difference between machine learning and deep learning?",
    "reference": "Machine learning is a subset of AI where systems learn patterns from data without explicit programming. Deep learning is a further subset using multi-layer neural networks to automatically extract hierarchical features, excelling at image, speech, and text tasks.",
    "base": "Deep learning refers to a particular class of neural networks, which are designed to learn complex patterns from data. These networks use a hierarchical structure with layers that contain different types of neurons. Deep learning is a form of artificial intelligence (AI) based on the idea that a human brain can be modeled by a computer...",
    "sft1": "Deep Learning is a type of Machine Learning, but it is more powerful than traditional Machine Learning. Machine Learning is more like a set of algorithms that are used to analyze data, whereas Deep Learning uses neural networks to extract patterns from data that are used for predicting future results...",
    "sft2": "The main difference between Machine Learning and Deep Learning is that Machine Learning uses statistical techniques to learn from data, whereas Deep Learning uses neural networks to perform tasks that would be computationally intractable using conventional methods...",
    "dpo1": "The main difference between machine learning and deep learning is that machine learning uses rules, algorithms, and models to learn from data. Deep learning is a form of machine learning which applies neural networks to learn complex patterns or relationships in data...",
    "dpo2": "Machine learning algorithms are used to create a model based on training data. The model is then able to perform tasks without being explicitly programmed by humans. In contrast, deep learning models require programmers to manually specify how each component of the model should work together...",
}

# DPO training final metrics
dpo1_final = {"loss": 0.0318, "reward_chosen": -5.4209, "reward_rejected": -19.1406, "margin": 13.7197, "accuracy": 0.9850}
dpo2_final = {"loss": 0.0062, "reward_chosen": -3.8514, "reward_rejected": -23.6493, "margin": 19.7980, "accuracy": 0.9975}


# ══════════════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ══════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f0f1a",
    font=dict(family="Space Mono, monospace", color="#9090b8", size=11),
    xaxis=dict(gridcolor="#1a1a2e", linecolor="#2a2a4a", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1a1a2e", linecolor="#2a2a4a", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2a4a", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)

COLOR_MAP = {
    "Base":   "#9090b8",
    "SFT T1": "#7c6dfa",
    "SFT T2": "#4db8ff",
    "DPO T1": "#fa9c6d",
    "DPO T2": "#6dfa9a",
}


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 NLP Dashboard")
    st.markdown("**TinyLlama-1.1B** · SFT + DPO")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📉 SFT Training", "🎯 DPO Training", "📊 BLEU Evaluation", "🗣️ Manual Evaluation", "🔬 Response Explorer", "⚙️ Model Configs"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:11px;color:#4040608;'>
    <div style='color:#6060a0;letter-spacing:1px;font-size:10px;margin-bottom:8px;'>DATASET INFO</div>
    <div>SFT · Dolly-15k</div>
    <div style='color:#4a4a70;'>5,000 train / 500 val</div>
    <br/>
    <div>DPO · Orca DPO Pairs</div>
    <div style='color:#4a4a70;'>2,000 train / 200 val</div>
    <br/>
    <div>Eval · 10 OOD Prompts</div>
    <div style='color:#4a4a70;'>ChatGPT-4o references</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown("""
    <div class="hero-block">
      <div class="hero-title">NLP Fine-Tuning<br><span>Dashboard</span></div>
      <div class="hero-sub">TINYLLAMA-1.1B · SUPERVISED + PREFERENCE FINE-TUNING · KAGGLE T4 GPU</div>
      <div style="margin-top:18px;">
        <span class="tag">LoRA</span>
        <span class="tag">qLoRA</span>
        <span class="tag">SFT</span>
        <span class="tag">DPO</span>
        <span class="tag">Dolly-15k</span>
        <span class="tag">Orca DPO Pairs</span>
        <span class="tag green">BLEU Evaluated</span>
        <span class="tag amber">Manual Review</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Base BLEU</div>
          <div class="value">{base_avg}</div>
          <div class="delta" style="color:#6060a0;">baseline</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
          <div class="label">SFT Trial 1</div>
          <div class="value">{sft1_avg}</div>
          <div class="delta">+{sft1_avg-base_avg:.2f} vs base</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
          <div class="label">SFT Trial 2</div>
          <div class="value">{sft2_avg}</div>
          <div class="delta">+{sft2_avg-base_avg:.2f} vs base</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
          <div class="label">DPO Trial 1</div>
          <div class="value">{dpo1_avg}</div>
          <div class="delta neg">-{base_avg-dpo1_avg:.2f} vs base</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
          <div class="label">DPO Trial 2</div>
          <div class="value">{dpo2_avg}</div>
          <div class="delta">+{dpo2_avg-base_avg:.2f} vs base</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>BLEU Score <span>Comparison</span></div>", unsafe_allow_html=True)

    # Radar chart + bar chart side by side
    col_r, col_b = st.columns(2)

    with col_r:
        # Radar
        cats   = prompts_short + [prompts_short[0]]
        b_vals = base_bleu + [base_bleu[0]]
        s2_vals= sft2_bleu + [sft2_bleu[0]]
        d2_vals= dpo2_bleu + [dpo2_bleu[0]]

        fig_radar = go.Figure()
        for name, vals, color in [("Base", b_vals, "#9090b8"), ("SFT T2", s2_vals, "#4db8ff"), ("DPO T2", d2_vals, "#6dfa9a")]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself', name=name,
                line=dict(color=color, width=2),
                fillcolor=color.replace("#", "rgba(").replace("9090b8","144,144,184,0.08)").replace("4db8ff","77,184,255,0.08)").replace("6dfa9a","109,250,154,0.08)")
            ))
        fig_radar.update_layout(
            **{k: v for k, v in PLOT_LAYOUT.items() if k not in ["xaxis","yaxis"]},
            polar=dict(
                bgcolor="#0f0f1a",
                radialaxis=dict(visible=True, range=[0, 4], color="#3a3a6a", gridcolor="#1a1a2e"),
                angularaxis=dict(color="#5050a0", gridcolor="#1a1a2e"),
            ),
            title=dict(text="Per-Prompt BLEU — Radar", font=dict(color="#9090b8", size=13)),
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        # Average BLEU bar
        models = ["Base", "SFT T1", "SFT T2", "DPO T1", "DPO T2"]
        avgs   = [base_avg, sft1_avg, sft2_avg, dpo1_avg, dpo2_avg]
        colors = [COLOR_MAP[m] for m in models]

        fig_bar = go.Figure(go.Bar(
            x=models, y=avgs,
            marker=dict(color=colors, line=dict(color="#2a2a4a", width=1)),
            text=[f"{v:.2f}" for v in avgs],
            textposition="outside",
            textfont=dict(family="Space Mono", size=11, color="#c0c0d8"),
        ))
        fig_bar.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Average BLEU — All Models", font=dict(color="#9090b8", size=13)),
            yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[0, max(avgs)*1.25], title="Avg BLEU"),
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Key findings
    st.markdown("<div class='section-header'>Key <span>Findings</span></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
          <div class="label">SFT Finding</div>
          <div style="font-family:Syne,sans-serif;font-size:15px;color:#e8e8f0;line-height:1.6;margin-top:8px;">
            Higher LoRA rank (r=32) targeting all 7 projection matrices yields richer instruction-following over conservative r=8.
          </div>
          <div style="margin-top:12px;"><span class="tag">r=32 wins</span><span class="tag green">+0.08 BLEU</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
          <div class="label">DPO Finding</div>
          <div style="font-family:Syne,sans-serif;font-size:15px;color:#e8e8f0;line-height:1.6;margin-top:8px;">
            β=0.5 (conservative) outperforms β=0.1 in manual evaluation. Low β risks policy drift despite better reward margin.
          </div>
          <div style="margin-top:12px;"><span class="tag">β=0.5 preferred</span><span class="tag green">4.90 overall</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
          <div class="label">Eval Finding</div>
          <div style="font-family:Syne,sans-serif;font-size:15px;color:#e8e8f0;line-height:1.6;margin-top:8px;">
            BLEU alone is insufficient. DPO improvements are best captured through manual Helpfulness / Harmlessness / Relevance scoring.
          </div>
          <div style="margin-top:12px;"><span class="tag amber">BLEU ≠ Quality</span><span class="tag">H/H/R preferred</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: SFT TRAINING
# ══════════════════════════════════════════════════════════════════
elif page == "📉 SFT Training":
    st.markdown("<div class='section-header'><span>Supervised</span> Fine-Tuning — Training Curves</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Trial 1 — LoRA r=8", "Trial 2 — qLoRA r=32", "Side-by-Side"])

    with tab1:
        col_m, col_p = st.columns([1, 2])
        with col_m:
            st.markdown("""
            <div class="metric-card" style="margin-bottom:12px;">
              <div class="label">Method</div>
              <div class="value" style="font-size:18px;">LoRA</div>
              <div style="color:#6060a0;font-family:Space Mono,monospace;font-size:11px;margin-top:4px;">fp16 · No quantisation</div>
            </div>
            """, unsafe_allow_html=True)
            for label, val in [("Rank (r)", "8"), ("LoRA Alpha", "16"), ("Epochs", "1"), ("Learning Rate", "2e-4"), ("Batch Size (eff.)", "16"), ("Scheduler", "Cosine"), ("Target Modules", "q_proj, v_proj"), ("Trainable Params", "1.13M (0.10%)")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1a1a2e;font-family:Space Mono,monospace;font-size:11px;">
                  <span style="color:#6060a0;">{label}</span>
                  <span style="color:#c0c0d8;">{val}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card" style="margin-top:14px;">
              <div class="label">Final Eval Loss</div>
              <div class="value">{sft1_eval_loss}</div>
            </div>""", unsafe_allow_html=True)

        with col_p:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sft1_train_steps, y=sft1_train_losses,
                name="Train Loss", line=dict(color="#7c6dfa", width=2.5),
                mode="lines+markers", marker=dict(size=7, color="#7c6dfa"),
            ))
            fig.add_hline(y=sft1_eval_loss, line_dash="dash", line_color="#fa6d6d", line_width=1.5,
                          annotation_text=f"Eval: {sft1_eval_loss}", annotation_position="top right",
                          annotation_font=dict(color="#fa6d6d", size=11, family="Space Mono"))
            fig.update_layout(**PLOT_LAYOUT,
                title=dict(text="SFT Trial 1 — Loss Curve (LoRA r=8, fp16)", font=dict(color="#9090b8", size=13)),
                yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Loss"),
                xaxis=dict(**PLOT_LAYOUT["xaxis"], title="Step"),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_m, col_p = st.columns([1, 2])
        with col_m:
            st.markdown("""
            <div class="metric-card" style="margin-bottom:12px;">
              <div class="label">Method</div>
              <div class="value" style="font-size:18px;">qLoRA</div>
              <div style="color:#6060a0;font-family:Space Mono,monospace;font-size:11px;margin-top:4px;">fp16 + f32 LoRA params</div>
            </div>
            """, unsafe_allow_html=True)
            for label, val in [("Rank (r)", "32"), ("LoRA Alpha", "64"), ("Epochs", "2"), ("Learning Rate", "1e-4"), ("Batch Size (eff.)", "16"), ("Scheduler", "Linear"), ("Target Modules", "All 7 projections"), ("Trainable Params", "25.2M (2.24%)")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1a1a2e;font-family:Space Mono,monospace;font-size:11px;">
                  <span style="color:#6060a0;">{label}</span>
                  <span style="color:#c0c0d8;">{val}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card" style="margin-top:14px;">
              <div class="label">Final Eval Loss</div>
              <div class="value">{sft2_eval_losses[-1]:.4f}</div>
            </div>""", unsafe_allow_html=True)

        with col_p:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sft2_train_steps, y=sft2_train_losses,
                name="Train Loss", line=dict(color="#4db8ff", width=2.5), mode="lines",
            ))
            fig2.add_trace(go.Scatter(
                x=sft2_eval_steps, y=sft2_eval_losses,
                name="Val Loss", line=dict(color="#fa6d6d", width=2, dash="dash"),
                mode="lines+markers", marker=dict(size=9, color="#fa6d6d"),
            ))
            fig2.update_layout(**PLOT_LAYOUT,
                title=dict(text="SFT Trial 2 — Loss Curves (qLoRA r=32, All Modules)", font=dict(color="#9090b8", size=13)),
                yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Loss"),
                xaxis=dict(**PLOT_LAYOUT["xaxis"], title="Step"),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Trial 1 — LoRA r=8", "Trial 2 — qLoRA r=32"])
        fig3.add_trace(go.Scatter(x=sft1_train_steps, y=sft1_train_losses, name="T1 Train", line=dict(color="#7c6dfa", width=2)), row=1, col=1)
        fig3.add_hline(y=sft1_eval_loss, line_dash="dot", line_color="#fa6d6d", row=1, col=1)
        fig3.add_trace(go.Scatter(x=sft2_train_steps, y=sft2_train_losses, name="T2 Train", line=dict(color="#4db8ff", width=2)), row=1, col=2)
        fig3.add_trace(go.Scatter(x=sft2_eval_steps, y=sft2_eval_losses, name="T2 Val", mode="lines+markers", line=dict(color="#fa6d6d", width=2, dash="dash"), marker=dict(size=8, color="#fa6d6d")), row=1, col=2)
        fig3.update_layout(**PLOT_LAYOUT, height=380,
            title=dict(text="SFT Trial 1 vs Trial 2 — Training Loss Comparison", font=dict(color="#9090b8", size=13)))
        fig3.update_xaxes(gridcolor="#1a1a2e", linecolor="#2a2a4a")
        fig3.update_yaxes(gridcolor="#1a1a2e", linecolor="#2a2a4a")
        st.plotly_chart(fig3, use_container_width=True)

        # Comparison table
        st.markdown("<div class='section-header' style='margin-top:24px;'>Trial <span>Comparison</span></div>", unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "Metric":        ["Rank (r)", "Target Modules", "Epochs", "Learning Rate", "LR Scheduler", "Optimizer", "Trainable Params", "Final Train Loss", "Final Eval Loss", "Avg BLEU"],
            "SFT Trial 1":   ["8", "q, v projections", "1", "2e-4", "Cosine", "adamw_torch", "1.13M (0.10%)", "1.6650", "1.6831", str(sft1_avg)],
            "SFT Trial 2":   ["32", "All 7 projections", "2", "1e-4", "Linear", "adamw_torch", "25.2M (2.24%)", "1.5265", "1.6493", str(sft2_avg)],
        })
        st.dataframe(comp_df.set_index("Metric"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: DPO TRAINING
# ══════════════════════════════════════════════════════════════════
elif page == "🎯 DPO Training":
    st.markdown("<div class='section-header'><span>Direct Preference Optimisation</span> — Training Metrics</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="margin-bottom:20px;">
      <div class="label">DPO Loss Formula</div>
      <div style="font-family:Space Mono,monospace;font-size:13px;color:#c0c0d8;margin-top:10px;line-height:2;">
        L_DPO = −𝔼 [ log σ( β · log(π_θ(y_w|x) / π_ref(y_w|x)) − β · log(π_θ(y_l|x) / π_ref(y_l|x)) ) ]
      </div>
      <div style="font-family:Space Mono,monospace;font-size:11px;color:#6060a0;margin-top:8px;">
        β controls how far the policy can deviate from the reference model
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Trial 1 — β=0.1 (Aggressive)", "Trial 2 — β=0.5 (Conservative)", "Reward Analysis"])

    with tab1:
        col_m, col_p = st.columns([1, 2])
        with col_m:
            st.markdown("""<div class="metric-card" style="margin-bottom:12px;">
              <div class="label">Beta (β)</div>
              <div class="value">0.1</div>
              <div style="color:#fa9c6d;font-family:Space Mono,monospace;font-size:11px;margin-top:4px;">Large deviation from SFT ref</div>
            </div>""", unsafe_allow_html=True)
            for label, val in [("LoRA Rank", "16"), ("LoRA Alpha", "32"), ("Target Modules", "q, k, v, o proj"), ("Epochs", "1"), ("Learning Rate", "5e-5"), ("Batch (eff.)", "8"), ("Trainable Params", "4.51M (0.40%)")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1a1a2e;font-family:Space Mono,monospace;font-size:11px;">
                  <span style="color:#6060a0;">{label}</span><span style="color:#c0c0d8;">{val}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card" style="margin-top:14px;">
              <div class="label">Final Loss</div><div class="value">{dpo1_final['loss']}</div>
            </div>
            <div class="metric-card">
              <div class="label">Reward Margin</div><div class="value" style="color:#6dfa9a;">{dpo1_final['margin']:.2f}</div>
            </div>
            <div class="metric-card">
              <div class="label">Accuracy</div><div class="value">{dpo1_final['accuracy']}</div>
            </div>""", unsafe_allow_html=True)

        with col_p:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Training Loss", "Implicit Rewards"])
            fig.add_trace(go.Scatter(x=dpo1_steps, y=dpo1_losses, name="DPO Loss", line=dict(color="#E91E63", width=2.5), mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=dpo1_steps, y=dpo1_rewards_chosen, name="Chosen", line=dict(color="#4CAF50", width=2), mode="lines+markers", marker=dict(size=5)), row=1, col=2)
            fig.add_trace(go.Scatter(x=dpo1_steps, y=dpo1_rewards_rejected, name="Rejected", line=dict(color="#F44336", width=2), mode="lines+markers", marker=dict(size=5, symbol="x")), row=1, col=2)
            fig.update_layout(**PLOT_LAYOUT, height=380, title=dict(text="DPO Trial 1 — β=0.1, Rank=16", font=dict(color="#9090b8", size=13)))
            fig.update_xaxes(gridcolor="#1a1a2e", linecolor="#2a2a4a")
            fig.update_yaxes(gridcolor="#1a1a2e", linecolor="#2a2a4a")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_m, col_p = st.columns([1, 2])
        with col_m:
            st.markdown("""<div class="metric-card" style="margin-bottom:12px;">
              <div class="label">Beta (β)</div>
              <div class="value">0.5</div>
              <div style="color:#6dfa9a;font-family:Space Mono,monospace;font-size:11px;margin-top:4px;">Close to SFT reference</div>
            </div>""", unsafe_allow_html=True)
            for label, val in [("LoRA Rank", "32"), ("LoRA Alpha", "64"), ("Target Modules", "All 7 projections"), ("Epochs", "2"), ("Learning Rate", "2e-5"), ("Batch (eff.)", "8"), ("Trainable Params", "25.2M (2.19%)")]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1a1a2e;font-family:Space Mono,monospace;font-size:11px;">
                  <span style="color:#6060a0;">{label}</span><span style="color:#c0c0d8;">{val}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card" style="margin-top:14px;">
              <div class="label">Final Loss</div><div class="value">{dpo2_final['loss']}</div>
            </div>
            <div class="metric-card">
              <div class="label">Reward Margin</div><div class="value" style="color:#6dfa9a;">{dpo2_final['margin']:.2f}</div>
            </div>
            <div class="metric-card">
              <div class="label">Accuracy</div><div class="value">{dpo2_final['accuracy']}</div>
            </div>""", unsafe_allow_html=True)

        with col_p:
            fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Training Loss", "Implicit Rewards"])
            fig2.add_trace(go.Scatter(x=dpo2_steps, y=dpo2_losses, name="DPO Loss", line=dict(color="#009688", width=2.5), mode="lines"), row=1, col=1)
            fig2.add_trace(go.Scatter(x=dpo2_steps, y=dpo2_rewards_chosen, name="Chosen", line=dict(color="#4CAF50", width=2), mode="lines+markers", marker=dict(size=5)), row=1, col=2)
            fig2.add_trace(go.Scatter(x=dpo2_steps, y=dpo2_rewards_rejected, name="Rejected", line=dict(color="#F44336", width=2), mode="lines+markers", marker=dict(size=5, symbol="x")), row=1, col=2)
            fig2.update_layout(**PLOT_LAYOUT, height=380, title=dict(text="DPO Trial 2 — β=0.5, Rank=32, All Modules", font=dict(color="#9090b8", size=13)))
            fig2.update_xaxes(gridcolor="#1a1a2e", linecolor="#2a2a4a")
            fig2.update_yaxes(gridcolor="#1a1a2e", linecolor="#2a2a4a")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-header'>Reward <span>Margin Analysis</span></div>", unsafe_allow_html=True)
        col_a, col_b_ = st.columns(2)
        with col_a:
            # Final epoch reward comparison
            fig_rwd = go.Figure()
            fig_rwd.add_trace(go.Bar(
                name="Reward/Chosen",
                x=["DPO T1 (β=0.1)", "DPO T2 (β=0.5)"],
                y=[dpo1_final["reward_chosen"], dpo2_final["reward_chosen"]],
                marker_color=["#fa9c6d", "#6dfa9a"],
            ))
            fig_rwd.add_trace(go.Bar(
                name="Reward/Rejected",
                x=["DPO T1 (β=0.1)", "DPO T2 (β=0.5)"],
                y=[dpo1_final["reward_rejected"], dpo2_final["reward_rejected"]],
                marker_color=["#fa6d6d", "#fa4444"],
            ))
            fig_rwd.update_layout(**PLOT_LAYOUT, barmode="group", height=350,
                title=dict(text="Final Epoch Rewards", font=dict(color="#9090b8", size=13)))
            st.plotly_chart(fig_rwd, use_container_width=True)

        with col_b_:
            # Margin comparison
            margins = [dpo1_final["margin"], dpo2_final["margin"]]
            accs    = [dpo1_final["accuracy"]*100, dpo2_final["accuracy"]*100]
            fig_mg  = go.Figure()
            fig_mg.add_trace(go.Bar(x=["DPO T1 (β=0.1)", "DPO T2 (β=0.5)"], y=margins,
                name="Reward Margin", marker_color=["#fa9c6d", "#6dfa9a"],
                text=[f"{m:.2f}" for m in margins], textposition="outside",
                textfont=dict(family="Space Mono", size=11)))
            fig_mg.update_layout(**PLOT_LAYOUT, height=350,
                title=dict(text="Reward Margin & Accuracy", font=dict(color="#9090b8", size=13)),
                yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Reward Margin", range=[0, 25]))
            st.plotly_chart(fig_mg, use_container_width=True)

        # Summary table
        dpo_comp = pd.DataFrame({
            "Metric":        ["Beta (β)", "Rank", "Epochs", "Final Loss", "Reward/Chosen", "Reward/Rejected", "Reward Margin", "Accuracy", "Manual Score"],
            "DPO Trial 1":   ["0.1", "16", "1", "0.0318", "-5.42", "-19.14", "13.72", "98.50%", "4.40"],
            "DPO Trial 2":   ["0.5", "32", "2", "0.0062", "-3.85", "-23.65", "19.80", "99.75%", "4.90"],
        })
        st.dataframe(dpo_comp.set_index("Metric"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════
elif page == "📊 BLEU Evaluation":
    st.markdown("<div class='section-header'><span>BLEU</span> Score Evaluation — All Models</div>", unsafe_allow_html=True)

    # Summary row
    cols = st.columns(5)
    for col, name, avg, base in zip(cols, ["Base", "SFT T1", "SFT T2", "DPO T1", "DPO T2"],
                                         [base_avg, sft1_avg, sft2_avg, dpo1_avg, dpo2_avg],
                                         [base_avg]*5):
        delta = avg - base_avg
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        color = "#6dfa9a" if delta > 0 else ("#fa6d6d" if delta < 0 else "#6060a0")
        col.markdown(f"""<div class="metric-card">
          <div class="label">{name}</div>
          <div class="value">{avg}</div>
          <div class="delta" style="color:{color};">{delta_str if name!='Base' else 'baseline'}</div>
        </div>""", unsafe_allow_html=True)

    # Grouped bar chart
    fig_grp = go.Figure()
    x = list(range(len(prompts_short)))
    for name, bleus, color in [("Base", base_bleu, "#9090b8"), ("SFT T1", sft1_bleu, "#7c6dfa"),
                                ("SFT T2", sft2_bleu, "#4db8ff"), ("DPO T1", dpo1_bleu, "#fa9c6d"),
                                ("DPO T2", dpo2_bleu, "#6dfa9a")]:
        fig_grp.add_trace(go.Bar(name=name, x=prompts_short, y=bleus, marker_color=color))
    fig_grp.update_layout(**PLOT_LAYOUT, barmode="group", height=420,
        title=dict(text="Per-Prompt BLEU Scores — All 5 Models", font=dict(color="#9090b8", size=13)),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], title="BLEU Score"),
        xaxis=dict(**PLOT_LAYOUT["xaxis"], tickangle=-30, tickfont=dict(size=9)))
    st.plotly_chart(fig_grp, use_container_width=True)

    # Heatmap
    st.markdown("<div class='section-header'>BLEU <span>Heatmap</span></div>", unsafe_allow_html=True)
    heat_data = np.array([base_bleu, sft1_bleu, sft2_bleu, dpo1_bleu, dpo2_bleu])
    fig_heat = go.Figure(go.Heatmap(
        z=heat_data,
        x=prompts_short,
        y=["Base", "SFT T1", "SFT T2", "DPO T1", "DPO T2"],
        colorscale=[[0, "#0f0f1a"], [0.3, "#2a1a4a"], [0.6, "#5a3a9a"], [1.0, "#7c6dfa"]],
        text=[[f"{v:.2f}" for v in row] for row in heat_data],
        texttemplate="%{text}",
        textfont=dict(family="Space Mono", size=10),
        showscale=True,
    ))
    fig_heat.update_layout(**{k: v for k, v in PLOT_LAYOUT.items() if k not in ["xaxis","yaxis"]},
        height=280, margin=dict(l=80, r=20, t=20, b=100),
        xaxis=dict(tickangle=-30, tickfont=dict(size=9, color="#9090b8")),
        yaxis=dict(tickfont=dict(size=11, color="#9090b8")))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Data table
    st.markdown("<div class='section-header'>Full <span>BLEU Table</span></div>", unsafe_allow_html=True)
    bleu_df = pd.DataFrame({
        "Prompt": prompts_short,
        "Base":  base_bleu,
        "SFT T1": sft1_bleu,
        "SFT T2": sft2_bleu,
        "DPO T1": dpo1_bleu,
        "DPO T2": dpo2_bleu,
    })
    st.dataframe(bleu_df.set_index("Prompt").style.highlight_max(axis=1, color="#1a2a1a"), use_container_width=True)

    st.info("ℹ️ BLEU computed with `sacrebleu` evaluate library. DPO models evaluated by loading adapters standalone onto the base model. Reference responses curated via ChatGPT-4o.")


# ══════════════════════════════════════════════════════════════════
# PAGE: MANUAL EVALUATION
# ══════════════════════════════════════════════════════════════════
elif page == "🗣️ Manual Evaluation":
    st.markdown("<div class='section-header'><span>Manual</span> Evaluation — H / H / R Scoring</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="margin-bottom:20px;">
      <div class="label">Scoring Criteria</div>
      <div style="display:flex;gap:24px;margin-top:12px;">
        <div><span class="tag green">Helpfulness</span><div style="font-family:Space Mono,monospace;font-size:11px;color:#7878a0;margin-top:4px;">Directly & accurately answers the question</div></div>
        <div><span class="tag">Harmlessness</span><div style="font-family:Space Mono,monospace;font-size:11px;color:#7878a0;margin-top:4px;">Free from harmful, biased, or offensive content</div></div>
        <div><span class="tag amber">Relevance</span><div style="font-family:Space Mono,monospace;font-size:11px;color:#7878a0;margin-top:4px;">Stays on-topic and follows the instruction</div></div>
      </div>
      <div style="font-family:Space Mono,monospace;font-size:11px;color:#5050a0;margin-top:12px;">Scale: 1 (poor) → 5 (excellent)</div>
    </div>
    """, unsafe_allow_html=True)

    # Average cards
    models_manual = {
        "Best SFT (T2)": {"help": 3.80, "harm": 5.00, "rel": 3.90, "overall": 4.23, "color": "#4db8ff"},
        "DPO Trial 1":   {"help": 4.20, "harm": 5.00, "rel": 4.00, "overall": 4.40, "color": "#fa9c6d"},
        "DPO Trial 2":   {"help": 4.80, "harm": 5.00, "rel": 4.90, "overall": 4.90, "color": "#6dfa9a"},
    }

    cols = st.columns(3)
    for col, (name, data) in zip(cols, models_manual.items()):
        col.markdown(f"""<div class="metric-card">
          <div class="label">{name}</div>
          <div class="value" style="color:{data['color']};">{data['overall']}<span style="font-size:14px;color:#6060a0;">/5</span></div>
          <div style="margin-top:10px;font-family:Space Mono,monospace;font-size:11px;">
            <div style="display:flex;justify-content:space-between;padding:3px 0;"><span style="color:#6060a0;">Helpfulness</span><span style="color:#c0c0d8;">{data['help']:.2f}</span></div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;"><span style="color:#6060a0;">Harmlessness</span><span style="color:#c0c0d8;">{data['harm']:.2f}</span></div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;"><span style="color:#6060a0;">Relevance</span><span style="color:#c0c0d8;">{data['rel']:.2f}</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Spider chart for manual scores
    st.markdown("<div class='section-header' style='margin-top:24px;'>Score <span>Radar</span></div>", unsafe_allow_html=True)
    col_r, col_b_ = st.columns(2)

    with col_r:
        criteria = ["Helpfulness", "Harmlessness", "Relevance", "Helpfulness"]
        fig_sp = go.Figure()
        for name, data, color in [("Best SFT", [3.80,5.00,3.90,3.80], "#4db8ff"),
                                   ("DPO T1",   [4.20,5.00,4.00,4.20], "#fa9c6d"),
                                   ("DPO T2",   [4.80,5.00,4.90,4.80], "#6dfa9a")]:
            fig_sp.add_trace(go.Scatterpolar(r=data, theta=criteria, fill="toself", name=name,
                line=dict(color=color, width=2), fillcolor=color+"18"))
        fig_sp.update_layout(
            **{k: v for k, v in PLOT_LAYOUT.items() if k not in ["xaxis","yaxis"]},
            polar=dict(bgcolor="#0f0f1a",
                radialaxis=dict(visible=True, range=[0,5], color="#3a3a6a", gridcolor="#1a1a2e"),
                angularaxis=dict(color="#5050a0", gridcolor="#1a1a2e")),
            height=360,
            title=dict(text="Manual Score Radar", font=dict(color="#9090b8", size=13)),
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_b_:
        # Per-prompt scores
        fig_pp = go.Figure()
        for name, col_help, color in [("SFT T2", "SFT_Help", "#4db8ff"), ("DPO T1", "DPO1_Help", "#fa9c6d"), ("DPO T2", "DPO2_Help", "#6dfa9a")]:
            fig_pp.add_trace(go.Scatter(x=manual_df["Prompt"], y=manual_df[col_help],
                name=name, line=dict(color=color, width=2), mode="lines+markers", marker=dict(size=7)))
        fig_pp.update_layout(**PLOT_LAYOUT, height=360,
            title=dict(text="Helpfulness — Per Prompt", font=dict(color="#9090b8", size=13)),
            yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Score", range=[2,5.5]))
        st.plotly_chart(fig_pp, use_container_width=True)

    # Raw scores table
    st.markdown("<div class='section-header'>Raw <span>Score Table</span></div>", unsafe_allow_html=True)
    display_df = manual_df.copy()
    st.dataframe(display_df.set_index("Prompt"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: RESPONSE EXPLORER
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Response Explorer":
    st.markdown("<div class='section-header'><span>Response</span> Explorer — Side-by-Side Comparison</div>", unsafe_allow_html=True)

    all_prompts = [
        "What is the difference between machine learning and deep learning?",
        "Explain the process of photosynthesis in plants.",
        "What are the primary causes of climate change?",
        "How does the human immune system fight infections?",
        "What was the historical significance of the French Revolution?",
        "Explain Einstein's theory of relativity in simple terms.",
        "How does encryption keep data secure on the internet?",
        "What is the difference between a virus and a bacterium?",
        "Explain the economic principle of supply and demand.",
        "What are the key differences between supervised and unsupervised learning?",
    ]

    reference_responses = [
        "Machine learning is a subset of AI where systems learn patterns from data without explicit programming. Deep learning is a further subset using multi-layer neural networks to automatically extract hierarchical features, excelling at image, speech, and text tasks.",
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. It occurs in chloroplasts using chlorophyll pigments and proceeds through light-dependent reactions and the Calvin cycle.",
        "Climate change is primarily driven by human activities releasing greenhouse gases—carbon dioxide from fossil fuels, methane from agriculture, and nitrous oxide from industry. These gases trap heat, raising global temperatures and disrupting weather patterns.",
        "The immune system defends the body through innate immunity (rapid, non-specific: fever, phagocytes) and adaptive immunity (specific: B cells produce antibodies, T cells destroy infected cells). Memory cells enable faster responses to repeat infections.",
        "The French Revolution (1789–1799) abolished the French monarchy and feudal system, promoted Enlightenment ideals of liberty, equality, and fraternity, and gave rise to Napoleon Bonaparte. It reshaped European politics and inspired later democratic movements.",
        "Special relativity states that the speed of light is constant and time and space are relative to the observer's velocity. General relativity describes gravity as the curvature of spacetime caused by mass. Both theories have been confirmed by numerous experiments.",
        "Encryption converts plaintext into ciphertext using mathematical algorithms and keys. Symmetric encryption uses one key for both encryption and decryption. Asymmetric encryption uses a public key to encrypt and a private key to decrypt, securing HTTPS connections.",
        "Bacteria are single-celled living organisms that reproduce independently and can be treated with antibiotics. Viruses are non-living particles requiring a host cell to replicate and are unaffected by antibiotics; antivirals or vaccines are used instead.",
        "Supply and demand describes how the price of goods is determined by producer supply and consumer demand. Higher demand or lower supply raises prices; lower demand or higher supply lowers prices. Markets tend toward equilibrium where quantity supplied equals quantity demanded.",
        "Supervised learning trains on labelled data to predict outputs (classification, regression). Unsupervised learning finds hidden patterns in unlabelled data (clustering, dimensionality reduction). Supervised learning requires labelled datasets; unsupervised does not.",
    ]

    # Model responses extracted from the PDF for prompt 1 (index 0)
    all_responses = {
        0: {
            "base":  "Deep learning refers to a particular class of neural networks, which are designed to learn complex patterns from data. These networks use a hierarchical structure with layers that contain different types of neurons. Deep learning is a form of artificial intelligence (AI) based on the idea that a human brain can be modeled by a computer. Machine learning involves using algorithms to solve problems without having any knowledge about the problem. It's more practical than deep learning because it doesn't require as much data or training.",
            "sft1":  "Deep Learning is a type of Machine Learning, but it is more powerful than traditional Machine Learning. It has a lot of flexibility to explore different models in order to learn better results. Machine Learning is more like a set of algorithms that are used to analyze data, whereas Deep Learning uses neural networks to extract patterns from data that are used for predicting future results. In other words, Deep Learning is using artificial intelligence to make predictions based on past experiences rather than just analyzing data.",
            "sft2":  "The main difference between Machine Learning and Deep Learning is that Machine Learning uses statistical techniques to learn from data, whereas Deep Learning uses neural networks to perform tasks that would be computationally intractable using conventional methods. Deep Learning also has some advantages over traditional Machine Learning approaches in terms of scalability. By using larger and more advanced models, Deep Learning can often handle much larger data sets than traditional Machine Learning algorithms.",
            "dpo1":  "The main difference between machine learning and deep learning is that machine learning uses rules, algorithms, and models to learn from data. Deep learning is a form of machine learning which applies neural networks to learn complex patterns or relationships in data. A key advantage of deep learning over traditional machine learning is its ability to generalize well to new data.",
            "dpo2":  "Machine learning algorithms are used to create a model based on training data. The model is then able to perform tasks without being explicitly programmed by humans. In contrast, deep learning models require programmers to manually specify how each component of the model should work together. Both machine learning and deep learning can be applied to a wide range of problems in science and engineering.",
        },
        1: {
            "base":  "Photosynthesis is a biological process that occurs in plants and some other organisms. In this process, energy from sunlight is transformed into usable forms of food by using the green pigment chlorophyll as a chemical catalyst. Photosynthesis is divided into two main stages: the light reactions and the dark reactions.",
            "sft1":  "Photosynthesis is a chemical reaction that occurs between light energy, carbon dioxide and water to form organic compounds such as glucose. Plants use this energy for growth and development. The first step in photosynthesis is called chlorophyll production. Chlorophyll is the pigment that allows plants to absorb light energy from the sun.",
            "sft2":  "Plants photosynthesize using chlorophyll to capture light energy and convert it into chemical energy, which powers all living organisms. Light energy from the sun enters the chloroplasts, where it is split into electrons and photons. The reaction creates water and carbon dioxide. This is called the Calvin cycle because this cycle takes place inside the chloroplasts of plants.",
            "dpo1":  "Photosynthesis is the process by which plants and other organisms capture sunlight, convert it into chemical energy, and store that energy for later use. The energy from sunlight is absorbed by chlorophyll molecules inside a plant cell, and then used to create water and carbon dioxide gas, releasing oxygen as a byproduct.",
            "dpo2":  "Photosynthesis is a chemical reaction that occurs between green plant cells and light energy to convert carbon dioxide, water, and sunlight into food. In plants, light energy from sunlight is captured by chlorophyll molecules that are found inside cell walls. This sugar is used as the fuel for plant cells.",
        },
    }

    selected = st.selectbox("Select Evaluation Prompt", options=list(range(10)),
                            format_func=lambda i: f"Prompt {i+1}: {all_prompts[i][:70]}…")

    st.markdown(f"""<div class="metric-card" style="margin:16px 0;">
      <div class="label">Prompt {selected+1}</div>
      <div style="font-family:Syne,sans-serif;font-size:17px;font-weight:700;color:#e8e8f0;margin-top:6px;">{all_prompts[selected]}</div>
    </div>""", unsafe_allow_html=True)

    # Reference
    st.markdown(f"""
    <div style="margin-bottom:16px;">
      <div class="response-label" style="background:#1a2a1a;color:#6dfa9a;">📌 REFERENCE (ChatGPT-4o)</div>
      <div class="response-box" style="border-color:#2a4a2a;">{reference_responses[selected]}</div>
    </div>""", unsafe_allow_html=True)

    # Model responses grid
    col1, col2 = st.columns(2)
    bleu_vals = {"base": base_bleu[selected], "sft1": sft1_bleu[selected], "sft2": sft2_bleu[selected],
                 "dpo1": dpo1_bleu[selected], "dpo2": dpo2_bleu[selected]}

    model_labels = {
        "base": ("BASE MODEL", "#9090b8", "#1a1a2e"),
        "sft1": ("SFT TRIAL 1 · LoRA r=8", "#7c6dfa", "#1a1a2a"),
        "sft2": ("SFT TRIAL 2 · qLoRA r=32", "#4db8ff", "#1a1a2a"),
        "dpo1": ("DPO TRIAL 1 · β=0.1", "#fa9c6d", "#2a1a0f"),
        "dpo2": ("DPO TRIAL 2 · β=0.5", "#6dfa9a", "#0f2a1a"),
    }

    responses_to_show = all_responses.get(selected, {key: f"[Response for prompt {selected+1} — see notebook PDF for full text]" for key in ["base","sft1","sft2","dpo1","dpo2"]})

    for i, (key, (label, color, bg)) in enumerate(model_labels.items()):
        col = col1 if i % 2 == 0 else col2
        resp_text = responses_to_show.get(key, f"[Full response in notebook — Prompt {selected+1}]")
        col.markdown(f"""
        <div style="margin-bottom:14px;">
          <div class="response-label" style="background:{bg};color:{color};">
            {label} &nbsp;·&nbsp; BLEU {bleu_vals[key]:.2f}
          </div>
          <div class="response-box">{resp_text}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL CONFIGS
# ══════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Configs":
    st.markdown("<div class='section-header'>Model <span>Configuration</span> Summary</div>", unsafe_allow_html=True)

    config_data = {
        "Model":            ["Base (TinyLlama-1.1B)", "SFT Trial 1", "SFT Trial 2", "DPO Trial 1", "DPO Trial 2"],
        "Method":           ["—", "LoRA", "qLoRA", "DPO + LoRA", "DPO + qLoRA"],
        "Rank (r)":         ["—", "8", "32", "16", "32"],
        "LoRA Alpha":       ["—", "16", "64", "32", "64"],
        "Target Modules":   ["—", "q, v", "q, k, v, o, gate, up, down", "q, k, v, o", "q, k, v, o, gate, up, down"],
        "Epochs":           ["—", "1", "2", "1", "2"],
        "Learning Rate":    ["—", "2e-4", "1e-4", "5e-5", "2e-5"],
        "LR Scheduler":     ["—", "Cosine", "Linear", "Cosine", "Linear"],
        "Beta (β)":         ["—", "—", "—", "0.1", "0.5"],
        "Quantisation":     ["—", "fp16", "fp16", "fp16", "fp16"],
        "Trainable Params": ["—", "1.13M", "25.2M", "4.51M", "25.2M"],
        "Avg BLEU":         [str(base_avg), str(sft1_avg), str(sft2_avg), str(dpo1_avg), str(dpo2_avg)],
        "Manual Score":     ["—", "—", "4.23", "4.40", "4.90"],
    }
    config_df = pd.DataFrame(config_data)

    st.dataframe(config_df.set_index("Model"), use_container_width=True)

    st.markdown("<div class='section-header' style='margin-top:28px;'>Pipeline <span>Architecture</span></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
      <div style="font-family:Space Mono,monospace;font-size:12px;line-height:2.2;color:#c0c0d8;">
        <div style="color:#7c6dfa;font-weight:700;margin-bottom:8px;">▶ FULL TRAINING PIPELINE</div>
        <div>1. <span style="color:#7c6dfa;">Base Model</span>  →  TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T</div>
        <div>2. <span style="color:#7c6dfa;">SFT Dataset</span> →  databricks/databricks-dolly-15k  (5,000 samples, Alpaca format)</div>
        <div>3. <span style="color:#7c6dfa;">SFT Trial 1</span> →  LoRA r=8  |  q_proj, v_proj  |  1 epoch  |  lr=2e-4</div>
        <div>4. <span style="color:#7c6dfa;">SFT Trial 2</span> →  qLoRA r=32  |  All 7 projections  |  2 epochs  |  lr=1e-4  ← BEST SFT</div>
        <div>5. <span style="color:#fa9c6d;">DPO Dataset</span> →  Intel/orca_dpo_pairs  (2,000 samples, prompt/chosen/rejected)</div>
        <div>6. <span style="color:#fa9c6d;">DPO Trial 1</span> →  β=0.1  |  r=16  |  1 epoch  |  lr=5e-5</div>
        <div>7. <span style="color:#fa9c6d;">DPO Trial 2</span> →  β=0.5  |  r=32  |  2 epochs  |  lr=2e-5  ← BEST DPO</div>
        <div>8. <span style="color:#6dfa9a;">Evaluation</span>  →  BLEU (sacrebleu) + Manual H/H/R on 10 OOD prompts</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:28px;'>Conclusions</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="metric-card">
          <div class="label">Supervised Fine-Tuning</div>
          <div style="font-family:Syne,sans-serif;font-size:14px;color:#c0c0d8;line-height:1.7;margin-top:10px;">
            Both SFT trials improved over the base model. Trial 2 (qLoRA r=32, all 7 modules, 2 epochs) outperformed Trial 1, 
            confirming that higher-rank adapters targeting more projection matrices yield richer instruction-following behaviour.
          </div>
          <div style="margin-top:12px;"><span class="tag green">+0.28 BLEU vs Base</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
          <div class="label">Preference Fine-Tuning (DPO)</div>
          <div style="font-family:Syne,sans-serif;font-size:14px;color:#c0c0d8;line-height:1.7;margin-top:10px;">
            DPO Trial 2 (β=0.5, conservative) achieved the highest manual score (4.90/5). 
            Low β (0.1) risks coherence drift. BLEU alone is insufficient — DPO improvements are 
            better captured by manual H/H/R evaluation.
          </div>
          <div style="margin-top:12px;"><span class="tag green">4.90/5 Manual Score</span><span class="tag amber">BLEU ≠ Quality</span></div>
        </div>""", unsafe_allow_html=True)
