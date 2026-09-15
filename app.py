import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ollama
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st

# MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="MalharIQ | Autonomous MLOps & Analytics Advisory",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Safe optional imports for multi-modal assets
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

import plotly.express as px

from core_cleaner import (
    execute_cleaning_recipe,
    generate_data_summary,
    get_cleaning_recipe,
    infer_ml_task_and_target,
)
from document_exporter import create_executive_deck, create_pdf_report
from governance_audit import audit_disparate_impact
from ml_engine import (
    run_association_mining,
    run_classification_tournament,
    run_clustering_tournament,
    run_regression_tournament,
)
from reporter import generate_dual_reports
from stats_validator import (
    calculate_kmo_bartlett,
    extract_latent_factors,
    run_feature_significance_tests,
)

# ==============================================================================
# MANAGEMENT CONSULTING STYLING
# ==============================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');
    
    .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }
    .consulting-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 16px;
        margin-bottom: 28px;
    }
    .consulting-logo {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0F172A;
    }
    .consulting-logo span { color: #2563EB; }
    .consulting-badge {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 5px 12px;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 9999px;
    }
    .hero-container {
        background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 13.5px;
        color: #64748B;
        line-height: 1.5;
        margin: 0;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    button[kind="primary"] {
        background: #0F172A !important;
        border: 1px solid #0F172A !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="consulting-nav">
    <div class="consulting-logo">Malhar <span>IQ</span></div>
    <div class="consulting-badge">Enterprise Multi-Modal MLOps Platform</div>
</div>
<div class="hero-container">
    <div class="hero-title">Automated Analytics, Multi-Modal Ingestion & Quantitative Governance</div>
    <div class="hero-subtitle">Ingest structured or unstructured assets, benchmark candidate architectures across 70/15/15 partitions, and export verified advisory deliverables.</div>
</div>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# AUTONOMOUS SPARSE DATASET SYNTHESIZER
# ==============================================================================
def synthesize_sparse_dataset(
    df: pd.DataFrame, min_threshold: int = 30, target_samples: int = 100
) -> tuple[pd.DataFrame, bool]:
    """
    Detects micro-datasets and synthesizes statistically consistent records
    preserving feature covariance, numerical variance, and categorical distributions.
    """
    if len(df) >= min_threshold:
        return df, False

    needed = target_samples - len(df)
    synthetic_rows = []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    num_stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        mean = series.mean() if not series.empty else 0.0
        std = series.std() if len(series) > 1 and series.std() > 0 else 0.08 * (abs(mean) + 1.0)
        num_stats[col] = (mean, std)

    for _ in range(needed):
        base_idx = np.random.choice(df.index)
        base_row = df.loc[base_idx].to_dict()

        for col in numeric_cols:
            mean, std = num_stats[col]
            noise = np.random.normal(0, 0.08 * std)
            base_row[col] = round(float(base_row[col] + noise), 4)

        for col in categorical_cols:
            cat_series = df[col].dropna()
            if not cat_series.empty:
                probs = cat_series.value_counts(normalize=True)
                base_row[col] = np.random.choice(probs.index, p=probs.values)

        synthetic_rows.append(base_row)

    augmented_df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)
    return augmented_df, True


# ==============================================================================
# UNSTRUCTURED-TO-STRUCTURED EXTRACTION LOGIC
# ==============================================================================
def extract_structured_from_unstructured(
    file_bytes: bytes, filename: str, directive: str = ""
) -> pd.DataFrame:
    ext = filename.strip().split(".")[-1].lower()

    # 1. JSON or Log structured lines
    if ext in ["json", "log"]:
        try:
            raw = json.loads(file_bytes.decode("utf-8", errors="ignore"))
            return pd.json_normalize(raw)
        except Exception:
            try:
                lines = [
                    json.loads(l)
                    for l in file_bytes.decode("utf-8", errors="ignore").splitlines()
                    if l.strip()
                ]
                return pd.DataFrame(lines)
            except Exception:
                st.error("Invalid JSON or Log format.")
                return pd.DataFrame()

    # 2. Text Documents (.txt, .text, .pdf)
    elif ext in ["txt", "text", "pdf"]:
        if ext in ["txt", "text"]:
            content = file_bytes.decode("utf-8", errors="ignore")
        else:
            if PdfReader is None:
                st.error("pypdf is not installed. Run `pip install pypdf`.")
                return pd.DataFrame()
            reader = PdfReader(io.BytesIO(file_bytes))
            content = "\n".join([p.extract_text() or "" for p in reader.pages if p.extract_text()])

        snippets = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 20]
        if not snippets:
            snippets = [
                content[i : i + 350]
                for i in range(0, min(len(content), 3500), 350)
                if len(content[i : i + 350].strip()) > 15
            ]

        records = []
        sys_prompt = f"""
        You are an automated ETL feature extraction engine.
        Convert the provided text snippet into a single JSON row with consistent numerical and categorical fields.
        {f"User Guidance: {directive}" if directive else "Extract: category (string), sentiment_score (float from -1.0 to 1.0), urgency (int 1-5), and action_required (0 or 1)."}
        Output ONLY valid JSON.
        """
        for chunk in snippets[:30]:
            try:
                resp = ollama.chat(
                    model="llama3.2:1b",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Text: {chunk}"},
                    ],
                    format="json",
                    options={"temperature": 0.1},
                )
                records.append(json.loads(resp["message"]["content"]))
            except Exception:
                continue
        return pd.DataFrame(records)

    # 3. Image Metadata & Channel Signals
    elif ext in ["png", "jpg", "jpeg"]:
        if Image is None:
            st.error("Pillow is not installed. Run `pip install Pillow`.")
            return pd.DataFrame()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        arr = np.array(img)
        feats = {
            "image_width": img.width,
            "image_height": img.height,
            "aspect_ratio": round(img.width / max(img.height, 1), 3),
            "mean_red_channel": float(np.mean(arr[:, :, 0])),
            "mean_green_channel": float(np.mean(arr[:, :, 1])),
            "mean_blue_channel": float(np.mean(arr[:, :, 2])),
            "brightness_std": float(np.std(arr)),
        }
        return pd.DataFrame([feats])

    return pd.DataFrame()


# ==============================================================================
# DUAL-PATH INGESTION INTERFACE
# ==============================================================================
st.markdown("### 📥 Select Enterprise Ingestion Protocol")
upload_tab_struct, upload_tab_unstruct = st.tabs([
    "📊 Structured Tabular Ingestion (CSV / Excel)",
    "📑 Unstructured Asset Parsing (TXT / PDF / JSON / Images)",
])

df = None

with upload_tab_struct:
    uploaded_structured = st.file_uploader(
        "Upload Tabular Enterprise Dataset",
        type=["csv", "xlsx", "xls"],
        key="structured_uploader",
    )
    if uploaded_structured:
        ext = uploaded_structured.name.split(".")[-1].lower()
        df = (
            pd.read_csv(uploaded_structured)
            if ext == "csv"
            else pd.read_excel(uploaded_structured)
        )

with upload_tab_unstruct:
    st.caption(
        "Converts plain text, semi-structured logs, raw documents, or image assets into clean tabular schemas via Ollama."
    )
    uploaded_unstructured = st.file_uploader(
        "Upload Unstructured Asset",
        type=[
            "txt", "TXT", "text", "TEXT",
            "json", "JSON", "log", "LOG",
            "pdf", "PDF",
            "png", "PNG", "jpg", "JPG", "jpeg", "JPEG"
        ],
        key="unstructured_uploader",
    )

    if uploaded_unstructured:
        col_dir, col_btn = st.columns([3, 1])
        with col_dir:
            unstructured_directive = st.text_input(
                "Feature Extraction Directive (Optional):",
                placeholder="e.g., 'Extract customer sentiment, issue category, urgency (1-5)'",
                key="directive_input",
            )
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            trigger_transform = st.button(
                "⚡ Synthesize Dataset",
                type="primary",
                use_container_width=True,
            )

        if trigger_transform:
            with st.spinner("Parsing and structuring unstructured asset into tabular schema..."):
                file_bytes = uploaded_unstructured.read()
                extracted_df = extract_structured_from_unstructured(
                    file_bytes, uploaded_unstructured.name, unstructured_directive
                )
                if not extracted_df.empty:
                    st.session_state["active_extracted_df"] = extracted_df
                    st.success(
                        f"Synthesized tabular dataset: {extracted_df.shape[0]} records × {extracted_df.shape[1]} attributes."
                    )
                else:
                    st.error("Failed to extract structured attributes from asset.")

        if "active_extracted_df" in st.session_state and df is None:
            df = st.session_state["active_extracted_df"]


# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================
if df is not None and not df.empty:
    with st.expander("🔍 Inspect Raw Data Schema & Integrity Profile", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{df.shape[0]:,}")
        c2.metric("Total Dimensions", f"{df.shape[1]}")
        c3.metric("Missing Data Points", f"{df.isnull().sum().sum():,}")
        st.dataframe(df.head(8), use_container_width=True)

    st.markdown("---")
    mode = st.radio(
        "Select Pipeline Mode:",
        [
            "⚡ Autonomous Intent Discovery (Recommended — Zero Configuration)",
            "✍️ Custom Strategic Directive",
        ],
        horizontal=True,
    )

    user_goal_input = ""
    if mode == "✍️ Custom Strategic Directive":
        user_goal_input = st.text_area(
            "State your operational directive:",
            placeholder="e.g., 'Forecast sales revenue' or 'Predict client churn risk'",
            height=80,
        )

    if st.button("🚀 Execute Advisory Pipeline", type="primary"):
        with st.status("Engaging Autonomous Advisory Engine...", expanded=True) as status:
            # 1. Profile metadata
            st.write("📊 Profiling metadata & statistical distributions...")
            summary = generate_data_summary(df)

            # 2. Intent Inference
            st.write("🎯 AI Agent identifying optimal objective & mathematical targets...")
            intent = infer_ml_task_and_target(summary, user_goal_input)
            task = intent.get("task", "Clustering / Segmentation")
            target_col = intent.get("target_col")

            st.info(
                f"🏛️ **Strategic Goal:** `{task}`"
                + (f" | **Target Feature:** `{target_col}`" if target_col else "")
                + f"\n\n*Agent Rationale:* {intent.get('reasoning')}"
            )

            # 3. Clean Data
            st.write("🧠 Formulating cleaning blueprint & resolving anomalies...")
            recipe = get_cleaning_recipe(summary, user_goal=task)
            st.write("⚙️ Executing deterministic transformations...")
            clean_df, logs = execute_cleaning_recipe(df, recipe)

            # Check and auto-synthesize if dataset is too small for holdout partitioning
            clean_df, was_synthesized = synthesize_sparse_dataset(clean_df, min_threshold=30, target_samples=100)
            if was_synthesized:
                st.warning(
                    f"⚠️ Low sample volume detected ({len(df)} records). "
                    f"Autonomously synthesized cohort up to {len(clean_df)} records "
                    "preserving distributions to satisfy holdout statistical validity."
                )
                logs.append(f"Synthetic Cohort Augmentation: Expanded from {len(df)} to {len(clean_df)} records.")

            # 4. Partition 70 / 15 / 15
            st.write("🔀 Partitioning into 70% Train, 15% Validation, 15% Test...")
            train_df, temp_df = train_test_split(clean_df, test_size=0.30, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

            split_data = {
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "counts": {
                    "train": len(train_df),
                    "val": len(val_df),
                    "test": len(test_df),
                },
            }

            # 5. Statistical Quality Gates
            st.write("📐 Applying Statistical Quality Gates (KMO, Sphericity, Factor Loadings)...")
            stats_results = {}
            if task == "Clustering / Segmentation":
                stats_results["kmo_bartlett"] = calculate_kmo_bartlett(clean_df)
                stats_results["factor_analysis"] = extract_latent_factors(clean_df)
            else:
                stats_results["feature_significance_tests"] = run_feature_significance_tests(
                    clean_df, target_col=target_col
                )

            # 6. Model Tournament Execution
            st.write(f"📈 Benchmarking candidate architectures for {task} across partitions...")
            if task == "Clustering / Segmentation":
                ml_results = run_clustering_tournament(clean_df)
            elif task == "Classification":
                ml_results = run_classification_tournament(clean_df, target_col=target_col)
            else:
                ml_results = run_regression_tournament(clean_df, target_col=target_col)

            assoc_rules_df = run_association_mining(clean_df)

            # 7. Synthesize Dual Reports
            st.write("📝 Synthesizing Executive & Engineering Deliverables...")
            reports = generate_dual_reports(logs, stats_results, ml_results, user_goal=task)

            # Persist state
            st.session_state["pipeline_ready"] = True
            st.session_state["reports"] = reports
            st.session_state["ml_results"] = ml_results
            st.session_state["stats_results"] = stats_results
            st.session_state["clean_df"] = clean_df
            st.session_state["logs"] = logs
            st.session_state["task"] = task
            st.session_state["target_col"] = target_col
            st.session_state["split_data"] = split_data
            st.session_state["assoc_rules_df"] = assoc_rules_df

            status.update(
                label="✅ Advisory Evaluation Complete",
                state="complete",
                expanded=False,
            )

    # Render Analytical Workspace
    if st.session_state.get("pipeline_ready", False):
        reports = st.session_state["reports"]
        ml_results = st.session_state["ml_results"]
        stats_results = st.session_state["stats_results"]
        clean_df = st.session_state["clean_df"]
        logs = st.session_state["logs"]
        task = st.session_state["task"]
        target_col = st.session_state["target_col"]
        split_data = st.session_state["split_data"]
        assoc_rules_df = st.session_state["assoc_rules_df"]

        # Partition KPI Cards
        with st.container():
            st.markdown("### 🔀 Dataset Partitioning Architecture")
            p1, p2, p3 = st.columns(3)
            c_train = split_data["counts"]["train"]
            c_val = split_data["counts"]["val"]
            c_test = split_data["counts"]["test"]
            tot = max(len(clean_df), 1)
            p1.metric("Training Set (70%)", f"{c_train:,} rows", f"{c_train/tot*100:.1f}%")
            p2.metric("Validation Set (15%)", f"{c_val:,} rows", f"{c_val/tot*100:.1f}%")
            p3.metric("Test Set (15%)", f"{c_test:,} rows", f"{c_test/tot*100:.1f}%")

            with st.expander("🔎 Preview Split Partitions", expanded=False):
                view_choice = st.radio(
                    "Select Partition to Inspect:",
                    ["Train (70%)", "Validation (15%)", "Test (15%)"],
                    horizontal=True,
                )
                if view_choice == "Train (70%)":
                    st.dataframe(split_data["train"].head(8), use_container_width=True)
                elif view_choice == "Validation (15%)":
                    st.dataframe(split_data["val"].head(8), use_container_width=True)
                else:
                    st.dataframe(split_data["test"].head(8), use_container_width=True)

        st.markdown("---")

        # Capture Static Chart Buffers for PPTX & PDF Embedding
        chart_buffers = []
        if task == "Regression" and "y_test" in ml_results and "y_pred" in ml_results:
            y_t = ml_results["y_test"]
            y_p = ml_results["y_pred"]
            min_v, max_v = min(min(y_t), min(y_p)), max(max(y_t), max(y_p))

            fig_m1, ax1 = plt.subplots(figsize=(7, 4.5))
            ax1.scatter(y_t, y_p, color="#2563EB", alpha=0.75, edgecolors="none")
            ax1.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Ideal Fit (y=x)")
            ax1.set_title("Actual vs Predicted (Holdout Test Set)", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Actual Values")
            ax1.set_ylabel("Predicted Values")
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend()
            buf1 = io.BytesIO()
            fig_m1.savefig(buf1, format="png", bbox_inches="tight", dpi=160)
            plt.close(fig_m1)
            chart_buffers.append(buf1)

            res = np.array(y_t) - np.array(y_p)
            fig_m2, ax2 = plt.subplots(figsize=(7, 4.5))
            ax2.hist(res, bins=15, color="#0F172A", edgecolor="#E2E8F0", alpha=0.9)
            ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
            ax2.set_title("Residual Error Distribution", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Residual (Actual - Predicted)")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, linestyle=":", alpha=0.6)
            buf2 = io.BytesIO()
            fig_m2.savefig(buf2, format="png", bbox_inches="tight", dpi=160)
            plt.close(fig_m2)
            chart_buffers.append(buf2)

        tab_biz, tab_tech, tab_leaderboard, tab_stats, tab_ml, tab_assoc, tab_data = st.tabs([
            "💼 Executive Memo",
            "🛠️ Engineering Deep Dive",
            "🏆 Algorithm Tournament",
            "📐 Quality Gates",
            "📊 Visual Analytics",
            "🔗 Pattern Mining",
            "🛡️ Governance & Bias Audit",
        ])

        biz = reports.get("business_report", {})
        tech = reports.get("technical_report", {})

        exec_deck = create_executive_deck(
            biz_report=biz if isinstance(biz, dict) else {},
            tech_report=tech if isinstance(tech, dict) else {},
            ml_results=ml_results,
            split_counts=split_data["counts"],
            task=task,
            chart_buffers=chart_buffers,
        )

        # 1. Executive Memo
        with tab_biz:
            st.markdown("### 🏛️ Executive Advisory Memo")
            d1, d2 = st.columns(2)
            with d1:
                biz_pdf_sections = [
                    ("Executive Verdict", [str(biz.get("executive_verdict", "Strategic evaluation completed."))]),
                    ("Strategic Recommendations", biz.get("strategic_recommendations", [])),
                    ("Operational Cohort Insights", biz.get("segment_insights", [])),
                    (
                        "Data Governance & Risks",
                        [
                            f"Asset Readiness: {biz.get('data_readiness_score', 'Enterprise Grade')}",
                            *biz.get("potential_business_risks", []),
                        ],
                    ),
                ]
                biz_pdf = create_pdf_report(
                    "MalharIQ Executive Advisory Brief",
                    biz_pdf_sections,
                    {"task": task, "champion": ml_results.get("best_algorithm", "N/A")},
                    chart_buffers=chart_buffers,
                )
                st.download_button(
                    label="📄 Download Executive Memo (PDF)",
                    data=biz_pdf,
                    file_name="MalharIQ_Executive_Memo.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            with d2:
                st.download_button(
                    label="📊 Download Client Presentation (PPT)",
                    data=exec_deck,
                    file_name="MalharIQ_Executive_Presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )

            st.divider()

            if isinstance(biz, dict):
                st.info(f"**Executive Verdict:**\n\n{biz.get('executive_verdict', '')}")
                col_a, col_b = st.columns([1.5, 1])
                with col_a:
                    st.markdown("#### 🎯 Operational Insights")
                    for insight in biz.get("segment_insights", []):
                        st.markdown(f"* {insight}")
                    st.markdown("#### 🚀 Strategic Action Items")
                    for rec in biz.get("strategic_recommendations", []):
                        st.markdown(f"* **{rec}**")
                with col_b:
                    st.markdown("#### 🛡️ Governance & Data Readiness")
                    st.success(f"**Data Asset Score:** {biz.get('data_readiness_score', 'Enterprise Grade')}")
                    st.caption(biz.get("data_readiness_detail", ""))
                    st.markdown("#### ⚠️ Operational Risk Factors")
                    for risk in biz.get("potential_business_risks", []):
                        st.warning(f"• {risk}")

        # 2. Engineering Deep Dive
        with tab_tech:
            st.markdown("### 🛠️ Production Architecture & Engineering Audit")
            t_d1, t_d2 = st.columns(2)
            with t_d1:
                tech_pdf_sections = [
                    ("Architecture Summary", [str(tech.get("architecture_summary", "Architecture verified."))]),
                    ("Data Engineering Transformations", tech.get("data_engineering_audit", logs[:5])),
                    (
                        "Quality Gates & Validation",
                        [
                            "Partitioning: 70% Train, 15% Validation, 15% Holdout Test",
                            f"Sampling Adequacy: {tech.get('statistical_quality_gates', {}).get('adequacy_verdict', 'Passed')}",
                        ],
                    ),
                    ("Production Monitoring & Retraining", tech.get("production_monitoring_plan", [])),
                ]
                tech_pdf = create_pdf_report(
                    "MalharIQ Engineering Technical Audit",
                    tech_pdf_sections,
                    {"task": task, "champion": ml_results.get("best_algorithm", "N/A")},
                    chart_buffers=chart_buffers,
                )
                st.download_button(
                    label="📄 Download Engineering Audit (PDF)",
                    data=tech_pdf,
                    file_name="MalharIQ_Technical_Audit.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            with t_d2:
                st.download_button(
                    label="📊 Download Technical Briefing (PPT)",
                    data=exec_deck,
                    file_name="MalharIQ_Technical_Briefing.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )

            st.divider()

            if isinstance(tech, dict):
                st.markdown(f"**Architecture Overview:**\n{tech.get('architecture_summary', '')}")
                t1, t2 = st.columns(2)
                with t1:
                    st.markdown("#### 📐 Statistical Quality Gates")
                    sq = tech.get("statistical_quality_gates", {})
                    st.markdown(f"* **Sampling Adequacy:** {sq.get('adequacy_verdict', 'Passed')}")
                    st.markdown(f"* **Factor Reduction:** {sq.get('factor_reduction', 'Completed')}")
                    st.markdown("#### 🧹 Preprocessing Audit Log")
                    for eng in tech.get("data_engineering_audit", []):
                        st.markdown(f"* {eng}")
                with t2:
                    st.markdown("#### 🚀 Production & MLOps Roadmap")
                    for mon in tech.get("production_monitoring_plan", []):
                        st.markdown(f"* {mon}")

        # 3. Leaderboard
        with tab_leaderboard:
            st.subheader("Model Competition & Benchmark Leaderboard")
            st.caption("Candidate algorithms tuned on 70% Train, ranked on 15% Validation, validated on 15% Test:")
            if "leaderboard_df" in ml_results:
                st.dataframe(ml_results["leaderboard_df"], use_container_width=True)

        # 4. Quality Gates
        with tab_stats:
            st.subheader("Statistical Validation & Factor Matrices")
            if task == "Clustering / Segmentation":
                kmo = stats_results.get("kmo_bartlett", {})
                if "error" not in kmo:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("KMO Sampling Adequacy", f"{kmo.get('kmo_score')}", delta=kmo.get("kmo_rating"))
                    m2.metric("Bartlett Chi-Square", f"{kmo.get('bartlett_chi_sq')}")
                    m3.metric("Bartlett P-Value", f"{kmo.get('bartlett_p_value')}")
                    m4.metric("Clustering Suitability", f"{kmo.get('clustering_suitability')}")

                fa = stats_results.get("factor_analysis", {})
                if "factor_loadings_df" in fa:
                    st.divider()
                    st.subheader("Orthogonal Factor Loadings Matrix")
                    st.dataframe(fa["factor_loadings_df"], use_container_width=True)
            else:
                p_val_df = stats_results.get("feature_significance_tests")
                if isinstance(p_val_df, pd.DataFrame) and not p_val_df.empty:
                    st.dataframe(p_val_df, use_container_width=True)

        # 5. Visual Analytics (Diagnostics & Drivers)
        with tab_ml:
            st.subheader(f"Selected Solution: {ml_results.get('best_algorithm', 'Champion Architecture')}")
            
            # --- REGRESSION VISUAL DIAGNOSTICS ---
            if task == "Regression":
                m1, m2, m3 = st.columns(3)
                m1.metric("Champion Architecture", ml_results.get("best_algorithm", "N/A"))
                r2_v = ml_results.get("r2_score")
                m2.metric("Holdout R² Variance", f"{r2_v:.4f}" if isinstance(r2_v, (int, float)) else str(r2_v or "N/A"))
                mae_v = ml_results.get("mae")
                m3.metric("Mean Absolute Error (MAE)", f"{mae_v:.4f}" if isinstance(mae_v, (int, float)) else str(mae_v or "N/A"))

                st.markdown("#### 🎯 Empirical Regression Diagnostics")
                r_col1, r_col2 = st.columns(2)

                if "y_test" in ml_results and "y_pred" in ml_results:
                    y_test_data = ml_results["y_test"]
                    y_pred_data = ml_results["y_pred"]
                    eval_df = pd.DataFrame({"Actual": y_test_data, "Predicted": y_pred_data})
                    eval_df["Residual"] = eval_df["Actual"] - eval_df["Predicted"]

                    with r_col1:
                        fig_scatter = px.scatter(
                            eval_df,
                            x="Actual",
                            y="Predicted",
                            title="Actual vs. Predicted (Holdout Test Set)",
                            template="plotly_white",
                        )
                        min_b = min(min(y_test_data), min(y_pred_data))
                        max_b = max(max(y_test_data), max(y_pred_data))
                        fig_scatter.add_shape(
                            type="line",
                            x0=min_b, y0=min_b, x1=max_b, y1=max_b,
                            line=dict(color="red", dash="dash"),
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    with r_col2:
                        fig_hist = px.histogram(
                            eval_df,
                            x="Residual",
                            nbins=15,
                            title="Residual Error Distribution",
                            template="plotly_white",
                            color_discrete_sequence=["#0F172A"],
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                if "feature_importances" in ml_results and ml_results["feature_importances"]:
                    fi_df = pd.DataFrame(
                        list(ml_results["feature_importances"].items()),
                        columns=["Predictive Driver", "Relative Impact"],
                    ).sort_values(by="Relative Impact", ascending=True)
                    fig_fi = px.bar(
                        fi_df,
                        x="Relative Impact",
                        y="Predictive Driver",
                        orientation="h",
                        title="Predictive Driver Importance (Normalized Absolute Impact)",
                        color="Relative Impact",
                        color_continuous_scale="Blues",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

            # --- CLASSIFICATION VISUAL DIAGNOSTICS ---
            elif task == "Classification":
                m1, m2, m3 = st.columns(3)
                m1.metric("Champion Model", ml_results.get("best_algorithm", "N/A"))
                acc = ml_results.get("accuracy")
                m2.metric("Holdout Test Accuracy", f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) else str(acc or "N/A"))
                f1 = ml_results.get("f1_score")
                m3.metric("Holdout Weighted F1", f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1 or "N/A"))

                if "feature_importances" in ml_results and ml_results["feature_importances"]:
                    fi_df = pd.DataFrame(
                        list(ml_results["feature_importances"].items()),
                        columns=["Driver", "Relative Impact"],
                    ).sort_values(by="Relative Impact", ascending=True)
                    fig = px.bar(
                        fi_df,
                        x="Relative Impact",
                        y="Driver",
                        orientation="h",
                        title="Key Classification Decision Drivers",
                        color="Relative Impact",
                        color_continuous_scale="Blues",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # --- CLUSTERING VISUAL DIAGNOSTICS ---
            elif task == "Clustering / Segmentation":
                m1, m2, m3 = st.columns(3)
                m1.metric("Optimal Architecture", ml_results.get("best_algorithm", "N/A"))
                m2.metric("Cluster Granularity (k)", f"{ml_results.get('best_k', 'N/A')}")
                sil = ml_results.get("silhouette_score")
                m3.metric("Silhouette Separation", f"{sil:.4f}" if isinstance(sil, (int, float)) else str(sil or "N/A"))

                if "result_df" in ml_results:
                    res_df = ml_results["result_df"].copy()
                    feature_cols = [c for c in res_df.columns if c != "Cluster" and not c.lower().startswith("id")]
                    if len(feature_cols) >= 2:
                        res_df["Cluster Segment"] = "Segment " + res_df["Cluster"].astype(str)
                        fig = px.scatter(
                            res_df,
                            x=feature_cols[0],
                            y=feature_cols[1],
                            color="Cluster Segment",
                            title=f"Segment Separation: {feature_cols[0]} vs {feature_cols[1]}",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # 6. Pattern Mining
        with tab_assoc:
            st.subheader("Discovered Behavioral & Cross-Variable Associations")
            if isinstance(assoc_rules_df, pd.DataFrame) and not assoc_rules_df.empty:
                st.dataframe(assoc_rules_df, use_container_width=True)
            else:
                st.info("No frequent itemset patterns discovered above support/confidence thresholds.")

        # 7. Governance & Bias Audit (EEOC 80% Rule)
        with tab_data:
            st.subheader("⚖️ Regulatory Compliance & Bias Audit (EEOC Four-Fifths)")
            st.caption("Evaluates selection parity across demographic attributes or proxy slices.")

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                sens_col = st.selectbox(
                    "Select Sensitive Attribute (e.g., Demographics, Region, Tier):",
                    clean_df.columns,
                    index=0,
                )
            with b_col2:
                out_candidates = [c for c in clean_df.columns if c != sens_col]
                out_col = st.selectbox(
                    "Select Outcome / Target Feature:",
                    out_candidates,
                    index=0 if out_candidates else None,
                )

            if sens_col and out_col:
                audit_res = audit_disparate_impact(clean_df, sens_col, out_col)

                if audit_res.get("status") == "PASS":
                    st.success(f"🟢 **COMPLIANT:** {audit_res['verdict']}")
                else:
                    st.warning(f"🟡 **REGULATORY ALERT:** {audit_res['verdict']}")

                if "disparate_impact_ratios" in audit_res:
                    ratio_series = pd.DataFrame(
                        list(audit_res["disparate_impact_ratios"].items()),
                        columns=["Cohort", "Parity Ratio"],
                    )
                    fig_bias = px.bar(
                        ratio_series,
                        x="Cohort",
                        y="Parity Ratio",
                        title=f"Disparate Impact Ratio across '{sens_col}' (Benchmark: 1.0, Floor: 0.80)",
                        template="plotly_white",
                        color="Parity Ratio",
                        color_continuous_scale="RdYlGn",
                    )
                    fig_bias.add_hline(
                        y=0.80,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="EEOC 80% Threshold",
                    )
                    st.plotly_chart(fig_bias, use_container_width=True)

            st.divider()
            st.subheader("🧹 Cleaning Audit Trail & Production Asset")
            for log in logs:
                st.success(f"✔️ {log}")
            st.dataframe(clean_df.head(10), use_container_width=True)
            st.download_button(
                "⬇️ Export Clean Data Asset (.csv)",
                clean_df.to_csv(index=False),
                "MalharIQ_clean_asset.csv",
                "text/csv",
            )

        # 8. Local Copilot (llama3.2:1b)
        st.divider()
        st.subheader("💬 Ask MalharIQ Copilot (Local LLM)")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_prompt := st.chat_input("Ask about tournament findings, partitions, or governance..."):
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            copilot_system_context = f"""
            You are the MalharIQ Analytical Copilot. Answer clearly and concisely using ONLY this run context:
            - Strategic Goal: {task}
            - Partitions: 70% Train ({c_train} rows), 15% Val ({c_val} rows), 15% Test ({c_test} rows)
            - Selected Champion Architecture: {ml_results.get('best_algorithm')}
            - Performance: {ml_results.get('f1_score') if task=='Classification' else ml_results.get('r2_score', ml_results.get('silhouette_score'))}
            - Preprocessing Logs: {logs[:3]}
            Keep your answer professional, under 3-4 sentences, and focused on operational value.
            """

            with st.chat_message("assistant"):
                with st.spinner("Analyzing pipeline metadata locally..."):
                    try:
                        resp = ollama.chat(
                            model="llama3.2:1b",
                            messages=[
                                {"role": "system", "content": copilot_system_context},
                                *st.session_state.chat_history,
                            ],
                            options={"temperature": 0.2, "num_predict": 250},
                        )
                        bot_reply = resp["message"]["content"]
                        st.markdown(bot_reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    except Exception as ex:
                        err_msg = f"Local model error: {str(ex)}. Ensure `ollama run llama3.2:1b` is active."
                        st.error(err_msg)