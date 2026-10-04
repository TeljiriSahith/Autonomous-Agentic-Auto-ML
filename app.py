import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    mean_squared_error, r2_score
)
import tempfile
import os
import pickle

# Backend imports
from data_analysis import analyze_and_plan
from cleaning_preprocessing import execute_cleaning_pipeline
from model_training import train_all_models, select_best_model_with_llm
from optimizer import optimize_best_model

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Autonomous Agentic AutoML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CLEAN ENTERPRISE DARK THEME (HIGH CONTRAST & LEGIBILITY)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background Canvas */
    .stApp {
        background: #0b0f19;
        color: #f1f5f9;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
    }
    .hero-title {
        color: #38bdf8;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }

    /* Cards */
    .dashboard-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Custom Metric Badges (Fixes Native Invisible Text Bug) */
    .metric-box {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-box-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-box-val {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-box-delta {
        color: #34d399;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Fix Table Text Contrast */
    [data-testid="stTable"] table {
        color: #f8fafc !important;
        background-color: rgba(17, 24, 39, 0.85) !important;
        border-radius: 8px;
    }
    [data-testid="stTable"] th {
        color: #38bdf8 !important;
        background-color: rgba(30, 41, 59, 0.95) !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #0284c7 !important;
    }
    [data-testid="stTable"] td {
        color: #e2e8f0 !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15) !important;
    }

    /* Expander Text & Icon Fix */
    div[data-testid="stExpander"] {
        background-color: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid rgba(75, 85, 99, 0.4) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] summary {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #0284c7, #4f46e5) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0369a1, #4338ca) !important;
    }

    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #059669, #10b981) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "state" not in st.session_state:
    st.session_state.state = {
        "df_raw": None,
        "plan": None,
        "cleaned_df": None,
        "benchmark_results": None,
        "trained_models": None,
        "verdict": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "tuned_model": None,
        "tuned_score": None,
        "improved": False,
        "best_params": {},
        "audit_report": {},
        "optimization_run": False
    }

# =====================================================================
# HEADER
# =====================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Autonomous Agentic AutoML</div>
    <p class="hero-subtitle">
        Local LLM Reasoning • Adaptive Preprocessing • Benchmark Evaluation • Optuna Fine-Tuning
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Control")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    user_prompt = st.text_area(
        "Agent Objective",
        placeholder="e.g., Predict whether a customer will churn based on precision",
        height=100
    )
    
    execute_button = st.button("🚀 Run Pipeline", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Engine:** Ollama Llama 3.2 (Local)")
    st.markdown("**Status:** Ready")

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if execute_button:
    if not uploaded_file:
        st.error("Please upload a CSV dataset first.")
    elif not user_prompt.strip():
        st.error("Please provide an objective prompt.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.status("Executing AutoML Pipeline...", expanded=True) as status:
            st.write("🔍 Stage 1: Inspecting schema and determining primary metric...")
            df_raw = pd.read_csv(tmp_path)
            plan = analyze_and_plan(tmp_path, user_prompt)
            
            st.write(f"🧹 Stage 2: Applying {len(plan.steps)} data cleaning & encoding steps...")
            cleaned_df = execute_cleaning_pipeline(df_raw, [s.model_dump() for s in plan.steps])
            
            st.write("🏋️ Stage 3: Benchmarking candidate ML algorithms...")
            results, models = train_all_models(
                cleaned_df=cleaned_df,
                problem_type=plan.problem_type,
                target_column=plan.target_column
            )
            
            st.write("🧠 Stage 4: Selecting champion model based on benchmark metric...")
            verdict = select_best_model_with_llm(
                results_table=results,
                problem_type=plan.problem_type,
                user_goal=user_prompt,
                primary_metric=plan.primary_metric
            )
            
            X = cleaned_df.drop(columns=[plan.target_column])
            y = cleaned_df[plan.target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.session_state.state.update({
                "df_raw": df_raw,
                "plan": plan,
                "cleaned_df": cleaned_df,
                "benchmark_results": results,
                "trained_models": models,
                "verdict": verdict,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "tuned_model": None,
                "tuned_score": None,
                "improved": False,
                "best_params": {},
                "audit_report": {},
                "optimization_run": False
            })
            
            status.update(label="Pipeline Execution Completed!", state="complete", expanded=False)
            os.remove(tmp_path)

# =====================================================================
# RESULTS DISPLAY
# =====================================================================
if st.session_state.state["plan"] is not None:
    state = st.session_state.state
    plan = state["plan"]
    verdict = state["verdict"]

    # 1. Preprocessing Summary
    st.markdown("### 1. Preprocessing & Feature Transformations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Ingested Data**")
        st.dataframe(state["df_raw"].head(5), use_container_width=True)
    with c2:
        st.markdown(f"**Cleaned Feature Space** (Shape: `{state['cleaned_df'].shape}`)")
        st.dataframe(state["cleaned_df"].head(5), use_container_width=True)

    with st.expander("🔎 View Applied Transformations Log", expanded=True):
        plan_table = pd.DataFrame([{
            "Feature": step.column,
            "Operation": step.operation,
            "Rationale": step.reason
        } for step in plan.steps])
        st.table(plan_table)
        
        # High-contrast custom banner replacing st.info
        st.markdown(f"""
        <div style="
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid #38bdf8;
            border-radius: 8px;
            padding: 12px 18px;
            margin-top: 10px;
            display: flex;
            gap: 24px;
            align-items: center;
        ">
            <span style="color: #94a3b8; font-size: 0.95rem;">Target: <b style="color: #ffffff;">{plan.target_column}</b></span>
            <span style="color: #94a3b8; font-size: 0.95rem;">Task: <b style="color: #38bdf8;">{plan.problem_type.upper()}</b></span>
            <span style="color: #94a3b8; font-size: 0.95rem;">Priority Metric: <b style="color: #34d399; font-weight: 700;">{plan.primary_metric.upper()}</b></span>
        </div>
        """, unsafe_allow_html=True)

    # 2. Benchmark Horizontal Bar Graph
    st.markdown("---")
    st.markdown(f"### 2. Candidate Algorithm Benchmarks (Optimizing: {verdict.primary_metric_name})")
    
    df_bench = pd.DataFrame(state["benchmark_results"]).T.reset_index().rename(columns={"index": "Model"})
    df_bench_sorted = df_bench.sort_values(by=verdict.primary_metric_name, ascending=True)
    
    fig = px.bar(
        df_bench_sorted,
        x=verdict.primary_metric_name,
        y="Model",
        orientation="h",
        color=verdict.primary_metric_name,
        color_continuous_scale="tealgrn",
        text=verdict.primary_metric_name,
        template="plotly_dark",
        title=f"Benchmark Standings ({verdict.primary_metric_name})"
    )
    fig.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 3. Model Verdict Card
    st.markdown("---")
    st.markdown("### 3. Champion Model Selection")
    st.markdown(f"""
    <div class="dashboard-card">
        <h3 style="color: #38bdf8; margin: 0 0 8px 0;">Selected Champion: {verdict.best_model_name}</h3>
        <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 12px;">
            Validation Benchmark: <b>{verdict.primary_metric_name}</b> = <code style="color: #34d399; font-size: 1.1rem;">{verdict.primary_metric_value:.4f}</code>
        </p>
        <p style="color: #94a3b8; line-height: 1.6; margin: 0;"><b>Reasoning:</b> {verdict.reasoning}</p>
    </div>
    """, unsafe_allow_html=True)

    # 4. Generalization Fit Curves
    st.markdown("---")
    st.markdown("### 4. Baseline Fit: Train vs. Test Evaluation")
    
    baseline_model = state["trained_models"][verdict.best_model_name]
    X_train, X_test = state["X_train"], state["X_test"]
    y_train, y_test = state["y_train"], state["y_test"]

    col_fit1, col_fit2 = st.columns(2)

    if plan.problem_type == "classification":
        with col_fit1:
            st.markdown("**ROC-AUC Curve (Train vs. Test)**")
            fig_roc = go.Figure()

            if hasattr(baseline_model, "predict_proba"):
                y_train_prob = baseline_model.predict_proba(X_train)[:, 1]
                fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_prob)
                fig_roc.add_trace(go.Scatter(x=fpr_tr, y=tpr_tr, name=f"Train ROC (AUC: {auc(fpr_tr, tpr_tr):.3f})", line=dict(color="#38bdf8", dash="dash")))

                y_test_prob = baseline_model.predict_proba(X_test)[:, 1]
                fpr_te, tpr_te, _ = roc_curve(y_test, y_test_prob)
                fig_roc.add_trace(go.Scatter(x=fpr_te, y=tpr_te, name=f"Test ROC (AUC: {auc(fpr_te, tpr_te):.3f})", line=dict(color="#f43f5e", width=2)))

            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline", line=dict(color="#64748b", dash="dot")))
            fig_roc.update_layout(template="plotly_dark", height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_fit2:
            st.markdown("**Test Split Confusion Matrix**")
            y_pred_test = baseline_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm = px.imshow(
                cm, text_auto=True, 
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Label", y="True Label"),
                x=["Class 0", "Class 1"],
                y=["Class 0", "Class 1"],
                template="plotly_dark"
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        with col_fit1:
            st.markdown("**Parity Plot (Train vs. Test)**")
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=y_train, y=baseline_model.predict(X_train), mode="markers", name="Train Split", opacity=0.6, marker=dict(color="#38bdf8")))
            fig_reg.add_trace(go.Scatter(x=y_test, y=baseline_model.predict(X_test), mode="markers", name="Test Split", opacity=0.8, marker=dict(color="#f43f5e")))
            fig_reg.update_layout(template="plotly_dark", height=380, xaxis_title="Actual Values", yaxis_title="Predicted Values")
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_fit2:
            st.markdown("**Residual Error Distribution**")
            residuals = y_test - baseline_model.predict(X_test)
            fig_res = px.histogram(residuals, nbins=20, template="plotly_dark", title="Test Residuals")
            fig_res.update_layout(height=380)
            st.plotly_chart(fig_res, use_container_width=True)

    # 5. Hyperparameter Tuning Section
    st.markdown("---")
    st.markdown("### 5. Hyperparameter Tuning & Bayesian Optimization")
    st.write(f"The baseline **{verdict.best_model_name}** achieved a score of `{verdict.primary_metric_value:.4f}`.")

    opt_col, _ = st.columns([1, 2])
    with opt_col:
        optimize_trigger = st.button("✨ Optimize Champion Model", use_container_width=True)

    if optimize_trigger or state["optimization_run"]:
        if not state["optimization_run"]:
            with st.spinner(f"Running Optuna Bayesian Optimization on {verdict.best_model_name} targeting {plan.primary_metric.upper()}..."):
                final_model, final_score, improved, best_params, audit_report = optimize_best_model(
                    model_name=verdict.best_model_name,
                    baseline_model=baseline_model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    problem_type=plan.problem_type,
                    primary_metric=plan.primary_metric,
                    n_trials=15
                )
                state["tuned_model"] = final_model
                state["tuned_score"] = final_score
                state["improved"] = improved
                state["best_params"] = best_params
                state["audit_report"] = audit_report
                state["optimization_run"] = True

        st.success("Optimization Process Completed!")

        audit = state.get("audit_report", {})
        delta_val = state["tuned_score"] - verdict.primary_metric_value
        delta_str = f"+{delta_val:.4f}" if delta_val >= 0 else f"{delta_val:.4f}"
        safeguard_str = "Adopted Tuned" if state["improved"] else "Retained Baseline"

        # Three Clear KPI Metric Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Baseline ({plan.primary_metric.upper()})</div>
                <div class="metric-box-val">{verdict.primary_metric_value:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Optimized ({plan.primary_metric.upper()})</div>
                <div class="metric-box-val">{state['tuned_score']:.4f}</div>
                <div class="metric-box-delta">{delta_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Safeguard Decision</div>
                <div class="metric-box-val" style="font-size: 1.3rem; padding-top: 6px; color: {'#34d399' if state['improved'] else '#38bdf8'};">{safeguard_str}</div>
            </div>
            """, unsafe_allow_html=True)

        # Full Technical Explanation & Decision Breakdown Card
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 20px; margin-top: 18px;">
            <h4 style="color: #38bdf8; margin: 0 0 10px 0;">🔍 Optimization Audit & Decision Rationale</h4>
            <p style="color: #e2e8f0; font-size: 0.95rem; margin-bottom: 8px;">
                <b>Methodology Used:</b> {audit.get('technique', 'Bayesian TPE Search')}
            </p>
            <p style="color: #e2e8f0; font-size: 0.95rem; margin-bottom: 8px;">
                <b>Optimization Objective:</b> Maximizing <code>{audit.get('target_metric', plan.primary_metric.upper())}</code> across <b>{audit.get('trials_evaluated', 15)} trials</b>.
            </p>
            <p style="color: #e2e8f0; font-size: 0.95rem; margin-bottom: 8px;">
                <b>Trial Score Spread:</b> Highest trial scored <b>{audit.get('best_trial_score', 0):.4f}</b> | Lowest trial scored <b>{audit.get('lowest_trial_score', 0):.4f}</b>.
            </p>
            <div style="background: rgba(30, 41, 59, 0.8); border-left: 4px solid {'#34d399' if state['improved'] else '#f59e0b'}; padding: 12px 16px; border-radius: 6px; margin-top: 12px;">
                <b style="color: {'#34d399' if state['improved'] else '#f59e0b'};">System Action:</b>
                <span style="color: #cbd5e1; font-size: 0.93rem;"> {audit.get('decision_verdict', '')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if state["best_params"]:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🛠️ Best Discovered Hyperparameter Configuration", expanded=False):
                st.json(state["best_params"])

        # Post-Optimization Fit Comparison
        st.markdown("#### Post-Optimization Fit Comparison")
        active_model = state["tuned_model"] if state["improved"] else baseline_model
        
        if plan.problem_type == "classification" and hasattr(active_model, "predict_proba"):
            fig_post = go.Figure()
            y_test_prob_opt = active_model.predict_proba(X_test)[:, 1]
            fpr_opt, tpr_opt, _ = roc_curve(y_test, y_test_prob_opt)
            
            fig_post.add_trace(go.Scatter(
                x=fpr_opt, y=tpr_opt, 
                name=f"Final Model Test ROC (AUC: {auc(fpr_opt, tpr_opt):.3f})", 
                line=dict(color="#10b981", width=3)
            ))
            fig_post.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#64748b", dash="dot"), showlegend=False))
            fig_post.update_layout(template="plotly_dark", height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_post, use_container_width=True)

    # 6. Save & Download Model Artifact (.pkl)
    st.markdown("---")
    st.markdown("### 6. Export Trained Model Artifact")
    
    model_to_export = state["tuned_model"] if (state["optimization_run"] and state["improved"]) else baseline_model
    model_bytes = pickle.dumps(model_to_export)
    
    col_dl1, col_dl2 = st.columns([2, 1])
    with col_dl1:
        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(75, 85, 99, 0.4); border-radius: 10px; padding: 18px;">
            <b style="color: #38bdf8;">Production-Ready Artifact:</b> Download the finalized <code>{verdict.best_model_name}</code> model serialized in standard Python Pickle format (<code>.pkl</code>).
        </div>
        """, unsafe_allow_html=True)
    with col_dl2:
        st.download_button(
            label=f"💾 Download {verdict.best_model_name}.pkl",
            data=model_bytes,
            file_name=f"{verdict.best_model_name.lower()}_model.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )