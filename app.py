import warnings
warnings.filterwarnings("ignore")

import os
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Future of Jobs | AI Impact Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    [data-testid='stHeader'] { background-color: rgba(0,0,0,0); }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    /* Style for the sidebar filters to make them look more organized */
    .stMultiSelect div[role='listbox'] { background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Constants & Config ---
CSV_FILE = "data/Future of Jobs AI Dataset.csv"
PRIMARY_COLOR = "#6366f1"
COLOR_PALETTE = px.colors.qualitative.Prism

# --- Data Engine ---
@st.cache_data
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    df = pd.read_csv(file_path)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    return df

# --- UI Components ---
def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e293b, #334155); padding: 22px; border-radius: 15px; border-left: 6px solid {PRIMARY_COLOR}; margin-bottom: 15px;">
            <div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500; text-transform: uppercase;">{label}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-top: 4px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Sidebar Filters (Updated with Dropdowns) ---
def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("🔍 Data Navigator")
    st.sidebar.markdown("Use the dropdowns below to slice the data.")
    
    # 1. Geography & Roles Dropdown
    with st.sidebar.expander("🌍 Geography & Roles", expanded=False):
        countries = st.multiselect(
            "Select Countries", 
            options=sorted(df["country"].unique()), 
            default=df["country"].unique()
        )
        jobs = st.multiselect(
            "Job Roles", 
            options=sorted(df["job_title"].unique()), 
            default=df["job_title"].unique()
        )
    
    # 2. Experience & Risk Dropdown
    with st.sidebar.expander("📈 Experience & Risk", expanded=False):
        exps = st.multiselect(
            "Experience Level", 
            options=df["experience_level"].unique(), 
            default=df["experience_level"].unique()
        )
        year_range = st.slider(
            "Timeline", 
            int(df["year"].min()), 
            int(df["year"].max()), 
            (int(df["year"].min()), int(df["year"].max()))
        )
        risks = st.multiselect(
            "AI Risk Level", 
            options=df["ai_risk_category"].unique(), 
            default=df["ai_risk_category"].unique()
        )
    
    st.sidebar.divider()
    if st.sidebar.button("Reset All Filters", use_container_width=True):
        st.rerun()
        
    return countries, jobs, exps, year_range, risks

# --- Main Render Functions ---
def render_overview(df: pd.DataFrame):
    st.subheader("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Analysis Sample", f"{len(df):,}")
    with c2: metric_card("Avg Market Salary", f"${df['salary'].mean():,.0f}")
    with c3: metric_card("Mean AI Risk", f"{df['ai_risk_score'].mean():.2f}")
    with c4: metric_card("Avg Skill Demand", f"{df['skill_demand_score'].mean():.1f}")

    left, right = st.columns([2, 1])
    with left:
        trend = df.groupby("year")["salary"].mean().reset_index()
        fig = px.area(trend, x="year", y="salary", title="Salary Trajectory Over Time", color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(template="plotly_white", yaxis_title="Salary ($)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        risk_dist = df["ai_risk_category"].value_counts().reset_index()
        fig = px.pie(risk_dist, names="ai_risk_category", values="count", title="Workforce Risk Distribution", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

def render_eda_tab(df: pd.DataFrame):
    st.subheader("In-Depth Exploratory Analysis")
    eda_tabs = st.tabs(["💰 Compensation", "⚠️ AI Risk Profiling", "📊 Distributions"])
    
    with eda_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="experience_level", y="salary", color="experience_level", title="Salary by Seniority")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x="education_level", y="salary", color="education_level", title="Education Premium")
            st.plotly_chart(fig, use_container_width=True)
        
        avg_sal_job = df.groupby("job_title")["salary"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_sal_job, x="job_title", y="salary", color="salary", title="Benchmarks by Role", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

    with eda_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="ai_risk_score", y="salary", color="job_title", size="skill_demand_score", title="Risk-Salary Correlation")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.violin(df, x="job_title", y="ai_risk_score", color="job_title", box=True, title="AI Risk Density")
            st.plotly_chart(fig, use_container_width=True)

    with eda_tabs[2]:
        feature = st.selectbox("Select metric", ["salary", "ai_risk_score", "skill_demand_score"])
        fig = px.histogram(df, x=feature, nbins=40, marginal="box", color_discrete_sequence=['#10b981'])
        st.plotly_chart(fig, use_container_width=True)

def render_insights_tab(df: pd.DataFrame):
    st.subheader("Strategic Insights")
    top_skill = df.groupby('primary_skill')['skill_demand_score'].mean().idxmax()
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"🚀 **Market Leader:** {top_skill}")
    with c2:
        st.info(f"🛡️ **Stability Index:** {df['job_survival_class'].mean():.2f}")
    st.divider()
    st.dataframe(df, use_container_width=True)

# --- Application Entry Point ---
def main():
    try:
        df_raw = load_and_clean_data(CSV_FILE)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}"); st.stop()

    # Sidebar
    countries, jobs, exps, years, risks = render_sidebar(df_raw)
    
    # Filter Logic
    mask = (
        df_raw["country"].isin(countries) & 
        df_raw["job_title"].isin(jobs) & 
        df_raw["experience_level"].isin(exps) & 
        df_raw["year"].between(years[0], years[1]) & 
        df_raw["ai_risk_category"].isin(risks)
    )
    df = df_raw[mask]

    if df.empty:
        st.warning("No data matches the selected filters."); st.stop()

    # Tabs
    main_tabs = st.tabs(["📊 Overview", "🔍 Deep Analysis (EDA)", "💡 Strategic Insights"])
    with main_tabs[0]: render_overview(df)
    with main_tabs[1]: render_eda_tab(df)
    with main_tabs[2]: render_insights_tab(df)

if __name__ == "__main__":
    main()
