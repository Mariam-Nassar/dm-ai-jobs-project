import warnings
warnings.filterwarnings("ignore")

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="AI Jobs Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_FILE = "data/Future of Jobs AI Dataset.csv"

EXPECTED_COLUMNS = [
    "job_title",
    "country",
    "experience_level",
    "education_level",
    "year",
    "salary",
    "ai_risk_score",
    "primary_skill",
    "skill_demand_score",
    "job_openings",
    "job_survival_class",
    "salary_bucket",
    "ai_risk_category",
]

NUMERIC_CANDIDATES = [
    "year",
    "salary",
    "ai_risk_score",
    "skill_demand_score",
    "job_openings",
    "job_survival_class",
]

CATEGORICAL_CANDIDATES = [
    "job_title",
    "country",
    "experience_level",
    "education_level",
    "primary_skill",
    "salary_bucket",
    "ai_risk_category",
]

TARGET_CLASS = "job_survival_class"
TARGET_REG = "salary"
LEAKAGE_COLUMNS = ["salary_bucket", "ai_risk_category"]


@st.cache_data

def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Put the CSV in the same folder as app.py."
        )
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


@st.cache_data

def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "ai_risk_score" in data.columns and "ai_risk_category" not in data.columns:
        data["ai_risk_category"] = pd.cut(
            data["ai_risk_score"],
            bins=[-0.01, 0.33, 0.66, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )

    if "salary" in data.columns and "salary_bucket" not in data.columns:
        q1 = data["salary"].quantile(0.33)
        q2 = data["salary"].quantile(0.66)
        data["salary_bucket"] = pd.cut(
            data["salary"],
            bins=[-np.inf, q1, q2, np.inf],
            labels=["Low", "Medium", "High"],
        )

    if {"salary", "job_openings"}.issubset(data.columns):
        data["salary_per_opening"] = data["salary"] / data["job_openings"].replace(0, np.nan)

    if {"ai_risk_score", "skill_demand_score"}.issubset(data.columns):
        data["risk_demand_gap"] = data["skill_demand_score"] - (data["ai_risk_score"] * 100)

    return data


@st.cache_data

def get_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_values": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values,
        "unique_values": df.nunique(dropna=False).values,
    })
    return quality


@st.cache_data

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


@st.cache_data

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


@st.cache_data

def build_filtered_data(df: pd.DataFrame, countries, job_titles, exp_levels, years, risk_categories):
    data = df.copy()

    if countries and "country" in data.columns:
        data = data[data["country"].isin(countries)]
    if job_titles and "job_title" in data.columns:
        data = data[data["job_title"].isin(job_titles)]
    if exp_levels and "experience_level" in data.columns:
        data = data[data["experience_level"].isin(exp_levels)]
    if years and "year" in data.columns:
        data = data[data["year"].between(years[0], years[1])]
    if risk_categories and "ai_risk_category" in data.columns:
        data = data[data["ai_risk_category"].isin(risk_categories)]

    return data


@st.cache_resource


def business_insights(df: pd.DataFrame) -> List[str]:
    insights = []

    if {"job_title", "ai_risk_score"}.issubset(df.columns):
        top_risky = (
            df.groupby("job_title", as_index=False)["ai_risk_score"]
            .mean()
            .sort_values("ai_risk_score", ascending=False)
            .head(1)
        )
        if not top_risky.empty:
            insights.append(
                f"Highest average AI risk appears in {top_risky.iloc[0]['job_title']} with an average risk score of {top_risky.iloc[0]['ai_risk_score']:.2f}."
            )

    if {"country", "salary"}.issubset(df.columns):
        top_salary = (
            df.groupby("country", as_index=False)["salary"]
            .mean()
            .sort_values("salary", ascending=False)
            .head(1)
        )
        if not top_salary.empty:
            insights.append(
                f"The highest average salary is in {top_salary.iloc[0]['country']} at about ${top_salary.iloc[0]['salary']:,.0f}."
            )

    if {"ai_risk_score", "salary"}.issubset(df.columns):
        corr = df[["ai_risk_score", "salary"]].corr().iloc[0, 1]
        insights.append(
            f"The correlation between AI risk score and salary is {corr:.2f}. This shows whether higher AI exposure is linked to salary increase or decline."
        )

    if {"primary_skill", "skill_demand_score"}.issubset(df.columns):
        top_skill = (
            df.groupby("primary_skill", as_index=False)["skill_demand_score"]
            .mean()
            .sort_values("skill_demand_score", ascending=False)
            .head(1)
        )
        if not top_skill.empty:
            insights.append(
                f"{top_skill.iloc[0]['primary_skill']} has the strongest average demand score at {top_skill.iloc[0]['skill_demand_score']:.1f}."
            )

    return insights[:4]



def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #111827, #4c1d95); padding: 18px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-size: 14px; color: #d1d5db;">{label}</div>
            <div style="font-size: 28px; font-weight: 700; color: white; margin-top: 6px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def sidebar_filters(df: pd.DataFrame):
    st.sidebar.title("Dashboard Controls")
    st.sidebar.caption("Global filters update the full dashboard")

    countries = st.sidebar.multiselect(
        "Country",
        options=sorted(df["country"].dropna().unique()) if "country" in df.columns else [],
        default=sorted(df["country"].dropna().unique()) if "country" in df.columns else [],
    )

    job_titles = st.sidebar.multiselect(
        "Job Title",
        options=sorted(df["job_title"].dropna().unique()) if "job_title" in df.columns else [],
        default=sorted(df["job_title"].dropna().unique()) if "job_title" in df.columns else [],
    )

    exp_levels = st.sidebar.multiselect(
        "Experience Level",
        options=sorted(df["experience_level"].dropna().unique()) if "experience_level" in df.columns else [],
        default=sorted(df["experience_level"].dropna().unique()) if "experience_level" in df.columns else [],
    )

    if "year" in df.columns:
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=(int(df["year"].min()), int(df["year"].max())),
        )
    else:
        year_range = None

    risk_categories = st.sidebar.multiselect(
        "AI Risk Category",
        options=sorted(df["ai_risk_category"].dropna().astype(str).unique()) if "ai_risk_category" in df.columns else [],
        default=sorted(df["ai_risk_category"].dropna().astype(str).unique()) if "ai_risk_category" in df.columns else [],
    )

    return countries, job_titles, exp_levels, year_range, risk_categories



def render_overview(df: pd.DataFrame):
    st.subheader("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Records", f"{len(df):,}")
    with c2:
        metric_card("Features", f"{df.shape[1]}")
    with c3:
        metric_card("Countries", f"{df['country'].nunique() if 'country' in df.columns else 'N/A'}")
    with c4:
        metric_card("Job Titles", f"{df['job_title'].nunique() if 'job_title' in df.columns else 'N/A'}")

    st.write("")

    left, right = st.columns([1.2, 1])

    with left:
        if {"year", "salary"}.issubset(df.columns):
            trend = df.groupby("year", as_index=False)["salary"].mean()
            fig = px.line(
                trend,
                x="year",
                y="salary",
                markers=True,
                title="Average Salary Trend Over Time",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if "job_survival_class" in df.columns:
            fig = px.pie(
                df,
                names="job_survival_class",
                title="Job Survival Class Distribution",
                hole=0.5,
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    if {"job_title", "ai_risk_score", "salary"}.issubset(df.columns):
        fig = px.scatter(
            df,
            x="ai_risk_score",
            y="salary",
            color="job_title",
            size="job_openings" if "job_openings" in df.columns else None,
            hover_data=[c for c in ["country", "experience_level", "primary_skill", "year"] if c in df.columns],
            title="Salary vs AI Risk Score",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)



def render_data_quality(df: pd.DataFrame):
    st.subheader("Data Quality and Structure")
    quality = get_data_quality(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Missing Values", f"{int(df.isna().sum().sum()):,}")
    with c2:
        metric_card("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    with c3:
        metric_card("Numeric Columns", f"{len(get_numeric_columns(df))}")
    with c4:
        metric_card("Categorical Columns", f"{len(get_categorical_columns(df))}")

    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(quality, use_container_width=True, height=420)
    with right:
        missing_chart = quality.sort_values("missing_values", ascending=False)
        fig = px.bar(
            missing_chart,
            x="column",
            y="missing_values",
            title="Missing Values by Column",
        )
        fig.update_layout(height=420, xaxis_title="Column", yaxis_title="Missing Values")
        st.plotly_chart(fig, use_container_width=True)

    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)



def render_exploration(df: pd.DataFrame):
    st.subheader("Interactive Exploration")

    if {"country", "salary"}.issubset(df.columns):
        left, right = st.columns(2)
        with left:
            fig = px.box(df, x="country", y="salary", color="country", title="Salary Distribution by Country")
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                df.groupby("job_title", as_index=False)["salary"].mean().sort_values("salary", ascending=False),
                x="job_title",
                y="salary",
                color="salary",
                title="Average Salary by Job Title",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    if {"primary_skill", "skill_demand_score"}.issubset(df.columns):
        fig = px.treemap(
            df.groupby("primary_skill", as_index=False)["skill_demand_score"].mean(),
            path=["primary_skill"],
            values="skill_demand_score",
            color="skill_demand_score",
            title="Skill Demand Landscape",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    if {"year", "ai_risk_score", "job_title"}.issubset(df.columns):
        fig = px.line(
            df.groupby(["year", "job_title"], as_index=False)["ai_risk_score"].mean(),
            x="year",
            y="ai_risk_score",
            color="job_title",
            title="AI Risk Trend by Job Title",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def render_insights(df: pd.DataFrame):
    st.subheader("Business Insights Panel")
    insights = business_insights(df)

    for i, insight in enumerate(insights, start=1):
        st.markdown(
            f"""
            <div style="background: rgba(76, 29, 149, 0.12); border: 1px solid rgba(76, 29, 149, 0.25); padding: 16px; border-radius: 16px; margin-bottom: 12px;">
                <div style="font-size: 15px; font-weight: 700; margin-bottom: 8px;">Insight {i}</div>
                <div style="font-size: 15px;">{insight}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Show filtered dataset"):
        st.dataframe(df, use_container_width=True, height=400)



def main():
    st.title("AI Impact on Jobs and Salary Trends Dashboard")
    st.markdown(
        "A CRISP-DM based interactive dashboard for exploring AI exposure, salary trends, job survival, and model results."
    )

    try:
        raw_df = load_data(CSV_FILE)
        df = enrich_data(raw_df)
    except Exception as e:
        st.error(str(e))
        st.info(
            "Make sure the CSV file is in the same project folder and the filename is exactly: Future of Jobs AI Dataset.csv"
        )
        st.stop()

    countries, job_titles, exp_levels, year_range, risk_categories = sidebar_filters(df)
    filtered_df = build_filtered_data(df, countries, job_titles, exp_levels, year_range, risk_categories)

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Change the filters and try again.")
        st.stop()

        tabs = st.tabs([
        "Overview",
        "Data Quality",
        "EDA",
        "Insights",
    ])
    
    with tabs[0]:
        render_overview(filtered_df)
    with tabs[1]:
        render_data_quality(filtered_df)
    with tabs[2]:
        render_exploration(filtered_df)
    with tabs[3]:
        render_insights(filtered_df)


if __name__ == "__main__":
    main()
