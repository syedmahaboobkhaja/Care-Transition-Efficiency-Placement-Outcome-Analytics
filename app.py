# ============================================================
# Care Transition Efficiency & Placement Outcome Analytics
# Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PART 1: PAGE SETUP + IMPORTS
# ============================================================

st.set_page_config(
    page_title="Care Transition Analytics",
    page_icon="🏥",
    layout="wide"
)


st.title("🏥 Care Transition Efficiency & Placement Outcome Analytics")

st.markdown(
    """
    This dashboard analyzes care transition flow,
    placement efficiency, backlog patterns,
    and operational insights.
    """
)



# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "HHS_Unaccompanied_Alien_Children_Program.csv"
    )

    return df



try:

    df = load_data()

except:

    st.error(
        "Dataset not found. Upload CSV file in repository."
    )

    st.stop()



# ============================================================
# DATA CLEANING
# ============================================================


df_clean = df.copy()


# Remove duplicates

df_clean.drop_duplicates(inplace=True)



# Fill missing values

for col in df_clean.select_dtypes(include="object"):

    df_clean[col] = df_clean[col].fillna(
        df_clean[col].mode()[0]
    )


for col in df_clean.select_dtypes(include=np.number):

    df_clean[col] = df_clean[col].fillna(
        df_clean[col].median()
    )



# ============================================================
# SIDEBAR FILTERS
# ============================================================


st.sidebar.header("Dashboard Filters")


selected_columns = st.sidebar.multiselect(
    "Select Columns",
    df_clean.columns,
    default=list(df_clean.columns[:5])
)



# ============================================================
# KPI DASHBOARD
# ============================================================


st.subheader("📊 Key Performance Indicators")


col1,col2,col3,col4 = st.columns(4)


with col1:

    st.metric(
        "Total Records",
        len(df_clean)
    )


with col2:

    st.metric(
        "Total Features",
        df_clean.shape[1]
    )


with col3:

    missing = df.isnull().sum().sum()

    st.metric(
        "Missing Values",
        missing
    )


with col4:

    duplicates = df.duplicated().sum()

    st.metric(
        "Duplicate Records",
        duplicates
    )



# ============================================================
# PART 2: EDA CHARTS
# ============================================================


st.header("📈 Exploratory Data Analysis")



# Numerical Distribution


numeric_columns = df_clean.select_dtypes(
    include=np.number
).columns



if len(numeric_columns)>0:


    selected_num = st.selectbox(
        "Select Numerical Feature",
        numeric_columns
    )


    fig = px.histogram(
        df_clean,
        x=selected_num,
        title=f"Distribution of {selected_num}"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )




# ============================================================
# CORRELATION HEATMAP
# ============================================================


st.subheader("🔥 Correlation Heatmap")


if len(numeric_columns)>1:


    corr = df_clean[
        numeric_columns
    ].corr()


    fig, ax = plt.subplots(
        figsize=(10,6)
    )


    sns.heatmap(
        corr,
        annot=True,
        ax=ax
    )


    st.pyplot(fig)



# ============================================================
# MONTHLY & WEEKLY ANALYSIS
# ============================================================


st.header("📅 Time Based Analysis")



date_columns = []


for col in df_clean.columns:

    if "date" in col.lower():

        date_columns.append(col)



if len(date_columns)>0:


    date_col = st.selectbox(
        "Select Date Column",
        date_columns
    )


    df_clean[date_col] = pd.to_datetime(
        df_clean[date_col],
        errors="coerce"
    )


    monthly = (
        df_clean
        .groupby(
            df_clean[date_col].dt.month
        )
        .size()
        .reset_index(name="Count")
    )


    fig = px.line(
        monthly,
        x=date_col,
        y="Count",
        title="Monthly Trend"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "No date column detected for monthly analysis."
    )



# ============================================================
# PART 3: TRANSFER EFFICIENCY
# ============================================================


st.header("🚚 Transfer Efficiency Analysis")



if len(numeric_columns)>0:


    efficiency_feature = st.selectbox(
        "Select Efficiency Metric",
        numeric_columns,
        key="eff"
    )


    avg_value = df_clean[
        efficiency_feature
    ].mean()


    st.metric(
        "Average Transfer Metric",
        round(avg_value,2)
    )


    fig = px.box(
        df_clean,
        y=efficiency_feature,
        title="Transfer Efficiency Distribution"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# ============================================================
# BACKLOG ANALYSIS
# ============================================================


st.header("⏳ Backlog Analysis")


if len(numeric_columns)>0:


    backlog_feature = st.selectbox(
        "Select Backlog Feature",
        numeric_columns,
        key="backlog"
    )


    backlog = df_clean[
        backlog_feature
    ].sum()


    st.metric(
        "Total Backlog Value",
        round(backlog,2)
    )



# ============================================================
# INTERACTIVE PLOTLY CHART
# ============================================================


st.header("📊 Interactive Analytics")


if len(selected_columns)>=2:


    fig = px.scatter(
        df_clean,
        x=selected_columns[0],
        y=selected_columns[1],
        title="Interactive Relationship Analysis"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# ============================================================
# PROJECT INSIGHTS
# ============================================================


st.header("💡 Project Insights")


st.success(
"""
• Identifies care transition patterns.

• Highlights operational delays and backlog areas.

• Supports data-driven placement decisions.

• Improves visibility of transition efficiency.

• Helps optimize resource allocation.
"""
)



# ============================================================
# PART 4: RECOMMENDATIONS
# ============================================================


st.header("✅ Recommendations")


recommendations = [

"Monitor high delay transition stages regularly",

"Improve resource allocation for high backlog areas",

"Use predictive analytics for placement planning",

"Automate reporting using dashboards",

"Track monthly transition performance"

]


for item in recommendations:

    st.write(
        "✔️",
        item
    )



# ============================================================
# FOOTER
# ============================================================


st.divider()


st.caption(
"Care Transition Efficiency & Placement Outcome Analytics | Streamlit Dashboard"
)
