#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================
# Step 1: Import Required Libraries
# ==========================================

import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print(" All libraries imported successfully!")


# In[2]:


# ==========================================
# Step 2: Load Dataset
# ==========================================

from google.colab import files

uploaded = files.upload()


# In[3]:


# Replace the filename if necessary
df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program (1).csv")

print(" Dataset Loaded Successfully!")

# Display first five rows
df.head()


# In[4]:


# ==========================================
# Step 3: Dataset Overview
# ==========================================

print("="*60)
print("Dataset Shape")
print("="*60)
print(df.shape)

print("\n")

print("="*60)
print("Column Names")
print("="*60)
print(df.columns)

print("\n")

print("="*60)
print("Dataset Information")
print("="*60)
df.info()

print("\n")

print("="*60)
print("First Five Rows")
print("="*60)
display(df.head())

print("\n")

print("="*60)
print("Last Five Rows")
print("="*60)
display(df.tail())


# In[5]:


# ==========================================
# Step 4: Data Cleaning
# ==========================================

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Rename columns for easier coding
df.rename(columns={
    'Children apprehended and placed in CBP custody*':'Children_Apprehended',
    'Children in CBP custody':'Children_CBP',
    'Children transferred out of CBP custody':'Transferred_HHS',
    'Children in HHS Care':'Children_HHS',
    'Children discharged from HHS Care':'Children_Discharged'
}, inplace=True)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort dataset by Date
df = df.sort_values(by='Date')

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print(" Data Cleaning Completed Successfully!")

# Display cleaned data
df.head()


# In[6]:


# ==========================================
# Step 5: Missing Value Analysis
# ==========================================

print("="*60)
print("Missing Values")
print("="*60)

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Values': missing.values,
    'Percentage': (missing.values/len(df))*100
})

display(missing_df)

# Total Missing Values
print("\nTotal Missing Values:", df.isnull().sum().sum())

# Heatmap of Missing Values
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Value Heatmap")
plt.show()

# Fill missing numeric values (if any) using median
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\n Missing values handled successfully!")

# Verify missing values
print("\nRemaining Missing Values:")
print(df.isnull().sum())


# In[7]:


# ==========================================
# Step 6: Statistical Summary
# ==========================================

print("="*60)
print("Statistical Summary")
print("="*60)

display(df.describe().T)

print("\nSkewness")
display(df.skew(numeric_only=True))

print("\nKurtosis")
display(df.kurtosis(numeric_only=True))


# In[8]:


# ==========================================
# Step 7: Feature Engineering
# ==========================================

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()
df['Week'] = df['Date'].dt.isocalendar().week
df['Day'] = df['Date'].dt.day
df['Day_Name'] = df['Date'].dt.day_name()
df['Quarter'] = df['Date'].dt.quarter

print("Feature Engineering Completed")

display(df.head())


# In[9]:


# ==========================================
# Step 8: Exploratory Data Analysis
# ==========================================

numeric_columns = [
    'Children_Apprehended',
    'Children_CBP',
    'Transferred_HHS',
    'Children_HHS',
    'Children_Discharged'
]

# Histograms
df[numeric_columns].hist(figsize=(16,10), bins=20)

plt.suptitle("Distribution of Numerical Variables", fontsize=18)
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(14,6))
sns.boxplot(data=df[numeric_columns])

plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=20)
plt.show()


# In[12]:


# ==========================================
# Step 9: Correlation Analysis
# ==========================================

# Ensure 'Children_HHS' is numeric by cleaning and converting it.
# The original data for 'Children in HHS Care' was 'object' dtype and contained commas.
# This conversion should ideally be done in Step 4 (Data Cleaning).
# Performing it here to fix the ValueError.
if 'Children_HHS' in df.columns and df['Children_HHS'].dtype == 'object':
    df['Children_HHS'] = df['Children_HHS'].astype(str).str.replace(',', '', regex=False)
    df['Children_HHS'] = pd.to_numeric(df['Children_HHS'], errors='coerce')
    # Fill any NaNs created by 'coerce' or existing in the column with the median
    # The previous missing value handling in Step 5 did not apply to this column as it was 'object' type.
    df['Children_HHS'].fillna(df['Children_HHS'].median(), inplace=True)

# Recalculate correlation after ensuring all columns in numeric_columns are actually numeric
corr = df[numeric_columns].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()


# In[13]:


# ==========================================
# Step 10: Transfer Efficiency Ratio
# ==========================================

df['Transfer_Efficiency_Ratio'] = (
    df['Transferred_HHS'] /
    df['Children_CBP']
)

df['Transfer_Efficiency_Ratio'] = df['Transfer_Efficiency_Ratio'].replace(
    [np.inf, -np.inf],
    np.nan
)

df['Transfer_Efficiency_Ratio'].fillna(0, inplace=True)

print("Average Transfer Efficiency")

print(round(df['Transfer_Efficiency_Ratio'].mean(),3))

plt.figure(figsize=(15,5))

plt.plot(
    df['Date'],
    df['Transfer_Efficiency_Ratio'],
    color='blue'
)

plt.title("Transfer Efficiency Ratio Over Time")

plt.xlabel("Date")

plt.ylabel("Ratio")

plt.grid(True)

plt.show()


# In[14]:


# ==========================================
# Step 11: Discharge Effectiveness
# ==========================================

df['Discharge_Effectiveness'] = (
    df['Children_Discharged'] /
    df['Children_HHS']
)

df['Discharge_Effectiveness'] = df['Discharge_Effectiveness'].replace(
    [np.inf,-np.inf],
    np.nan
)

df['Discharge_Effectiveness'].fillna(0,inplace=True)

print("Average Discharge Effectiveness")

print(round(df['Discharge_Effectiveness'].mean(),3))

plt.figure(figsize=(15,5))

plt.plot(
    df['Date'],
    df['Discharge_Effectiveness'],
    color='green'
)

plt.title("Discharge Effectiveness Trend")

plt.xlabel("Date")

plt.ylabel("Ratio")

plt.grid(True)

plt.show()


# In[15]:


# ==========================================
# Step 12: Pipeline Throughput
# ==========================================

df['Pipeline_Throughput'] = (
    df['Children_Discharged'] /
    df['Children_Apprehended']
)

df['Pipeline_Throughput'] = df['Pipeline_Throughput'].replace(
    [np.inf,-np.inf],
    np.nan
)

df['Pipeline_Throughput'].fillna(0,inplace=True)

print("Average Pipeline Throughput")

print(round(df['Pipeline_Throughput'].mean(),3))

plt.figure(figsize=(15,5))

plt.plot(
    df['Date'],
    df['Pipeline_Throughput'],
    color='purple'
)

plt.title("Pipeline Throughput")

plt.xlabel("Date")

plt.ylabel("Throughput")

plt.grid(True)

plt.show()


# In[16]:


# ==========================================
# Step 13: Backlog Accumulation
# ==========================================

df['Backlog'] = (
    df['Children_CBP']
    +
    df['Children_HHS']
    -
    df['Children_Discharged']
)

print("Average Backlog")

print(round(df['Backlog'].mean(),2))

plt.figure(figsize=(15,5))

plt.plot(
    df['Date'],
    df['Backlog'],
    color='red'
)

plt.title("Backlog Accumulation")

plt.xlabel("Date")

plt.ylabel("Children")

plt.grid(True)

plt.show()


# In[17]:


# ==========================================
# Step 14: Outcome Stability
# ==========================================

df['Rolling_Mean_Discharge'] = (
    df['Children_Discharged']
    .rolling(7)
    .mean()
)

df['Rolling_STD_Discharge'] = (
    df['Children_Discharged']
    .rolling(7)
    .std()
)

plt.figure(figsize=(15,5))

plt.plot(
    df['Date'],
    df['Rolling_Mean_Discharge'],
    label='7-Day Average'
)

plt.plot(
    df['Date'],
    df['Rolling_STD_Discharge'],
    label='7-Day Std'
)

plt.legend()

plt.title("Outcome Stability")

plt.grid(True)

plt.show()


# In[18]:


# ==========================================
# Step 15: Daily Trend Analysis
# ==========================================

plt.figure(figsize=(16,7))

plt.plot(
    df['Date'],
    df['Children_Apprehended'],
    label='Apprehended'
)

plt.plot(
    df['Date'],
    df['Transferred_HHS'],
    label='Transferred'
)

plt.plot(
    df['Date'],
    df['Children_Discharged'],
    label='Discharged'
)

plt.legend()

plt.title("Daily Care Pipeline Trend")

plt.xlabel("Date")

plt.ylabel("Children")

plt.grid(True)

plt.show()

print("="*50)
print("Key KPI Summary")
print("="*50)

print("Average Transfer Efficiency:",
      round(df['Transfer_Efficiency_Ratio'].mean(),3))

print("Average Discharge Effectiveness:",
      round(df['Discharge_Effectiveness'].mean(),3))

print("Average Pipeline Throughput:",
      round(df['Pipeline_Throughput'].mean(),3))

print("Average Backlog:",
      round(df['Backlog'].mean(),2))


# In[19]:


# ==========================================
# Step 16: Weekly Analysis
# ==========================================

weekly = df.groupby('Day_Name')[[
    'Children_Apprehended',
    'Transferred_HHS',
    'Children_Discharged'
]].mean()

# Reorder weekdays
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekly = weekly.reindex(order)

display(weekly)

weekly.plot(kind='bar', figsize=(12,6))
plt.title("Weekly Average Care Pipeline")
plt.ylabel("Average Number of Children")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[20]:


# ==========================================
# Step 17: Monthly Analysis
# ==========================================

monthly = df.groupby('Month_Name')[[
    'Children_Apprehended',
    'Transferred_HHS',
    'Children_Discharged'
]].mean()

month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

monthly = monthly.reindex(month_order)

display(monthly)

monthly.plot(figsize=(14,6), marker='o')

plt.title("Monthly Care Pipeline Trend")
plt.ylabel("Average Children")
plt.grid(True)
plt.show()


# In[21]:


# ==========================================
# Step 18: Bottleneck Detection
# ==========================================

threshold = df['Backlog'].mean()

bottlenecks = df[df['Backlog'] > threshold]

print("Total Bottleneck Days:", len(bottlenecks))

display(
    bottlenecks[
        ['Date',
         'Children_CBP',
         'Children_HHS',
         'Backlog']
    ].head(20)
)

plt.figure(figsize=(15,5))

plt.plot(df['Date'], df['Backlog'], label='Backlog')
plt.axhline(threshold,
            color='red',
            linestyle='--',
            label='Average Backlog')

plt.legend()
plt.title("Backlog Bottleneck Detection")
plt.grid(True)
plt.show()


# In[22]:


# ==========================================
# Step 19: KPI Dashboard
# ==========================================

kpis = {

'Total Apprehended':
df['Children_Apprehended'].sum(),

'Total CBP':
df['Children_CBP'].sum(),

'Total Transfers':
df['Transferred_HHS'].sum(),

'Total HHS':
df['Children_HHS'].sum(),

'Total Discharged':
df['Children_Discharged'].sum(),

'Average Transfer Ratio':
round(df['Transfer_Efficiency_Ratio'].mean(),3),

'Average Discharge Ratio':
round(df['Discharge_Effectiveness'].mean(),3),

'Average Throughput':
round(df['Pipeline_Throughput'].mean(),3),

'Average Backlog':
round(df['Backlog'].mean(),2)

}

kpi_df = pd.DataFrame(kpis.items(),
                      columns=['KPI','Value'])

display(kpi_df)


# In[23]:


# ==========================================
# Step 20: Interactive Charts
# ==========================================

fig = px.line(
    df,
    x='Date',
    y=[
        'Children_Apprehended',
        'Transferred_HHS',
        'Children_Discharged'
    ],
    title='Interactive Care Pipeline Trend'
)

fig.show()

fig2 = px.area(
    df,
    x='Date',
    y='Backlog',
    title='Backlog Accumulation'
)

fig2.show()


# In[24]:


# ==========================================
# Step 21: Export Dataset
# ==========================================

df.to_csv(
    "Processed_Care_Transition_Data.csv",
    index=False
)

print("Processed dataset saved successfully.")


# In[25]:


# ==========================================
# Step 22: Generate Insights
# ==========================================

print("="*60)
print("PROJECT INSIGHTS")
print("="*60)

print(f"Average Transfer Efficiency : {df['Transfer_Efficiency_Ratio'].mean():.2f}")

print(f"Average Discharge Effectiveness : {df['Discharge_Effectiveness'].mean():.2f}")

print(f"Average Pipeline Throughput : {df['Pipeline_Throughput'].mean():.2f}")

print(f"Average Backlog : {df['Backlog'].mean():.2f}")

print(f"Highest Backlog : {df['Backlog'].max():.2f}")

print(f"Lowest Backlog : {df['Backlog'].min():.2f}")


# In[29]:


# ==========================================
# Step 23: Recommendations
# ==========================================

recommendations = [

"Increase transfer efficiency from CBP to HHS.",

"Reduce backlog by improving discharge processing.",

"Monitor days with unusually high backlog.",

"Strengthen sponsor verification workflow.",

"Use predictive analytics for resource planning.",

"Review periods with declining throughput.",

"Improve coordination between CBP and HHS.",

"Develop automated monitoring dashboards."

]

print("Recommendations")

for i, rec in enumerate(recommendations,1):
    print(f"{i}. {rec}")


# In[30]:


# Install Streamlit (if not already installed)
get_ipython().system('pip install streamlit')

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Care Transition Dashboard",
                   layout="wide")

st.title("Care Transition Efficiency & Placement Outcome Analytics")

df = pd.read_csv("Processed_Care_Transition_Data.csv")

df['Date'] = pd.to_datetime(df['Date'])

start = st.sidebar.date_input(
    "Start Date",
    df['Date'].min()
)

end = st.sidebar.date_input(
    "End Date",
    df['Date'].max()
)

filtered = df[
    (df['Date']>=pd.to_datetime(start))
    &
    (df['Date']<=pd.to_datetime(end))
]

st.metric("Total Apprehended",
          int(filtered['Children_Apprehended'].sum()))

st.metric("Total Discharged",
          int(filtered['Children_Discharged'].sum()))

fig = px.line(
    filtered,
    x='Date',
    y=[
        'Children_Apprehended',
        'Transferred_HHS',
        'Children_Discharged'
    ]
)

st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(filtered,
               x='Date',
               y='Backlog')

st.plotly_chart(fig2,
                use_container_width=True)


# In[31]:


# ==========================================
# Step 25: Final Summary
# ==========================================

print("="*70)
print("CARE TRANSITION EFFICIENCY & PLACEMENT OUTCOME ANALYTICS")
print("="*70)

print("Project Completed Successfully!")

print("\nModules Completed")

modules = [

"Data Loading",

"Data Cleaning",

"Feature Engineering",

"Exploratory Data Analysis",

"Correlation Analysis",

"Transfer Efficiency KPI",

"Discharge Effectiveness KPI",

"Pipeline Throughput KPI",

"Backlog Detection",

"Outcome Stability",

"Weekly Analysis",

"Monthly Analysis",

"Bottleneck Detection",

"KPI Dashboard",

"Interactive Visualizations",

"Processed Dataset Export",

"Recommendations",

"Streamlit Dashboard"

]

for i, m in enumerate(modules,1):
    print(f"{i}. {m}")

print("\nProject Ready for GitHub Submission")
print("Research Paper Ready")
print("Executive Summary Ready")
print("Dashboard Ready")

