import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Flight Risk Analysis", layout="wide")

st.title("US Flight Operational Risk Analysis (2023)")
st.markdown("""
This dashboard presents the results of the Machine Learning Term Project (CS5805).
It covers Airport Clustering, Association Rules for Delays, Classification, and Regression models.
""")

# --- Helper to load data safely ---
@st.cache_data
def load_csv(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

# --- 1. Clustering Results ---
st.header("1. Airport Performance Tiers (Clustering)")
st.markdown("Airports clustered into 5 tiers based on metrics like delay frequency and weather impact.")
df_clusters = load_csv("airport_performance_tiers_enriched.csv")
if df_clusters is not None:
    st.dataframe(df_clusters)
else:
    st.info("Clustering data not found. Run Cluster.py first.")

# --- 2. Association Rules ---
st.header("2. Association Rules for Severe Delays")
st.markdown("Frequent itemsets and association rules discovered using Apriori algorithm.")

col1, col2 = st.columns([1, 1])

with col1:
    df_rules = load_csv("all_association_rules_full.csv")
    if df_rules is not None:
        st.dataframe(df_rules)
    elif os.path.exists("association_rules_results.csv"):
        st.dataframe(load_csv("association_rules_results.csv"))
    else:
        st.info("Rules data not found. Run Rules.py first.")

with col2:
    if os.path.exists("rules_plot.png"):
        st.image("rules_plot.png", caption="Association Rules Scatter Plot", use_container_width=True)
    else:
        st.info("Rules plot not found.")

# --- 3. Feature Importance & Classification ---
st.header("3. Classification Feature Importance")
st.markdown("Importance of features for predicting severe delays.")
df_feat = load_csv("feature_importance.csv")
if df_feat is not None:
    # Get top 15 features for better visualization
    st.bar_chart(df_feat.head(15).set_index("Feature")["Importance"])
else:
    st.info("Feature importance data not found. Run Classify.py first.")
    
# --- 4. Regression Analysis ---
st.header("4. Arrival Delay Regression Models")
st.markdown("Performance of regression models aimed at predicting exact delay minutes.")
df_reg = load_csv("regression_metrics.csv")
if df_reg is not None:
    st.dataframe(df_reg)
else:
    st.info("Regression metrics not found. Run Reg.py first.")
