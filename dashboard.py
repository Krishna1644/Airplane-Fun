import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

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
    import folium
    import streamlit.components.v1 as components
    
    # Map tier to colors matching the screenshot
    tier_colors = {
        4: 'orange', # Mega-Hub
        3: 'green',  # Efficient
        0: 'purple', # Secondary
        2: 'cadetblue', # Underperforming
        1: 'red'     # High Risk
    }
    tier_names = {
        4: 'Mega-Hub (Tier 4)',
        3: 'Efficient (Tier 3)',
        0: 'Secondary (Tier 0)',
        2: 'Underperforming (Tier 2)',
        1: 'High Risk (Tier 1)'
    }
    
    # Create the map centered on the US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="CartoDB positron")
    
    for idx, row in df_clusters.iterrows():
        tier = int(row['Performance_Tier'])
        color = tier_colors.get(tier, 'black')
        
        popup_text = f"<b>{row['AIRPORT']} ({row['Dep_Airport']})</b><br>Tier: {tier_names.get(tier, str(tier))}<br>Avg Delay: {row['avg_dep_delay']:.1f} min"
        
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=6,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(m)
        
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;
     ">
     <b>Performance Tiers</b><br>
     &nbsp; <i class="fa fa-circle" style="color:orange"></i> Mega-Hub (Tier 4)<br>
     &nbsp; <i class="fa fa-circle" style="color:green"></i> Efficient (Tier 3)<br>
     &nbsp; <i class="fa fa-circle" style="color:purple"></i> Secondary (Tier 0)<br>
     &nbsp; <i class="fa fa-circle" style="color:cadetblue"></i> Underperforming (Tier 2)<br>
     &nbsp; <i class="fa fa-circle" style="color:red"></i> High Risk (Tier 1)
      </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    map_html = m._repr_html_()
    components.html(map_html, height=800)
    
    with st.expander("View Raw Clustering Data"):
        st.dataframe(df_clusters, use_container_width=True)
else:
    st.info("Clustering data not found. Run Cluster.py first.")

# --- 2. Association Rules ---
st.header("2. Association Rules for Severe Delays")
st.markdown("Top association rules discovered using the Apriori algorithm.")

df_rules = load_csv("all_association_rules_full.csv")
if df_rules is None and os.path.exists("association_rules_results.csv"):
    df_rules = load_csv("association_rules_results.csv")
    
if df_rules is not None:
    # Clean up the ugly frozenset formatting
    def clean_rule(text):
        if pd.isna(text): return ""
        return str(text).replace("frozenset({", "").replace("})", "").replace("'", "")
        
    df_rules['antecedents'] = df_rules['antecedents'].apply(clean_rule)
    df_rules['consequents'] = df_rules['consequents'].apply(clean_rule)
    df_rules['Rule'] = df_rules['antecedents'] + "  ➔  " + df_rules['consequents']

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Top 15 Rules by Lift")
        top_rules = df_rules.nlargest(15, 'lift')
        fig_bar = px.bar(
            top_rules,
            x="lift",
            y="Rule",
            orientation="h",
            color="confidence",
            hover_data=["support"],
            labels={"lift": "Lift", "confidence": "Confidence", "Rule": ""},
            color_continuous_scale="Sunsetdark"
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("### Rules Distribution")
        fig_scatter = px.scatter(
            df_rules,
            x="support",
            y="confidence",
            color="lift",
            hover_data=["Rule"],
            labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with st.expander("View Cleaned Rules Dataset"):
        st.dataframe(
            df_rules[["Rule", "antecedent support", "consequent support", "support", "confidence", "lift"]], 
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Rules data not found. Run Rules.py first.")

# --- 3. Feature Importance & Classification ---
st.header("3. Classification Feature Importance")
st.markdown("Importance of features for predicting severe delays.")
df_feat = load_csv("feature_importance.csv")
if df_feat is not None:
    top_features = df_feat.head(15).sort_values(by="Importance", ascending=True)
    fig_feat = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Most Important Features",
        color="Importance",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.info("Feature importance data not found. Run Classify.py first.")
    
# --- 4. Regression Analysis ---
st.header("4. Arrival Delay Regression Models")
st.markdown("Performance of regression models aimed at predicting exact delay minutes.")
df_reg = load_csv("regression_metrics.csv")
if df_reg is not None:
    # Display top model metrics as KPIs
    best_model = df_reg.loc[df_reg['RMSE'].idxmin()]
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(label="Best Model", value=best_model['Model'])
    with col_kpi2:
        st.metric(label="Best RMSE", value=f"{best_model['RMSE']:.2f} mins")
    with col_kpi3:
        st.metric(label="Best R²", value=f"{best_model['R2']:.3f}")
        
    st.markdown("### Model Comparison")
    
    # Optional bar chart for RMSE comparison
    fig_reg = px.bar(
        df_reg,
        x="Model",
        y="RMSE",
        title="Root Mean Squared Error (RMSE) by Model - Lower is Better",
        color="Model"
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    
    with st.expander("View Raw Metrics Data"):
        st.dataframe(df_reg, use_container_width=True, hide_index=True)
else:
    st.info("Regression metrics not found. Run Reg.py first.")
