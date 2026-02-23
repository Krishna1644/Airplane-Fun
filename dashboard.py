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

st.info("""
### 💡 Key Project Conclusions & Takeaways
Based on our Machine Learning pipelines, we discovered three actionable insights regarding nationwide flight delays:
1. **Not All Airports Fail Equally (Clustering):** Massive flight volume does not guarantee delays (e.g., ATL is a hyper-efficient 'Mega-Hub'), while certain smaller regional airports consistently fail under pressure ('High Risk'). Geographic topology and facility design are just as important as size.
2. **Cascading Failure Conditions (Apriori Rules):** Severe delays are rarely caused by a single isolated variable. Our rules engine proved that specific combinations—like *Alaska Airlines encountering Freezing Weather*—create compounding risk multipliers (Lift > 2.0) that predictably cascade into system-wide delays.
3. **Predictable Operational Risk (Classification/Regression):** A flight's delay is not random; it is highly deterministic. Our Random Forest classification models successfully proved that simply knowing the Departure Airport, Time of Day, and Carrier allows us to accurately estimate the tangible delay risk *before* the plane ever boards.
""")

# --- Helper to load data safely ---
@st.cache_data
def load_csv(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

# --- 1. Clustering Results ---
st.header("1. Airport Categories (Clustering)")
st.markdown("""
**What this shows:** We used K-Means clustering to analyze historical flight data and group US airports into 5 distinct categories based on their flight volumes, delay frequencies, and volatility. 
- **Mega-Hubs (Orange):** Massive airports with extremely high flight volumes (e.g., ATL, ORD, DFW).
- **Efficient Regional (Green):** Airports with excellent performance and low average delays.
- **Secondary Hubs (Purple):** Medium-to-large transit airports with average operational performance.
- **Underperforming (Blue):** Airports experiencing worse-than-average delays relative to their size.
- **High Risk / Chaotic (Red):** Airports that historically suffer from the most severe, systemic delays and unpredictability.
**Why it matters:** This helps airlines and passengers immediately visualize the reliability of different airports across the country, shifting focus from raw numbers to actionable operational archetypes.
""")
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
        4: 'Mega-Hub',
        3: 'Efficient Regional',
        0: 'Secondary Hub',
        2: 'Underperforming',
        1: 'High Risk / Chaotic'
    }
    
    # Create the map centered on the US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="CartoDB positron")
    
    for idx, row in df_clusters.iterrows():
        tier = int(row['Performance_Tier'])
        color = tier_colors.get(tier, 'black')
        
        popup_text = f"<b>{row['AIRPORT']} ({row['Dep_Airport']})</b><br>Category: {tier_names.get(tier, str(tier))}<br>Total Flights: {row['total_flights']:,}<br>Avg Delay: {row['avg_dep_delay']:.1f} min"
        
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
     bottom: 50px; left: 50px; width: 220px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;
     ">
     <b>Airport Categories</b><br>
     &nbsp; <i class="fa fa-circle" style="color:orange"></i> Mega-Hub<br>
     &nbsp; <i class="fa fa-circle" style="color:green"></i> Efficient Regional<br>
     &nbsp; <i class="fa fa-circle" style="color:purple"></i> Secondary Hub<br>
     &nbsp; <i class="fa fa-circle" style="color:cadetblue"></i> Underperforming<br>
     &nbsp; <i class="fa fa-circle" style="color:red"></i> High Risk / Chaotic
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
st.markdown("""
**What this shows:** We used the Apriori algorithm to discover common "recipes" or conditions that lead to severe flight delays. 
- **Rule (A ➔ B):** If condition A happens, then B is likely to happen.
- **Lift:** A multiplier showing how much more likely B is to happen when A occurs, compared to normal. A Lift > 1 means a strong relationship.
**Why it matters:** It shifts our perspective from looking at single variables to understanding how combinations (like *Snowy Weather + a specific Airline*) trigger cascading delays.
""")

df_rules = load_csv("all_association_rules_full.csv")
if df_rules is None and os.path.exists("association_rules_results.csv"):
    df_rules = load_csv("association_rules_results.csv")
    
if df_rules is not None:
    # Clean up the ugly frozenset formatting
    def clean_rule(text):
        if pd.isna(text): return ""
        return str(text).replace("frozenset({", "").replace("})", "").replace("'", "")
        
    df_rules['antecedents_clean'] = df_rules['antecedents'].apply(clean_rule)
    df_rules['consequents_clean'] = df_rules['consequents'].apply(clean_rule)
    
    # Remove A -> B and B -> A duplicates by creating a sorted set string
    def create_itemset_key(row):
        # Merge the lists of items from antecedents and consequents
        items = row['antecedents_clean'].split(', ') + row['consequents_clean'].split(', ')
        # Sort them and join to create a unique identifier for the complete itemset
        return " | ".join(sorted([i.strip() for i in items if i.strip()]))
        
    df_rules['itemset_key'] = df_rules.apply(create_itemset_key, axis=1)
    
    # Sort by lift so we keep the rule direction with the highest lift when dropping duplicates
    df_rules = df_rules.sort_values(by='lift', ascending=False)
    df_rules = df_rules.drop_duplicates(subset=['itemset_key'], keep='first')
        
    df_rules['Rule'] = df_rules['antecedents_clean'] + "  ➔  " + df_rules['consequents_clean']

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Top 10 Unique Rules by Lift")
        top_rules = df_rules.head(10) # Grab the top 10 since we already sorted by lift
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
st.markdown("""
**What this shows:** We trained Machine Learning classification models (like Random Forests) to predict whether a specific flight will be delayed. This chart shows which variables the AI found most critical when making its predictions.
**Why it matters:** It proves that certain factors (like the specific Departure Airport or the Departure Time) play a disproportionately massive role in whether your flight will be on time.
""")
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
st.markdown("""
**What this shows:** While classification predicts *if* a flight will be delayed, Regression models attempt to predict *exactly how many minutes* the flight will be delayed. 
- **RMSE (Root Mean Squared Error):** Indicates the model's average error in minutes. A lower RMSE is better.
- **R² (R-squared):** Indicates how much of the delay variance the model successfully predicted. 
**Why it matters:** It demonstrates the project's capability to assign a tangible, numerical operational risk to any individual flight based on current conditions.
""")
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
