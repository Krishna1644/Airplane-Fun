import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    print("mlxtend not found. Please install it using 'pip install mlxtend'")
    exit(1)

# Configuration
FLIGHT_DATA = 'US_flights_2023.csv'
TIER_DATA = 'airport_performance_tiers_enriched.csv'
WEATHER_DATA = 'weather_meteo_by_airport.csv'
OUTPUT_RULES_FILE = 'association_rules_results.csv'
OUTPUT_PLOT_FILE = 'rules_plot.png'

print("--- 1. Loading & Merging Master Dataset ---")

# 1. Load Main Flight Data
cols_flight = ['FlightDate', 'Airline', 'Dep_Airport', 'Dep_Delay']
try:
    df = pd.read_csv(FLIGHT_DATA, usecols=cols_flight, low_memory=False)
    # SAMPLING to avoid Memory Error
    SAMPLE_SIZE = 500000
    if len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} rows from {len(df)} for Association Mining...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
except FileNotFoundError:
    print(f"Error: {FLIGHT_DATA} not found.")
    exit(1)

# 2. Load Performance Tiers
try:
    df_tiers = pd.read_csv(TIER_DATA)
    df_tiers = df_tiers[['Dep_Airport', 'Performance_Tier']]
except FileNotFoundError:
    print(f"Error: {TIER_DATA} not found. Please run clustering script first.")
    exit(1)

# 3. Load Weather Data
try:
    df_weather = pd.read_csv(WEATHER_DATA)
    cols_weather = ['time', 'airport_id', 'prcp', 'snow', 'wspd', 'tmin']
    df_weather = df_weather[cols_weather]
except FileNotFoundError:
    print(f"Error: {WEATHER_DATA} not found.")
    exit(1)

print("... Merging Datasets ...")

# Merge Tiers
df_merged = df.merge(df_tiers, on='Dep_Airport', how='left')

# Merge Weather (Match Date + Airport)
# Ensure date formats match
df_merged['FlightDate'] = pd.to_datetime(df_merged['FlightDate'])
df_weather['time'] = pd.to_datetime(df_weather['time'])

df_merged = df_merged.merge(
    df_weather,
    left_on=['FlightDate', 'Dep_Airport'],
    right_on=['time', 'airport_id'],
    how='left'
)

# Clean up
df_merged.drop(columns=['time', 'airport_id'], inplace=True)
df_merged.dropna(subset=['Dep_Delay', 'Performance_Tier', 'wspd'], inplace=True)

print(f"Success. Master Dataset created: {len(df_merged):,} flights.")

print("\n--- 2. Creating Risk Factors ---")

# 1. Map Tier Names
tier_mapping = {
    0: 'Tier_0_Secondary',
    1: 'Tier_1_HighRisk',
    2: 'Tier_2_Underperforming',
    3: 'Tier_3_Efficient',
    4: 'Tier_4_MegaHub'
}
df_merged['Tier_Name'] = df_merged['Performance_Tier'].map(tier_mapping)

# 2. Discretize Delays (Target)
bins = [-np.inf, 14, 60, np.inf]
labels = ['OnTime', 'Late', 'Severe']
df_merged['Delay_Class'] = pd.cut(df_merged['Dep_Delay'], bins=bins, labels=labels)

# 3. Create Weather Flags (True/False)
df_merged['Weather_Snow'] = df_merged['snow'] > 0
df_merged['Weather_Rain'] = df_merged['prcp'] > 5
df_merged['Weather_Wind'] = df_merged['wspd'] > 25
df_merged['Weather_Freezing'] = df_merged['tmin'] < 0

# 4. Select Final Columns
mining_df = df_merged[[
    'Airline',
    'Tier_Name',
    'Weather_Snow',
    'Weather_Rain',
    'Weather_Wind',
    'Weather_Freezing',
    'Delay_Class'
]].copy()

print("Feature Engineering Complete.")

print("\n--- 3. Running Apriori (Full Dataset) ---")

# 1. One-Hot Encoding
df_ohe = pd.get_dummies(mining_df, prefix_sep='=').astype(bool)

# 2. Run Apriori Algorithm
print("Mining Frequent Patterns (This may take a few minutes)...")
# Using a slightly higher min_support to ensure speed if dataset is huge, or stick to 0.001
frequent_itemsets = apriori(df_ohe, min_support=0.001, use_colnames=True)

# 3. Generate Rules
print("Generating Association Rules...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print(f"Computation Complete. Found {len(rules)} association rules.")

print("\n--- 4. Analysis: Predictors of Severe Delays ---")

# Filter: Consequent must be SEVERE Delay
# Note: consequents is a frozenset
# We need to filter where 'Delay_Class=Severe' is in consequents
target = 'Delay_Class=Severe'
severe_rules = rules[rules['consequents'].apply(lambda x: target in x)].copy()

# Sort by Lift
severe_rules = severe_rules.sort_values('lift', ascending=False)

# Save results
severe_rules.to_csv(OUTPUT_RULES_FILE, index=False)
print(f"Saved {len(severe_rules)} severe delay rules to {OUTPUT_RULES_FILE}")

# Display Top 15
cols_show = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
print("TOP RISK FACTORS (Apriori Results):")
print(severe_rules[cols_show].head(15))

# Visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="support",
    y="confidence",
    size="lift",
    data=severe_rules,
    hue="lift",
    palette="magma",
    sizes=(20, 200),
    alpha=0.8
)
plt.title('Operational Risk Map (Apriori): What Causes Severe Delays?')
plt.xlabel('Support (Frequency)')
plt.ylabel('Confidence (Likelihood)')
plt.legend(title='Lift')
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_PLOT_FILE)
print(f"Saved plot to {OUTPUT_PLOT_FILE}")
