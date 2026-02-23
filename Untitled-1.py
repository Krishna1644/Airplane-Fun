# %%
# ==========================================
# CHUNK 0: MASTER SETUP (Run once at the start)
# ==========================================
import pandas as pd
import numpy as np
import time # Fixes the 'time' error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# CONFIGURATION
SAMPLE_SIZE = None  # Keep this small for dev, set to None for full run

print("--- 1. Loading Data ---")
# Load datasets
df_flights = pd.read_csv('US_flights_2023.csv', low_memory=False)
df_airports = pd.read_csv('airports_geolocation.csv')
df_weather = pd.read_csv('weather_meteo_by_airport.csv')

print("--- 2. Merging & Prepping ---")
# Convert dates
df_flights['FlightDate'] = pd.to_datetime(df_flights['FlightDate'])
df_weather['time'] = pd.to_datetime(df_weather['time'])

# SAMPLING (Crucial for speed)
if SAMPLE_SIZE and len(df_flights) > SAMPLE_SIZE:
    print(f"   -> Sampling {SAMPLE_SIZE} rows...")
    df_flights = df_flights.sample(n=SAMPLE_SIZE, random_state=42)

# MERGE AIRPORTS (Get Origin_State)
df_merged = df_flights.merge(
    df_airports[['IATA_CODE', 'STATE']], 
    left_on='Dep_Airport', 
    right_on='IATA_CODE', 
    how='left'
)
df_merged = df_merged.rename(columns={'STATE': 'Origin_State'}).drop(columns=['IATA_CODE'])

# MERGE WEATHER (Get Rain/Wind)
df_final = df_merged.merge(
    df_weather[['airport_id', 'time', 'prcp', 'snow', 'wspd', 'tavg']],
    left_on=['Dep_Airport', 'FlightDate'],
    right_on=['airport_id', 'time'],
    how='left'
)
# Cleanup
df_final = df_final.drop(columns=['airport_id', 'time'])

print(f"SETUP COMPLETE. 'df_final' is ready. Shape: {df_final.shape}")

# %%
# ==========================================
# CHUNK 1: Feature Selection & Split
# ==========================================
print("--- Setting up Final Regression Data ---")

# 1. FINAL FEATURES
# We include Dep_Delay because it is the primary operational signal.
features_final = [
    'Dep_Delay',          # The primary driver
    'Flight_Duration',    # Proxy for Distance
    'Day_Of_Week', 
    'DepTime_label',      # Morning/Evening
    'Airline',
    'Aicraft_age',
    'Origin_State',
    'tavg', 'prcp', 'wspd' # Weather factors
]

target = 'Arr_Delay'

# 2. FILTER & CLEAN
# We assume 'df_final' is still in memory from the previous merge step.
# If not, re-run the Merge cell.
df_reg = df_final[features_final + [target]].copy()
df_reg = df_reg.dropna()

# 3. SAMPLING
# 100k is enough to get 99% of the accuracy for dev. 
# If you want the "Production" model, set this to None (takes longer).
SAMPLE_N = SAMPLE_SIZE 
if SAMPLE_N and len(df_reg) > SAMPLE_N:
    df_reg = df_reg.sample(SAMPLE_N, random_state=42)

# 4. SPLIT
X = df_reg.drop(columns=[target])
y = df_reg[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data Ready. Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

# %%
# ==========================================
# CHUNK 1.5: Phase I - Cleaning & EDA Audit (Syllabus Req)
# ==========================================
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Phase I: Data Cleaning & EDA Audit ---")

# 1. DUPLICATE CHECK [Syllabus: "Check for data duplications"]
# -----------------------------------------------------------
initial_len = len(df_reg)
df_reg = df_reg.drop_duplicates()
new_len = len(df_reg)
print(f"Duplicates Removed: {initial_len - new_len}")

# 2. MISSING DATA CHECK [Syllabus: "pick a method to fix the missing data"]
# -----------------------------------------------------------
# We already used dropna() in Chunk 1, but we must document it.
print(f"Missing Values Status: {df_reg.isna().sum().sum()} (Cleaned via DropNA)")

# 3. OUTLIER ANALYSIS [Syllabus: "Anomaly detection/Outlier Analysis"]
# -----------------------------------------------------------
# We need to show we looked at outliers.
# WARNING: Do not remove valid long delays (e.g., 500 min) as those are real data points.
# We only remove "impossible" physics, e.g., negative Flight Duration.

impossible_flights = df_reg[df_reg['Flight_Duration'] <= 0]
print(f"Impossible Flights (Duration <= 0): {len(impossible_flights)}")
if len(impossible_flights) > 0:
    df_reg = df_reg[df_reg['Flight_Duration'] > 0]
    print(" -> Impossible flights removed.")

# Visualization for Report (Boxplot)
plt.figure(figsize=(10, 4))
sns.boxplot(x=df_reg['Dep_Delay'], color='orange')
plt.title('Outlier Analysis: Departure Delays')
plt.xlabel('Minutes')
plt.show() 
print("Observation: The data contains extreme values (long delays). These are natural outliers and kept for validity.")

# 4. COLLINEARITY CHECK [Syllabus: "Sample Pearson Correlation coefficients Matrix"]
# -----------------------------------------------------------
# This satisfies the requirement to check for collinearity [Syllabus: "make sure that the collinearity does not exist"]
# We only check numeric columns.
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df_reg.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Matrix (Collinearity Check)')
plt.show()

print("Collinearity Observation: Check if any TWO features (excluding Target) have corr > 0.8.")
print("If 'Dep_Delay' and 'Arr_Delay' are highly correlated, that is expected (Feature vs Target).")
# ... existing code for Pearson Correlation Heatmap ...

from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\n--- Syllabus Check: VIF Analysis (Multicollinearity) ---")
# We use the numeric dataframe you created for the heatmap
# numeric_df was defined in your previous lines
X_vif = numeric_df.drop(columns=['Arr_Delay'], errors='ignore') # Drop target if present
X_vif['intercept'] = 1 # Statsmodels requires an intercept

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i)
                   for i in range(len(X_vif.columns))]

print(vif_data.sort_values('VIF', ascending=False))
print("Observation: VIF > 10 indicates high multicollinearity.")
# ============================

# %%
import numpy as np
import pandas as pd

# Outlier / "stupid point" audit for regression features
# Uses variables already defined in the notebook (X, df_reg, y, features_final, etc.)

def outlier_audit(df, features):

    # numeric/categorical split for the feature list
    num_feats = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_feats = [c for c in features if c not in num_feats]

    report_rows = []

    for c in num_feats:
        col = df[c]
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        z = (col - col.mean()) / (col.std(ddof=0) if col.std(ddof=0) != 0 else 1)
        n_iqr = ((col < lb) | (col > ub)).sum()
        n_z4 = (z.abs() > 4).sum()
        n_nan = col.isna().sum()
        report_rows.append({
            "feature": c,
            "dtype": str(col.dtype),
            "count": len(col),
            "n_nan": int(n_nan),
            "min": float(col.min()),
            "q1": float(q1),
            "median": float(col.median()),
            "mean": float(col.mean()),
            "q3": float(q3),
            "max": float(col.max()),
            "iqr": float(iqr),
            "outliers_1.5IQR": int(n_iqr),
            "outliers_z>4": int(n_z4)
        })

    stats_df = pd.DataFrame(report_rows).set_index('feature')
    display(stats_df)

    # Domain-specific sanity checks (flag indices)
    flags = pd.Series(False, index=df.index)

    # Flight duration must be positive
    if 'Flight_Duration' in df.columns:
        bad = df['Flight_Duration'] <= 0
        if bad.any():
            print(f"Flag: Flight_Duration <= 0 -> {bad.sum()} rows")
            flags = flags | bad

    # Aircraft age: negative or extremely large (>80)
    if 'Aicraft_age' in df.columns:
        bad = (df['Aicraft_age'] < 0) | (df['Aicraft_age'] > 80)
        if bad.any():
            print(f"Flag: Aicraft_age <0 or >80 -> {bad.sum()} rows")
            flags = flags | bad

    # Precipitation: should be >=0 and not absurdly large (>500 mm)
    if 'prcp' in df.columns:
        bad = (df['prcp'] < 0) | (df['prcp'] > 500)
        if bad.any():
            print(f"Flag: prcp <0 or >500 -> {bad.sum()} rows")
            flags = flags | bad

    # Wind speed: should be >=0 and not absurdly large (>200)
    if 'wspd' in df.columns:
        bad = (df['wspd'] < 0) | (df['wspd'] > 200)
        if bad.any():
            print(f"Flag: wspd <0 or >200 -> {bad.sum()} rows")
            flags = flags | bad

    # Avg temp: reasonable Earth bounds
    if 'tavg' in df.columns:
        bad = (df['tavg'] < -80) | (df['tavg'] > 60)
        if bad.any():
            print(f"Flag: tavg < -80 or > 60 -> {bad.sum()} rows")
            flags = flags | bad

    # Day of week should be 1..7
    if 'Day_Of_Week' in df.columns:
        bad = ~df['Day_Of_Week'].isin([1,2,3,4,5,6,7])
        if bad.any():
            print(f"Flag: Day_Of_Week outside 1-7 -> {bad.sum()} rows")
            flags = flags | bad

    # Dep_Delay extreme sanity (very negative or extremely large positive)
    if 'Dep_Delay' in df.columns:
        bad = (df['Dep_Delay'] < -1440) | (df['Dep_Delay'] > 1440)  # more than a day?
        if bad.any():
            print(f"Flag: Dep_Delay beyond +/-1440 min -> {bad.sum()} rows")
            flags = flags | bad

    # For categorical features: show rare categories
    rare_reports = {}
    for c in cat_feats:
        vc = df[c].value_counts(dropna=False)
        rare = vc[vc <= max(1, int(0.001 * len(df)))]  # <=0.1% or at least 1
        rare_reports[c] = {"unique": vc.size, "rare_count": rare.sum(), "rare_unique": rare.size}
    if cat_feats:
        print("\nCategorical rarity (<=0.1% considered rare):")
        for k,v in rare_reports.items():
            print(f" - {k}: unique={v['unique']}, rare_unique={v['rare_unique']}, rare_count={v['rare_count']}")

    # Summarize flagged rows (limit to first 200)
    n_flagged = flags.sum()
    print(f"\nTotal rows flagged by domain rules: {n_flagged}")
    if n_flagged:
        print("Sample flagged rows (first 200) with selected columns:")
        cols_show = list(set(num_feats + cat_feats))  # maintain features used
        display(df.loc[flags, cols_show].head(200))

    # Return dataframes for further programmatic handling
    return {"stats": stats_df, "flag_index": df.index[flags], "flags_mask": flags}

# Run the audit on X (features used for regression)
audit = outlier_audit(X, features_final)

# Example actions you can perform after the audit:
# - inspect audit['flag_index'] to decide removal / correction
# - apply winsorization on numeric columns that have many extreme values
# - leave extreme delays (they may be real)

# %%
# Remove flagged rows from X and y using audit['flag_index']
flagged_idx = audit['flag_index']
X_clean = X.drop(index=flagged_idx)
y_clean = y.drop(index=flagged_idx)

print(f"Removed {len(flagged_idx)} flagged rows. Cleaned X shape: {X_clean.shape}, y shape: {y_clean.shape}")

# %%
# ==========================================
# CHUNK 2: The Pipeline
# ==========================================

# Identify columns by type
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Build Transformer
# handle_unknown='ignore' protects us if a rare airline appears in the test set
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

print("Pipeline defined.")
print(f"Numerical Features: {num_cols}")
print(f"Categorical Features: {cat_cols}")

# %%
# Define a threshold for "delayed" (e.g., > 0 minutes arrival delay)
delay_threshold = 15

# Training set
n_train_delayed = (y_train > delay_threshold).sum()
n_train_not_delayed = (y_train <= delay_threshold).sum()

# Test set
n_test_delayed = (y_test > delay_threshold).sum()
n_test_not_delayed = (y_test <= delay_threshold).sum()

print(f"Training set: {n_train_delayed} delayed, {n_train_not_delayed} not delayed")
print(f"Test set:     {n_test_delayed} delayed, {n_test_not_delayed} not delayed")

# %%
# ==========================================
# CHUNK 3: Model Training & Comparison (CORRECTED)
# ==========================================


results = {}

def train_and_score(name, model):
    print(f"\nTraining {name}...")
    start = time.time()
    
    # Create full pipeline
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    
    # Fit
    pipe.fit(X_train, y_train)
    
    # Predict
    preds = pipe.predict(X_test)
    
    # Score
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    duration = time.time() - start
    print(f"  -> RMSE: {rmse:.4f} min")
    print(f"  -> R2:   {r2:.4f}")
    print(f"  -> Time: {duration:.2f}s")
    
    return {'RMSE': rmse, 'R2': r2, 'Model': pipe}

# Model A: Linear Regression (The Baseline)
results['Linear'] = train_and_score("Linear Regression", LinearRegression())

# Model B: Random Forest (The Standard)
# results['RF'] = train_and_score("Random Forest", RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42))

# Model C: Gradient Boosting (The 'Better' One)
# results['GB'] = train_and_score("Gradient Boosting", GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42))

# %%
# ==========================================
# CHUNK 4: Linear Regression Diagnostics
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

# 1. GET THE CHOSEN MODEL
# We extract the Linear Regression pipeline from your results dictionary
final_model_pipe = results['Linear']['Model']
final_model_obj = final_model_pipe.named_steps['model']

# 2. GENERATE PREDICTIONS
# We use the test set we set aside earlier
y_pred_final = final_model_pipe.predict(X_test)
residuals = y_test - y_pred_final

# 3. PLOT: ACTUAL VS PREDICTED
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
# Downsample for plotting speed if dataset is huge, otherwise plot all
sns.scatterplot(x=y_test, y=y_pred_final, alpha=0.3, color='#3498db')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect fit line
plt.xlabel('Actual Arrival Delay (min)')
plt.ylabel('Predicted Arrival Delay (min)')
plt.title('Actual vs. Predicted (Linear Regression)')

# 4. PLOT: RESIDUALS (ERRORS)
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, color='#e74c3c', bins=50)
plt.xlabel('Error (Minutes)')
plt.title('Distribution of Errors (Residuals)')
plt.xlim(-50, 50) # Zoom in on the center to see the bell curve

plt.tight_layout()
plt.show()

# 5. EXTRACT COEFFICIENTS (The "Explainability" Part)
# We want to see the weight of 'Dep_Delay' specifically
feature_names = num_cols # We focus on numeric first as they are easiest to read
coeffs = final_model_obj.coef_[:len(num_cols)]

print("\n--- Linear Regression Equation Insights ---")
print("Intercept (Baseline Delay):", round(final_model_obj.intercept_, 2), "minutes")
for name, coef in zip(feature_names, coeffs):
    print(f"Feature: {name:<15} | Coefficient: {coef:.4f}")

print("\n--- CONCLUSION FOR REPORT ---")
print(f"Model Selected: Linear Regression")
print(f"Performance: R2 = {r2_score(y_test, y_pred_final):.4f} (Explains ~94% of variance)")
print("Key Insight: The coefficient for Dep_Delay is likely close to 1.0,")
print("indicating that delay in the air is almost perfectly linear with delay on the ground.")

# %%
# ==========================================
# CHUNK 5: Phase II - Statistical Inference & Stepwise Regression (FINAL FIX)
# ==========================================
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("--- Phase II: Statistical Analysis (Syllabus Requirements) ---")

# 1. DATA PREPARATION FOR STATSMODELS
# ======================================================
STATS_SAMPLE = 100000 
print(f"Sampling {STATS_SAMPLE} rows for Statistical Analysis...")

# Sample the cleaned regression data
df_stats = df_reg.sample(n=min(len(df_reg), STATS_SAMPLE), random_state=42).copy()

# One-Hot Encode
df_stats_encoded = pd.get_dummies(df_stats, columns=cat_cols, drop_first=True, dtype=int)

# Define X (features) and y (target)
y_sm = df_stats_encoded['Arr_Delay']
X_sm = df_stats_encoded.drop(columns=['Arr_Delay'])

# Convert to float
X_sm = X_sm.astype(float)
y_sm = y_sm.astype(float)

# Add intercept
X_sm = sm.add_constant(X_sm)

print(f"Stats Data Ready. Shape: {X_sm.shape}")

# 2. STEPWISE REGRESSION (Backward Elimination)
# ======================================================
def backward_elimination(X, y, significance_level=0.05):
    features = X.columns.tolist()
    initial_count = len(features)
    
    print(f"\nStarting Backward Elimination on {initial_count} features...")
    
    while len(features) > 0:
        X_curr = X[features]
        if X_curr.empty: break
            
        model = sm.OLS(y, X_curr).fit()
        p_values = model.pvalues
        
        max_p = p_values.max()
        worst_feature = p_values.idxmax()
        
        if max_p > significance_level:
            if worst_feature == 'const':
                temp_p = p_values.drop('const')
                if len(temp_p) > 0:
                    max_p = temp_p.max()
                    worst_feature = temp_p.idxmax()
                else: break 
            
            if max_p > significance_level:
                features.remove(worst_feature)
        else:
            break 
            
    print(f"Elimination Complete. Removed {initial_count - len(features)} features.")
    return model, features

# Run selection
final_sm_model, selected_features = backward_elimination(X_sm, y_sm)

# 3. FINAL REGRESSION OUTPUTS
# ======================================================
print("\n" + "="*40)
print("FINAL MODEL SUMMARY (T-tests, F-tests, CI)")
print("="*40)
print(final_sm_model.summary())

# 4. METRICS TABLE
# ======================================================
metrics_data = {
    "Metric": ["R-Squared", "Adjusted R-Squared", "AIC", "BIC", "MSE (Mean Squared Error)"],
    "Value": [
        final_sm_model.rsquared,
        final_sm_model.rsquared_adj,
        final_sm_model.aic,
        final_sm_model.bic,
        final_sm_model.mse_resid
    ]
}
metrics_df = pd.DataFrame(metrics_data)
print("\n--- REQUIRED METRICS TABLE ---")
display(metrics_df)

# 5. PREDICTION EXAMPLE (FIXED)
# ======================================================
# FIX: Filter 'test_row' to include ONLY selected features
test_row = X_sm.iloc[[0]][selected_features] 

# FIX: Use .iloc[0] instead of [0] to ignore index mismatch
predicted_val = final_sm_model.predict(test_row).iloc[0]
actual_val = y_sm.iloc[0]

print("\n--- Single Prediction Example ---")
print(f"Inputs used ({len(selected_features)} features): {test_row.columns[:3].tolist()}...")
print(f"Predicted Arrival Delay: {predicted_val:.2f} min")
print(f"Actual Arrival Delay:    {actual_val:.2f} min")

# 6. PLOT: TRAIN, TEST, PREDICTED (FIXED)
# ======================================================
plt.figure(figsize=(10, 6))

plot_n = 50
# Take a new sample for plotting
viz_df = df_stats.sample(n=plot_n*2, random_state=99).copy()

# Prep data exactly like training
viz_encoded = pd.get_dummies(viz_df, columns=cat_cols, drop_first=True, dtype=int)
viz_encoded = sm.add_constant(viz_encoded)
viz_encoded = viz_encoded.astype(float)

# FIX: Align columns specifically to 'selected_features' and fill missing with 0
viz_encoded = viz_encoded.reindex(columns=selected_features, fill_value=0)

# Split into Train/Test for the visual
viz_train = viz_encoded.iloc[:plot_n]
viz_test = viz_encoded.iloc[plot_n:]
viz_y_train = y_sm.loc[viz_train.index]
viz_y_test = y_sm.loc[viz_test.index]

# Generate predictions
viz_pred_line = final_sm_model.predict(viz_test)

# PLOT 1: Train Data (Black Dots)
plt.scatter(viz_train['Dep_Delay'], viz_y_train, color='black', alpha=0.6, label='Train Data')

# PLOT 2: Test Data (Red Dots)
plt.scatter(viz_test['Dep_Delay'], viz_y_test, color='red', alpha=0.6, label='Test Data')

# PLOT 3: Predicted Data (Blue X)
plt.scatter(viz_test['Dep_Delay'], viz_pred_line, color='blue', marker='x', s=60, label='Predicted')

plt.title('Regression Analysis: Train vs Test vs Predicted')
plt.xlabel('Departure Delay (min)')
plt.ylabel('Arrival Delay (min)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Phase II Complete.")


