US Flight Operational Risk Analysis (2023)

1. Prerequisites
You need Python 3.8+ and the following libraries installed. You can install them via pip:

pip install pandas numpy matplotlib seaborn scikit-learn statsmodels folium mlxtend

2. Required Data Files
Ensure the following 3 raw CSV files are in the same directory as the scripts:

- US_flights_2023.csv (Main flight data)
- airports_geolocation.csv (Airport coordinates)
- weather_meteo_by_airport.csv (Weather data)

3. Execution Order (Critical)
You must run the scripts in this specific order to avoid "File Not Found" errors.

Step 1: Airport Clustering (Run First)
File: Cluster.py
Action: Runs K-Means clustering to categorize airports into 5 tiers based on performance.
Output: Generates airport_performance_tiers_enriched.csv.
Note: Classify.py and Rules.py require this output file to run.

Step 2: Association Rule Mining
File: Rules.py
Action: Uses the Apriori algorithm to find "recipes" for severe delays (e.g., Rain + Tier 1 Airport = Severe Delay).
Output: Generates all_association_rules_full.csv.

Step 3: Classification Modeling
File: Classify.py
Action: Trains multiple models (Random Forest, Gradient Boosting, etc.) to predict if a specific flight will be delayed.
Output: Prints model performance metrics (AUC, Precision, Recall) and feature importance charts to the console.

Step 4: Regression Analysis (Independent)
File: Reg.py
Action: Runs a Linear Regression model to estimate the exact minutes of arrival delay.
Note: This script is technically independent and can be run at any time, but it is best run after you have reviewed the clustering results.

4. Troubleshooting common errors

FileNotFoundError: airport_performance_tiers_enriched.csv:
Cause: You tried to run Classify.py or Rules.py before Cluster.py.
Fix: Run Cluster.py and wait for it to finish saving the CSV.

Memory Errors (MemoryError):
Cause: The dataset (6M+ rows) is too large for your RAM.
Fix: Open the scripts and look for the SAMPLE_SIZE variable (usually near the top). Change it from 100000 to 50000 or smaller.
