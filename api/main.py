from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import io
import ast
import requests
import math
from datetime import datetime

app = FastAPI(title="Flight Risk Evaluator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to models and data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1_MODEL_PATH = os.path.join(BASE_DIR, "stage1_model.joblib")
STAGE2_MODEL_PATH = os.path.join(BASE_DIR, "stage2_model.joblib")
RULES_CSV_PATH = os.path.join(BASE_DIR, "all_association_rules_full.csv")
TIERS_CSV_PATH = os.path.join(BASE_DIR, "airport_performance_tiers_enriched.csv")

# Load models and rules
try:
    stage1_pipe = joblib.load(STAGE1_MODEL_PATH)
    stage2_pipe = joblib.load(STAGE2_MODEL_PATH)
    rules_df = pd.read_csv(RULES_CSV_PATH)
    tiers_df = pd.read_csv(TIERS_CSV_PATH)
except Exception as e:
    print(f"Error loading models or rules: {e}")
    stage1_pipe = None
    stage2_pipe = None
    rules_df = None
    tiers_df = None

# Create a mapping for airport tiers and historical data
airport_data = {}
if tiers_df is not None:
    for _, row in tiers_df.iterrows():
        airport_data[row['Dep_Airport']] = {
            'tier': row['Performance_Tier'],
            'avg_dep_delay': row['avg_dep_delay'] if not pd.isna(row['avg_dep_delay']) else 0.0,
            'lat': row['LATITUDE'],
            'lon': row['LONGITUDE'],
            'state': row['STATE'] if not pd.isna(row['STATE']) else 'TX'
        }

class FlightRequest(BaseModel):
    Origin_Airport: str
    Destination_Airport: str
    Carrier: str
    Departure_Time: str  # E.g. "Morning", "Afternoon", "Evening", "Night"
    Date: str            # E.g. "YYYY-MM-DD"

@app.get("/")
def read_root():
    return {"message": "Flight Risk API is running"}

@app.post("/predict")
def predict_flight_risk(request: FlightRequest):
    if stage1_pipe is None or stage2_pipe is None or rules_df is None:
        raise HTTPException(status_code=500, detail="Models or rules not loaded properly.")

    # Get airport specific historical data
    origin_info = airport_data.get(request.Origin_Airport, {
        'tier': 2, 'avg_dep_delay': 5.0, 'lat': 39.8283, 'lon': -98.5795, 'state': 'TX'
    })
    dest_info = airport_data.get(request.Destination_Airport, {
        'tier': 2, 'avg_dep_delay': 5.0, 'lat': 39.8283, 'lon': -98.5795, 'state': 'TX'
    })

    # Calculate Haversine Flight Duration
    def haversine(lat1, lon1, lat2, lon2):
        R = 3958.8 # Earth radius in miles
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        return distance
    
    dist_miles = haversine(origin_info['lat'], origin_info['lon'], dest_info['lat'], dest_info['lon'])
    flight_duration_mins = (dist_miles / 500.0) * 60.0 # pure flight time
    flight_duration_mins += 40.0 # Add 40 minutes for taxi/takeoff/landing logic

    # Weather Defaults (if fetch fails)
    prcp = 0.0
    snow = 0.0
    wspd = 10.0
    tmin = 50.0
    tavg = 50.0

    # Fetch Weather from Open-Meteo
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={origin_info['lat']}&longitude={origin_info['lon']}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,windspeed_10m_max&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=auto&start_date={request.Date}&end_date={request.Date}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            weather = resp.json().get('daily', {})
            if weather and len(weather.get('temperature_2m_max', [])) > 0:
                tmax = weather['temperature_2m_max'][0]
                tmin = weather['temperature_2m_min'][0]
                # Fallbacks in case api returned nulls
                if tmax is not None and tmin is not None: tavg = (tmax + tmin) / 2
                if weather['precipitation_sum'][0] is not None: prcp = weather['precipitation_sum'][0]
                if weather['snowfall_sum'][0] is not None: snow = weather['snowfall_sum'][0]
                if weather['windspeed_10m_max'][0] is not None: wspd = weather['windspeed_10m_max'][0]
    except Exception as e:
        print(f"Weather Fetch Error: {e}")

    # Calculate Day Of Week (1 = Monday, 7 = Sunday to match typical datasets)
    try:
        dt = datetime.strptime(request.Date, "%Y-%m-%d")
        day_of_week = dt.isoweekday() 
    except:
        day_of_week = 3

    # Prepare base features mapping
    # Note: Our pipelines use OrdinalEncoder internally, so we just pass the strings!
    base_features = {
        'Dep_Airport': [request.Origin_Airport],
        'Arr_Airport': [request.Destination_Airport],
        'Month': [dt.month if 'dt' in locals() else 1], 
        'Day_Of_Week': [day_of_week],         
        'Airline': [request.Carrier],
        'DepTime_label': [request.Departure_Time], 
        'Flight_Duration': [flight_duration_mins], 
        'tavg': [tavg],
        'prcp': [prcp],
        'wspd': [wspd],
        'Aicraft_age': [12.0], # Hardcoded assumption for now
    }
    
    # 1. Stage 1: Predict Expected Departure Delay
    df_stage1 = pd.DataFrame(base_features)
    try:
        expected_dep_delay = stage1_pipe.predict(df_stage1)[0]
    except Exception as e:
        expected_dep_delay = origin_info['avg_dep_delay']
        print(f"Stage 1 error: {e}")

    # 2. Stage 2: Predict Expected Arrival Delay
    # Stage 2 requires 'Dep_Delay' as an input feature
    stage2_features = base_features.copy()
    stage2_features['Dep_Delay'] = [expected_dep_delay]
    df_stage2 = pd.DataFrame(stage2_features)
    
    try:
        expected_arr_delay = stage2_pipe.predict(df_stage2)[0]
    except Exception as e:
        expected_arr_delay = expected_dep_delay
        print(f"Stage 2 error: {e}")

    # 3. Association Rules (Severe Risk Index)
    tier_name_mapping = {
        0: 'Tier_0_Secondary',
        1: 'Tier_1_HighRisk',
        2: 'Tier_2_Underperforming',
        3: 'Tier_3_Efficient',
        4: 'Tier_4_MegaHub'
    }
    tier_name = f"Tier_Name={tier_name_mapping.get(origin_info['tier'], 'Tier_2_Underperforming')}"
    
    # Get Weather condition strings based on typical weather risk thresholds
    weather_snow = "Weather_Snow" if snow > 0 else None
    weather_rain = "Weather_Rain" if prcp > 0.2 else None
    weather_wind = "Weather_Wind" if wspd > 25 else None
    weather_freez = "Weather_Freezing" if tmin < 32 else None
    
    airline_cond = f"Airline={request.Carrier}"
    
    current_conditions = set([tier_name, airline_cond])
    if weather_snow: current_conditions.add(weather_snow)
    if weather_rain: current_conditions.add(weather_rain)
    if weather_wind: current_conditions.add(weather_wind)
    if weather_freez: current_conditions.add(weather_freez)

    max_risk_score = 0.0
    matched_rule = None
    
    for _, row in rules_df.iterrows():
        try:
             ant_str = row['antecedents']
             if "frozenset" in ant_str:
                 set_str = ant_str.replace("frozenset(", "")[:-1]
                 antecedent_set = set(ast.literal_eval(set_str))
             else: continue
                 
             cons_str = row['consequents']
             if "frozenset" in cons_str:
                 set_str_cons = cons_str.replace("frozenset(", "")[:-1]
                 consequent_set = set(ast.literal_eval(set_str_cons))
             else: continue
                 
             if 'Delay_Class=Severe' in consequent_set:
                 if antecedent_set.issubset(current_conditions):
                     # Blended Risk Score = Confidence * Lift
                     rule_score = row['confidence'] * row['lift']
                     if rule_score > max_risk_score:
                         max_risk_score = rule_score
                         matched_rule = antecedent_set
        except: continue

    # Normalize Score to 0-100 (Assuming a max reasonable score of ~4.0 for conf*lift)
    risk_index_raw = (max_risk_score / 4.0) * 100
    risk_index = min(max(risk_index_raw, 5.0), 100.0) # Cap at 100, floor at 5

    # Generate curved flight path
    def generate_flight_arc(lat1, lon1, lat2, lon2, num_points=50):
        if lat1 == lat2 and lon1 == lon2:
            return [[lat1, lon1]]
        points = []
        dLat = lat2 - lat1
        dLon = lon2 - lon1
        dist = math.sqrt(dLat**2 + dLon**2)
        
        nx = -dLon / dist
        ny = dLat / dist
        
        # Always curve towards North relative to the path
        if nx < 0:
            nx = -nx
            ny = -ny
            
        for i in range(num_points + 1):
            t = i / num_points
            lat_t = lat1 + t * dLat
            lon_t = lon1 + t * dLon
            
            # Parabolic offset
            offset = dist * 0.15 * 4 * (t * (1 - t))
            
            lat_arc = lat_t + offset * nx
            lon_arc = lon_t + offset * ny
            points.append([lat_arc, lon_arc])
        return points

    arc_path = generate_flight_arc(origin_info['lat'], origin_info['lon'], dest_info['lat'], dest_info['lon'])
        
    return {
        "expected_dep_delay": round(float(expected_dep_delay), 2),
        "expected_arr_delay": round(float(expected_arr_delay), 2),
        "risk_index": round(float(risk_index), 1),
        "matched_conditions": list(matched_rule) if matched_rule else ["Baseline Risk"],
        "weather_used": {
            "precipitation_inches": prcp,
            "wind_speed_mph": wspd,
            "snowfall_inches": snow,
            "min_temp_f": tmin,
            "avg_temp_f": tavg
        },
        "airport_avg_delay": round(origin_info['avg_dep_delay'], 2),
        "flight_path": arc_path
    }
