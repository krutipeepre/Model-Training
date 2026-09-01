import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings("ignore")

# 1. Load your verified real-world infrastructure data
print("Loading infrastructure datasets...")
lamps_df = pd.read_csv("delhi_real_street_lamps.csv")
police_df = pd.read_csv("delhi_real_police_stations.csv")

# Optional: Agar police_df mein crime_weight column nahi hai, toh default 1.0 set kar do
if 'crime_weight' not in police_df.columns:
    police_df['crime_weight'] = 1.0  # Isko baad mein historical data ke mutabiq customize kar sakti ho

# 2. Convert GPS coordinates to radians (Required for Haversine distance math)
lamps_rad = np.deg2rad(lamps_df[['Latitude', 'Longitude']].values)
police_rad = np.deg2rad(police_df[['Latitude', 'Longitude']].values)

# 3. Build the Spatial Trees
print("Training Spatial Trees...")
lamp_tree = BallTree(lamps_rad, metric='haversine')
police_tree = BallTree(police_rad, metric='haversine')

# 4. Define the Intelligence: Enhanced Risk Scoring Function with Crime Weights
def calculate_risk_score(lat, lon):
    R = 6371000  # Earth's radius in meters
    point_rad = np.deg2rad([[lat, lon]])
    
    # Nearest Lamp Query
    dist_lamp_rad, _ = lamp_tree.query(point_rad, k=1)
    dist_lamp_meters = dist_lamp_rad[0][0] * R
    
    # Nearest Police Station Query (along with index to fetch its weight)
    dist_police_rad, police_ind = police_tree.query(point_rad, k=1)
    dist_police_meters = dist_police_rad[0][0] * R
    
    # Fetch the corresponding crime weight of the nearest police station area
    nearest_station_idx = police_ind[0][0]
    station_crime_multiplier = police_df.iloc[nearest_station_idx].get('crime_weight', 1.0)
    
    # Base isolation check (The "Soonsaan" zone logic)
    is_isolated = (dist_lamp_meters > 50) and (dist_police_meters > 2000)
    
    # Enhanced Risk Score Calculation (Distance factored with historical crime multipliers)
    enhanced_risk_score = round((dist_police_meters / 1000) * station_crime_multiplier, 2)
    
    # Determine risk level category
    if is_isolated or enhanced_risk_score > 3.0:
        risk_level = "High Risk (Red Zone)"
    elif enhanced_risk_score > 1.5:
        risk_level = "Moderate Risk (Orange Zone)"
    else:
        risk_level = "Safe (Green Zone)"

    return {
        "coordinate": [lat, lon],
        "distance_to_nearest_lamp_m": round(dist_lamp_meters, 1),
        "distance_to_nearest_police_m": round(dist_police_meters, 1),
        "crime_weight_multiplier": station_crime_multiplier,
        "enhanced_risk_score": enhanced_risk_score,
        "risk_category": risk_level,
        "is_isolated_soonsaan": is_isolated
    }

# 5. Test the model near Hauz Khas
test_lat, test_lon = 28.5494, 77.2001
result = calculate_risk_score(test_lat, test_lon)

print("\n--- Vyaghri ML Enhanced Spatial Risk Assessment ---")
for key, value in result.items():
    print(f"{key}: {value}")