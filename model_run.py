import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings("ignore")

# 1. Load your verified real-world infrastructure data
print("Loading infrastructure datasets...")
lamps_df = pd.read_csv("delhi_real_street_lamps.csv")
police_df = pd.read_csv("delhi_real_police_stations.csv")

# 2. Convert GPS coordinates to radians (Required for Haversine distance math)
lamps_rad = np.deg2rad(lamps_df[['Latitude', 'Longitude']].values)
police_rad = np.deg2rad(police_df[['Latitude', 'Longitude']].values)

# 3. Build the Spatial Trees
print("Training Spatial Trees...")
lamp_tree = BallTree(lamps_rad, metric='haversine')
police_tree = BallTree(police_rad, metric='haversine')

# 4. Define the Intelligence: The Risk Scoring Function
def calculate_risk_score(lat, lon):
    R = 6371000  # Earth's radius in meters
    point_rad = np.deg2rad([[lat, lon]])
    
    dist_lamp_rad, _ = lamp_tree.query(point_rad, k=1)
    dist_lamp_meters = dist_lamp_rad[0][0] * R
    
    dist_police_rad, _ = police_tree.query(point_rad, k=1)
    dist_police_meters = dist_police_rad[0][0] * R
    
    is_risk_zone = (dist_lamp_meters > 50) and (dist_police_meters > 2000)
    
    return {
        "coordinate": [lat, lon],
        "distance_to_nearest_lamp_m": round(dist_lamp_meters, 1),
        "distance_to_nearest_police_m": round(dist_police_meters, 1),
        "high_risk_zone": is_risk_zone
    }

# 5. Test the model near Hauz Khas
test_lat, test_lon = 28.5494, 77.2001
result = calculate_risk_score(test_lat, test_lon)

print("\n--- Vyaghri ML Risk Assessment ---")
print(result)