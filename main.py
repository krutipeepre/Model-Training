# Have to comment before sharing with backend engineer

from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Project Vyaghri API")

# 1. Load data and train spatial trees once when the server starts up
print("Booting up Vyaghri ML Engine & loading infrastructure...")
lamps_df = pd.read_csv("delhi_real_street_lamps.csv")
police_df = pd.read_csv("delhi_real_police_stations.csv")

lamps_rad = np.deg2rad(lamps_df[['Latitude', 'Longitude']].values)
police_rad = np.deg2rad(police_df[['Latitude', 'Longitude']].values)

lamp_tree = BallTree(lamps_rad, metric='haversine')
police_tree = BallTree(police_rad, metric='haversine')

@app.get("/")
def home():
    return {"status": "Vyaghri backend & ML spatial engine are live!"}

# 2. Live Risk Assessment Endpoint for the Mobile App
@app.get("/api/check-risk")
def check_risk(lat: float, lon: float):
    R = 6371000  # Earth's radius in meters
    point_rad = np.deg2rad([[lat, lon]])
    
    dist_lamp_rad, _ = lamp_tree.query(point_rad, k=1)
    dist_lamp_meters = float(dist_lamp_rad[0][0] * R)
    
    dist_police_rad, _ = police_tree.query(point_rad, k=1)
    dist_police_meters = float(dist_police_rad[0][0] * R)
    
    is_risk_zone = (dist_lamp_meters > 50) and (dist_police_meters > 2000)
    
    return {
        "coordinate": [lat, lon],
        "distance_to_nearest_lamp_m": round(dist_lamp_meters, 1),
        "distance_to_nearest_police_m": round(dist_police_meters, 1),
        "high_risk_zone": is_risk_zone
    }