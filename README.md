# Spatial Risk Assessment & Machine Learning Engine

A production-ready spatial proximity analysis and risk assessment backend built with Python, FastAPI, and Scikit-learn. This system evaluates safety and proximity risk scores based on GPS coordinates using spatial indexing trees (`BallTree`) and real-world infrastructure datasets (hospitals, police stations, and street lamps).

---

## Features

- **High-Performance Spatial Queries**: Uses Scikit-learn's `BallTree` with the Haversine metric for fast $O(\log n)$ geographic neighbor searches.
- **FastAPI REST Endpoints**: Asynchronous, lightweight API for real-time risk checking.
- **Dual-Threshold Proximity Logic**: Analyzes critical safety radiuses (e.g., immediate 50m and broader 2000m thresholds) for emergency response assets.
- **Team Handover Ready**: Complete with structured data ingestion pipelines and modular code organization.

---

## Tech Stack

- **Language**: Python
- **Framework**: FastAPI, Uvicorn
- **Machine Learning / Spatial Math**: NumPy, Scikit-learn (`BallTree`)
- **Data Handling**: Pandas

---

## Project Structure

```text
Model-Training/
│
├── main.py                     # FastAPI application and endpoint definitions
├── model_run.py                # Core spatial logic and BallTree calculations
├── model_test.ipynb            # Jupyter notebook for model testing and validation
├── security.ipynb              # Security and data sanitization checks
├── README.md                   # Project documentation and API reference
│
├── delhi_real_hospitals.csv      # Hospital infrastructure dataset (Latitude, Longitude)
├── delhi_real_police_stations.csv# Police station dataset (Latitude, Longitude)
└── delhi_real_street_lamps.csv   # Street lamp infrastructure dataset

Getting Started Locally

1. Clone the Repository

Bash

git clone [https://github.com/krutipeepre/Model-Training.git](https://github.com/krutipeepre/Model-Training.git)
cd Model-Training

2. Set Up Virtual Environment & Dependencies
Create and activate your virtual environment, then install the required dependencies:

Bash

```pip install fastapi uvicorn scikit-learn pandas numpy```

3. Run the FastAPI Server
Start the Uvicorn development server with live reload enabled:

Bash

```uvicorn main:app --reload```

API Reference
Check Location Risk
Evaluates the proximity risk score for a specific set of geographical coordinates.

Endpoint: /api/check-risk

Method: GET

Parameters:

lat (float): Latitude of the target location.

lon (float): Longitude of the target location.

Example Request:
HTTP
GET /api/check-risk?lat=28.5494&lon=77.2001
Example Response:
JSON
{
  "latitude": 28.5494,
  "longitude": 77.2001,
  "risk_score": "Moderate",
  "nearest_police_station_meters": 420.5,
  "nearest_hospital_meters": 850.2,
  "street_lamps_within_50m": 3
}

Author
Maintained by krutipeepre.