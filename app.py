from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json
import pandas as pd
app = FastAPI(title="Flight Delay Predictor", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model          = joblib.load("flight_delay_model.pkl")
encoders       = joblib.load("encoders.pkl")
route_delay_   = joblib.load("route_delay.pkl")
carrier_delay_ = joblib.load("carrier_delay.pkl")
top_airports_  = joblib.load("top_airports.pkl")
base_rate      = joblib.load("base_rate.pkl")

with open("feature_cols.json") as f:
    feature_cols = json.load(f)

route_lookup   = route_delay_.set_index(["Origin","Dest"])["route_delay_rate"].to_dict()
carrier_lookup = carrier_delay_.set_index("Reporting_Airline")["carrier_delay_rate"].to_dict()

class Flight(BaseModel):
    CRSDepTime      : int
    DayOfWeek       : int
    Month           : int
    Quarter         : int
    Distance        : float
    Airline         : str
    Origin          : str
    Dest            : str
    TaxiOut         : float = 15.0
    DistanceGroup   : int   = 3
    prior_leg_delay : float = 0.0

@app.get("/")
def root():
    return {"message": "Flight Delay Predictor API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(flight: Flight):
    try:
        dep_hour    = int(str(flight.CRSDepTime).zfill(4)[:2])
        dist_bucket = "short" if flight.Distance < 500 else "medium" if flight.Distance < 1500 else "long"
        airline_enc = encoders["Reporting_Airline"].transform([flight.Airline])[0]
        origin_enc  = encoders["Origin"].transform([flight.Origin])[0]
        dest_enc    = encoders["Dest"].transform([flight.Dest])[0]
        dist_enc    = encoders["distance_bucket"].transform([dist_bucket])[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Unknown airline or airport code")

    route_rate   = route_lookup.get((flight.Origin, flight.Dest), base_rate)
    carrier_rate = carrier_lookup.get(flight.Airline, base_rate)

    f = {
        "dep_hour"              : dep_hour,
        "dep_weekday"           : flight.DayOfWeek,
        "dep_month"             : flight.Month,
        "dep_quarter"           : flight.Quarter,
        "is_weekend"            : int(flight.DayOfWeek in [6, 7]),
        "is_rush_hour"          : int((6 <= dep_hour <= 9) or (16 <= dep_hour <= 19)),
        "Distance"              : flight.Distance,
        "distance_bucket_enc"   : dist_enc,
        "route_delay_rate"      : route_rate,
        "is_busy_airport"       : int(flight.Origin in top_airports_),
        "Reporting_Airline_enc" : airline_enc,
        "carrier_delay_rate"    : carrier_rate,
        "prior_leg_delay"       : flight.prior_leg_delay,
        "Origin_enc"            : origin_enc,
        "Dest_enc"              : dest_enc,
        "TaxiOut"               : flight.TaxiOut,
        "DistanceGroup"         : flight.DistanceGroup,
    }

    row   = pd.DataFrame([[f[c] for c in feature_cols]], columns=feature_cols)
    proba = float(model.predict_proba(row)[0][1])

    return {
        "airline"           : flight.Airline,
        "route"             : f"{flight.Origin} → {flight.Dest}",
        "delay_probability" : round(proba, 3),
        "prediction"        : "DELAYED" if proba > 0.5 else "ON TIME",
        "confidence"        : f"{max(proba, 1-proba)*100:.1f}%"
    }
