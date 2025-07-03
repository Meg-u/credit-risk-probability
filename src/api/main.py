# src/api/main.py

from fastapi import FastAPI
from src.api.pydantic_models import CustomersRequest
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('models/credit_risk_model.pkl')

# FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predict credit risk probability for BNPL service",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Credit Risk Scoring API is running 🚀"}

@app.post("/predict")
def predict_credit_risk(request: CustomersRequest):
    # Convert input to DataFrame
    data = [customer.dict() for customer in request.customers]
    df_new = pd.DataFrame(data)

    # === Do any necessary feature engineering ===
    # For demo, drop IDs, convert types if needed:
    X_new = df_new.drop(['CustomerId'], axis=1, errors='ignore')

    # Predict
    preds = model.predict(X_new)
    probs = model.predict_proba(X_new)[:, 1]

    # Return as JSON
    results = []
    for i, row in df_new.iterrows():
        results.append({
            "CustomerId": row['CustomerId'],
            "PredictedClass": int(preds[i]),
            "RiskProbability": float(probs[i])
        })

    return {"predictions": results}
