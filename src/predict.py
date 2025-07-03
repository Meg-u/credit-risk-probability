# src/predict.py

import pandas as pd
import joblib

df = pd.read_csv('data/processed/processed_train_data.csv')
df.sample(5).to_csv('data/processed/new_customers.csv', index=False)


def load_model(model_path='models/credit_risk_model.pkl'):
    """Load trained model"""
    model = joblib.load(model_path)
    return model

def load_data(new_data_path):
    """Load new customer data to score"""
    df_new = pd.read_csv(new_data_path)
    return df_new

def predict(model, X_new):
    """Make predictions & risk scores"""
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]  # probability of 'bad' class
    return predictions, probabilities

if __name__ == "__main__":
    # Load trained model
    model = load_model()
    print("Model loaded successfully")

    # Load new unseen data (same format as training features)
    df_new = load_data('data/processed/new_customers.csv')  # adjust path!
    print(f"New data shape: {df_new.shape}")

    # Drop IDs if needed
    if 'CustomerId' in df_new.columns:
        X_new = df_new.drop(['CustomerId'], axis=1)
    else :
        X_new = df_new.copy()

    if 'FraudResult' in df_new.columns:
        X_new = df_new.drop(['FraudResult'], axis=1)
    else :
        X_new = df_new.copy()
    # Predict
    preds, probs = predict(model, X_new)

    # Save results
    df_new['PredictedClass'] = preds
    df_new['RiskProbability'] = probs

    df_new.to_csv('data/processed/new_customers_scored.csv', index=False)
    print("Predictions saved to data/processed/new_customers_scored.csv")

    print(df_new.head())
