# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

def load_data():
    """Load processed data."""
    df = pd.read_csv('data/processed/processed_train_data.csv')
    print(f"Loaded data shape: {df.shape}")
    return df

def clean_data(df):
    """Clean and prepare dataset."""
    # Drop rows with missing target
    df = df.dropna(subset=['FraudResult'])

    # Remove object (non-numeric) columns like IDs
    non_numeric_cols = df.select_dtypes(include='object').columns.tolist()

    # Keep 'FraudResult' for target
    columns_to_drop = [col for col in non_numeric_cols if col != 'FraudResult']
    X = df.drop(columns=columns_to_drop + ['FraudResult'], errors='ignore')
    y = df['FraudResult']

    return X, y

def train_baseline_model(X_train, y_train):
    """Train simple Logistic Regression."""
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_complex_model(X_train, y_train):
    """Train Gradient Boosting as alternative."""
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")

if __name__ == "__main__":
    # Load and clean data
    df = load_data()
    X, y = clean_data(df)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train baseline model
    print("\n=== Baseline Logistic Regression ===")
    baseline_model = train_baseline_model(X_train, y_train)
    evaluate(baseline_model, X_test, y_test)

    # Train complex model
    print("\n=== Gradient Boosting Classifier ===")
    complex_model = train_complex_model(X_train, y_train)
    evaluate(complex_model, X_test, y_test)

    # Save best model
    joblib.dump(complex_model, 'models/credit_risk_model.pkl')
    print("\nModel saved to models/credit_risk_model.pkl")