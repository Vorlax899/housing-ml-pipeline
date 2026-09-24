import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_pipeline():
    print("1. Loading Data...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Target renaming
    df.rename(columns={'MedHouseVal': 'Price'}, inplace=True)
    
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")

    # Building Pipeline
    numeric_features = X.columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Model Training
    print("2. Training RandomForest Model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluation
    print("3. Evaluating Model Performance...")
    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Performance Metrics ---")
    print(f"MAE:  ${mae * 100000:.2f}")
    print(f"RMSE: ${rmse * 100000:.2f}")
    print(f"R² Score: {r2:.4f}\n")

    # Save Pipeline
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "housing_pipeline.joblib")
    joblib.dump(model_pipeline, model_path)
    print(f"4. Pipeline successfully saved to: {model_path}")

if __name__ == "__main__":
    run_pipeline()