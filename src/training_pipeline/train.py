import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from pathlib import Path
import joblib


DEFAULT_TRAIN_PATH = Path("data/processed/HouseTS_train_FE.csv")
DEFAULT_VALIDATION_PATH = Path("data/processed/HouseTS_validation_FE.csv")
OUTPUT_MODEL_PATH = Path("models")

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using RMSE, MAE, and R2 metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    return rmse, mae, r2

def train_model():
    """Train XGBoost regression model on training data and evaluate on validation data"""
    # Load training and validation data
    train_df = pd.read_csv(DEFAULT_TRAIN_PATH)
    validation_df = pd.read_csv(DEFAULT_VALIDATION_PATH)

    # Separate features and target
    target = "price"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_validation = validation_df.drop(columns=[target])
    y_validation = validation_df[target]

    # Initialize and train XGBoost model
    xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_validation)

    rmse, mae, r2 = evaluate_model(y_validation, y_pred)

    # save model
    OUTPUT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_file = OUTPUT_MODEL_PATH / "xgb_model.pkl"
    joblib.dump(xgb_model, model_file)
    print(f"Model saved to {model_file}")

    return xgb_model, (rmse, mae, r2)


if __name__ == "__main__":
    train_model()