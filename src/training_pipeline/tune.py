from pathlib import Path
import pandas as pd
import xgboost as xgb
import optuna
import joblib
import mlflow
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import sqrt


DEFAULT_TRAIN_PATH = Path("data/processed/HouseTS_train_FE.csv")
DEFAULT_VALIDATION_PATH = Path("data/processed/HouseTS_validation_FE.csv")
OUTPUT_MODEL_PATH = Path("models/xgb_best_model.pkl")

def tune_model():
    """Tune XGBoost regression model hyperparameters using Optuna and MLflow"""

    # load training and validation data
    train_df = pd.read_csv(DEFAULT_TRAIN_PATH)
    validation_df = pd.read_csv(DEFAULT_VALIDATION_PATH)

    # separate features and target
    target = "price"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_validation = validation_df.drop(columns=[target])
    y_validation = validation_df[target]

    def optuna_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run():
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(X_train, y_train)
            
            y_pred = xgb_model.predict(X_validation)
            mae = mean_absolute_error(y_validation, y_pred)
            rmse = sqrt(mean_squared_error(y_validation, y_pred))
            r2 = r2_score(y_validation, y_pred)

            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        return rmse
    
    # optimize hyperparameters
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=15)
    print("Best params:", study.best_trial.params)

    # training final model with best hyperparameters
    best_params = study.best_trial.params
    xgb_best_model = xgb.XGBRegressor(**best_params)
    xgb_best_model.fit(X_train, y_train)

    y_pred = xgb_best_model.predict(X_validation)
    mae = mean_absolute_error(y_validation, y_pred)
    rmse = sqrt(mean_squared_error(y_validation, y_pred))
    r2 = r2_score(y_validation, y_pred)
    print(f"Final Model Performance on Validation Set: RMSE={rmse}, MAE={mae}, R2={r2}")

    # log final model
    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(
        xgb_best_model,
        artifact_path="xgboost_model"
    )
    
    # save the best model
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_best_model, OUTPUT_MODEL_PATH)
    print(f"Best model saved to {OUTPUT_MODEL_PATH}")
        
    return xgb_best_model, (rmse, mae, r2)

if __name__ == "__main__":
    tune_model()
