import pandas as pd
from pathlib import Path
from joblib import load
import argparse, sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.FE_pipeline.preprocess import clean_merge_citydata, drop_duplicates, remove_outliers
from src.FE_pipeline.feature_eng import add_date_features, drop_columns
from src.training_pipeline.train import evaluate_model


DEFAULT_MODEL_PATH = project_root / "models/xgb_best_model.pkl"
DEFAULT_ZIP_ENCODER_PATH = project_root / "models/zipcode_target_encoder.joblib"
DEFAULT_TARGET_ENCODER_PATH = project_root / "models/city_full_target_encoder.joblib"
DEFAULT_OUTPUT_PATH = project_root / "output/predictions.csv"
DEFAULT_INPUT_DATA_PATH = project_root / "data/raw/HouseTS_test.csv"
DEFAULT_METRO_DATA_PATH = project_root / "data/raw/usmetros.csv"


# Get feature columns used during training
def get_feature_columns(model_path: Path) -> list:
    """Load the trained model and return the feature columns used during training"""
    model = load(model_path)
    return model.get_booster().feature_names

def predict(
        df: pd.DataFrame,
        ):
    
    # preprocess data
    df = clean_merge_citydata(df, metro_path=DEFAULT_METRO_DATA_PATH)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    # feature engineering
    if 'date' in df.columns:
        df = add_date_features(df)
    
    # load custom encoders
    if 'zipcode' in df.columns and Path.exists(DEFAULT_ZIP_ENCODER_PATH): # apply zipcode target encoding
        zip_encoder_data = load(DEFAULT_ZIP_ENCODER_PATH)
        zipcode_target = zip_encoder_data["zipcode_target"]
        global_mean = zip_encoder_data["global_mean"]
        df['zipcode_encoded'] = df['zipcode'].map(zipcode_target)
        df['zipcode_encoded'].fillna(global_mean, inplace=True)

    if 'city_full' in df.columns and Path.exists(DEFAULT_TARGET_ENCODER_PATH): # apply city target encoding
        city_te = load(DEFAULT_TARGET_ENCODER_PATH)
        df["city_full_enc"] = city_te.transform(df["city_full"])
    
    # drop unused columns
    df, _, _ = drop_columns(df, df, df)

    # drop price column if exists & use it for evaluation
    if 'price' in df.columns:
        print("Warning: 'price' column found in input data - Evaluation mode enabled.")
        true_prices = df['price']
        df = df.drop(columns=['price'])
        evaluate = True
    else:
        evaluate = False

    
    # check feature columns
    feature_columns = get_feature_columns(DEFAULT_MODEL_PATH)
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing required feature columns: {missing_cols}")
    
    # load model
    model = load(DEFAULT_MODEL_PATH)
    predictions = model.predict(df[feature_columns])

    # evaluate if true prices are available
    if evaluate:
        rmse, mae, r2 = evaluate_model(true_prices, predictions)
        print(f"Inference Performance: RMSE={rmse}, MAE={mae}, R2={r2}")

    # save predictions
    output_df = df.copy()
    output_df['predicted_price'] = predictions
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(DEFAULT_OUTPUT_PATH, index=False)
    print(f"Predictions saved to {DEFAULT_OUTPUT_PATH}")
    return output_df


if __name__ == "__main__":
    print("path:", DEFAULT_INPUT_DATA_PATH)
    
    parser = argparse.ArgumentParser(description="Run inference pipeline on input data")
    parser.add_argument(
        "--input_data_path",
        type=str,
        default=str(DEFAULT_INPUT_DATA_PATH),
        help="Path to input data CSV file for predictions",
    )
    args = parser.parse_args()
    input_data_path = Path(args.input_data_path)
    test_df = pd.read_csv(input_data_path)
    predict(test_df)



