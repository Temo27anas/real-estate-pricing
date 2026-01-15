import pandas as pd
from category_encoders import TargetEncoder
from pathlib import Path
from joblib import dump

MODELS_DIR = Path("models/")
OUTPUT_DIR = Path("data/processed/")


def add_date_features(df):
    """Extract year, month, quarter from date column"""
    # ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # extract features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    return df

def zip_encoder(train_df, test_df, validation_df, target='price'):
    """Apply target encoding to zipcode column"""
    # calculate mean price for each zipcode from training data
    zipcode_target = train_df.groupby('zipcode')[target].mean().to_dict()
    global_mean = train_df[target].mean()

    
    # apply encoding to all datasets
    train_df['zipcode_encoded'] = train_df['zipcode'].map(zipcode_target)
    test_df['zipcode_encoded'] = test_df['zipcode'].map(zipcode_target)
    validation_df['zipcode_encoded'] = validation_df['zipcode'].map(zipcode_target)
    
    # handle unseen zipcodes with global mean
    train_df['zipcode_encoded'].fillna(global_mean, inplace=True)
    validation_df['zipcode_encoded'].fillna(global_mean, inplace=True)
    test_df['zipcode_encoded'].fillna(global_mean, inplace=True)

    # save custom encoder
    dump(
    {
        "zipcode_target": zipcode_target,
        "global_mean": global_mean
    },
    MODELS_DIR / "zipcode_target_encoder.joblib"
    )
    
    return train_df, test_df, validation_df


def city_encoder(train_df, test_df, validation_df):
    """Apply target encoding to city_full column"""
    te = TargetEncoder(cols=["city_full"])

    # fit and transform on training data, transform on test and validation data
    train_df["city_full_enc"] = te.fit_transform(train_df["city_full"], train_df["price"])
    test_df["city_full_enc"] = te.transform(test_df["city_full"])
    validation_df["city_full_enc"] = te.transform(validation_df["city_full"])
    
    # save encoder
    dump(te, MODELS_DIR / "city_full_target_encoder.joblib")
    return train_df, test_df, validation_df


def drop_columns(train_df, test_df, validation_df):
    """Drop specified columns from all datasets"""
    columns = ['date', 'zipcode', 'city_full', 'city', 'median_sale_price']
    
    # drop columns
    train_df = train_df.drop(columns=columns)
    test_df = test_df.drop(columns=columns)
    validation_df = validation_df.drop(columns=columns)
    
    return train_df, test_df, validation_df


# --- Pipeline  ---
def feature_engineering_pipeline(train_df, test_df, validation_df, target='price', output_dir=OUTPUT_DIR):
    """Complete feature engineering pipeline"""
    # Add date features
    train_df = add_date_features(train_df)
    test_df = add_date_features(test_df)
    validation_df = add_date_features(validation_df)
    
    # Apply zipcode encoding
    train_df, test_df, validation_df = zip_encoder(train_df, test_df, validation_df, target)
    
    # Apply city_full encoding
    train_df, test_df, validation_df = city_encoder(train_df, test_df, validation_df)
    
    # Drop unnecessary columns
    train_df, test_df, validation_df = drop_columns(train_df, test_df, validation_df)
    
    # Save engineered datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "HouseTS_train_FE.csv", index=False)
    validation_df.to_csv(output_dir / "HouseTS_validation_FE.csv", index=False)
    test_df.to_csv(output_dir / "HouseTS_test_FE.csv", index=False)

    # print summary
    print(f"Feature engineered datasets saved to {output_dir}")
    print(f"Train set shape after FE: {train_df.shape}")
    print(f"Validation set shape after FE: {validation_df.shape}")
    print(f"Test set shape after FE: {test_df.shape}")

    return train_df, test_df, validation_df

if __name__ == "__main__":
    train_path = OUTPUT_DIR / "HouseTS_train_processed.csv"
    validation_path = OUTPUT_DIR / "HouseTS_validation_processed.csv"
    test_path = OUTPUT_DIR / "HouseTS_test_processed.csv"

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    feature_engineering_pipeline(train_df, test_df, validation_df, target='price', output_dir=OUTPUT_DIR)