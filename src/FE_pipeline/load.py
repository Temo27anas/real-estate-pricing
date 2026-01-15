import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/")

def load_and_split(raw_data_path, output_dir = RAW_DIR):
     
    df = pd.read_csv(raw_data_path)

    # Fix date column to datetime & sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")

    # Split into train, validation, test 
    train_ratio, validation_ratio, test_ratio = 0.7, 0.15, 0.15
    n = len(df)

    train_df = df.iloc[: int(n * train_ratio)]
    validation_df = df.iloc[int(n * train_ratio) + 1 : int(n * (train_ratio + validation_ratio))]
    test_df = df.iloc[int(n * (train_ratio + validation_ratio)) + 1 :]

    print(f"Train set shape: {train_df.shape}\n Date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}\n\n")
    print(f"Validation set shape: {validation_df.shape}\n Date range: {validation_df['date'].min().date()} to {validation_df['date'].max().date()}\n\n")
    print(f"Test set shape: {test_df.shape}\n Date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}\n\n")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "HouseTS_train.csv", index=False)
    validation_df.to_csv(output_dir / "HouseTS_validation.csv", index=False)
    test_df.to_csv(output_dir / "HouseTS_test.csv", index=False)

    print(f"Data splits saved to {output_dir}")

    return train_df, validation_df, test_df

if __name__ == "__main__":
    raw_data_path = RAW_DIR / "HouseTS.csv"
    load_and_split(raw_data_path, output_dir=RAW_DIR)
