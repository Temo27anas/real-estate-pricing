from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/")
PROCESSED_DIR = Path("data/processed/")
METRO_DATA_PATH = RAW_DIR / 'usmetros.csv'

city_mapping = {
    'Las Vegas-Henderson-Paradise': 'Las Vegas-Henderson-North Las Vegas',
    'Denver-Aurora-Lakewood': 'Denver-Aurora-Centennial',
    'Houston-The Woodlands-Sugar Land': 'Houston-Pasadena-The Woodlands',
    'Austin-Round Rock-Georgetown': 'Austin-Round Rock-San Marcos',
    'Miami-Fort Lauderdale-Pompano Beach': 'Miami-Fort Lauderdale-West Palm Beach',
    'San Francisco-Oakland-Berkeley': 'San Francisco-Oakland-Fremont',
    'DC_Metro': 'Washington-Arlington-Alexandria',
    'Atlanta-Sandy Springs-Alpharetta': 'Atlanta-Sandy Springs-Roswell'
}


def clean_merge_citydata(df):
    """Merge latitude and longitude data based on city/metro names."""

    if 'city_full' not in df.columns:
        raise ValueError("DataFrame must contain 'city_full' column.")
    
    metros_df = pd.read_csv(METRO_DATA_PATH)

    # Keep only the primary metro name before the comma
    metros_df['metro_full'] = metros_df['metro_full'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x)

    # Replace city names based on the mapping
    df['city_full'] = df['city_full'].replace(city_mapping)
    df = df.merge(metros_df[['metro_full', 'lat', 'lng']]
                  , how='left', left_on='city_full', right_on='metro_full')
    df = df.drop(columns=['metro_full'])

    # Check for missing lat/lng after merge
    missing = df[df['lat'].isnull()]['city_full'].unique()
    if len(missing) > 0:
        print(f"Warning: Missing lat/lng for cities: {missing}")
    else :
        print("Success: All cities merged with lat/lng data.")
    return df

def drop_duplicates(df):
    """Find and display duplicate rows in the DataFrame."""
    duplicates = df[df.duplicated()]
    duplicates_without_date = df[df.duplicated(subset=df.columns.difference(['date']))]

    print("Number of duplicate rows (all columns):", len(duplicates))
    print("Number of duplicate rows (excluding date):", len(duplicates_without_date))
    return df


def remove_outliers(df):
    """Remove outliers based on median_list_price threshold."""
    if "median_list_price" not in df.columns:
        raise ValueError("DataFrame must contain 'median_list_price' column.")

    print(f"Data shape before outlier removal: {df.shape}")
    df = df[df['median_list_price'] <= 16_000_000].copy()
    print(f"Data shape after outlier removal: {df.shape}")
    return df

def preprocess_split(split:str , metro_path=METRO_DATA_PATH, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    """Run preprocessing for a split and save to processed_dir."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_path = raw_dir / f"HouseTS_{split}.csv"
    df = pd.read_csv(split_path)
    df = clean_merge_citydata(df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    processed_path = processed_dir / f"HouseTS_{split}_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"Processed {split} data saved to {processed_path} - ({df.shape})")
    return df

if __name__ == "__main__":
    for split in ['train', 'validation', 'test']:
        preprocess_split(split)