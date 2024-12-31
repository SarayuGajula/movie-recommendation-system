import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File {path} not found.")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning steps:
      - Remove duplicates
      - Drop rows where 'title' is missing
      - (Optional) Any additional cleaning
    """
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['title'], inplace=True)

    logging.info(f"Dropped duplicates and missing titles. Shape went from {initial_shape} to {df.shape}.")
    return df

