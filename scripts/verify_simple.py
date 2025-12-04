import polars as pl
import os
import sys

def verify_simple():
    print("Verifying Simple Data Loading...")
    data_path = "data/all/dataset.parquet"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return False

    try:
        # Test Polars only first
        df_pl = pl.read_parquet(data_path)
        print(f"Polars loaded data. Shape: {df_pl.shape}")
        return True
    except Exception as e:
        print(f"Polars verification failed: {e}")
        return False

if __name__ == "__main__":
    if verify_simple():
        print("Simple verification SUCCESS")
        sys.exit(0)
    else:
        print("Simple verification FAILED")
        sys.exit(1)
