import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def __():
    import polars as pl
    import modin.pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    
    # Initialize Ray for Modin if needed (Modin usually handles this, but good to be explicit if needed)
    # import ray
    # ray.init()
    return np, os, pd, pl, plt, sns


@app.cell
def __(pd, os):
    # Load data
    # Note: Replace with actual path after data download
    data_path = "../data/train.parquet"
    if os.path.exists(data_path):
        try:
            # Modin read_parquet
            df = pd.read_parquet(data_path)
            print("Data loaded successfully with Modin")
        except Exception as e:
            print(f"Error loading data: {e}")
            df = None
    else:
        print(f"Data file not found at {data_path}")
        df = None
    return df,


@app.cell
def __(df, plt, sns):
    if df is not None:
        # Sepsis Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x="SepsisLabel", data=df)
        plt.title("Distribution of Sepsis Labels")
        plt.show()
    return


@app.cell
def __(df):
    if df is not None:
        print(df.describe())
    return


if __name__ == "__main__":
    app.run()
