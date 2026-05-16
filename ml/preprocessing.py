import pandas as pd

def load_and_clean_data():
    print("STEP 1: Load & Cleaning Data")

    df = pd.read_csv("data/raw/online_retail_II.csv")

    df = df.dropna()

    df = df.drop_duplicates()

    df = df[~df['Invoice'].astype(str).str.contains('C')]

    df.to_csv("data/processed/clean_data.csv", index=False)

    print("clean_data.csv berhasil dibuat")
    return df