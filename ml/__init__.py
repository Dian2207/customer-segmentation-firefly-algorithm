from .preprocessing import load_and_clean_data
from .rfm import create_rfm
from .split_data import split_dataset

def run_pipeline():
    print("Menjalankan pipeline data...")

    df = load_and_clean_data()
    rfm = create_rfm(df)
    split_dataset(rfm)

    print("Pipeline selesai!")