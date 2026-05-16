from app import create_app

from ml.preprocessing import load_and_clean_data
from ml.rfm import create_rfm
from ml.split_data import split_dataset

app = create_app()

print("=== MEMULAI PIPELINE DATA ===")

df_clean = load_and_clean_data()
rfm_scaled = create_rfm(df_clean)
split_dataset(rfm_scaled)

print("=== PIPELINE SELESAI ===")

if __name__ == "__main__":
    app.run(debug=True)