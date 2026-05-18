from ml.preprocessing import load_and_clean_data
from ml.rfm import create_rfm
from ml.split_data import split_dataset
from ml.eda import run_eda
from ml.evaluate import run_full_evaluation

print("======================================")
print("   TRAINING CUSTOMER SEGMENTATION")
print("======================================")

df_clean = load_and_clean_data()

rfm_raw, rfm_scaled = create_rfm(df_clean)

split_dataset(rfm_scaled, rfm_raw)

run_eda()

silhouette, best_k, cluster_summary = run_full_evaluation()

print("\n======================================")
print("TRAINING SELESAI")
print("Best K:", best_k)
print("Silhouette:", silhouette)
print("======================================")