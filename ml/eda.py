import os
import pandas as pd
import matplotlib.pyplot as plt


def run_eda():

    print("===================================")
    print("STEP 4 : EXPLORATORY DATA ANALYSIS")
    print("===================================")

    # =========================================
    # MEMBUAT FOLDER OUTPUT
    # =========================================
    output_folder = "app/static/images/eda"

    os.makedirs(output_folder, exist_ok=True)

    # =========================================
    # LOAD DATA RFM
    # EDA DILAKUKAN PADA DATA RFM
    # =========================================
    rfm = pd.read_csv(
        "data/processed/rfm.csv"
    )

    print("\nData RFM berhasil dibaca")
    print(rfm.head())

    # =========================================
    # INFO DATA
    # =========================================
    print("\nINFO DATA")
    print(rfm.info())

    # =========================================
    # DESKRIPSI STATISTIK
    # =========================================
    print("\nDESKRIPSI STATISTIK")
    print(rfm.describe())

    # =========================================
    # CEK NULL VALUE
    # =========================================
    print("\nNULL VALUE")
    print(rfm.isnull().sum())

    # =========================================
    # HISTOGRAM RECENCY
    # =========================================
    plt.figure(figsize=(8, 5))

    plt.hist(
        rfm['Recency'],
        bins=20
    )

    plt.title("Distribusi Recency")
    plt.xlabel("Recency")
    plt.ylabel("Jumlah Customer")

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/histogram_recency.png"
    )

    plt.close()

    print("Histogram Recency berhasil dibuat")

    # =========================================
    # HISTOGRAM FREQUENCY
    # =========================================
    plt.figure(figsize=(8, 5))

    plt.hist(
        rfm['Frequency'],
        bins=20
    )

    plt.title("Distribusi Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Jumlah Customer")

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/histogram_frequency.png"
    )

    plt.close()

    print("Histogram Frequency berhasil dibuat")

    # =========================================
    # HISTOGRAM MONETARY
    # =========================================
    plt.figure(figsize=(8, 5))

    plt.hist(
        rfm['Monetary'],
        bins=20
    )

    plt.title("Distribusi Monetary")
    plt.xlabel("Monetary")
    plt.ylabel("Jumlah Customer")

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/histogram_monetary.png"
    )

    plt.close()

    print("Histogram Monetary berhasil dibuat")

    # =========================================
    # CORRELATION MATRIX
    # =========================================
    correlation = rfm.corr()

    print("\nCORRELATION MATRIX")
    print(correlation)

    # =========================================
    # HEATMAP KORELASI RFM
    # =========================================
    plt.figure(figsize=(7, 6))

    plt.imshow(correlation)

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.colorbar()

    plt.title("Correlation Heatmap RFM")

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/heatmap_rfm.png"
    )

    plt.close()

    print("Heatmap Korelasi berhasil dibuat")

    print("\n===================================")
    print("EDA SELESAI")
    print("===================================")